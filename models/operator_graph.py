"""
models/operator_graph.py

Data structures for representing a DNN as a computation graph
of operators with tensor shapes, FLOPs, and memory requirements.

This is the central data structure the entire tool works on.
Users either:
  (a) Build it from a PyTorch model automatically
  (b) Define it manually via JSON/YAML config
  (c) Use the pre-built model zoo
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import json


class OperatorType(str, Enum):
    GEMM            = "gemm"
    LINEAR          = "linear"
    CONV2D          = "conv2d"
    ATTENTION       = "attention"
    LAYERNORM       = "layernorm"
    BATCHNORM       = "batchnorm"
    RELU            = "relu"
    GELU            = "gelu"
    SOFTMAX         = "softmax"
    ADD             = "add"
    MULTIPLY        = "multiply"
    EMBEDDING       = "embedding"
    POOLING         = "pooling"
    FFN             = "ffn"           # Combined FFN (2 linear + activation)
    DROPOUT         = "dropout"
    RESHAPE         = "reshape"
    TRANSPOSE       = "transpose"
    CONCAT          = "concat"
    OTHER           = "other"


@dataclass
class TensorShape:
    """Represents a tensor's shape and data type."""
    dims: List[int]
    dtype: str = "fp16"              # fp16 | bf16 | fp32 | fp8

    @property
    def num_elements(self) -> int:
        result = 1
        for d in self.dims:
            result *= d
        return result

    @property
    def bytes_per_element(self) -> int:
        mapping = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}
        return mapping.get(self.dtype.lower(), 2)

    @property
    def size_bytes(self) -> int:
        return self.num_elements * self.bytes_per_element

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 ** 2)

    def __str__(self) -> str:
        return f"[{', '.join(str(d) for d in self.dims)}] {self.dtype}"


@dataclass
class OperatorNode:
    """
    Represents a single operator in the computation graph.

    Fields are populated by the FLOPs counter and memory estimator.
    """
    op_id: str                                  # Unique identifier e.g. "layer_0_gemm"
    op_type: OperatorType
    layer_index: int                            # Which layer this belongs to
    layer_name: str = ""

    # Tensor shapes
    input_shapes: List[TensorShape] = field(default_factory=list)
    output_shapes: List[TensorShape] = field(default_factory=list)
    weight_shapes: List[TensorShape] = field(default_factory=list)

    # Operator-specific parameters
    params: Dict = field(default_factory=dict)
    # e.g. for CONV2D: {"kernel": [3,3], "stride": 1, "padding": 1, "groups": 1}
    # e.g. for GEMM: {"M": 512, "K": 1024, "N": 512}
    # e.g. for ATTENTION: {"num_heads": 16, "head_dim": 64, "seq_len": 512}

    # Computed values (filled by FLOPs counter)
    flops_forward: float = 0.0                  # FLOPs for forward pass
    flops_backward: float = 0.0                 # FLOPs for backward pass

    # Memory requirements (filled by memory module)
    activation_bytes: int = 0                   # Input + output tensor bytes
    weight_bytes: int = 0                       # Weight tensor bytes
    gradient_bytes: int = 0                     # Gradient tensor bytes
    working_set_bytes: int = 0                  # Total on-chip working set

    # Graph connectivity
    predecessor_ids: List[str] = field(default_factory=list)
    successor_ids: List[str] = field(default_factory=list)

    # Fusion eligibility
    fusable_with_next: bool = False             # Can be fused into streaming pipeline
    fusion_group_id: Optional[int] = None       # Which fusion group this belongs to

    @property
    def total_flops(self) -> float:
        return self.flops_forward + self.flops_backward

    @property
    def total_bytes_accessed(self) -> int:
        """Total bytes read + written for this operator."""
        return (
            sum(s.size_bytes for s in self.input_shapes)
            + sum(s.size_bytes for s in self.output_shapes)
            + sum(s.size_bytes for s in self.weight_shapes)
        )

    @property
    def arithmetic_intensity(self) -> float:
        """
        Arithmetic Intensity = FLOPs / bytes accessed.
        This is the C3 metric (AI) from the research plan.
        """
        total_bytes = self.total_bytes_accessed
        if total_bytes == 0:
            return 0.0
        return self.flops_forward / total_bytes

    def is_compute_bound(self, ridge_point: float) -> bool:
        """
        Returns True if operator is compute-bound on hardware
        with given ridge_point (FLOP/byte).
        """
        return self.arithmetic_intensity > ridge_point

    def __str__(self) -> str:
        return (
            f"OperatorNode(id={self.op_id}, type={self.op_type.value}, "
            f"layer={self.layer_index}, "
            f"flops_fwd={self.flops_forward:.2e}, "
            f"AI={self.arithmetic_intensity:.2f} FLOP/byte)"
        )


@dataclass
class ComputationGraph:
    """
    Full computation graph of a DNN model.
    Contains all operators in topological order.
    """
    model_name: str
    model_type: str                             # 'cnn' | 'transformer' | 'mlp'
    operators: List[OperatorNode] = field(default_factory=list)

    # Training configuration
    batch_size: int = 32
    dtype: str = "fp16"
    num_devices: int = 1

    # Model-level stats (computed after graph is built)
    total_params: int = 0
    total_flops_forward: float = 0.0
    total_flops_backward: float = 0.0
    total_weight_bytes: int = 0
    total_activation_bytes: int = 0
    total_gradient_bytes: int = 0

    def add_operator(self, op: OperatorNode):
        self.operators.append(op)

    def compute_totals(self):
        """Aggregate model-level statistics from all operators."""
        self.total_flops_forward = sum(op.flops_forward for op in self.operators)
        self.total_flops_backward = sum(op.flops_backward for op in self.operators)
        self.total_weight_bytes = sum(op.weight_bytes for op in self.operators)
        self.total_activation_bytes = sum(
            sum(s.size_bytes for s in op.output_shapes) for op in self.operators
        )
        # Gradients = same size as weights + activations needed for backward
        self.total_gradient_bytes = self.total_weight_bytes

    @property
    def total_flops(self) -> float:
        return self.total_flops_forward + self.total_flops_backward

    @property
    def total_params_M(self) -> float:
        if self.total_weight_bytes > 0:
            bytes_per_param = 2 if self.dtype in ["fp16", "bf16"] else 4
            return self.total_weight_bytes / (bytes_per_param * 1e6)
        return self.total_params / 1e6

    def get_operators_by_type(self, op_type: OperatorType) -> List[OperatorNode]:
        return [op for op in self.operators if op.op_type == op_type]

    def get_fusion_groups(self) -> Dict[int, List[OperatorNode]]:
        """Return operators grouped by their fusion group ID."""
        groups: Dict[int, List[OperatorNode]] = {}
        for op in self.operators:
            if op.fusion_group_id is not None:
                groups.setdefault(op.fusion_group_id, []).append(op)
        return groups

    def summary(self) -> str:
        self.compute_totals()
        lines = [
            f"ComputationGraph: {self.model_name} ({self.model_type})",
            f"  Operators       : {len(self.operators)}",
            f"  Batch size      : {self.batch_size}",
            f"  Data type       : {self.dtype}",
            f"  Total FLOPs fwd : {self.total_flops_forward:.3e}",
            f"  Total FLOPs bwd : {self.total_flops_backward:.3e}",
            f"  Total FLOPs     : {self.total_flops:.3e}",
            f"  Weight bytes    : {self.total_weight_bytes / 1e9:.3f} GB",
            f"  Activation bytes: {self.total_activation_bytes / 1e9:.3f} GB",
            f"  Gradient bytes  : {self.total_gradient_bytes / 1e9:.3f} GB",
        ]
        # Operator type distribution
        type_counts: Dict[str, int] = {}
        for op in self.operators:
            type_counts[op.op_type.value] = type_counts.get(op.op_type.value, 0) + 1
        lines.append("  Operator types  :")
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {t:<20} {count}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        self.compute_totals()
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "batch_size": self.batch_size,
            "dtype": self.dtype,
            "num_devices": self.num_devices,
            "total_flops_forward": self.total_flops_forward,
            "total_flops_backward": self.total_flops_backward,
            "total_weight_bytes": self.total_weight_bytes,
            "total_activation_bytes": self.total_activation_bytes,
            "total_gradient_bytes": self.total_gradient_bytes,
            "num_operators": len(self.operators),
        }
