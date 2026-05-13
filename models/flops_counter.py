"""
models/flops_counter.py

Computes FLOPs(l,j) for every operator type analytically.
All formulas derived from first principles as discussed in research plan.

Rules:
  - FLOPs = 2 * MACs  (each MAC = 1 multiply + 1 add)
  - Forward FLOPs computed from tensor shapes
  - Backward FLOPs = 2 * forward FLOPs (standard approximation)
  - Memory bytes computed from tensor shapes and dtype
"""

from typing import List, Dict, Tuple, Optional
from models.operator_graph import OperatorNode, OperatorType, TensorShape, ComputationGraph


class FLOPsCounter:
    """
    Computes FLOPs and memory bytes for each operator node
    in a computation graph.
    """

    BYTES_PER_DTYPE = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "fp8":  1,
        "int8": 1,
    }

    def annotate_graph(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Annotate all operators in the graph with FLOPs and memory values.
        Returns the same graph with all values populated.
        """
        for op in graph.operators:
            self._annotate_operator(op, graph.dtype)
        graph.compute_totals()
        return graph

    def _annotate_operator(self, op: OperatorNode, dtype: str):
        """Dispatch to the correct FLOPs formula based on operator type."""
        dispatch = {
            OperatorType.GEMM:       self._gemm_flops,
            OperatorType.LINEAR:     self._linear_flops,
            OperatorType.CONV2D:     self._conv2d_flops,
            OperatorType.ATTENTION:  self._attention_flops,
            OperatorType.FFN:        self._ffn_flops,
            OperatorType.LAYERNORM:  self._layernorm_flops,
            OperatorType.BATCHNORM:  self._batchnorm_flops,
            OperatorType.RELU:       self._elementwise_flops,
            OperatorType.GELU:       self._gelu_flops,
            OperatorType.SOFTMAX:    self._softmax_flops,
            OperatorType.ADD:        self._elementwise_flops,
            OperatorType.MULTIPLY:   self._elementwise_flops,
            OperatorType.EMBEDDING:  self._embedding_flops,
            OperatorType.POOLING:    self._pooling_flops,
            OperatorType.DROPOUT:    self._dropout_flops,
            OperatorType.RESHAPE:    self._zero_flops,
            OperatorType.TRANSPOSE:  self._zero_flops,
            OperatorType.CONCAT:     self._zero_flops,
            OperatorType.OTHER:      self._zero_flops,
        }
        fn = dispatch.get(op.op_type, self._zero_flops)
        fn(op)
        self._compute_memory_bytes(op, dtype)

    # ─────────────────────────────────────────────────────────
    # COMPUTE (FLOPs) FORMULAS
    # ─────────────────────────────────────────────────────────

    def _gemm_flops(self, op: OperatorNode):
        """
        GEMM: A(M×K) × B(K×N) = C(M×N)
        FLOPs = 2 * M * K * N
        """
        p = op.params
        M = p.get("M", 1)
        K = p.get("K", 1)
        N = p.get("N", 1)
        BS = p.get("batch_size", 1)
        op.flops_forward = 2.0 * BS * M * K * N
        op.flops_backward = 2.0 * op.flops_forward

    def _linear_flops(self, op: OperatorNode):
        """
        Linear/FC: (BS × S × d_in) × (d_in × d_out)
        FLOPs = 2 * BS * S * d_in * d_out  (+bias: BS*S*d_out)
        """
        p = op.params
        BS  = p.get("batch_size", 1)
        S   = p.get("seq_len", 1)
        din = p.get("in_features", 1)
        dout= p.get("out_features", 1)
        op.flops_forward = 2.0 * BS * S * din * dout + BS * S * dout
        op.flops_backward = 2.0 * op.flops_forward

    def _conv2d_flops(self, op: OperatorNode):
        """
        Conv2D: Input(BS×Cin×H×W), Kernel(Cout×Cin×Kh×Kw)
        FLOPs = 2 * BS * Cout * Hout * Wout * Cin * Kh * Kw
        """
        p = op.params
        BS   = p.get("batch_size", 1)
        Cin  = p.get("in_channels", 1)
        H    = p.get("input_height", 1)
        W    = p.get("input_width", 1)
        Cout = p.get("out_channels", 1)
        Kh   = p.get("kernel_h", 3)
        Kw   = p.get("kernel_w", 3)
        stride = p.get("stride", 1)
        padding = p.get("padding", 0)
        groups = p.get("groups", 1)

        Hout = (H - Kh + 2 * padding) // stride + 1
        Wout = (W - Kw + 2 * padding) // stride + 1

        # Adjust for grouped convolution
        Cin_per_group = Cin // groups
        op.flops_forward = (
            2.0 * BS * Cout * Hout * Wout * Cin_per_group * Kh * Kw
        )
        op.flops_backward = 2.0 * op.flops_forward

        # Store output dims for memory calculation
        op.params["output_height"] = Hout
        op.params["output_width"] = Wout

    def _attention_flops(self, op: OperatorNode):
        """
        Multi-head Self-Attention.

        Components:
          QKV projections : 3 * 2 * BS * S * d * d = 6 * BS * S * d^2
          QK scores       : 2 * BS * S^2 * d
          Softmax         : 5 * BS * h * S * S  (approx)
          AV weighted sum : 2 * BS * S^2 * d
          Output proj     : 2 * BS * S * d^2

        Total ≈ BS * S * (8*d^2 + 4*S*d)
        """
        p = op.params
        BS  = p.get("batch_size", 1)
        S   = p.get("seq_len", 512)
        d   = p.get("hidden_dim", 1024)
        h   = p.get("num_heads", 16)

        flops_qkv   = 6.0 * BS * S * d * d
        flops_qk    = 2.0 * BS * S * S * d
        flops_softmax = 5.0 * BS * h * S * S
        flops_av    = 2.0 * BS * S * S * d
        flops_out   = 2.0 * BS * S * d * d

        op.flops_forward = (
            flops_qkv + flops_qk + flops_softmax + flops_av + flops_out
        )
        op.flops_backward = 2.0 * op.flops_forward

    def _ffn_flops(self, op: OperatorNode):
        """
        FFN block: two linear layers with hidden_dim → ffn_dim → hidden_dim
        Default ffn_dim = 4 * hidden_dim
        FLOPs = 2 * (2 * BS * S * d * ffn_dim)
              = 16 * BS * S * d^2  (when ffn_dim = 4d)
        """
        p = op.params
        BS       = p.get("batch_size", 1)
        S        = p.get("seq_len", 512)
        d        = p.get("hidden_dim", 1024)
        ffn_dim  = p.get("ffn_dim", 4 * d)

        op.flops_forward = (
            2.0 * (2.0 * BS * S * d * ffn_dim)  # two linear layers
            + BS * S * ffn_dim                    # activation (GeLU approx)
        )
        op.flops_backward = 2.0 * op.flops_forward

    def _layernorm_flops(self, op: OperatorNode):
        """
        LayerNorm: mean(d) + var(2d) + normalize(2d) + scale+shift(2d) = 7d
        FLOPs = 7 * BS * S * d
        """
        p = op.params
        BS = p.get("batch_size", 1)
        S  = p.get("seq_len", 1)
        d  = p.get("hidden_dim", p.get("num_features", 1))
        op.flops_forward  = 7.0 * BS * S * d
        op.flops_backward = 2.0 * op.flops_forward

    def _batchnorm_flops(self, op: OperatorNode):
        """
        BatchNorm: similar to LayerNorm but over batch dimension
        FLOPs ≈ 5 * BS * C * H * W
        """
        p = op.params
        BS = p.get("batch_size", 1)
        C  = p.get("num_channels", p.get("num_features", 1))
        H  = p.get("height", 1)
        W  = p.get("width", 1)
        op.flops_forward  = 5.0 * BS * C * H * W
        op.flops_backward = 2.0 * op.flops_forward

    def _elementwise_flops(self, op: OperatorNode):
        """
        Element-wise ops (ReLU, Add, Multiply): 1 FLOP per element
        """
        if op.input_shapes:
            n_elements = op.input_shapes[0].num_elements
        else:
            p = op.params
            n_elements = (
                p.get("batch_size", 1)
                * p.get("seq_len", 1)
                * p.get("hidden_dim", 1)
            )
        op.flops_forward  = float(n_elements)
        op.flops_backward = float(n_elements)

    def _gelu_flops(self, op: OperatorNode):
        """
        GeLU: approximately 8 FLOPs per element (exp, multiply, add, etc.)
        """
        self._elementwise_flops(op)
        op.flops_forward  *= 8.0
        op.flops_backward *= 8.0

    def _softmax_flops(self, op: OperatorNode):
        """
        Softmax over last dimension of size N:
        3 passes: max subtraction + exp + normalization
        FLOPs = 3 * total_elements_in_output
        """
        if op.input_shapes:
            n_elements = op.input_shapes[0].num_elements
        else:
            p = op.params
            BS = p.get("batch_size", 1)
            h  = p.get("num_heads", 1)
            S  = p.get("seq_len", 1)
            n_elements = BS * h * S * S
        op.flops_forward  = 3.0 * n_elements
        op.flops_backward = 2.0 * op.flops_forward

    def _embedding_flops(self, op: OperatorNode):
        """
        Embedding lookup: essentially a memory access, minimal FLOPs.
        FLOPs ≈ 0 (table lookup only)
        """
        op.flops_forward  = 0.0
        op.flops_backward = 0.0

    def _pooling_flops(self, op: OperatorNode):
        """
        Pooling (max/avg): 1 comparison per element in kernel window.
        """
        p = op.params
        BS  = p.get("batch_size", 1)
        C   = p.get("channels", 1)
        Hout = p.get("output_height", 1)
        Wout = p.get("output_width", 1)
        Kh   = p.get("kernel_h", 2)
        Kw   = p.get("kernel_w", 2)
        op.flops_forward  = float(BS * C * Hout * Wout * Kh * Kw)
        op.flops_backward = float(BS * C * Hout * Wout)

    def _dropout_flops(self, op: OperatorNode):
        """Dropout: 1 comparison per element during training."""
        self._elementwise_flops(op)

    def _zero_flops(self, op: OperatorNode):
        """Ops with no meaningful FLOPs (reshape, transpose, concat)."""
        op.flops_forward  = 0.0
        op.flops_backward = 0.0

    # ─────────────────────────────────────────────────────────
    # MEMORY BYTES
    # ─────────────────────────────────────────────────────────

    def _compute_memory_bytes(self, op: OperatorNode, dtype: str):
        """
        Compute activation, weight, and gradient bytes for an operator.
        These feed directly into the memory module metrics.
        """
        bpe = self.BYTES_PER_DTYPE.get(dtype.lower(), 2)

        # Activation bytes = input tensors + output tensors
        input_bytes  = sum(s.size_bytes for s in op.input_shapes)
        output_bytes = sum(s.size_bytes for s in op.output_shapes)
        op.activation_bytes = input_bytes + output_bytes

        # Weight bytes
        op.weight_bytes = sum(s.size_bytes for s in op.weight_shapes)

        # Gradient bytes = same size as weights (one gradient per weight)
        # + output gradients for backward pass
        op.gradient_bytes = op.weight_bytes + output_bytes

        # Working set = everything needed simultaneously on-chip
        op.working_set_bytes = (
            input_bytes          # inputs needed to compute
            + output_bytes       # outputs being produced
            + op.weight_bytes    # weights needed for computation
            # Note: gradient storage NOT included in working set
            # (gradients are accumulated, not all needed at once)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def compute_flops(graph: ComputationGraph) -> ComputationGraph:
    """Annotate an entire computation graph with FLOPs and memory bytes."""
    counter = FLOPsCounter()
    return counter.annotate_graph(graph)
