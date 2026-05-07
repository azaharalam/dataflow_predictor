"""
core/memory_module.py

Memory Module: Predicts the three memory metrics.

  M1 — MBU: Memory Bandwidth Utilization
  M2 — SRE: Scratchpad Reuse Efficiency
  M3 — OMT: Off-Chip Memory Traffic

All formulas are fully analytical.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math

from ..models.operator_graph import ComputationGraph, OperatorNode
from ..hardware.base_hardware import HardwareSpec


@dataclass
class MemoryTierUsage:
    """Memory usage breakdown per memory tier."""
    tier_name: str
    capacity_gb: float
    bandwidth_gbps: float
    bytes_accessed: int = 0
    utilization: float = 0.0              # fraction of bandwidth used


@dataclass
class MemoryMetrics:
    """All memory module metrics."""
    model_name: str
    hardware_name: str
    batch_size: int
    dtype: str

    # M1 — MBU: Memory Bandwidth Utilization
    total_bytes_accessed: int = 0
    total_bytes_onchip: int = 0
    total_bytes_offchip: int = 0
    peak_bandwidth_gbps: float = 0.0
    achieved_bandwidth_gbps: float = 0.0
    mbu: float = 0.0                      # [0, 1]

    # M2 — SRE: Scratchpad Reuse Efficiency
    theoretical_min_bytes: int = 0        # Min bytes if perfect reuse
    actual_bytes_served_onchip: int = 0   # Bytes actually from scratchpad
    sre: float = 0.0                      # [0, 1]
    # Per-operator reuse factors
    reuse_factor_per_operator: Dict[str, float] = field(default_factory=dict)

    # M3 — OMT: Off-Chip Memory Traffic
    minimum_offchip_bytes: int = 0        # Unavoidable off-chip traffic
    actual_offchip_bytes: int = 0         # Actual predicted off-chip traffic
    spill_bytes: int = 0                  # Spill from on-chip → off-chip
    otaf: float = 1.0                     # Off-chip Traffic Amplification Factor
    # Breakdown
    offchip_weights_bytes: int = 0
    offchip_activations_bytes: int = 0
    offchip_gradients_bytes: int = 0

    # Memory tier utilization
    tier_usage: List[MemoryTierUsage] = field(default_factory=list)

    # Total memory demand
    total_weight_bytes: int = 0
    total_activation_bytes: int = 0
    total_gradient_bytes: int = 0
    total_optimizer_bytes: int = 0        # Adam: 2x weights for m,v states
    peak_memory_demand_gb: float = 0.0
    fits_on_chip: bool = False
    recommended_batch_size: int = 0

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  MEMORY MODULE RESULTS",
            f"  Model   : {self.model_name}",
            f"  Hardware: {self.hardware_name}",
            f"{'='*60}",
            f"\n[M1] Memory Bandwidth Utilization (MBU)",
            f"  Total bytes accessed        : {self.total_bytes_accessed/1e9:.3f} GB",
            f"    On-chip                   : {self.total_bytes_onchip/1e9:.3f} GB",
            f"    Off-chip                  : {self.total_bytes_offchip/1e9:.3f} GB",
            f"  Peak bandwidth              : {self.peak_bandwidth_gbps:.1f} GB/s",
            f"  Achieved bandwidth          : {self.achieved_bandwidth_gbps:.1f} GB/s",
            f"  MBU                         : {self.mbu*100:.2f}%",
            f"\n[M2] Scratchpad Reuse Efficiency (SRE)",
            f"  Theoretical min bytes       : {self.theoretical_min_bytes/1e9:.3f} GB",
            f"  Bytes served from scratchpad: {self.actual_bytes_served_onchip/1e9:.3f} GB",
            f"  SRE                         : {self.sre*100:.2f}%",
            f"\n[M3] Off-Chip Memory Traffic (OMT)",
            f"  Minimum off-chip bytes      : {self.minimum_offchip_bytes/1e9:.3f} GB",
            f"  Actual off-chip bytes       : {self.actual_offchip_bytes/1e9:.3f} GB",
            f"  Spill bytes                 : {self.spill_bytes/1e9:.3f} GB",
            f"  OTAF (amplification factor) : {self.otaf:.3f}x",
            f"    Weights off-chip          : {self.offchip_weights_bytes/1e9:.3f} GB",
            f"    Activations off-chip      : {self.offchip_activations_bytes/1e9:.3f} GB",
            f"    Gradients off-chip        : {self.offchip_gradients_bytes/1e9:.3f} GB",
            f"\n[Memory Capacity]",
            f"  Weights                     : {self.total_weight_bytes/1e9:.3f} GB",
            f"  Activations                 : {self.total_activation_bytes/1e9:.3f} GB",
            f"  Gradients                   : {self.total_gradient_bytes/1e9:.3f} GB",
            f"  Optimizer states (Adam)     : {self.total_optimizer_bytes/1e9:.3f} GB",
            f"  Peak memory demand          : {self.peak_memory_demand_gb:.3f} GB",
            f"  Fits on-chip                : {'YES' if self.fits_on_chip else 'NO'}",
            f"  Recommended batch size      : {self.recommended_batch_size}",
        ]
        return "\n".join(lines)


class MemoryModule:
    """
    Computes all three memory metrics for a DNN model on a given platform.
    """

    # Reuse factor constants by operator type (fraction of accesses from scratchpad)
    # These are derived from the execution model of each accelerator type
    REUSE_FACTORS_DATAFLOW = {
        "gemm": 0.85,        # Weights cached in PMU between tiles
        "linear": 0.80,
        "conv2d": 0.75,      # Partial reuse depending on kernel size
        "attention": 0.65,   # QKV weights reused, but attention scores large
        "ffn": 0.80,
        "layernorm": 0.40,   # Memory-bound, low reuse
        "batchnorm": 0.40,
        "relu": 0.50,
        "gelu": 0.50,
        "softmax": 0.30,     # Input matrix read twice effectively
        "add": 0.60,
        "multiply": 0.60,
        "embedding": 0.20,   # Random access, poor reuse
        "pooling": 0.50,
        "dropout": 0.50,
        "other": 0.40,
    }

    REUSE_FACTORS_BSP = {
        # Graphcore: per-tile scratchpad, reuse within BSP compute phase
        "gemm": 0.90,
        "linear": 0.85,
        "conv2d": 0.80,
        "attention": 0.60,
        "ffn": 0.82,
        "layernorm": 0.35,
        "batchnorm": 0.35,
        "relu": 0.70,         # Same tile can keep activation
        "gelu": 0.65,
        "softmax": 0.25,
        "add": 0.70,
        "multiply": 0.70,
        "embedding": 0.15,
        "pooling": 0.55,
        "dropout": 0.55,
        "other": 0.35,
    }

    REUSE_FACTORS_WAFER = {
        # Cerebras: all weights on-chip → near-perfect weight reuse
        "gemm": 0.98,
        "linear": 0.97,
        "conv2d": 0.95,
        "attention": 0.90,
        "ffn": 0.96,
        "layernorm": 0.80,   # Still memory-bound but on-chip
        "batchnorm": 0.80,
        "relu": 0.90,
        "gelu": 0.88,
        "softmax": 0.75,
        "add": 0.85,
        "multiply": 0.85,
        "embedding": 0.70,
        "pooling": 0.75,
        "dropout": 0.75,
        "other": 0.70,
    }

    def __init__(self, hardware: HardwareSpec):
        self.hw = hardware
        self.reuse_factors = self._get_reuse_factors()

    def _get_reuse_factors(self) -> Dict[str, float]:
        model = self.hw.execution_model
        if model == "wafer_scale":
            return self.REUSE_FACTORS_WAFER
        elif model == "bsp":
            return self.REUSE_FACTORS_BSP
        else:
            return self.REUSE_FACTORS_DATAFLOW

    def compute(self, graph: ComputationGraph,
                t_compute_s: float = 1.0) -> MemoryMetrics:
        """
        Main entry point.

        Args:
            graph: Annotated ComputationGraph
            t_compute_s: Compute time from ComputeModule (used for MBU)

        Returns:
            MemoryMetrics with all three metrics populated
        """
        metrics = MemoryMetrics(
            model_name=graph.model_name,
            hardware_name=self.hw.name,
            batch_size=graph.batch_size,
            dtype=graph.dtype,
        )

        self._compute_memory_demand(graph, metrics)
        self._compute_sre(graph, metrics)
        self._compute_omt(graph, metrics)
        self._compute_mbu(graph, metrics, t_compute_s)
        self._compute_batch_size_recommendation(graph, metrics)

        return metrics

    # ─────────────────────────────────────────────────────────
    # Memory Demand (prerequisite for all metrics)
    # ─────────────────────────────────────────────────────────

    def _compute_memory_demand(self, graph: ComputationGraph, metrics: MemoryMetrics):
        """
        Compute total memory demand for weights, activations, gradients,
        and optimizer states.
        """
        metrics.total_weight_bytes     = graph.total_weight_bytes
        metrics.total_activation_bytes = graph.total_activation_bytes
        metrics.total_gradient_bytes   = graph.total_gradient_bytes

        # Optimizer state memory (Adam = 2 copies of weights: m and v)
        metrics.total_optimizer_bytes  = 2 * graph.total_weight_bytes

        # Peak memory = weights + activations + gradients + optimizer
        peak_bytes = (
            metrics.total_weight_bytes
            + metrics.total_activation_bytes
            + metrics.total_gradient_bytes
            + metrics.total_optimizer_bytes
        )
        metrics.peak_memory_demand_gb = peak_bytes / 1e9

        # Check if model fits on-chip
        total_onchip = self.hw.total_onchip_sram_gb * 1e9
        metrics.fits_on_chip = peak_bytes <= total_onchip

    # ─────────────────────────────────────────────────────────
    # M2: Scratchpad Reuse Efficiency
    # ─────────────────────────────────────────────────────────

    def _compute_sre(self, graph: ComputationGraph, metrics: MemoryMetrics):
        """
        M2 — Scratchpad Reuse Efficiency.

        SRE = bytes_served_from_scratchpad / total_bytes_required

        For each operator:
          reuse_factor = f(op_type, execution_model, working_set / onchip_budget)
          bytes_from_scratchpad = reuse_factor * total_bytes_accessed(op)

        The reuse factor is reduced when:
          working_set > on-chip budget (data must spill)
        """
        total_bytes_required = 0
        total_bytes_onchip   = 0
        onchip_capacity      = self.hw.total_onchip_sram_gb * 1e9

        for op in graph.operators:
            total_bytes = op.total_bytes_accessed
            if total_bytes == 0:
                metrics.reuse_factor_per_operator[op.op_id] = 1.0
                continue

            # Base reuse factor from execution model
            base_rf = self.reuse_factors.get(op.op_type.value, 0.5)

            # Capacity-adjusted reuse factor
            # If working set > on-chip capacity, reuse is reduced proportionally
            if op.working_set_bytes > onchip_capacity and onchip_capacity > 0:
                capacity_ratio = onchip_capacity / op.working_set_bytes
                # Reuse factor scales linearly with how much fits on-chip
                adjusted_rf = base_rf * capacity_ratio
            else:
                adjusted_rf = base_rf

            # For Cerebras weight streaming mode:
            # Activations must come from off-chip (MemoryX) but weights are on-chip
            if self.hw.execution_model == "wafer_scale" and self.hw.supports_weight_streaming:
                # If model doesn't fit on chip, activations stream from MemoryX
                if not metrics.fits_on_chip:
                    act_bytes  = sum(s.size_bytes for s in op.output_shapes)
                    wt_bytes   = op.weight_bytes
                    if total_bytes > 0:
                        # Weights still get good reuse, activations do not
                        wt_rf  = min(base_rf, 0.98)
                        act_rf = 0.20   # activations stream in from MemoryX
                        adjusted_rf = (
                            (wt_bytes * wt_rf + act_bytes * act_rf)
                            / total_bytes
                            if total_bytes > 0 else base_rf
                        )

            metrics.reuse_factor_per_operator[op.op_id] = adjusted_rf
            bytes_from_scratchpad = int(adjusted_rf * total_bytes)

            total_bytes_required += total_bytes
            total_bytes_onchip   += bytes_from_scratchpad

        metrics.theoretical_min_bytes      = total_bytes_required
        metrics.actual_bytes_served_onchip = total_bytes_onchip
        metrics.total_bytes_accessed       = total_bytes_required
        metrics.total_bytes_onchip         = total_bytes_onchip
        metrics.total_bytes_offchip        = total_bytes_required - total_bytes_onchip

        metrics.sre = (
            total_bytes_onchip / total_bytes_required
            if total_bytes_required > 0 else 0.0
        )

    # ─────────────────────────────────────────────────────────
    # M3: Off-Chip Memory Traffic
    # ─────────────────────────────────────────────────────────

    def _compute_omt(self, graph: ComputationGraph, metrics: MemoryMetrics):
        """
        M3 — Off-Chip Memory Traffic.

        Minimum off-chip traffic = unavoidable bytes:
          - Input batch data (must come from storage)
          - Output predictions (must go to storage)
          - Gradient updates (must be written back)

        Actual off-chip traffic = minimum + spill:
          Spill = (1 - SRE) * total_bytes_accessed

        OTAF = Actual / Minimum   (amplification factor, ≥ 1.0)
        """
        # Minimum necessary off-chip traffic
        # Weights must be loaded once per iteration (even with good reuse,
        # they must be fetched initially)
        # For Cerebras layer-pipelined: weights stay on chip → min ≈ activation only
        # For weight streaming mode: weights must stream in every forward pass

        if self.hw.execution_model == "wafer_scale" and not self.hw.supports_weight_streaming:
            # All weights resident on chip — only activations and gradients off-chip
            min_offchip = (
                metrics.total_activation_bytes  # Input/output activations
                + metrics.total_gradient_bytes  # Gradient checkpointing
            )
        elif self.hw.execution_model == "wafer_scale" and self.hw.supports_weight_streaming:
            if metrics.fits_on_chip:
                min_offchip = metrics.total_activation_bytes
            else:
                # Weight streaming: all weights stream from MemoryX
                min_offchip = (
                    metrics.total_weight_bytes      # Weights streamed in
                    + metrics.total_activation_bytes
                    + metrics.total_gradient_bytes
                )
        else:
            # Standard dataflow/BSP: weights + activations + gradients
            # At minimum, each must cross the off-chip boundary once
            min_offchip = (
                metrics.total_weight_bytes
                + metrics.total_activation_bytes
                + metrics.total_gradient_bytes
            )

        # Spill bytes = data that had to leave on-chip and come back
        # = (1 - SRE) * total_accessed   minus what was already in min_offchip
        spill_bytes = max(0, int(
            (1.0 - metrics.sre) * metrics.total_bytes_accessed
            - min_offchip
        ))

        actual_offchip = min_offchip + spill_bytes

        # Breakdown by data category
        # Weights: if SRE < 1, some weight accesses miss scratchpad
        wt_bytes = metrics.total_weight_bytes
        act_bytes = metrics.total_activation_bytes
        grad_bytes = metrics.total_gradient_bytes

        if self.hw.execution_model == "wafer_scale" and not metrics.fits_on_chip:
            offchip_wt  = int(wt_bytes * (1.0 - 0.95))  # Small spill even streaming
            offchip_act = act_bytes
            offchip_grad = grad_bytes
        else:
            weight_rf   = sum(
                v for k, v in metrics.reuse_factor_per_operator.items()
            ) / max(len(metrics.reuse_factor_per_operator), 1)
            offchip_wt   = int(wt_bytes   * (1.0 - weight_rf))
            offchip_act  = int(act_bytes  * (1.0 - weight_rf * 0.7))
            offchip_grad = grad_bytes  # Gradients always written off-chip

        metrics.minimum_offchip_bytes      = min_offchip
        metrics.actual_offchip_bytes       = actual_offchip
        metrics.spill_bytes                = spill_bytes
        metrics.offchip_weights_bytes      = offchip_wt
        metrics.offchip_activations_bytes  = offchip_act
        metrics.offchip_gradients_bytes    = offchip_grad
        metrics.otaf = (
            actual_offchip / min_offchip
            if min_offchip > 0 else 1.0
        )

    # ─────────────────────────────────────────────────────────
    # M1: Memory Bandwidth Utilization
    # ─────────────────────────────────────────────────────────

    def _compute_mbu(self, graph: ComputationGraph,
                     metrics: MemoryMetrics, t_compute_s: float):
        """
        M1 — Memory Bandwidth Utilization.

        MBU = achieved_bandwidth / peak_bandwidth

        achieved_bandwidth = total_bytes_offchip / T_compute
        peak_bandwidth = hardware peak off-chip bandwidth

        When MBU is high and MFU is low → memory-bandwidth-bound
        When MFU is high and MBU is low → compute-bound
        """
        peak_bw = self.hw.effective_offchip_bandwidth()  # bytes/sec
        metrics.peak_bandwidth_gbps = peak_bw / 1e9

        if t_compute_s > 0 and peak_bw > 0:
            # Bandwidth achieved based on off-chip traffic over compute time
            achieved_bw = metrics.total_bytes_offchip / t_compute_s
            metrics.achieved_bandwidth_gbps = achieved_bw / 1e9
            metrics.mbu = min(achieved_bw / peak_bw, 1.0)
        else:
            metrics.achieved_bandwidth_gbps = 0.0
            metrics.mbu = 0.0

        # Build tier usage breakdown
        for tier_name, cap_gb, bw_gbps in self.hw.memory_tiers():
            if tier_name == "onchip_sram":
                bytes_accessed = metrics.total_bytes_onchip
            else:
                bytes_accessed = metrics.total_bytes_offchip

            util = (
                (bytes_accessed / t_compute_s) / (bw_gbps * 1e9)
                if t_compute_s > 0 and bw_gbps > 0 else 0.0
            )
            metrics.tier_usage.append(MemoryTierUsage(
                tier_name=tier_name,
                capacity_gb=cap_gb,
                bandwidth_gbps=bw_gbps,
                bytes_accessed=bytes_accessed,
                utilization=min(util, 1.0)
            ))

    # ─────────────────────────────────────────────────────────
    # Batch Size Recommendation
    # ─────────────────────────────────────────────────────────

    def _compute_batch_size_recommendation(
        self, graph: ComputationGraph, metrics: MemoryMetrics
    ):
        """
        Find the maximum batch size that fits within on-chip memory.
        This extends Centimani's binary fit/no-fit to a recommendation.
        """
        onchip_bytes = self.hw.total_onchip_sram_gb * 1e9

        # Memory per sample (approximately linear in batch size)
        if graph.batch_size > 0:
            mem_per_sample = (
                metrics.total_activation_bytes / graph.batch_size
                + metrics.total_gradient_bytes / graph.batch_size
            )
            # Weights are fixed cost (not per sample)
            fixed_mem = metrics.total_weight_bytes + metrics.total_optimizer_bytes

            if mem_per_sample > 0 and onchip_bytes > fixed_mem:
                max_bs = int((onchip_bytes - fixed_mem) / mem_per_sample)
                metrics.recommended_batch_size = max(1, max_bs)
            else:
                metrics.recommended_batch_size = graph.batch_size
        else:
            metrics.recommended_batch_size = 1


def run_memory_module(
    graph: ComputationGraph,
    hardware: HardwareSpec,
    t_compute_s: float = 1.0
) -> MemoryMetrics:
    """Run the full memory module and return all metrics."""
    module = MemoryModule(hardware)
    return module.compute(graph, t_compute_s)
