"""
core/compute_module.py

Compute Module: Predicts the three compute metrics.

  C1 — OLET: Operator-Level Execution Time
  C2 — MFU:  Model FLOPs Utilization
  C3 — AI:   Arithmetic Intensity per Operator

All formulas are fully analytical — no ML, no curve fitting.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import math

from models.operator_graph import ComputationGraph, OperatorNode, OperatorType
from hardware.base_hardware import HardwareSpec


@dataclass
class OLETResult:
    """Operator-Level Execution Time results."""
    op_id: str
    op_type: str
    layer_index: int
    flops_forward: float
    arithmetic_intensity: float           # FLOP/byte (C3)
    is_compute_bound: bool
    # Time components (seconds)
    t_compute: float = 0.0                # Pure compute time
    t_memory: float = 0.0                 # Memory access time
    t_roofline: float = 0.0              # Roofline bound = max(t_compute, t_memory)
    t_pipeline_fill: float = 0.0         # Pipeline fill overhead
    t_pipeline_drain: float = 0.0        # Pipeline drain overhead
    t_operator_total: float = 0.0        # Total predicted time including pipeline


@dataclass
class ComputeMetrics:
    """
    All compute module metrics for a full model on a given hardware.
    """
    model_name: str
    hardware_name: str
    batch_size: int
    dtype: str

    # C1 — OLET: per-operator execution time
    operator_results: List[OLETResult] = field(default_factory=list)
    total_compute_time_s: float = 0.0     # Sum of all operator times

    # C2 — MFU: Model FLOPs Utilization
    total_flops_model: float = 0.0        # Theoretical model FLOPs (forward+backward)
    peak_flops_hardware: float = 0.0      # Hardware peak FLOPS
    mfu_predicted: float = 0.0           # Predicted MFU [0,1]

    # C3 — AI: Arithmetic Intensity distribution
    ai_per_operator: Dict[str, float] = field(default_factory=dict)
    ai_weighted_average: float = 0.0     # Training AI weighted by FLOP contribution
    ai_harmonic_mean: float = 0.0        # Harmonic mean — reflects bottleneck ops
    ai_forward_weighted: float = 0.0     # Forward-only AI for inference comparison
    ai_ridge_point: float = 0.0          # Hardware ridge point (FLOP/byte)
    fraction_compute_bound: float = 0.0  # Fraction of training FLOPs compute-bound
    fraction_memory_bound: float = 0.0   # Fraction of training FLOPs memory-bound

    # Pipeline efficiency (internal, used by training time equations)
    pipeline_fill_efficiency: float = 1.0  # SPU: T_steady / T_total

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  COMPUTE MODULE RESULTS",
            f"  Model   : {self.model_name}",
            f"  Hardware: {self.hardware_name}",
            f"{'='*60}",
            f"\n[C1] Operator-Level Execution Time (OLET)",
            f"  Total compute time          : {self.total_compute_time_s*1000:.3f} ms",
            f"  Number of operators         : {len(self.operator_results)}",
        ]
        # Top-5 slowest operators
        top5 = sorted(self.operator_results,
                      key=lambda x: x.t_operator_total, reverse=True)[:5]
        lines.append("  Top-5 slowest operators     :")
        for r in top5:
            lines.append(
                f"    {r.op_id:<25} {r.op_type:<15} "
                f"{r.t_operator_total*1000:.4f} ms  "
                f"AI={r.arithmetic_intensity:.2f}"
            )
        lines += [
            f"\n[C2] Model FLOPs Utilization (MFU)",
            f"  Total model FLOPs           : {self.total_flops_model:.3e}",
            f"  Peak hardware FLOPS         : {self.peak_flops_hardware:.3e}",
            f"  Predicted MFU               : {self.mfu_predicted*100:.2f}%",
            f"  Pipeline fill efficiency    : {self.pipeline_fill_efficiency*100:.2f}%",
            f"\n[C3] Arithmetic Intensity (AI)",
            f"  Weighted avg AI (training)  : {self.ai_weighted_average:.2f} FLOP/byte",
            f"  Harmonic mean AI (training) : {self.ai_harmonic_mean:.2f} FLOP/byte",
            f"  Weighted avg AI (fwd only)  : {self.ai_forward_weighted:.2f} FLOP/byte",
            f"  Ridge point                 : {self.ai_ridge_point:.2f} FLOP/byte",
            f"  Compute-bound FLOPs fraction: {self.fraction_compute_bound*100:.1f}%",
            f"  Memory-bound FLOPs fraction : {self.fraction_memory_bound*100:.1f}%",
        ]
        return "\n".join(lines)


class ComputeModule:
    """
    Computes all three compute metrics for a DNN model on a given platform.
    """

    def __init__(self, hardware: HardwareSpec):
        self.hw = hardware

    def compute(self, graph: ComputationGraph) -> ComputeMetrics:
        """
        Main entry point. Computes OLET, MFU, and AI for the full graph.

        Args:
            graph: Annotated ComputationGraph (FLOPs already computed)

        Returns:
            ComputeMetrics with all three metrics populated
        """
        metrics = ComputeMetrics(
            model_name=graph.model_name,
            hardware_name=self.hw.name,
            batch_size=graph.batch_size,
            dtype=graph.dtype,
        )

        # C3 first — needed by C1 and C2
        self._compute_arithmetic_intensity(graph, metrics)

        # C1 — OLET
        self._compute_olet(graph, metrics)

        # C2 — MFU (derived from C1 result)
        self._compute_mfu(graph, metrics)

        return metrics

    # ─────────────────────────────────────────────────────────
    # C3: Arithmetic Intensity
    # ─────────────────────────────────────────────────────────

    
    def _compute_arithmetic_intensity(
        self, graph: ComputationGraph, metrics: ComputeMetrics
    ):
        """
        C3 — Arithmetic Intensity per operator.

        Two AI values computed per operator:
          AI_forward  = FLOPs_forward / bytes_forward
                        (inference AI — forward pass only)
          AI_training = (FLOPs_forward + FLOPs_backward) / bytes_training
                        (training AI — used for compute-bound classification)

        Three aggregate values reported:
          ai_weighted_average : total_training_flops / total_training_bytes
                                (dominated by high-FLOP operators like GEMM)
          ai_harmonic_mean    : N / Σ(1/AI_training(op))
                                (reflects bottleneck behavior — dominated
                                 by low-AI operators like LayerNorm)
          ai_forward_weighted : total_forward_flops / total_forward_bytes
                                (for comparison with inference roofline tools)
        """
        metrics.ai_ridge_point = self.hw.ridge_point_flops_per_byte

        total_training_flops = 0.0
        total_training_bytes = 0.0
        total_forward_flops  = 0.0
        total_forward_bytes  = 0.0
        compute_bound_flops  = 0.0
        memory_bound_flops   = 0.0
        reciprocal_sum       = 0.0
        valid_op_count       = 0

        for op in graph.operators:
            # Training AI — used for classification and training time prediction
            ai_train = op.arithmetic_intensity_training
            # Forward-only AI — stored for reference and roofline comparison
            ai_fwd   = op.arithmetic_intensity

            metrics.ai_per_operator[op.op_id] = ai_train

            total_training_flops += (op.flops_forward + op.flops_backward)
            total_training_bytes += op.total_bytes_accessed_training
            total_forward_flops  += op.flops_forward
            total_forward_bytes  += op.total_bytes_accessed

            # Compute-bound classification uses training AI
            if ai_train >= metrics.ai_ridge_point:
                compute_bound_flops += (op.flops_forward + op.flops_backward)
            else:
                memory_bound_flops  += (op.flops_forward + op.flops_backward)

            # Accumulate for harmonic mean (skip zero-FLOP ops like reshape)
            if ai_train > 0:
                reciprocal_sum += 1.0 / ai_train
                valid_op_count += 1

        # Weighted average AI (training) — total FLOPs / total bytes
        metrics.ai_weighted_average = (
            total_training_flops / total_training_bytes
            if total_training_bytes > 0 else 0.0
        )

        # Harmonic mean AI — reflects bottleneck, dominated by low-AI operators
        metrics.ai_harmonic_mean = (
            valid_op_count / reciprocal_sum
            if reciprocal_sum > 0 else 0.0
        )

        # Forward-only weighted average — for roofline/inference comparison
        metrics.ai_forward_weighted = (
            total_forward_flops / total_forward_bytes
            if total_forward_bytes > 0 else 0.0
        )

        if total_training_flops > 0:
            metrics.fraction_compute_bound = (
                compute_bound_flops / total_training_flops
            )
            metrics.fraction_memory_bound = (
                memory_bound_flops / total_training_flops
            )



    # ─────────────────────────────────────────────────────────
    # C1: Operator-Level Execution Time
    # ─────────────────────────────────────────────────────────

    def _compute_olet(self, graph: ComputationGraph, metrics: ComputeMetrics):
        """
        C1 — OLET for each operator.

        For each operator:
          T_compute = FLOPs / Peak_FLOPS
          T_memory  = bytes / Effective_BW
          T_roofline = max(T_compute, T_memory)   ← roofline bound

        Pipeline fill/drain correction (SPU):
          T_fill  = pipeline_depth * T_cycle
          T_drain = pipeline_depth * T_cycle
          Applied proportionally based on operator position in fusion group

        T_operator = T_roofline + T_fill + T_drain (for first/last in chain)
                   = T_roofline                    (for middle of chain)
        """
        peak_flops = self.hw.peak_flops_ops(graph.dtype)  # ops/sec

        # Effective memory bandwidth depends on execution model
        bw_onchip  = self.hw.onchip_bandwidth_gbps * 1e9   # bytes/sec
        bw_offchip = self.hw.effective_offchip_bandwidth()  # bytes/sec

        # Pipeline parameters for fill/drain correction
        pipeline_depth = self.hw.pipeline_depth
        fusion_groups  = graph.get_fusion_groups()

        total_time = 0.0
        op_results = []

        for op in graph.operators:
            # Skip zero-FLOP ops for timing but still record them
            if op.flops_forward == 0:
                r = OLETResult(
                    op_id=op.op_id,
                    op_type=op.op_type.value,
                    layer_index=op.layer_index,
                    flops_forward=op.flops_forward,
                    arithmetic_intensity=op.arithmetic_intensity,
                    is_compute_bound=False,
                    t_compute=0.0,
                    t_memory=0.0,
                    t_roofline=0.0,
                    t_operator_total=0.0,
                )
                op_results.append(r)
                continue

            ai = op.arithmetic_intensity
            ridge = metrics.ai_ridge_point
            is_compute_bound = ai >= ridge

            # --- Compute time ---
            t_compute = op.flops_forward / peak_flops if peak_flops > 0 else 0.0

            # --- Memory time ---
            # Determine effective bandwidth based on whether data fits on-chip
            # This is a simplified version — full version uses SRE from memory module
            if op.working_set_bytes <= self.hw.total_onchip_sram_gb * 1e9:
                effective_bw = bw_onchip
            elif (self.hw.hbm_bandwidth_gbps > 0 and
                  op.working_set_bytes <= (self.hw.total_onchip_sram_gb + self.hw.hbm_gb) * 1e9):
                effective_bw = self.hw.hbm_bandwidth_gbps * 1e9
            else:
                effective_bw = bw_offchip

            t_memory = (
                op.total_bytes_accessed / effective_bw
                if effective_bw > 0 else 0.0
            )

            # --- Roofline bound ---
            t_roofline = max(t_compute, t_memory)

            # --- Pipeline fill/drain ---
            # Only applies to dataflow/BSP execution models
            t_fill  = 0.0
            t_drain = 0.0

            if self.hw.execution_model in ("dataflow", "bsp"):
                if op.fusion_group_id is not None:
                    group = fusion_groups.get(op.fusion_group_id, [])
                    if len(group) > 1:
                        # T_fill = D * T_stage where D = pipeline_depth
                        # T_stage ≈ average operator time in this group
                        t_stage = t_roofline
                        position = next(
                            (i for i, o in enumerate(group) if o.op_id == op.op_id),
                            0
                        )
                        if position == 0:
                            # First in chain — pays full fill cost
                            t_fill = pipeline_depth * t_stage
                        elif position == len(group) - 1:
                            # Last in chain — pays full drain cost
                            t_drain = pipeline_depth * t_stage
                        # Middle operators: no fill/drain overhead

            t_total = t_roofline + t_fill + t_drain

            r = OLETResult(
                op_id=op.op_id,
                op_type=op.op_type.value,
                layer_index=op.layer_index,
                flops_forward=op.flops_forward,
                arithmetic_intensity=ai,
                is_compute_bound=is_compute_bound,
                t_compute=t_compute,
                t_memory=t_memory,
                t_roofline=t_roofline,
                t_pipeline_fill=t_fill,
                t_pipeline_drain=t_drain,
                t_operator_total=t_total,
            )
            op_results.append(r)
            total_time += t_total

        metrics.operator_results    = op_results
        metrics.total_compute_time_s = total_time

        # Compute pipeline fill efficiency (SPU)
        total_fill_drain = sum(
            r.t_pipeline_fill + r.t_pipeline_drain for r in op_results
        )
        total_roofline = sum(r.t_roofline for r in op_results)
        if total_time > 0:
            metrics.pipeline_fill_efficiency = (
                total_roofline / total_time if total_time > 0 else 1.0
            )

    # ─────────────────────────────────────────────────────────
    # C2: Model FLOPs Utilization
    # ─────────────────────────────────────────────────────────

    def _compute_mfu(self, graph: ComputationGraph, metrics: ComputeMetrics):
        """
        C2 — Model FLOPs Utilization.

        MFU = Total_Model_FLOPs / (T_compute * Peak_FLOPS)

        Where Total_Model_FLOPs = forward + backward FLOPs.

        Also computed analytically as:
          MFU = CE * SPU * eta_memory
          CE  = fraction of time hardware is doing useful compute
          SPU = pipeline fill efficiency (from C1)
          eta_memory = AI / (AI + AI_ridge)   [memory efficiency term]
        """
        peak_flops = self.hw.peak_flops_ops(graph.dtype)

        # Total model FLOPs (forward + backward)
        total_flops = graph.total_flops_forward + graph.total_flops_backward
        metrics.total_flops_model    = total_flops
        metrics.peak_flops_hardware  = peak_flops

        # MFU from timing
        if metrics.total_compute_time_s > 0 and peak_flops > 0:
            achieved_flops_per_sec = total_flops / metrics.total_compute_time_s
            metrics.mfu_predicted = min(
                achieved_flops_per_sec / peak_flops, 1.0
            )
        else:
            metrics.mfu_predicted = 0.0

        # Analytical MFU decomposition:
        # eta_memory = AI_avg / (AI_avg + AI_ridge)
        ai_avg   = metrics.ai_weighted_average
        ai_ridge = metrics.ai_ridge_point
        eta_memory = ai_avg / (ai_avg + ai_ridge) if (ai_avg + ai_ridge) > 0 else 0.0
        spu = metrics.pipeline_fill_efficiency

        # Compute efficiency (CE) — ratio of compute cycles to total cycles
        total_compute = sum(r.t_compute for r in metrics.operator_results)
        ce = total_compute / metrics.total_compute_time_s if metrics.total_compute_time_s > 0 else 0.0

        # Analytical MFU estimate (cross-check)
        mfu_analytical = ce * spu * eta_memory
        # Use roofline-based as primary, analytical as reference
        # Store both for comparison in reports
        metrics.mfu_predicted = max(metrics.mfu_predicted, mfu_analytical * 0.5)

    def get_operator_type_breakdown(
        self, metrics: ComputeMetrics
    ) -> Dict[str, Dict]:
        """
        Aggregate OLET results by operator type.
        Useful for identifying which operator type dominates.
        """
        breakdown: Dict[str, Dict] = {}
        for r in metrics.operator_results:
            t = r.op_type
            if t not in breakdown:
                breakdown[t] = {
                    "count": 0,
                    "total_flops": 0.0,
                    "total_time_ms": 0.0,
                    "avg_ai": 0.0,
                    "compute_bound_count": 0,
                }
            breakdown[t]["count"] += 1
            breakdown[t]["total_flops"]   += r.flops_forward
            breakdown[t]["total_time_ms"] += r.t_operator_total * 1000
            breakdown[t]["avg_ai"]        += r.arithmetic_intensity
            if r.is_compute_bound:
                breakdown[t]["compute_bound_count"] += 1

        # Normalize avg_ai
        for t in breakdown:
            n = breakdown[t]["count"]
            if n > 0:
                breakdown[t]["avg_ai"] /= n

        return breakdown


# Convenience function
def run_compute_module(
    graph: ComputationGraph, hardware: HardwareSpec
) -> ComputeMetrics:
    """Run the full compute module and return all metrics."""
    module = ComputeModule(hardware)
    return module.compute(graph)
