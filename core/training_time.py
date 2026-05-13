"""
core/training_time.py

Predicts training iteration time using 7 distinct analytical equations.
Each equation uses a different modeling philosophy.

  Eq1 — Operator Decomposition Model
  Eq2 — Roofline-Bounded Model
  Eq3 — Streaming Pipeline Model
  Eq4 — Memory-Traffic-Driven Model
  Eq5 — MFU-Normalized Model
  Eq6 — Communication-Aware Scaling Model
  Eq7 — Hierarchical Bottleneck Model
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import math

from models.operator_graph import ComputationGraph
from hardware.base_hardware import HardwareSpec
from .compute_module import ComputeMetrics
from .memory_module import MemoryMetrics
from .communication_module import CommMetrics


@dataclass
class TrainingTimePrediction:
    """
    Training time predictions from all 7 equations.
    """
    model_name: str
    hardware_name: str
    batch_size: int
    num_devices: int

    # Individual equation predictions (milliseconds per iteration)
    eq1_operator_decomposition_ms: float = 0.0
    eq2_roofline_bounded_ms: float = 0.0
    eq3_streaming_pipeline_ms: float = 0.0
    eq4_memory_traffic_driven_ms: float = 0.0
    eq5_mfu_normalized_ms: float = 0.0
    eq6_comm_aware_scaling_ms: float = 0.0
    eq7_hierarchical_bottleneck_ms: float = 0.0

    # Derived throughput for each equation (samples/sec)
    throughput_eq1: float = 0.0
    throughput_eq2: float = 0.0
    throughput_eq3: float = 0.0
    throughput_eq4: float = 0.0
    throughput_eq5: float = 0.0
    throughput_eq6: float = 0.0
    throughput_eq7: float = 0.0

    # Which bottleneck dominates (from Eq7)
    bottleneck: str = "unknown"           # "compute" | "memory" | "communication"

    # Intermediate values used by equations
    alpha_backward_ratio: float = 2.0     # T_backward / T_forward
    t_weight_update_ms: float = 0.0
    t_overlap_ms: float = 0.0

    def predictions_dict(self) -> Dict[str, float]:
        return {
            "Eq1 Operator Decomposition": self.eq1_operator_decomposition_ms,
            "Eq2 Roofline Bounded": self.eq2_roofline_bounded_ms,
            "Eq3 Streaming Pipeline": self.eq3_streaming_pipeline_ms,
            "Eq4 Memory Traffic Driven": self.eq4_memory_traffic_driven_ms,
            "Eq5 MFU Normalized": self.eq5_mfu_normalized_ms,
            "Eq6 Comm-Aware Scaling": self.eq6_comm_aware_scaling_ms,
            "Eq7 Hierarchical Bottleneck": self.eq7_hierarchical_bottleneck_ms,
        }

    def throughput_dict(self) -> Dict[str, float]:
        return {
            "Eq1": self.throughput_eq1,
            "Eq2": self.throughput_eq2,
            "Eq3": self.throughput_eq3,
            "Eq4": self.throughput_eq4,
            "Eq5": self.throughput_eq5,
            "Eq6": self.throughput_eq6,
            "Eq7": self.throughput_eq7,
        }

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  TRAINING TIME PREDICTIONS",
            f"  Model   : {self.model_name}",
            f"  Hardware: {self.hardware_name}",
            f"  Batch   : {self.batch_size}  Devices: {self.num_devices}",
            f"{'='*60}",
            f"  {'Equation':<35} {'Time (ms)':>10}  {'Throughput':>12}",
            f"  {'-'*60}",
        ]
        for eq_name, t_ms in self.predictions_dict().items():
            thr = self.batch_size / (t_ms / 1000) if t_ms > 0 else 0
            lines.append(f"  {eq_name:<35} {t_ms:>10.3f}  {thr:>10.1f} s/s")
        lines += [
            f"  {'-'*60}",
            f"\n  Dominant bottleneck: {self.bottleneck.upper()}",
            f"  Alpha (bwd/fwd ratio): {self.alpha_backward_ratio:.2f}",
            f"  Weight update time: {self.t_weight_update_ms:.3f} ms",
            f"  Compute-comm overlap: {self.t_overlap_ms:.3f} ms",
        ]
        return "\n".join(lines)


class TrainingTimePredictor:
    """
    Predicts training time per iteration using 7 distinct equations.
    Takes metric results from all three modules as inputs.
    """

    def __init__(self, hardware: HardwareSpec):
        self.hw = hardware

    def predict(
        self,
        graph: ComputationGraph,
        compute_m: ComputeMetrics,
        memory_m: MemoryMetrics,
        comm_m: CommMetrics,
        num_devices: int = 1,
    ) -> TrainingTimePrediction:
        """
        Run all 7 training time equations.

        Args:
            graph: Computation graph
            compute_m: Results from ComputeModule
            memory_m: Results from MemoryModule
            comm_m: Results from CommunicationModule
            num_devices: Number of chips

        Returns:
            TrainingTimePrediction with all 7 equation results
        """
        pred = TrainingTimePrediction(
            model_name=graph.model_name,
            hardware_name=self.hw.name,
            batch_size=graph.batch_size,
            num_devices=num_devices,
        )

        # Compute shared intermediate values
        pred.alpha_backward_ratio = self._estimate_backward_ratio(graph)
        pred.t_weight_update_ms   = self._estimate_weight_update_time(
            graph, compute_m
        )
        pred.t_overlap_ms         = self._estimate_overlap(compute_m, comm_m)

        # Run all 7 equations
        pred.eq1_operator_decomposition_ms = self._eq1_operator_decomposition(
            compute_m, pred
        )
        pred.eq2_roofline_bounded_ms = self._eq2_roofline_bounded(
            graph, compute_m, memory_m, comm_m, pred
        )
        pred.eq3_streaming_pipeline_ms = self._eq3_streaming_pipeline(
            graph, compute_m, memory_m, pred
        )
        pred.eq4_memory_traffic_driven_ms = self._eq4_memory_traffic_driven(
            graph, compute_m, memory_m, pred
        )
        pred.eq5_mfu_normalized_ms = self._eq5_mfu_normalized(
            graph, compute_m, pred
        )
        pred.eq6_comm_aware_scaling_ms = self._eq6_comm_aware_scaling(
            pred, comm_m, num_devices
        )
        pred.eq7_hierarchical_bottleneck_ms, pred.bottleneck = (
            self._eq7_hierarchical_bottleneck(compute_m, memory_m, comm_m, pred)
        )

        # Compute throughput for each equation
        bs = graph.batch_size
        for eq_num, attr in enumerate([
            "eq1_operator_decomposition_ms",
            "eq2_roofline_bounded_ms",
            "eq3_streaming_pipeline_ms",
            "eq4_memory_traffic_driven_ms",
            "eq5_mfu_normalized_ms",
            "eq6_comm_aware_scaling_ms",
            "eq7_hierarchical_bottleneck_ms",
        ], 1):
            t_ms = getattr(pred, attr)
            thr  = bs / (t_ms / 1000) if t_ms > 0 else 0.0
            setattr(pred, f"throughput_eq{eq_num}", thr)

        return pred

    # ─────────────────────────────────────────────────────────
    # Shared intermediate values
    # ─────────────────────────────────────────────────────────

    def _estimate_backward_ratio(self, graph: ComputationGraph) -> float:
        """
        Estimate alpha = T_backward / T_forward.
        Not fixed at 2x — varies by model type and operator distribution.

        CNN models:        alpha ≈ 1.8–2.0  (backprop through convolutions)
        Transformer models: alpha ≈ 2.0–2.3  (attention backward is more expensive)
        MLP models:        alpha ≈ 1.9–2.1
        """
        model_type = graph.model_type
        if model_type == "transformer":
            return 2.1
        elif model_type == "cnn":
            return 1.9
        elif model_type == "mlp":
            return 2.0
        else:
            return 2.0

    def _estimate_weight_update_time(
        self, graph: ComputationGraph, compute_m: ComputeMetrics
    ) -> float:
        """
        Estimate weight update time (optimizer step) in milliseconds.

        Weight update touches all parameters once:
          T_wu = 2 * weight_bytes / effective_BW
          (factor 2: read weight + write updated weight)

        On Cerebras: weights on-chip → fast update
        On SambaNova: weights may be in HBM/DDR → slower
        On Graphcore: distributed per-tile update → fast
        """
        weight_bytes = graph.total_weight_bytes

        if self.hw.execution_model == "wafer_scale":
            bw = self.hw.onchip_bandwidth_gbps * 1e9
        elif self.hw.hbm_bandwidth_gbps > 0:
            bw = self.hw.hbm_bandwidth_gbps * 1e9
        else:
            bw = max(self.hw.onchip_bandwidth_gbps, 1.0) * 1e9

        t_wu = (2 * weight_bytes / bw) if bw > 0 else 0.0
        return t_wu * 1000  # Convert to ms

    def _estimate_overlap(
        self, compute_m: ComputeMetrics, comm_m: CommMetrics
    ) -> float:
        """
        Estimate compute-communication overlap in ms.

        For data parallel training: backward pass can overlap with
        all-reduce of already-computed gradients.
        Overlap ≈ min(T_backward, T_allreduce)
        """
        t_backward_ms = compute_m.total_compute_time_s * 1000 * 0.65
        t_allreduce_ms = comm_m.t_allreduce_s * 1000
        return min(t_backward_ms, t_allreduce_ms) * 0.70  # 70% overlap assumed

    # ─────────────────────────────────────────────────────────
    # EQUATION 1: Operator Decomposition
    # ─────────────────────────────────────────────────────────

    def _eq1_operator_decomposition(
        self, compute_m: ComputeMetrics, pred: TrainingTimePrediction
    ) -> float:
        """
        T_iter = (1 + alpha) * Σ T_op(l,j) + T_weight_update - T_overlap

        Sum of all operator times scaled by backward pass ratio,
        plus weight update, minus compute-comm overlap.
        """
        t_sum_ms = compute_m.total_compute_time_s * 1000
        t_iter = (
            (1 + pred.alpha_backward_ratio) * t_sum_ms
            + pred.t_weight_update_ms
            - pred.t_overlap_ms
        )
        return max(t_iter, 0.0)

    # ─────────────────────────────────────────────────────────
    # EQUATION 2: Roofline-Bounded
    # ─────────────────────────────────────────────────────────

    def _eq2_roofline_bounded(
        self,
        graph: ComputationGraph,
        compute_m: ComputeMetrics,
        memory_m: MemoryMetrics,
        comm_m: CommMetrics,
        pred: TrainingTimePrediction
    ) -> float:
        """
        T_iter = (1+alpha) * Σ max(T_compute(op), T_memory(op)) + T_sync

        Each operator's time is bounded by roofline.
        Memory bandwidth is adjusted by SRE (from memory module).

        BW_eff = SRE * BW_onchip + (1 - SRE) * BW_offchip
        """
        # BW_eff uses SRE from memory module
        sre  = memory_m.sre
        bw_on  = self.hw.onchip_bandwidth_gbps
        bw_off = self.hw.effective_offchip_bandwidth() / 1e9

        bw_eff_gbps = sre * bw_on + (1.0 - sre) * bw_off

        # Recompute operator times with SRE-adjusted bandwidth
        peak_flops = self.hw.peak_flops_ops(graph.dtype)
        bw_eff     = bw_eff_gbps * 1e9

        t_roofline_total = 0.0
        for r in compute_m.operator_results:
            if r.flops_forward == 0:
                continue
            t_comp = r.flops_forward / peak_flops if peak_flops > 0 else 0.0
            # Re-compute memory time with SRE-adjusted bandwidth
            op_bytes = (r.flops_forward / r.arithmetic_intensity
                        if r.arithmetic_intensity > 0 else 0.0)
            t_mem  = op_bytes / bw_eff if bw_eff > 0 else 0.0
            t_roofline_total += max(t_comp, t_mem)

        t_sync_ms = comm_m.t_sync_s * 1000

        t_iter = (
            (1 + pred.alpha_backward_ratio) * t_roofline_total * 1000
            + pred.t_weight_update_ms
            + t_sync_ms
            - pred.t_overlap_ms
        )
        return max(t_iter, 0.0)

    # ─────────────────────────────────────────────────────────
    # EQUATION 3: Streaming Pipeline
    # ─────────────────────────────────────────────────────────

    def _eq3_streaming_pipeline(
        self,
        graph: ComputationGraph,
        compute_m: ComputeMetrics,
        memory_m: MemoryMetrics,
        pred: TrainingTimePrediction
    ) -> float:
        """
        T_iter = T_fill + T_steady_state + T_drain + T_weight_update

        T_steady_state = N_tokens / BS * T_pipeline_cycle
        T_pipeline_cycle = max(T_stage_s for all pipeline stages)
        T_fill = T_drain = pipeline_depth * T_pipeline_cycle
        """
        if compute_m.operator_results:
            # Pipeline cycle time = slowest operator (bottleneck)
            t_bottleneck_s = max(
                r.t_roofline for r in compute_m.operator_results
                if r.flops_forward > 0
            ) if any(r.flops_forward > 0 for r in compute_m.operator_results) else 0.0
        else:
            t_bottleneck_s = compute_m.total_compute_time_s

        pipeline_depth = self.hw.pipeline_depth
        n_operators    = max(len(compute_m.operator_results), 1)

        # Fill and drain cost
        t_fill  = pipeline_depth * t_bottleneck_s
        t_drain = pipeline_depth * t_bottleneck_s

        # Steady-state: all N operators processed at bottleneck rate
        t_steady = n_operators * t_bottleneck_s

        # Forward + backward (backward pipeline is also filled/drained)
        t_fwd = t_fill + t_steady + t_drain
        t_bwd = pred.alpha_backward_ratio * t_fwd

        t_iter = (t_fwd + t_bwd) * 1000 + pred.t_weight_update_ms
        return max(t_iter, 0.0)

    # ─────────────────────────────────────────────────────────
    # EQUATION 4: Memory-Traffic-Driven
    # ─────────────────────────────────────────────────────────

    def _eq4_memory_traffic_driven(
        self,
        graph: ComputationGraph,
        compute_m: ComputeMetrics,
        memory_m: MemoryMetrics,
        pred: TrainingTimePrediction
    ) -> float:
        """
        T_iter = OMT_total / BW_eff + FLOPs_compute_bound / Peak_FLOPS

        Separates compute-bound operators (dominated by FLOPS)
        from memory-bound operators (dominated by bandwidth).
        """
        # SRE-adjusted effective bandwidth
        sre    = memory_m.sre
        bw_on  = self.hw.onchip_bandwidth_gbps * 1e9
        bw_off = self.hw.effective_offchip_bandwidth()
        bw_eff = sre * bw_on + (1.0 - sre) * bw_off

        # Off-chip memory traffic (from memory module M3)
        omt_bytes = memory_m.actual_offchip_bytes

        # FLOPS from compute-bound operators only
        flops_compute_bound = sum(
            r.flops_forward for r in compute_m.operator_results
            if r.is_compute_bound
        )
        peak_flops = self.hw.peak_flops_ops(graph.dtype)

        t_memory_s  = omt_bytes / bw_eff if bw_eff > 0 else 0.0
        t_compute_s = flops_compute_bound / peak_flops if peak_flops > 0 else 0.0

        t_fwd = t_memory_s + t_compute_s
        t_iter = (1 + pred.alpha_backward_ratio) * t_fwd * 1000 + pred.t_weight_update_ms
        return max(t_iter, 0.0)

    # ─────────────────────────────────────────────────────────
    # EQUATION 5: MFU-Normalized
    # ─────────────────────────────────────────────────────────

    def _eq5_mfu_normalized(
        self,
        graph: ComputationGraph,
        compute_m: ComputeMetrics,
        pred: TrainingTimePrediction
    ) -> float:
        """
        T_iter = FLOPs_total / (MFU * Peak_FLOPS)

        Where MFU is predicted analytically:
          MFU = CE * SPU * eta_memory
          CE  = compute fraction of total time
          SPU = pipeline fill efficiency
          eta_memory = AI / (AI + AI_ridge)
        """
        total_flops = graph.total_flops_forward + graph.total_flops_backward
        mfu         = max(compute_m.mfu_predicted, 0.01)  # avoid div by 0
        peak_flops  = self.hw.peak_flops_ops(graph.dtype)

        if peak_flops > 0 and mfu > 0:
            t_iter_s = total_flops / (mfu * peak_flops)
        else:
            t_iter_s = compute_m.total_compute_time_s * (1 + pred.alpha_backward_ratio)

        t_iter_ms = t_iter_s * 1000 + pred.t_weight_update_ms
        return max(t_iter_ms, 0.0)

    # ─────────────────────────────────────────────────────────
    # EQUATION 6: Communication-Aware Scaling
    # ─────────────────────────────────────────────────────────

    def _eq6_comm_aware_scaling(
        self,
        pred: TrainingTimePrediction,
        comm_m: CommMetrics,
        num_devices: int
    ) -> float:
        """
        T_iter(N) = T_iter(1) / (N * eta_scale) + T_sync

        eta_scale = 1 / (1 + CCR * (N-1)/N)
        T_iter(1) from Eq1 (best single-device prediction)
        """
        if num_devices <= 1:
            return pred.eq1_operator_decomposition_ms

        t_single_ms = pred.eq1_operator_decomposition_ms
        N    = num_devices
        ccr  = comm_m.ccr
        sof  = comm_m.sof
        eta  = comm_m.scaling_efficiency

        if eta > 0:
            t_scaled = t_single_ms / (N * eta)
        else:
            t_scaled = t_single_ms / N

        # Add synchronization overhead
        t_sync_ms = comm_m.t_sync_s * 1000
        t_iter    = t_scaled + t_sync_ms

        return max(t_iter, 0.0)

    # ─────────────────────────────────────────────────────────
    # EQUATION 7: Hierarchical Bottleneck
    # ─────────────────────────────────────────────────────────

    def _eq7_hierarchical_bottleneck(
        self,
        compute_m: ComputeMetrics,
        memory_m: MemoryMetrics,
        comm_m: CommMetrics,
        pred: TrainingTimePrediction,
    ) -> tuple:
        """
        T_iter = max(T_compute_bound, T_memory_bound, T_comm_bound) + T_sync

        T_compute_bound = FLOPs / (MFU * Peak_FLOPS)
        T_memory_bound  = OMT / BW_eff
        T_comm_bound    = Gradient_bytes / (BW_inter * CCBU)

        The dominant term identifies the bottleneck.
        """
        # Compute bound
        t_compute_ms = (
            compute_m.total_compute_time_s
            * (1 + pred.alpha_backward_ratio)
            * 1000
        )

        # Memory bound (SRE-adjusted)
        sre    = memory_m.sre
        bw_on  = self.hw.onchip_bandwidth_gbps * 1e9
        bw_off = self.hw.effective_offchip_bandwidth()
        bw_eff = sre * bw_on + (1.0 - sre) * bw_off

        omt_bytes = memory_m.actual_offchip_bytes
        t_memory_ms = (
            omt_bytes / bw_eff * 1000
            if bw_eff > 0 else t_compute_ms
        )
        t_memory_ms *= (1 + pred.alpha_backward_ratio)

        # Communication bound
        inter_bw  = self.hw.inter_chip_bandwidth_gbps * 1e9
        ccbu      = max(comm_m.ccbu, 0.01)
        grad_bytes = comm_m.gradient_bytes
        t_comm_ms = (
            grad_bytes / (inter_bw * ccbu) * 1000
            if inter_bw > 0 and comm_m.num_devices > 1 else 0.0
        )

        # Find bottleneck
        times = {
            "compute":       t_compute_ms,
            "memory":        t_memory_ms,
            "communication": t_comm_ms,
        }
        bottleneck = max(times, key=times.get)
        t_dominant = times[bottleneck]

        # Sync overhead
        t_sync_ms = comm_m.t_sync_s * 1000

        t_iter = t_dominant + t_sync_ms + pred.t_weight_update_ms
        return max(t_iter, 0.0), bottleneck


def predict_training_time(
    graph: ComputationGraph,
    hardware: HardwareSpec,
    compute_m: ComputeMetrics,
    memory_m: MemoryMetrics,
    comm_m: CommMetrics,
    num_devices: int = 1,
) -> TrainingTimePrediction:
    """Convenience function: run all 7 training time equations."""
    predictor = TrainingTimePredictor(hardware)
    return predictor.predict(graph, compute_m, memory_m, comm_m, num_devices)
