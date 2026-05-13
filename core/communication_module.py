"""
core/communication_module.py

Communication Module: Predicts three communication metrics.

  Comm1 — CCR:  Communication-to-Computation Ratio
  Comm2 — CCBU: Collective Communication Bandwidth Utilization
  Comm3 — SOF:  Synchronization Overhead Fraction

Uses LogGP model for all-reduce time prediction.
Models BSP barrier overhead for Graphcore,
pipeline sync for SambaNova, and SwarmX sync for Cerebras.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math

from models.operator_graph import ComputationGraph
from hardware.base_hardware import HardwareSpec


@dataclass
class CommMetrics:
    """All communication module metrics."""
    model_name: str
    hardware_name: str
    num_devices: int
    dtype: str

    # Comm1 — CCR: Communication-to-Computation Ratio
    t_compute_s: float = 0.0
    t_allreduce_s: float = 0.0
    t_intra_chip_s: float = 0.0           # On-chip data movement time
    t_comm_total_s: float = 0.0
    ccr: float = 0.0                      # T_comm / T_compute

    # Comm2 — CCBU: Collective Communication BW Utilization
    gradient_bytes: int = 0
    allreduce_bytes_total: int = 0
    peak_inter_chip_bw_gbps: float = 0.0
    achieved_comm_bw_gbps: float = 0.0
    ccbu: float = 0.0                     # achieved / peak

    # Comm3 — SOF: Synchronization Overhead Fraction
    t_sync_s: float = 0.0
    sof: float = 0.0                      # T_sync / T_total_iter
    t_bsp_barrier_s: float = 0.0          # Graphcore BSP barrier
    t_pipeline_stall_s: float = 0.0       # SambaNova pipeline stall
    t_inter_device_sync_s: float = 0.0    # Multi-device gradient sync

    # LogGP parameters used
    loggp_L: float = 0.0                  # Latency (seconds)
    loggp_BW: float = 0.0                 # Bandwidth (bytes/sec)

    # Scaling efficiency
    scaling_efficiency: float = 1.0       # T_single / (N * T_multi)

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  COMMUNICATION MODULE RESULTS",
            f"  Model   : {self.model_name}",
            f"  Hardware: {self.hardware_name}",
            f"  Devices : {self.num_devices}",
            f"{'='*60}",
            f"\n[Comm1] Communication-to-Computation Ratio (CCR)",
            f"  Compute time               : {self.t_compute_s*1000:.3f} ms",
            f"  All-reduce time            : {self.t_allreduce_s*1000:.3f} ms",
            f"  Intra-chip comm time       : {self.t_intra_chip_s*1000:.3f} ms",
            f"  Total comm time            : {self.t_comm_total_s*1000:.3f} ms",
            f"  CCR                        : {self.ccr:.4f}",
            f"\n[Comm2] Collective Communication BW Utilization (CCBU)",
            f"  Gradient bytes             : {self.gradient_bytes/1e9:.3f} GB",
            f"  All-reduce bytes           : {self.allreduce_bytes_total/1e9:.3f} GB",
            f"  Peak inter-chip BW         : {self.peak_inter_chip_bw_gbps:.1f} GB/s",
            f"  Achieved comm BW           : {self.achieved_comm_bw_gbps:.1f} GB/s",
            f"  CCBU                       : {self.ccbu*100:.2f}%",
            f"\n[Comm3] Synchronization Overhead Fraction (SOF)",
            f"  BSP barrier time           : {self.t_bsp_barrier_s*1000:.3f} ms",
            f"  Pipeline stall time        : {self.t_pipeline_stall_s*1000:.3f} ms",
            f"  Inter-device sync time     : {self.t_inter_device_sync_s*1000:.3f} ms",
            f"  Total sync time            : {self.t_sync_s*1000:.3f} ms",
            f"  SOF                        : {self.sof*100:.2f}%",
            f"\n[Scaling]",
            f"  Scaling efficiency (N={self.num_devices})   : {self.scaling_efficiency*100:.2f}%",
        ]
        return "\n".join(lines)


class CommunicationModule:
    """
    Computes all three communication metrics.
    """

    def __init__(self, hardware: HardwareSpec):
        self.hw = hardware

    def compute(
        self,
        graph: ComputationGraph,
        t_compute_s: float,
        gradient_bytes: int,
        num_devices: int = 1,
        parallelism: str = "data"
    ) -> CommMetrics:
        """
        Main entry point.

        Args:
            graph: Computation graph
            t_compute_s: Compute time from ComputeModule (seconds)
            gradient_bytes: Total gradient bytes to synchronize
            num_devices: Number of chips/devices
            parallelism: 'data' | 'model' | 'pipeline'

        Returns:
            CommMetrics with all three metrics populated
        """
        metrics = CommMetrics(
            model_name=graph.model_name,
            hardware_name=self.hw.name,
            num_devices=num_devices,
            dtype=graph.dtype,
        )
        metrics.t_compute_s   = t_compute_s
        metrics.gradient_bytes = gradient_bytes

        # Comm2 — all-reduce time (LogGP model)
        self._compute_allreduce_time(metrics, num_devices, parallelism)

        # Comm1 — intra-chip fabric time
        self._compute_intra_chip_time(graph, metrics, t_compute_s)

        # Total communication
        metrics.t_comm_total_s = metrics.t_allreduce_s + metrics.t_intra_chip_s

        # CCR
        metrics.ccr = (
            metrics.t_comm_total_s / t_compute_s
            if t_compute_s > 0 else 0.0
        )

        # CCBU
        self._compute_ccbu(metrics, num_devices)

        # SOF
        self._compute_sof(graph, metrics, t_compute_s, num_devices)

        # Scaling efficiency
        self._compute_scaling_efficiency(metrics, num_devices)

        return metrics

    # ─────────────────────────────────────────────────────────
    # Comm2: All-Reduce Time (LogGP Model)
    # ─────────────────────────────────────────────────────────

    def _compute_allreduce_time(
        self, metrics: CommMetrics, num_devices: int, parallelism: str
    ):
        """
        Comm2 — All-reduce time using LogGP model.

        Ring all-reduce (used when message > latency*BW crossover):
          T = 2 * (N-1)/N * [L + M/N * (1/BW)]

        Tree all-reduce (used for small messages):
          T = 2 * log2(N) * [L + M * (1/BW)]

        Crossover point: M_crossover = L * BW
        """
        if num_devices <= 1:
            metrics.t_allreduce_s = 0.0
            metrics.peak_inter_chip_bw_gbps = 0.0
            return

        M = metrics.gradient_bytes
        N = num_devices
        BW = self.hw.inter_chip_bandwidth_gbps * 1e9   # bytes/sec
        L  = self.hw.inter_chip_latency_us * 1e-6       # seconds

        metrics.loggp_L  = L
        metrics.loggp_BW = BW
        metrics.peak_inter_chip_bw_gbps = self.hw.inter_chip_bandwidth_gbps

        if BW == 0:
            metrics.t_allreduce_s = 0.0
            return

        # Crossover: use ring when M > L * BW
        crossover = L * BW
        if M > crossover:
            # Ring all-reduce
            t_allreduce = 2.0 * (N - 1) / N * (L + M / N / BW)
        else:
            # Tree all-reduce (small messages)
            t_allreduce = 2.0 * math.log2(max(N, 2)) * (L + M / BW)

        # Cerebras-specific: SwarmX fabric enables near-linear scaling
        # Approximated as ~10% of standard all-reduce overhead
        if self.hw.execution_model == "wafer_scale":
            t_allreduce *= 0.10

        metrics.t_allreduce_s          = t_allreduce
        metrics.allreduce_bytes_total  = int(2 * (N - 1) / N * M)  # Ring traffic

    # ─────────────────────────────────────────────────────────
    # Intra-Chip Communication
    # ─────────────────────────────────────────────────────────

    def _compute_intra_chip_time(
        self, graph: ComputationGraph, metrics: CommMetrics, t_compute_s: float
    ):
        """
        Intra-chip data movement between compute units and memory units.

        For SambaNova: PCU→PMU→PCU switch traversal
        For Graphcore: BSP exchange phase
        For Cerebras: 2D mesh routing (included in compute time)
        """
        if self.hw.execution_model == "wafer_scale":
            # Cerebras: on-chip BW is so high (20 PB/s) that intra-chip
            # communication is essentially free relative to compute
            metrics.t_intra_chip_s = t_compute_s * 0.01  # ~1% overhead

        elif self.hw.execution_model == "bsp":
            # Graphcore BSP exchange phase
            # Exchange time ≈ max_tile_exchange_bytes / exchange_fabric_bw
            # Approx: 15-25% of compute time for dense layers
            exchange_bw = self.hw.exchange_fabric_bandwidth_gbps * 1e9
            total_activation_bytes = graph.total_activation_bytes
            if exchange_bw > 0:
                metrics.t_intra_chip_s = total_activation_bytes / exchange_bw
            else:
                metrics.t_intra_chip_s = t_compute_s * 0.20

        else:
            # SambaNova: PCU-PMU-PCU streaming
            # On-chip BW is high; overhead mainly from switch latency
            onchip_bw = self.hw.onchip_bandwidth_gbps * 1e9
            total_tensor_bytes = (
                graph.total_activation_bytes + graph.total_weight_bytes
            )
            if onchip_bw > 0:
                metrics.t_intra_chip_s = total_tensor_bytes / onchip_bw
            else:
                metrics.t_intra_chip_s = t_compute_s * 0.05

    # ─────────────────────────────────────────────────────────
    # Comm2: CCBU
    # ─────────────────────────────────────────────────────────

    def _compute_ccbu(self, metrics: CommMetrics, num_devices: int):
        """
        Comm2 — Collective Communication Bandwidth Utilization.

        CCBU = achieved_comm_bandwidth / peak_inter_chip_bandwidth

        achieved_comm_bandwidth = allreduce_bytes / T_allreduce
        """
        if (metrics.t_allreduce_s > 0
                and metrics.peak_inter_chip_bw_gbps > 0
                and num_devices > 1):
            achieved_bw = (
                metrics.allreduce_bytes_total / metrics.t_allreduce_s
            )
            metrics.achieved_comm_bw_gbps = achieved_bw / 1e9
            metrics.ccbu = min(
                achieved_bw / (metrics.peak_inter_chip_bw_gbps * 1e9), 1.0
            )
        else:
            metrics.achieved_comm_bw_gbps = 0.0
            metrics.ccbu = 0.0

    # ─────────────────────────────────────────────────────────
    # Comm3: Synchronization Overhead Fraction
    # ─────────────────────────────────────────────────────────

    def _compute_sof(
        self,
        graph: ComputationGraph,
        metrics: CommMetrics,
        t_compute_s: float,
        num_devices: int
    ):
        """
        Comm3 — Synchronization Overhead Fraction.

        SOF = T_sync / T_total_iter

        T_sync varies by platform:

        Graphcore (BSP):
          T_bsp_barrier = imbalance_factor * T_compute_phase
          imbalance_factor ≈ 0.05–0.15 depending on operator heterogeneity

        SambaNova (pipeline stall):
          T_stall = (1 - PSB) * T_pipeline
          PSB ≈ 0.85 for typical models (15% of time in stalls)

        Cerebras (gradient sync):
          T_sync ≈ 0 for single device (SwarmX handles it)
          T_sync = T_allreduce for multi-device
        """
        t_sync = 0.0

        if self.hw.execution_model == "bsp":
            # Graphcore BSP barrier
            # Imbalance factor: how much the slowest tile exceeds average
            # For heterogeneous operator models (transformers), imbalance is higher
            is_transformer = graph.model_type == "transformer"
            imbalance_factor = 0.12 if is_transformer else 0.06
            t_bsp = imbalance_factor * t_compute_s
            metrics.t_bsp_barrier_s = t_bsp
            t_sync += t_bsp

        elif self.hw.execution_model == "dataflow":
            # SambaNova pipeline stall
            # Pipeline stall occurs when one PCU stage is slower than its neighbours
            # Stall fraction ≈ 0.08–0.15 depending on pipeline balance
            is_heterogeneous = (
                len(set(op.op_type.value for op in graph.operators)) > 4
            )
            stall_fraction = 0.12 if is_heterogeneous else 0.06
            t_stall = stall_fraction * t_compute_s
            metrics.t_pipeline_stall_s = t_stall
            t_sync += t_stall

        elif self.hw.execution_model == "wafer_scale":
            # Cerebras: minimal sync for single device
            # For multi-device: gradient sync via SwarmX
            t_sync += metrics.t_allreduce_s * 0.05  # SwarmX has very low overhead

        # Multi-device gradient sync barrier (all execution models)
        if num_devices > 1:
            # Time waiting at gradient sync barrier
            # ≈ 5% overhead on top of all-reduce time (barrier coordination)
            t_inter_sync = metrics.t_allreduce_s * 0.05
            metrics.t_inter_device_sync_s = t_inter_sync
            t_sync += t_inter_sync

        metrics.t_sync_s = t_sync

        # SOF = sync time / total iteration time
        t_total = t_compute_s + metrics.t_comm_total_s + t_sync
        metrics.sof = t_sync / t_total if t_total > 0 else 0.0

    # ─────────────────────────────────────────────────────────
    # Scaling Efficiency
    # ─────────────────────────────────────────────────────────

    def _compute_scaling_efficiency(
        self, metrics: CommMetrics, num_devices: int
    ):
        """
        Scaling efficiency using Amdahl's Law with communication overhead.

        η_scale(N) = 1 / (1 + CCR * (N-1)/N)

        For Cerebras (near-linear scaling):
          η_scale ≈ 1.0 due to SwarmX
        """
        if num_devices <= 1:
            metrics.scaling_efficiency = 1.0
            return

        if self.hw.execution_model == "wafer_scale":
            # Cerebras near-linear scaling
            metrics.scaling_efficiency = 0.98
        else:
            ccr_adjusted = metrics.ccr * (num_devices - 1) / num_devices
            metrics.scaling_efficiency = 1.0 / (1.0 + ccr_adjusted)


def run_communication_module(
    graph: ComputationGraph,
    hardware: HardwareSpec,
    t_compute_s: float,
    gradient_bytes: int,
    num_devices: int = 1,
    parallelism: str = "data"
) -> CommMetrics:
    """Run the full communication module and return all metrics."""
    module = CommunicationModule(hardware)
    return module.compute(
        graph, t_compute_s, gradient_bytes, num_devices, parallelism
    )
