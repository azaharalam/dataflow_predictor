"""
analysis/reporter.py

Generates clean tabular reports and CSV output
for all 9 metrics and 7 training time equations.
"""

import csv
import json
import os
from typing import List, Optional, Dict
from datetime import datetime

from ..core.compute_module import ComputeMetrics
from ..core.memory_module import MemoryMetrics
from ..core.communication_module import CommMetrics
from ..core.training_time import TrainingTimePrediction


class Reporter:
    """Formats and saves analysis results."""

    def __init__(self, output_dir: str = "./results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def print_full_report(
        self,
        compute_m: ComputeMetrics,
        memory_m:  MemoryMetrics,
        comm_m:    CommMetrics,
        pred:      TrainingTimePrediction,
    ):
        """Print complete analysis to stdout."""
        print(compute_m.summary())
        print(memory_m.summary())
        print(comm_m.summary())
        print(pred.summary())
        self._print_combined_table(compute_m, memory_m, comm_m, pred)

    def _print_combined_table(
        self,
        compute_m: ComputeMetrics,
        memory_m:  MemoryMetrics,
        comm_m:    CommMetrics,
        pred:      TrainingTimePrediction,
    ):
        """Print a single summary table of all 9 metrics."""
        print(f"\n{'='*60}")
        print(f"  9-METRIC SUMMARY TABLE")
        print(f"{'='*60}")
        print(f"  {'Metric':<45} {'Value':>12}")
        print(f"  {'-'*58}")

        rows = [
            ("COMPUTE MODULE", ""),
            ("  C1 - Total Compute Time (ms)",
             f"{compute_m.total_compute_time_s*1000:.3f}"),
            ("  C2 - MFU (%)",
             f"{compute_m.mfu_predicted*100:.2f}"),
            ("  C3 - Weighted Arithmetic Intensity (FLOP/byte)",
             f"{compute_m.ai_weighted_average:.2f}"),
            ("  C3 - Compute-bound FLOPs fraction (%)",
             f"{compute_m.fraction_compute_bound*100:.1f}"),
            ("  C3 - Pipeline Fill Efficiency / SPU (%)",
             f"{compute_m.pipeline_fill_efficiency*100:.2f}"),
            ("", ""),
            ("MEMORY MODULE", ""),
            ("  M1 - MBU (%)",
             f"{memory_m.mbu*100:.2f}"),
            ("  M2 - Scratchpad Reuse Efficiency SRE (%)",
             f"{memory_m.sre*100:.2f}"),
            ("  M3 - Off-Chip Traffic Amplification (OTAF)",
             f"{memory_m.otaf:.3f}x"),
            ("  M3 - Actual Off-Chip Traffic (GB)",
             f"{memory_m.actual_offchip_bytes/1e9:.3f}"),
            ("  Memory: Peak Demand (GB)",
             f"{memory_m.peak_memory_demand_gb:.3f}"),
            ("  Memory: Recommended Batch Size",
             f"{memory_m.recommended_batch_size}"),
            ("", ""),
            ("COMMUNICATION MODULE", ""),
            ("  Comm1 - CCR",
             f"{comm_m.ccr:.4f}"),
            ("  Comm2 - CCBU (%)",
             f"{comm_m.ccbu*100:.2f}"),
            ("  Comm3 - SOF (%)",
             f"{comm_m.sof*100:.2f}"),
            ("  Comm: All-Reduce Time (ms)",
             f"{comm_m.t_allreduce_s*1000:.3f}"),
            ("  Comm: Scaling Efficiency (%)",
             f"{comm_m.scaling_efficiency*100:.2f}"),
            ("", ""),
            ("TRAINING TIME PREDICTIONS (ms/iter)", ""),
        ]

        for label, value in rows:
            if value == "":
                if label == "":
                    print(f"  {'-'*58}")
                else:
                    print(f"\n  {label}")
            else:
                print(f"  {label:<45} {value:>12}")

        # Training time equations
        for eq_name, t_ms in pred.predictions_dict().items():
            thr = pred.batch_size / (t_ms / 1000) if t_ms > 0 else 0
            print(f"  {eq_name:<45} {t_ms:>10.3f}   ({thr:.0f} samp/s)")

        print(f"\n  Dominant Bottleneck: {pred.bottleneck.upper()}")
        print(f"{'='*60}")

    def save_json(
        self,
        compute_m: ComputeMetrics,
        memory_m:  MemoryMetrics,
        comm_m:    CommMetrics,
        pred:      TrainingTimePrediction,
        filename:  Optional[str] = None,
    ) -> str:
        """Save all results to a JSON file."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{pred.model_name}_{pred.hardware_name}_{ts}.json"
        filepath = os.path.join(self.output_dir, filename)

        data = {
            "metadata": {
                "model":    pred.model_name,
                "hardware": pred.hardware_name,
                "batch_size": pred.batch_size,
                "num_devices": pred.num_devices,
                "timestamp": datetime.now().isoformat(),
            },
            "compute_metrics": {
                "C1_total_compute_time_ms": compute_m.total_compute_time_s * 1000,
                "C2_mfu_percent": compute_m.mfu_predicted * 100,
                "C3_weighted_ai": compute_m.ai_weighted_average,
                "C3_ridge_point": compute_m.ai_ridge_point,
                "C3_compute_bound_fraction": compute_m.fraction_compute_bound,
                "C3_memory_bound_fraction": compute_m.fraction_memory_bound,
                "pipeline_fill_efficiency": compute_m.pipeline_fill_efficiency,
            },
            "memory_metrics": {
                "M1_mbu_percent": memory_m.mbu * 100,
                "M2_sre_percent": memory_m.sre * 100,
                "M3_otaf": memory_m.otaf,
                "M3_actual_offchip_gb": memory_m.actual_offchip_bytes / 1e9,
                "M3_minimum_offchip_gb": memory_m.minimum_offchip_bytes / 1e9,
                "M3_spill_gb": memory_m.spill_bytes / 1e9,
                "peak_memory_demand_gb": memory_m.peak_memory_demand_gb,
                "fits_on_chip": memory_m.fits_on_chip,
                "recommended_batch_size": memory_m.recommended_batch_size,
            },
            "communication_metrics": {
                "Comm1_ccr": comm_m.ccr,
                "Comm2_ccbu_percent": comm_m.ccbu * 100,
                "Comm3_sof_percent": comm_m.sof * 100,
                "allreduce_time_ms": comm_m.t_allreduce_s * 1000,
                "sync_time_ms": comm_m.t_sync_s * 1000,
                "scaling_efficiency_percent": comm_m.scaling_efficiency * 100,
            },
            "training_time_ms": pred.predictions_dict(),
            "throughput_samples_per_sec": pred.throughput_dict(),
            "bottleneck": pred.bottleneck,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Results saved to: {filepath}")
        return filepath

    def save_csv(
        self,
        results_list: List[Dict],
        filename: str = "comparison_results.csv",
    ) -> str:
        """
        Save comparison results for multiple model/hardware combos to CSV.
        Each item in results_list is a flat dict of metric values.
        """
        filepath = os.path.join(self.output_dir, filename)
        if not results_list:
            return filepath

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
            writer.writeheader()
            writer.writerows(results_list)

        print(f"\n  CSV saved to: {filepath}")
        return filepath
