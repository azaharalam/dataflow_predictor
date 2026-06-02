"""
validation/c1c2_validator.py

C1 (Operator-Level Execution Time / OLET) and
C2 (Model FLOPs Utilization / MFU) Validation.

What we validate:
  C1: predicted training throughput (samples/sec) vs measured GlobalRate
  C2: predicted MFU vs measured MFU derived from GlobalRate

Metric collected from each platform:
  Cerebras  — GlobalRate (samples/sec) from run.log
              "| Train ... GlobalRate=XX.XX samples/sec"
  SambaNova — samples/sec from SambaTune Section Report or training log
              "e2e samples_per_sec: XX.XX"
  Graphcore — samples/sec from training log or PopVision timeline

Validation outputs:
  - Relative error (%) between predicted and measured throughput
  - MFU error (%)
  - Equation-level breakdown: which of Eq1–Eq7 is closest to measured
  - Correction factor (measured / predicted) for roofline calibration
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThroughputMeasurement:
    """
    A single measured throughput data point from a hardware run.
    """
    platform: str               # "cerebras_wse2", "sambanova_sn30", "graphcore_bow_ipu"
    model_name: str
    measured_samples_per_sec: float
    batch_size: int
    seq_len: Optional[int] = None
    num_devices: int = 1
    dtype: str = "fp16"
    notes: str = ""


@dataclass
class EquationResult:
    """Predicted throughput from one of the 7 training time equations."""
    equation_id: int            # 1–7
    equation_name: str
    predicted_time_ms: float
    predicted_samples_per_sec: float
    relative_error_pct: float   # vs measured (signed: + = overpredicts time)
    throughput_error_pct: float # vs measured (signed: + = underpredicts throughput)


@dataclass
class C1C2ValidationReport:
    """
    Full C1/C2 validation report for one model × hardware run.
    """
    platform: str
    model_name: str
    hardware_name: str
    batch_size: int

    # Measured ground truth
    measured_samples_per_sec: float = 0.0
    measured_time_ms: float = 0.0       # derived: batch_size / samples_per_sec * 1000
    measured_mfu: float = 0.0           # derived from measured throughput

    # Predicted values
    predicted_samples_per_sec_eq1: float = 0.0   # Eq1 (Operator Decomposition)
    predicted_mfu: float = 0.0

    # Per-equation breakdown
    equation_results: List[EquationResult] = field(default_factory=list)

    # Best equation
    best_equation_id: int = -1
    best_equation_name: str = ""
    best_equation_error_pct: float = 999.0

    # Correction factor: measured / predicted_eq1
    correction_factor: float = 1.0

    # C1 error (using Eq1 as the canonical C1 prediction)
    c1_relative_error_pct: float = 0.0

    # C2 error
    c2_absolute_error_pct: float = 0.0   # |predicted_mfu - measured_mfu| in pct points

    notes: List[str] = field(default_factory=list)

    def print_report(self):
        sep = "─" * 68
        print(f"\n{sep}")
        print(f"  C1/C2 Throughput & MFU Validation")
        print(f"  Platform : {self.platform}")
        print(f"  Model    : {self.model_name}")
        print(f"  Hardware : {self.hardware_name}")
        print(f"  Batch    : {self.batch_size}")
        print(sep)

        print(f"\n  [C1] Throughput (samples/sec)")
        print(f"    Measured   : {self.measured_samples_per_sec:>10.2f} samp/s")
        print(f"    Predicted  : {self.predicted_samples_per_sec_eq1:>10.2f} samp/s  (Eq1)")
        err = self.c1_relative_error_pct
        direction = "overpredicts" if err < 0 else "underpredicts"
        print(f"    Error      : {abs(err):>9.1f}%  (model {direction} throughput)")
        print(f"    Corr factor: {self.correction_factor:>10.3f}x  (measured / predicted)")

        print(f"\n  [C2] MFU (%)")
        print(f"    Measured   : {self.measured_mfu*100:>9.2f}%")
        print(f"    Predicted  : {self.predicted_mfu*100:>9.2f}%")
        print(f"    Delta      : {self.c2_absolute_error_pct:>+9.2f} pct points")

        print(f"\n  [Equation Breakdown]")
        print(f"  {'Eq':<4} {'Name':<30} {'Pred(samp/s)':>12} {'Error%':>8}")
        print(f"  {'-'*58}")
        for eq in sorted(self.equation_results, key=lambda e: abs(e.throughput_error_pct)):
            marker = " ◀ best" if eq.equation_id == self.best_equation_id else ""
            print(f"  Eq{eq.equation_id:<3} {eq.equation_name:<30} "
                  f"{eq.predicted_samples_per_sec:>12.2f} "
                  f"{eq.throughput_error_pct:>+7.1f}%{marker}")

        print(f"\n  Best equation: Eq{self.best_equation_id} "
              f"({self.best_equation_name}) — "
              f"error = {abs(self.best_equation_error_pct):.1f}%")

        for note in self.notes:
            print(f"\n  NOTE: {note}")
        print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Log parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_cerebras_log(log_path: str) -> ThroughputMeasurement:
    """
    Parse a Cerebras run.log and extract stable GlobalRate (samples/sec).

    Looks for lines like:
      | Train Device=CSX, Step=N, Loss=X, Rate=X, GlobalRate=X samples/sec, ...

    Takes the median of the last 20 stable steps (after warmup) to avoid
    including the slow first steps where the pipeline is filling.

    Args:
        log_path: path to run.log file

    Returns:
        ThroughputMeasurement with measured_samples_per_sec set
    """
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Cerebras log not found: {log_path}")

    pattern = re.compile(
        r"GlobalRate=(\d+\.?\d*)\s+samples/sec"
    )
    # Also try to extract model name and batch size from the log header
    model_pattern = re.compile(r"model_dir[_/](\w+)")
    batch_pattern  = re.compile(r"train_batch_size[:\s=]+(\d+)")

    global_rates = []
    model_name = "unknown"
    batch_size = 1

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                global_rates.append(float(m.group(1)))
            mb = model_pattern.search(line)
            if mb:
                model_name = mb.group(1)
            bb = batch_pattern.search(line)
            if bb:
                batch_size = int(bb.group(1))

    if not global_rates:
        raise ValueError(
            f"No GlobalRate values found in {log_path}.\n"
            "Ensure the run has started producing training steps."
        )

    # Skip first 5 steps (pipeline warmup), take median of the rest
    stable = global_rates[5:] if len(global_rates) > 5 else global_rates
    stable_sorted = sorted(stable)
    mid = len(stable_sorted) // 2
    if len(stable_sorted) % 2 == 0:
        median_rate = (stable_sorted[mid - 1] + stable_sorted[mid]) / 2
    else:
        median_rate = stable_sorted[mid]

    return ThroughputMeasurement(
        platform="cerebras_wse2",
        model_name=model_name,
        measured_samples_per_sec=median_rate,
        batch_size=batch_size,
        notes=f"Parsed {len(global_rates)} steps; "
              f"median of {len(stable)} stable steps (skip first 5). "
              f"Min={min(stable):.1f}, Max={max(stable):.1f}, "
              f"Median={median_rate:.2f} samp/s"
    )


def parse_sambanova_log(log_path: str) -> ThroughputMeasurement:
    """
    Parse a SambaNova training log and extract samples/sec.

    Looks for lines like:
      inner train loop time : X.XX for N epochs, ..., e2e samples_per_sec: X.XX
    or:
      Throughput: X.XX samples/sec

    Args:
        log_path: path to SambaNova output log file

    Returns:
        ThroughputMeasurement
    """
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"SambaNova log not found: {log_path}")

    # SambaNova multi-node pattern
    pattern1 = re.compile(r"e2e samples_per_sec:\s*(\d+\.?\d*)")
    # General throughput pattern
    pattern2 = re.compile(r"[Tt]hroughput[:\s]+(\d+\.?\d*)\s*samples?/sec")
    # SambaTune pattern
    pattern3 = re.compile(r"samples_per_second[:\s=]+(\d+\.?\d*)")

    rates = []
    with open(log_path) as f:
        for line in f:
            for pat in [pattern1, pattern2, pattern3]:
                m = pat.search(line)
                if m:
                    rates.append(float(m.group(1)))
                    break

    if not rates:
        raise ValueError(
            f"No throughput values found in {log_path}.\n"
            "Expected 'e2e samples_per_sec: X.XX' lines from SambaNova log."
        )

    # Average across parallel instances (SambaNova reports per-instance)
    avg_rate = sum(rates) / len(rates)

    return ThroughputMeasurement(
        platform="sambanova_sn30",
        model_name="unknown",
        measured_samples_per_sec=avg_rate,
        batch_size=1,
        notes=f"Averaged {len(rates)} throughput readings. "
              f"Min={min(rates):.1f}, Max={max(rates):.1f}, "
              f"Avg={avg_rate:.2f} samp/s"
    )


def parse_graphcore_log(log_path: str) -> ThroughputMeasurement:
    """
    Parse a Graphcore IPU training log and extract samples/sec.

    Looks for lines like:
      throughput: X.XX samples/sec
      Throughput: X items/sec
      step X, loss X.XX, throughput=X.XX

    Args:
        log_path: path to Graphcore training log

    Returns:
        ThroughputMeasurement
    """
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Graphcore log not found: {log_path}")

    patterns = [
        re.compile(r"[Tt]hroughput[:\s=]+(\d+\.?\d*)\s*(?:samples?|items?)/sec"),
        re.compile(r"throughput=(\d+\.?\d*)"),
        re.compile(r"(\d+\.?\d*)\s+(?:samples?|items?)/s(?:ec)?"),
    ]

    rates = []
    with open(log_path) as f:
        for line in f:
            for pat in patterns:
                m = pat.search(line)
                if m:
                    rates.append(float(m.group(1)))
                    break

    if not rates:
        raise ValueError(
            f"No throughput values found in {log_path}.\n"
            "Expected 'throughput: X.XX samples/sec' lines."
        )

    stable = rates[3:] if len(rates) > 3 else rates
    avg = sum(stable) / len(stable)

    return ThroughputMeasurement(
        platform="graphcore_bow_ipu",
        model_name="unknown",
        measured_samples_per_sec=avg,
        batch_size=1,
        notes=f"Averaged {len(stable)} stable throughput readings. "
              f"Avg={avg:.2f} samp/s"
    )


def measurement_from_json(json_path: str, platform: str) -> ThroughputMeasurement:
    """
    Load a measurement from a JSON file (manually recorded or from script).

    Expected JSON format:
    {
      "model_name": "gpt3_2p7b",
      "platform": "cerebras_wse2",
      "measured_samples_per_sec": 59.2,
      "batch_size": 32,
      "seq_len": 2048,
      "num_devices": 1,
      "dtype": "fp16",
      "notes": "Step 100-500 median"
    }
    """
    with open(json_path) as f:
        d = json.load(f)
    return ThroughputMeasurement(
        platform=d.get("platform", platform),
        model_name=d.get("model_name", "unknown"),
        measured_samples_per_sec=float(d["measured_samples_per_sec"]),
        batch_size=int(d.get("batch_size", 1)),
        seq_len=d.get("seq_len"),
        num_devices=int(d.get("num_devices", 1)),
        dtype=d.get("dtype", "fp16"),
        notes=d.get("notes", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core validation function
# ─────────────────────────────────────────────────────────────────────────────

def validate_c1c2(
    measurement: ThroughputMeasurement,
    training_time_predictions,   # TrainingTimePrediction object or dict
    compute_metrics,
    hardware_spec,
) -> C1C2ValidationReport:
    """
    Validate C1 (throughput) and C2 (MFU) against measured data.

    Args:
        measurement:              ThroughputMeasurement from log parser
        training_time_predictions: TrainingTimePrediction object from
                                   predict_training_time(), or a dict
                                   {eq_id: (time_ms, name)}
        compute_metrics:          ComputeMetrics from run_compute_module()
        hardware_spec:            HardwareSpec for the platform

    Returns:
        C1C2ValidationReport
    """
    # Normalise input — accept TrainingTimePrediction object or raw dict
    if hasattr(training_time_predictions, 'eq1_operator_decomposition_ms'):
        p = training_time_predictions
        training_time_predictions = {
            1: (p.eq1_operator_decomposition_ms, "Operator Decomposition"),
            2: (p.eq2_roofline_bounded_ms,       "Roofline Bounded"),
            3: (p.eq3_streaming_pipeline_ms,     "Streaming Pipeline"),
            4: (p.eq4_memory_traffic_driven_ms,  "Memory Traffic Driven"),
            5: (p.eq5_mfu_normalized_ms,         "MFU Normalized"),
            6: (p.eq6_comm_aware_scaling_ms,     "Comm-Aware Scaling"),
            7: (p.eq7_hierarchical_bottleneck_ms,"Hierarchical Bottleneck"),
        }
    bs = measurement.batch_size
    measured_sps = measurement.measured_samples_per_sec
    measured_time_ms = (bs / measured_sps * 1000.0) if measured_sps > 0 else 0.0

    # Measured MFU: derived from measured throughput
    # MFU = (measured_samples/sec × FLOPs_per_sample) / peak_FLOPS
    total_flops = compute_metrics.total_flops_model   # forward + backward
    peak_flops  = hardware_spec.peak_flops_ops("fp16")
    flops_per_sample = total_flops / bs if bs > 0 else 0.0
    measured_mfu = (
        (measured_sps * flops_per_sample) / peak_flops
        if peak_flops > 0 else 0.0
    )

    report = C1C2ValidationReport(
        platform=measurement.platform,
        model_name=measurement.model_name or compute_metrics.model_name,
        hardware_name=hardware_spec.name,
        batch_size=bs,
        measured_samples_per_sec=measured_sps,
        measured_time_ms=measured_time_ms,
        measured_mfu=min(measured_mfu, 1.0),   # cap at 100%
        predicted_mfu=compute_metrics.mfu_predicted,
    )

    # Per-equation errors
    eq1_pred_sps = None
    best_err = 999.0
    best_eq_id = -1
    best_eq_name = ""

    for eq_id, (pred_time_ms, eq_name) in training_time_predictions.items():
        if pred_time_ms <= 0:
            continue
        pred_sps = bs / (pred_time_ms / 1000.0)

        # Relative error on throughput: (pred - meas) / meas × 100
        tp_err = (pred_sps - measured_sps) / measured_sps * 100.0 if measured_sps > 0 else 0.0
        # Relative error on time: (pred_time - meas_time) / meas_time × 100
        time_err = (pred_time_ms - measured_time_ms) / measured_time_ms * 100.0 if measured_time_ms > 0 else 0.0

        eq_result = EquationResult(
            equation_id=eq_id,
            equation_name=eq_name,
            predicted_time_ms=pred_time_ms,
            predicted_samples_per_sec=pred_sps,
            relative_error_pct=time_err,
            throughput_error_pct=tp_err,
        )
        report.equation_results.append(eq_result)

        if eq_id == 1:
            eq1_pred_sps = pred_sps

        if abs(tp_err) < abs(best_err):
            best_err = tp_err
            best_eq_id = eq_id
            best_eq_name = eq_name

    report.best_equation_id = best_eq_id
    report.best_equation_name = best_eq_name
    report.best_equation_error_pct = best_err

    # C1 error uses Eq1
    if eq1_pred_sps and eq1_pred_sps > 0:
        report.predicted_samples_per_sec_eq1 = eq1_pred_sps
        report.c1_relative_error_pct = (
            (eq1_pred_sps - measured_sps) / measured_sps * 100.0
        )
        report.correction_factor = measured_sps / eq1_pred_sps

    # C2 error: absolute difference in MFU percentage points
    report.c2_absolute_error_pct = (
        (compute_metrics.mfu_predicted - min(measured_mfu, 1.0)) * 100.0
    )

    # Notes
    report.notes.append(measurement.notes)
    report.notes.append(
        f"C1: Eq1 predicts {eq1_pred_sps:.1f} samp/s vs measured {measured_sps:.1f} samp/s "
        f"→ correction factor {report.correction_factor:.3f}x"
    )
    if report.correction_factor > 2.0:
        report.notes.append(
            f"Large correction factor ({report.correction_factor:.2f}x) suggests the model "
            "does not account for hardware-level pipeline overlap or execution hiding. "
            "Consider applying this factor as a platform-specific calibration constant."
        )
    report.notes.append(
        f"Best-fit equation: Eq{best_eq_id} ({best_eq_name}) with "
        f"{abs(best_err):.1f}% throughput error."
    )

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Multi-model summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_validation_summary(reports: List[C1C2ValidationReport]):
    """
    Print a compact summary table across multiple model validation reports.
    Useful for comparing prediction accuracy across all 8 benchmarks.
    """
    sep = "═" * 90
    print(f"\n{sep}")
    print(f"  C1/C2 VALIDATION SUMMARY — {reports[0].platform if reports else ''}")
    print(sep)
    print(f"  {'Model':<22} {'Meas(s/s)':>10} {'Pred(s/s)':>10} "
          f"{'C1 Err%':>9} {'CorrF':>7} "
          f"{'MFU_meas':>9} {'MFU_pred':>9} {'C2 Δppt':>8} {'BestEq':>7}")
    print(f"  {'-'*88}")
    for r in reports:
        print(
            f"  {r.model_name:<22} "
            f"{r.measured_samples_per_sec:>10.1f} "
            f"{r.predicted_samples_per_sec_eq1:>10.1f} "
            f"{r.c1_relative_error_pct:>+8.1f}% "
            f"{r.correction_factor:>7.3f}x "
            f"{r.measured_mfu*100:>8.1f}% "
            f"{r.predicted_mfu*100:>8.1f}% "
            f"{r.c2_absolute_error_pct:>+7.1f}pp "
            f"Eq{r.best_equation_id:>4}"
        )
    print(sep)

    if reports:
        avg_c1_err = sum(abs(r.c1_relative_error_pct) for r in reports) / len(reports)
        avg_corr   = sum(r.correction_factor for r in reports) / len(reports)
        avg_c2_err = sum(abs(r.c2_absolute_error_pct) for r in reports) / len(reports)
        print(f"\n  Mean absolute C1 error : {avg_c1_err:.1f}%")
        print(f"  Mean correction factor : {avg_corr:.3f}x")
        print(f"  Mean absolute C2 error : {avg_c2_err:.1f} pct points")
        print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: validate directly from a Cerebras run.log
# ─────────────────────────────────────────────────────────────────────────────

def validate_cerebras_from_log(
    log_path: str,
    model_name: str,
    batch_size: int,
    training_time_predictions: Dict,
    compute_metrics,
    hardware_spec,
) -> C1C2ValidationReport:
    """
    One-call convenience function: parse Cerebras log + validate C1/C2.

    Args:
        log_path:                  path to run.log
        model_name:                model identifier string
        batch_size:                batch size used in the run
        training_time_predictions: from run_training_time_module()
        compute_metrics:           from run_compute_module()
        hardware_spec:             HardwareSpec for cerebras_wse2

    Returns:
        C1C2ValidationReport
    """
    meas = parse_cerebras_log(log_path)
    meas.model_name = model_name
    meas.batch_size = batch_size
    return validate_c1c2(
        meas, training_time_predictions, compute_metrics, hardware_spec
    )