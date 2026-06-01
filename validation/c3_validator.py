"""
validation/c3_validator.py

C3 (Arithmetic Intensity) — Classification Accuracy Validation.

Validates whether your model correctly predicts compute-bound vs
memory-bound for each operator on each platform.

ONLY metric: Classification Accuracy
  = (operators correctly classified) / (total operators with nonzero FLOPs)

How ground-truth is collected per platform:

  Graphcore (PopVision Graph Analyser):
    - Per-operation cycle counts  (Execution Trace)
    - Per-operation FLOPs         (Operations Summary, requires profiler.includeFlopEstimates)
    - Ground truth classification: achieved_FLOPS/s vs peak_FLOPS/s
      If achieved >= threshold * peak → compute-bound
      Else                             → memory-bound
    - Granularity: per operator (best platform for C3 validation)

  SambaNova (SambaTune):
    - Per-section latency (ms) and DDR bandwidth (GB/s)  (Section Report)
    - Ground truth classification per SECTION (not per operator):
      If DDR utilization >= threshold * peak_DDR → memory-bound
      Else                                        → compute-bound
    - Granularity: per fused section (operators grouped by compiler)
    - You map sections → ops using SambaTune Stack Tracing report

  Cerebras (cstorch profiler):
    - Only model-level flops_utilization (single aggregate number)
    - Ground truth: model-level classification only (1 data point)
    - If flops_utilization >= threshold → model is compute-bound
    - Granularity: whole model only (weakest platform for C3 validation)

Input format for each platform is a plain CSV or dict — no special libraries.

Usage:
  from validation.c3_validator import (
      validate_graphcore, validate_sambanova, validate_cerebras,
      load_measurements_csv, print_classification_report
  )
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models.operator_graph import ComputationGraph, OperatorNode
from hardware.base_hardware import HardwareSpec
from core.compute_module import ComputeMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """
    Per-operator or per-section classification comparison.
    predicted_compute_bound : what your model predicted
    measured_compute_bound  : what the hardware profiler says
    correct                 : whether they agree
    """
    identifier: str            # op_id, section_id, or "model_level"
    op_type: str               # operator type string (GEMM, LAYERNORM, etc.)
    predicted_ai: float        # predicted AI_training (FLOP/byte)
    ridge_point: float         # hardware ridge point (FLOP/byte)
    predicted_compute_bound: bool
    measured_compute_bound: bool
    correct: bool
    # Extra measured values stored for transparency
    measured_value: float = 0.0   # cycles, DDR util fraction, or MFU
    measured_label: str = ""       # "cycles", "ddr_utilization", "flops_utilization"


@dataclass
class C3ValidationReport:
    """
    Full validation report for one model × hardware run.
    """
    platform: str
    model_name: str
    hardware_name: str
    ridge_point: float

    results: List[ClassificationResult] = field(default_factory=list)

    # Aggregate accuracy
    total_evaluated: int = 0
    total_correct: int = 0
    classification_accuracy: float = 0.0   # fraction [0, 1]

    # Breakdown by operator type
    accuracy_by_type: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # key = op_type string, value = (correct, total)

    notes: List[str] = field(default_factory=list)

    def compute_accuracy(self):
        """Compute accuracy from results list. Call after populating results."""
        self.total_evaluated = len(self.results)
        self.total_correct   = sum(1 for r in self.results if r.correct)
        self.classification_accuracy = (
            self.total_correct / self.total_evaluated
            if self.total_evaluated > 0 else 0.0
        )
        # Per-type breakdown
        by_type: Dict[str, List[bool]] = {}
        for r in self.results:
            by_type.setdefault(r.op_type, []).append(r.correct)
        self.accuracy_by_type = {
            t: (sum(v), len(v)) for t, v in by_type.items()
        }

    def print_report(self):
        """Print full validation report to terminal."""
        sep = "─" * 64
        print(f"\n{sep}")
        print(f"  C3 Classification Accuracy Validation")
        print(f"  Platform : {self.platform}")
        print(f"  Model    : {self.model_name}")
        print(f"  Hardware : {self.hardware_name}")
        print(f"  Ridge pt : {self.ridge_point:.1f} FLOP/byte")
        print(sep)
        print(f"  Overall accuracy : {self.classification_accuracy*100:.1f}%"
              f"  ({self.total_correct}/{self.total_evaluated})")
        print()

        if self.accuracy_by_type:
            print(f"  Accuracy by operator type:")
            print(f"  {'Type':<20} {'Correct':>8} {'Total':>7} {'Acc%':>7}")
            print(f"  {'-'*46}")
            for op_type, (correct, total) in sorted(
                self.accuracy_by_type.items(),
                key=lambda x: -x[1][1]
            ):
                acc = correct / total * 100 if total > 0 else 0.0
                print(f"  {op_type:<20} {correct:>8} {total:>7} {acc:>6.1f}%")
            print()

        # Show misclassified operators
        wrong = [r for r in self.results if not r.correct]
        if wrong:
            print(f"  Misclassified ({len(wrong)} operators):")
            print(f"  {'ID':<30} {'Type':<12} {'PredAI':>8} {'Ridge':>7}"
                  f" {'Pred':>7} {'Meas':>7}")
            print(f"  {'-'*75}")
            for r in wrong[:20]:   # cap at 20
                pred_label = "COMP" if r.predicted_compute_bound else "MEM"
                meas_label = "COMP" if r.measured_compute_bound else "MEM"
                print(f"  {r.identifier:<30} {r.op_type:<12} "
                      f"{r.predicted_ai:>8.2f} {r.ridge_point:>7.1f}"
                      f" {pred_label:>7} {meas_label:>7}")
            if len(wrong) > 20:
                print(f"  ... ({len(wrong) - 20} more misclassified)")
        else:
            print("  All operators correctly classified.")

        for note in self.notes:
            print(f"\n  NOTE: {note}")
        print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Graphcore validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_graphcore(
    compute_metrics: ComputeMetrics,
    graph: ComputationGraph,
    hw: HardwareSpec,
    measurements: List[Dict],
    compute_bound_threshold: float = 0.5,
) -> C3ValidationReport:
    """
    Validate C3 classification against PopVision per-operator measurements.

    Args:
        compute_metrics:  output of run_compute_module()
        graph:            annotated ComputationGraph
        hw:               HardwareSpec for graphcore_bow_ipu
        measurements:     list of dicts, one per operator, with keys:
                            op_name       (str)  — must match op.layer_name or op.op_id
                            cycles        (int)  — measured cycle count from PopVision
                            flops         (float)— measured FLOPs from PopVision
                                                   (requires profiler.includeFlopEstimates)
        compute_bound_threshold: fraction of peak FLOPS above which op is
                                 classified as compute-bound (default 0.5)

    Returns:
        C3ValidationReport

    How to collect measurements on Graphcore:
        1. Set engine option before compiling:
               import poplar
               poplar.set_engine_option("profiler.includeFlopEstimates", "true")
               poplar.set_engine_option("autoReport.outputExecutionReport", "true")
        2. Run your model. Profile files are written to profile.pop (or POPLAR_ENGINE_OPTIONS).
        3. Open profile.pop in PopVision Graph Analyser (desktop app).
        4. Operations Summary tab → export as CSV.
        5. Execution Trace tab → cycle counts per operation.
        6. Each row = one operation with: name, cycles, FLOPs.

    Clock speed used for FLOP/s calculation: Bow IPU = 1.85 GHz
    """
    clock_hz = 1.85e9   # Bow IPU clock (1.85 GHz confirmed from datasheet)
    peak_flops = hw.peak_flops_ops(graph.dtype)   # ops/sec
    ridge = hw.ridge_point_flops_per_byte

    report = C3ValidationReport(
        platform="graphcore_bow_ipu",
        model_name=graph.model_name,
        hardware_name=hw.name,
        ridge_point=ridge,
    )

    # Build lookup: predicted AI per operator, keyed by op_id and layer_name
    pred_ai_by_id:   Dict[str, float] = dict(compute_metrics.ai_per_operator)
    pred_bound_by_id: Dict[str, bool] = {}
    op_type_by_id:   Dict[str, str]   = {}
    for op in graph.operators:
        bound = op.is_compute_bound(ridge)
        pred_bound_by_id[op.op_id]     = bound
        pred_bound_by_id[op.layer_name] = bound
        op_type_by_id[op.op_id]        = op.op_type.value
        op_type_by_id[op.layer_name]   = op.op_type.value
        pred_ai_by_id[op.layer_name]   = op.arithmetic_intensity_training

    matched = 0
    unmatched = []

    for meas in measurements:
        op_name = str(meas.get("op_name", meas.get("name", "")))
        cycles  = float(meas.get("cycles", 0))
        flops   = float(meas.get("flops", meas.get("FLOPs", 0)))

        if cycles <= 0:
            continue

        # Match: try exact op_id, then layer_name substring
        key = None
        if op_name in pred_bound_by_id:
            key = op_name
        else:
            for k in pred_bound_by_id:
                if k and (k in op_name or op_name in k):
                    key = k
                    break

        if key is None:
            unmatched.append(op_name)
            continue

        # Ground truth: achieved FLOP/s vs peak
        time_s = cycles / clock_hz
        achieved_flops_per_sec = flops / time_s if time_s > 0 else 0.0
        measured_compute_bound = (
            achieved_flops_per_sec >= compute_bound_threshold * peak_flops
        )

        predicted_compute_bound = pred_bound_by_id[key]
        predicted_ai            = pred_ai_by_id.get(key, 0.0)
        op_type                 = op_type_by_id.get(key, "unknown")

        result = ClassificationResult(
            identifier=op_name,
            op_type=op_type,
            predicted_ai=predicted_ai,
            ridge_point=ridge,
            predicted_compute_bound=predicted_compute_bound,
            measured_compute_bound=measured_compute_bound,
            correct=(predicted_compute_bound == measured_compute_bound),
            measured_value=achieved_flops_per_sec / peak_flops,
            measured_label="achieved_flops_fraction",
        )
        report.results.append(result)
        matched += 1

    report.compute_accuracy()

    if unmatched:
        report.notes.append(
            f"{len(unmatched)} operators from PopVision had no match in predicted graph: "
            f"{unmatched[:5]}{'...' if len(unmatched) > 5 else ''}"
        )
    report.notes.append(
        f"Matched {matched}/{len(measurements)} PopVision operators to predicted graph."
    )
    report.notes.append(
        f"Compute-bound threshold: achieved FLOP/s >= {compute_bound_threshold*100:.0f}% of peak."
    )

    return report


# ─────────────────────────────────────────────────────────────────────────────
# SambaNova validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_sambanova(
    compute_metrics: ComputeMetrics,
    graph: ComputationGraph,
    hw: HardwareSpec,
    sections: List[Dict],
    ddr_bound_threshold: float = 0.7,
) -> C3ValidationReport:
    """
    Validate C3 classification against SambaTune section-level measurements.

    Args:
        compute_metrics:  output of run_compute_module()
        graph:            annotated ComputationGraph
        hw:               HardwareSpec for sambanova_sn30
        sections:         list of dicts, one per section, with keys:
                            section_id    (int)   — section number from SambaTune
                            latency_ms    (float) — section execution time
                            ddr_bw_gbps   (float) — measured DDR bandwidth for section
                            op_ids        (str)   — comma-separated op_ids in this section
                                                    (from SambaTune Stack Tracing report)
        ddr_bound_threshold: DDR utilization fraction above which section is
                             classified memory-bound (default 0.7 = 70% of peak DDR)

    Returns:
        C3ValidationReport

    How to collect measurements on SambaNova:
        1. Run: sambatune --yaml <your_app.yaml>
           (See ALCF docs: https://docs.alcf.anl.gov/ai-testbed/sambanova/sambatune/)
        2. Open SambaTune UI → Section Report tab.
           Record for each section: section_id, latency_ms, DDR bandwidth (GB/s).
        3. Open SambaTune UI → Stack Tracing tab.
           For each section, see which PyTorch ops/layers map to it.
           Record the op_ids (matching your graph's op.op_id values).
        4. Fill sections list (or load from CSV, see load_measurements_csv).

    Classification logic:
        DDR utilization = ddr_bw_gbps / peak_ddr_bw_gbps
        If DDR utilization >= threshold → section is memory-bound
        Else                            → section is compute-bound (PCU-bound)
    """
    ridge = hw.ridge_point_flops_per_byte
    peak_ddr_bw = hw.ddr_bandwidth_gbps if hw.ddr_bandwidth_gbps > 0 else 200.0

    report = C3ValidationReport(
        platform="sambanova_sn30",
        model_name=graph.model_name,
        hardware_name=hw.name,
        ridge_point=ridge,
    )

    # Build lookup: op_id → predicted bound and AI
    pred_bound_by_id: Dict[str, bool]  = {}
    pred_ai_by_id:    Dict[str, float] = {}
    op_type_by_id:    Dict[str, str]   = {}
    for op in graph.operators:
        bound = op.is_compute_bound(ridge)
        pred_bound_by_id[op.op_id]     = bound
        pred_bound_by_id[op.layer_name] = bound
        pred_ai_by_id[op.op_id]        = op.arithmetic_intensity_training
        pred_ai_by_id[op.layer_name]   = op.arithmetic_intensity_training
        op_type_by_id[op.op_id]        = op.op_type.value
        op_type_by_id[op.layer_name]   = op.op_type.value

    for sec in sections:
        section_id   = str(sec.get("section_id", "?"))
        latency_ms   = float(sec.get("latency_ms", 0))
        ddr_bw_gbps  = float(sec.get("ddr_bw_gbps", sec.get("ddr_bandwidth_gbps", 0)))
        op_ids_raw   = str(sec.get("op_ids", ""))

        if latency_ms <= 0:
            continue

        # Parse op_ids for this section
        op_ids = [x.strip() for x in op_ids_raw.split(",") if x.strip()]

        # Ground truth: DDR utilization determines memory-bound vs compute-bound
        ddr_util = ddr_bw_gbps / peak_ddr_bw
        measured_compute_bound = (ddr_util < ddr_bound_threshold)

        if not op_ids:
            # No op mapping — record section-level result with unknown op type
            # Aggregate predicted classification for this section:
            # section is compute-bound if majority of its FLOPs are compute-bound
            report.notes.append(
                f"Section {section_id} has no op_ids. "
                "Use SambaTune Stack Tracing to map sections to ops."
            )
            continue

        # Aggregate predicted AI and classification for this section
        sec_flops = 0.0
        sec_compute_flops = 0.0
        sec_ai_values = []
        matched_types = []

        for oid in op_ids:
            matched_key = None
            if oid in pred_bound_by_id:
                matched_key = oid
            else:
                for k in pred_bound_by_id:
                    if k and (k in oid or oid in k):
                        matched_key = k
                        break
            if matched_key is None:
                continue

            # Get the operator node for FLOPs
            op_node = next(
                (op for op in graph.operators
                 if op.op_id == matched_key or op.layer_name == matched_key),
                None
            )
            if op_node:
                op_flops = op_node.flops_forward + op_node.flops_backward
                sec_flops += op_flops
                if pred_bound_by_id[matched_key]:
                    sec_compute_flops += op_flops
                sec_ai_values.append(pred_ai_by_id[matched_key])
                matched_types.append(op_type_by_id[matched_key])

        if sec_flops == 0:
            predicted_compute_bound = False
            section_ai = 0.0
        else:
            # Section is predicted compute-bound if >50% of FLOPs are compute-bound
            predicted_compute_bound = (sec_compute_flops / sec_flops) >= 0.5
            section_ai = (
                sum(sec_ai_values) / len(sec_ai_values) if sec_ai_values else 0.0
            )

        dominant_type = max(set(matched_types), key=matched_types.count) if matched_types else "unknown"

        result = ClassificationResult(
            identifier=f"section_{section_id}",
            op_type=dominant_type,
            predicted_ai=section_ai,
            ridge_point=ridge,
            predicted_compute_bound=predicted_compute_bound,
            measured_compute_bound=measured_compute_bound,
            correct=(predicted_compute_bound == measured_compute_bound),
            measured_value=ddr_util,
            measured_label="ddr_utilization",
        )
        report.results.append(result)

    report.compute_accuracy()
    report.notes.append(
        f"SambaNova: validated at section granularity (fused operator groups)."
    )
    report.notes.append(
        f"Memory-bound threshold: DDR utilization >= {ddr_bound_threshold*100:.0f}% of "
        f"peak DDR BW ({peak_ddr_bw:.0f} GB/s)."
    )
    report.notes.append(
        "Map sections → ops using SambaTune UI → Stack Tracing tab."
    )

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Cerebras validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_cerebras(
    compute_metrics: ComputeMetrics,
    graph: ComputationGraph,
    hw: HardwareSpec,
    flops_utilization: float,
    compute_bound_threshold: float = 0.5,
) -> C3ValidationReport:
    """
    Validate C3 classification against Cerebras model-level flops_utilization.

    Cerebras only exposes a single model-level flops_utilization value.
    This means only ONE classification decision can be validated.

    Args:
        compute_metrics:       output of run_compute_module()
        graph:                 annotated ComputationGraph
        hw:                    HardwareSpec for cerebras_wse2
        flops_utilization:     from cstorch profiler:
                               executor.profiler.rate_tracker.flops_utilization
                               (float in [0, 1])
        compute_bound_threshold: above this MFU → model is compute-bound
                                 (default 0.5)

    Returns:
        C3ValidationReport with exactly 1 result (model-level)

    How to collect flops_utilization on Cerebras:
        Add to your training script after the executor loop:

            import json
            result = {
                "model": MODEL_NAME,
                "flops_utilization": executor.profiler.rate_tracker.flops_utilization,
                "samples_per_sec":   executor.profiler.rate_tracker.samples_per_sec,
                "batch_size":        BATCH_SIZE,
            }
            with open(f"cerebras_meas_{MODEL_NAME}.json", "w") as f:
                json.dump(result, f, indent=2)

    Important limitation:
        This gives you exactly 1 data point (model-level).
        Classification accuracy is either 0% or 100% — not meaningful statistics.
        Cerebras is the weakest platform for per-operator C3 validation.
        Use Graphcore as your primary validation platform.
    """
    ridge = hw.ridge_point_flops_per_byte

    report = C3ValidationReport(
        platform="cerebras_wse2",
        model_name=graph.model_name,
        hardware_name=hw.name,
        ridge_point=ridge,
    )

    # Predicted: is the overall model compute-bound?
    # Use weighted average AI vs ridge point
    predicted_compute_bound = (
        compute_metrics.ai_weighted_average >= ridge
    )

    # Measured: is the model compute-bound based on MFU?
    measured_compute_bound = (flops_utilization >= compute_bound_threshold)

    result = ClassificationResult(
        identifier="model_level",
        op_type="model",
        predicted_ai=compute_metrics.ai_weighted_average,
        ridge_point=ridge,
        predicted_compute_bound=predicted_compute_bound,
        measured_compute_bound=measured_compute_bound,
        correct=(predicted_compute_bound == measured_compute_bound),
        measured_value=flops_utilization,
        measured_label="flops_utilization",
    )
    report.results.append(result)
    report.compute_accuracy()

    predicted_label = "compute-bound" if predicted_compute_bound else "memory-bound"
    measured_label  = "compute-bound" if measured_compute_bound else "memory-bound"

    report.notes.append(
        f"Predicted: AI_weighted={compute_metrics.ai_weighted_average:.1f} "
        f"vs ridge={ridge:.1f} → {predicted_label}"
    )
    report.notes.append(
        f"Measured:  flops_utilization={flops_utilization:.3f} → {measured_label}"
    )
    report.notes.append(
        "Cerebras gives 1 model-level data point only. "
        "Per-operator validation is not possible on this platform."
    )

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CSV loader — shared across all platforms
# ─────────────────────────────────────────────────────────────────────────────

def load_measurements_csv(filepath: str) -> List[Dict]:
    """
    Load measurements from a CSV file.

    Graphcore CSV columns (from PopVision Operations Summary export):
        op_name, cycles, flops

    SambaNova CSV columns (filled manually from SambaTune Section Report):
        section_id, latency_ms, ddr_bw_gbps, op_ids

    Cerebras: single-row CSV or JSON with:
        flops_utilization, samples_per_sec, batch_size

    Args:
        filepath: path to CSV file

    Returns:
        list of dicts, one per row
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Measurement file not found: {filepath}\n"
            f"Generate it from the platform profiler — see function docstrings."
        )
    rows = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric strings to float where possible
            parsed = {}
            for k, v in row.items():
                k = k.strip()
                v = v.strip()
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV template generators — print what to collect from each platform
# ─────────────────────────────────────────────────────────────────────────────

def print_collection_guide(platform: str, model_name: str = "bert_large"):
    """
    Print exact steps to collect ground-truth measurements
    and the CSV format expected by the validator.
    """
    sep = "=" * 64

    if platform == "graphcore_bow_ipu":
        print(f"\n{sep}")
        print("  Graphcore PopVision — Measurement Collection Guide")
        print(sep)
        print("""
Step 1 — Enable FLOPs estimation before compiling your model:
    import poplar
    opts = poplar.OptionFlags()
    opts.set("profiler.includeFlopEstimates", "true")
    opts.set("autoReport.outputExecutionReport", "true")

    Or via environment variable before running:
    export POPLAR_ENGINE_OPTIONS='{"profiler.includeFlopEstimates":"true","autoReport.all":"true"}'

Step 2 — Run your model normally. PopVision profile files are
    written to: profile.pop (or the path in PVTI_OPTIONS)

Step 3 — Open profile.pop in PopVision Graph Analyser (desktop app).
    Download from: https://www.graphcore.ai/developer/popvision-tools

Step 4 — Go to: Operations Summary tab
    Click: Columns dropdown → ensure "Cycles" and "FLOPs" are enabled
    Click: Export as CSV

Step 5 — Save the CSV as: measurements/graphcore_<model>.csv

Expected CSV format (columns):
    op_name, cycles, flops
    layer0.q_proj, 12500, 268435456.0
    layer0.k_proj, 12500, 268435456.0
    ...

Column meanings:
    op_name  — must match your op.layer_name or op.op_id values
    cycles   — measured cycle count for this operation
    flops    — FLOPs estimate from PopVision (requires profiler.includeFlopEstimates)
""")

    elif platform == "sambanova_sn30":
        print(f"\n{sep}")
        print("  SambaNova SambaTune — Measurement Collection Guide")
        print(sep)
        print(f"""
Step 1 — Set up SambaTune YAML for your model.
    Template at: /opt/sambaflow/sambatune/configs/
    ALCF guide: https://docs.alcf.anl.gov/ai-testbed/sambanova/sambatune/

Step 2 — Run SambaTune:
    sambatune --yaml your_model.yaml

Step 3 — Open SambaTune UI → Section Report tab.
    For each section record:
      section_id     (integer, from Section column)
      latency_ms     (from Latency column, in ms)
      ddr_bw_gbps    (from DDR Bandwidth column, in GB/s)

Step 4 — Open SambaTune UI → Stack Tracing tab.
    For each section, click to see which PyTorch ops map to it.
    Record the op names — these should match your graph's layer_name values.

Step 5 — Save as: measurements/sambanova_{model_name}.csv

Expected CSV format:
    section_id, latency_ms, ddr_bw_gbps, op_ids
    0, 1.23, 145.2, "layer0.q_proj,layer0.k_proj,layer0.v_proj"
    1, 0.87, 188.4, "layer0.layernorm1,layer0.add1"
    ...

Column meanings:
    section_id   — section number from SambaTune Section Report
    latency_ms   — measured section execution time in milliseconds
    ddr_bw_gbps  — measured DDR bandwidth for this section (GB/s)
    op_ids       — comma-separated list of op_ids in this section
                   (from SambaTune Stack Tracing; match your graph op names)
""")

    elif platform == "cerebras_wse2":
        print(f"\n{sep}")
        print("  Cerebras cstorch Profiler — Measurement Collection Guide")
        print(sep)
        print(f"""
Step 1 — Add these lines to your Cerebras training script,
    after the executor loop completes:

    import json
    measurement = {{
        "model":             "{model_name}",
        "flops_utilization": executor.profiler.rate_tracker.flops_utilization,
        "samples_per_sec":   executor.profiler.rate_tracker.samples_per_sec,
        "batch_size":        BATCH_SIZE,
    }}
    with open("measurements/cerebras_{model_name}.json", "w") as f:
        json.dump(measurement, f, indent=2)

Step 2 — Run your training script on the Cerebras node at ALCF:
    ssh cer-usn-01.ai.alcf.anl.gov
    source /software/cerebras/venv_cerebras_pt/bin/activate
    python your_training_script.py

Step 3 — The JSON file will contain flops_utilization in [0, 1].

Expected JSON format:
    {{
      "model": "{model_name}",
      "flops_utilization": 0.312,
      "samples_per_sec": 847.3,
      "batch_size": 8
    }}

Important: Cerebras gives ONE model-level number.
    Per-operator validation is not possible.
    Use Graphcore as your primary C3 validation platform.
""")
    else:
        print(f"Unknown platform: {platform}")
        print(f"Supported: graphcore_bow_ipu, sambanova_sn30, cerebras_wse2")