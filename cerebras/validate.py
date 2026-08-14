"""Validation harness. Measured data enters here and nowhere else."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifacts import load_layer_cycles_json, load_measured_rate, load_param_breakdown, totals
from .streaming_model import CS3System, ModelCfg, Prediction, predict


@dataclass
class Report:
    model: str
    calibration_point: bool
    measured_step_s: float
    predicted_step_s: float
    mape_ours: float
    baseline_compiler_step_s: float | None
    mape_compiler: float | None
    tensor_inventory_check: dict
    lines: list[str]

    def __str__(self) -> str:
        return "\n".join(self.lines)


def _mape(pred: float, meas: float) -> float:
    return abs(pred - meas) / meas * 100.0


def validate(
    model_yaml: str,
    hardware_yaml: str,
    raw_dir: str,
    calibration_point: bool = True,
) -> Report:
    raw = Path(raw_dir)
    m = ModelCfg.from_yaml(model_yaml)
    hw = CS3System.from_yaml(hardware_yaml)
    pred: Prediction = predict(m, hw)

    rate = load_measured_rate(raw / "measured_rates.txt")
    measured_step = m.batch_size / rate["global_rate"]

    comp_step = mape_c = None
    lc = raw / next(
        (p.name for p in raw.glob("*layer_cycles.json")), "layer_cycles.json"
    )
    if lc.exists():
        t = totals(load_layer_cycles_json(lc))
        comp_step = t["cycles_with_io"] / hw.clock_hz
        mape_c = _mape(comp_step, measured_step)

    pb = raw / "param_breakdown.txt"
    check = {}
    if pb.exists():
        actual = load_param_breakdown(pb)["total_elems"]
        ours = pred.breakdown["params"]
        check = {
            "predicted": ours,
            "actual": actual,
            "match": ours == actual,
            "note": "spec-consistency guard, not a predictive metric",
        }

    mape = _mape(pred.t_step_s, measured_step)
    lines = [
        f"model                : {m.name}",
        f"regime               : {pred.regime}",
        f"traffic / step       : {pred.traffic_bytes/1e9:,.0f} GB "
        f"(k_in={hw.k_in}, k_out={hw.k_out}, boxes={m.n_boxes})",
        f"T_stream             : {pred.t_stream_s:8.2f} s",
        f"T_cmp lower bound    : {pred.t_cmp_lower_s:8.2f} s",
        f"predicted step       : {pred.t_step_s:8.2f} s  ({pred.throughput:.4f} samples/s)",
        f"measured step        : {measured_step:8.2f} s  ({rate['global_rate']:.4f} samples/s)",
    ]
    if calibration_point:
        lines += [
            f"agreement            : {mape:8.2f} %  [calibration closure, NOT a result]",
        ]
    else:
        lines += [
            f"MAPE (held-out)      : {mape:8.2f} %",
        ]
    if comp_step is not None:
        lines += [
            f"compiler baseline    : {comp_step:8.2f} s   MAPE {mape_c:.2f} %",
        ]
    if check:
        status = "PASS" if check["match"] else (
            f"FAIL ({check['predicted']:,} vs {check['actual']:,})"
        )
        lines += [f"tensor inventory     : {status}   [guard, not a metric]"]
    if calibration_point:
        lines += [
            "",
            "NOTE: bw_memx and k_in/k_out were calibrated on this run. The agreement",
            "figure above is arithmetic closure, not predictive accuracy, and must not",
            "be reported as a result. Held-out validation requires a model whose data",
            "never entered calibration (run with --held-out).",
        ]
    return Report(
        model=m.name,
        calibration_point=calibration_point,
        measured_step_s=measured_step,
        predicted_step_s=pred.t_step_s,
        mape_ours=mape,
        baseline_compiler_step_s=comp_step,
        mape_compiler=mape_c,
        tensor_inventory_check=check,
        lines=lines,
    )