"""Loaders for CS-3 compile artifacts and run logs. Validation-side only."""
from __future__ import annotations

import ast
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

_LOC = re.compile(r'^#(loc\d+) = loc\((?:fused\[)?"([^"]+)"')
_LAYER = re.compile(r"layers\.(\d+)\.(.+?)\.(fwd|bwd)/(.+)$")
_CYC = re.compile(r"(?<!_)\bcycles = (\d+)")
_CYC_IO = re.compile(r"\bcycles_with_io = (\d+)")
_IO = re.compile(r"(?<!_)\bio_bits = \[([\d,\s]+)\]")
_USE = re.compile(r"loc\((#loc\d+)\)")


@dataclass
class Group:
    cycles: int = 0
    cycles_with_io: int = 0
    io_bits: int = 0
    n_ops: int = 0


def parse_ws_rt_mlir(path: str | Path) -> dict[tuple, Group]:
    loc: dict[str, tuple] = {}
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("#loc"):
                m = _LOC.match(line)
                if not m:
                    continue
                name = m.group(2)
                lm = _LAYER.search(name)
                loc[m.group(1)] = (
                    (int(lm.group(1)), lm.group(2), lm.group(3))
                    if lm
                    else (None, name.split("/")[0], None)
                )
    agg: dict[tuple, Group] = {}
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u = _USE.search(line)
            if not u:
                continue
            key = loc.get(u.group(1)[1:])
            if key is None:
                continue
            g = agg.setdefault(key, Group())
            m = _CYC.search(line)
            if m:
                g.cycles += int(m.group(1))
            m = _CYC_IO.search(line)
            if m:
                g.cycles_with_io += int(m.group(1))
            m = _IO.search(line)
            if m:
                g.io_bits += sum(int(x) for x in m.group(1).split(","))
            g.n_ops += 1
    return agg


def load_layer_cycles_json(path: str | Path) -> dict[tuple, Group]:
    raw = json.load(open(path))
    out = {}
    for k, v in raw.items():
        out[ast.literal_eval(k)] = Group(*v)
    return out


def totals(groups: dict[tuple, Group]) -> dict:
    c = sum(g.cycles for g in groups.values())
    ci = sum(g.cycles_with_io for g in groups.values())
    b = sum(g.io_bits for g in groups.values())
    nl = sum(g.cycles_with_io for k, g in groups.items() if k[0] is None)
    return {
        "cycles": c,
        "cycles_with_io": ci,
        "io_bytes": b / 8,
        "io_over_compute": ci / c if c else None,
        "non_layer_fraction": nl / ci if ci else None,
    }


def load_param_breakdown(path: str | Path) -> dict:
    per_tensor, total_elems = {}, 0
    for line in open(path, errors="ignore"):
        if "parameter" not in line or "torch." not in line:
            continue
        cells = [c.strip() for c in line.split("│") if c.strip()]
        if len(cells) < 5:
            continue
        try:
            elems = int(cells[4].replace(",", ""))
        except ValueError:
            continue
        per_tensor[cells[1]] = elems
        total_elems += elems
    return {"per_tensor": per_tensor, "total_elems": total_elems}


def load_measured_rate(path: str | Path) -> dict:
    rates = [
        float(m.group(1))
        for line in open(path, errors="ignore")
        if (m := re.search(r"GlobalRate=([\d.]+) samples/sec", line))
    ]
    if not rates:
        raise ValueError(f"no GlobalRate lines in {path}")
    steady = rates[len(rates) // 4 :]
    return {"global_rate": statistics.median(steady), "n_samples": len(rates)}


def load_run_meta(path: str | Path) -> dict:
    d = json.load(open(path))
    ex = d["execute_jobs"][0]
    return {
        "software": ex["software_versions"]["appliance-client"],
        "n_systems_execute": len(ex["systems"]),
        "execute_time_s": ex["execution_time_s"],
        "compile_time_s": d["compile_jobs"][0]["execution_time_s"],
    }