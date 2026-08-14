"""CS-3 weight-streaming training-time model.

Inputs: model + hardware YAML only. No runtime measurement enters here;
calibrated constants (bw_memx_achieved, k_in, k_out) are provenance-tagged
in the hardware YAML and were derived once from the llama3-200B artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml

_DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1}


@dataclass
class CS3System:
    name: str
    clock_hz: float
    peak_flops: float
    bw_memx: float
    bw_memx_compiler: float
    k_in: float
    k_out: float
    sram_bytes_per_pe: float
    n_pe: float
    provenance: dict

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CS3System":
        d = yaml.safe_load(open(path))
        prov = {k: v["provenance"] for k, v in d.items() if isinstance(v, dict)}
        g = lambda k: float(d[k]["value"])
        return cls(
            name=d["name"],
            clock_hz=g("clock_hz"),
            peak_flops=g("peak_flops_fp16"),
            bw_memx=g("bw_memx_achieved"),
            bw_memx_compiler=g("bw_memx_compiler"),
            k_in=g("k_in"),
            k_out=g("k_out"),
            sram_bytes_per_pe=g("sram_bytes_per_pe"),
            n_pe=g("n_pe_total"),
            provenance=prov,
        )

    def fitted_params(self) -> list[str]:
        return [k for k, p in self.provenance.items() if p == "fitted"]


@dataclass
class ModelCfg:
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ffn: int
    ffn_activation: str
    vocab_size: int
    seq_len: int
    batch_size: int
    n_boxes: int
    weight_bytes_per_elem: int
    tie_embeddings: bool

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelCfg":
        d = yaml.safe_load(open(path))
        a, t = d["architecture"], d["training"]
        dec, emb = a["decoder"], a["embedding"]
        return cls(
            name=d["name"],
            n_layers=dec["num_layers"],
            d_model=dec["hidden_dim"],
            n_heads=dec["num_heads"],
            n_kv_heads=dec.get("num_kv_heads", dec["num_heads"]),
            d_ffn=dec["ffn_dim"],
            ffn_activation=dec.get("ffn_activation", "gelu"),
            vocab_size=emb["vocab_size"],
            seq_len=t["seq_len"],
            batch_size=t["batch_size"],
            n_boxes=t.get("n_boxes", 1),
            weight_bytes_per_elem=_DTYPE_BYTES[t.get("weights_stored_dtype", "fp32")],
            tie_embeddings=a.get("tie_embeddings", False),
        )


def weight_elems(m: ModelCfg) -> int:
    e, kv, f = m.d_model, m.n_kv_heads * (m.d_model // m.n_heads), m.d_ffn
    attn = 2 * e * e + 2 * kv * e
    n_up = 2 if m.ffn_activation == "swiglu" else 1
    ffn = n_up * f * e + e * f
    block = attn + ffn + 2 * e
    embed = m.vocab_size * e
    head = 0 if m.tie_embeddings else m.vocab_size * e
    return block * m.n_layers + embed + head + e


def train_flops_per_step(m: ModelCfg) -> float:
    p = weight_elems(m)
    tokens = m.batch_size * m.seq_len
    attn_extra = 12.0 * m.batch_size * m.n_layers * m.seq_len**2 * m.d_model
    return 6.0 * p * tokens + attn_extra


@dataclass
class Prediction:
    t_stream_s: float
    t_cmp_lower_s: float
    t_step_s: float
    regime: str
    traffic_bytes: float
    throughput: float
    breakdown: dict


def predict(m: ModelCfg, hw: CS3System) -> Prediction:
    assert not hw.fitted_params(), f"fitted params present: {hw.fitted_params()}"
    w_bytes = weight_elems(m) * m.weight_bytes_per_elem
    traffic = m.n_boxes * w_bytes * (hw.k_in + hw.k_out)
    t_stream = traffic / hw.bw_memx
    t_cmp = train_flops_per_step(m) / hw.peak_flops
    t_step = max(t_stream, t_cmp)
    return Prediction(
        t_stream_s=t_stream,
        t_cmp_lower_s=t_cmp,
        t_step_s=t_step,
        regime="streaming" if t_stream >= t_cmp else "compute",
        traffic_bytes=traffic,
        throughput=m.batch_size / t_step,
        breakdown={
            "weight_bytes": w_bytes,
            "params": weight_elems(m),
            "flops_per_step": train_flops_per_step(m),
            "batch_independent_traffic": True,
        },
    )