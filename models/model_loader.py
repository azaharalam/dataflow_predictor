"""
models/model_loader.py

Generic model loader that reads any model YAML from configs/models/
and builds a fully annotated ComputationGraph.

This is the key file that makes adding new models config-driven.
Users never need to touch Python code to add a new model.

Supported architecture types:
  - transformer_encoder  (BERT, RoBERTa, ViT, etc.)
  - transformer_decoder  (GPT, LLaMA, etc.)
  - resnet               (ResNet family)
  - mlp                  (Fully-connected networks)
  - custom               (Explicit operator list)
"""

import os
import yaml
from typing import Dict, Optional, List, Any

from .operator_graph import (
    ComputationGraph, OperatorNode, OperatorType, TensorShape
)
from .flops_counter import compute_flops


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def _models_dir() -> str:
    """Return path to configs/models/ directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "configs", "models"
    )


def list_available_models() -> Dict[str, str]:
    """
    Scan configs/models/ and return {model_id: display_name} for every
    YAML file found. No Python changes needed when new models are added.
    """
    models = {}
    d = _models_dir()
    if not os.path.isdir(d):
        return models
    for fname in sorted(os.listdir(d)):
        if fname.endswith(".yaml") and not fname.startswith("TEMPLATE"):
            fpath = os.path.join(d, fname)
            try:
                with open(fpath) as f:
                    cfg = yaml.safe_load(f)
                model_id = cfg.get("model_id", fname.replace(".yaml", ""))
                name     = cfg.get("name", model_id)
                models[model_id] = name
            except Exception:
                pass
    return models


def load_model_config(model_id: str) -> Dict:
    """
    Load a model's YAML config by its model_id.
    Raises FileNotFoundError if not found in configs/models/.
    """
    d = _models_dir()
    fpath = os.path.join(d, f"{model_id}.yaml")
    if not os.path.isfile(fpath):
        available = list(list_available_models().keys())
        raise FileNotFoundError(
            f"Model '{model_id}' not found in {d}.\n"
            f"Available models: {available}\n"
            f"To add a new model: copy TEMPLATE_new_model.yaml, "
            f"fill it in, save as {model_id}.yaml"
        )
    with open(fpath) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_from_config(
    model_id: str,
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    dtype: Optional[str] = None,
) -> ComputationGraph:
    """
    Build a ComputationGraph from a model YAML config file.

    CLI-provided args (batch_size, seq_len, dtype) override the
    config file defaults, allowing quick experimentation without
    editing files.

    Args:
        model_id:   Model identifier matching the YAML filename
        batch_size: Override training.batch_size from config
        seq_len:    Override training.seq_len from config
        dtype:      Override training.dtype from config

    Returns:
        Fully annotated ComputationGraph
    """
    cfg = load_model_config(model_id)

    # Merge CLI overrides with config file values
    train_cfg = cfg.get("training", {})
    bs    = batch_size or train_cfg.get("batch_size", 32)
    sl    = seq_len    or train_cfg.get("seq_len", 512)
    dt    = dtype      or train_cfg.get("dtype", "fp16")

    arch  = cfg.get("architecture", {})
    atype = arch.get("type", "custom")

    # Dispatch to appropriate builder
    builders = {
        "transformer_encoder": _build_transformer_encoder,
        "transformer_decoder": _build_transformer_decoder,
        "resnet":              _build_resnet,
        "mlp":                 _build_mlp,
        "custom":              _build_custom,
    }

    if atype not in builders:
        raise ValueError(
            f"Unknown architecture type '{atype}' in {model_id}.yaml.\n"
            f"Supported types: {list(builders.keys())}"
        )

    graph = builders[atype](cfg, bs, sl, dt)
    return compute_flops(graph)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture builders
# ─────────────────────────────────────────────────────────────────────────────

def _t(dims: list, dtype: str) -> TensorShape:
    return TensorShape(dims=dims, dtype=dtype)


def _build_transformer_encoder(cfg: Dict, BS: int, S: int, dt: str) -> ComputationGraph:
    """Build transformer encoder (BERT-style)."""
    arch  = cfg["architecture"]
    enc   = arch.get("encoder", arch.get("decoder", {}))
    emb   = arch.get("embedding", {})

    d           = enc.get("hidden_dim", emb.get("hidden_dim", 768))
    num_layers  = enc.get("num_layers", 12)
    num_heads   = enc.get("num_heads", 12)
    ffn_dim     = enc.get("ffn_dim", 4 * d)
    vocab_size  = emb.get("vocab_size", 30522)

    graph = ComputationGraph(
        model_name=cfg.get("name", "Transformer Encoder"),
        model_type="transformer",
        batch_size=BS, dtype=dt
    )
    op_id = 0

    def add(op_type, layer, name, in_shapes, out_shapes, wt_shapes, params):
        nonlocal op_id
        node = OperatorNode(
            op_id=f"op_{op_id}", op_type=op_type,
            layer_index=layer, layer_name=name,
            input_shapes=in_shapes, output_shapes=out_shapes,
            weight_shapes=wt_shapes, params=params
        )
        op_id += 1
        graph.add_operator(node)

    # Embedding
    add(OperatorType.EMBEDDING, 0, "embedding",
        [], [_t([BS, S, d], dt)], [_t([vocab_size, d], dt)],
        {"vocab_size": vocab_size, "hidden_dim": d})

    for layer in range(1, num_layers + 1):
        # Self-attention
        add(OperatorType.ATTENTION, layer, f"layer_{layer}_attn",
            [_t([BS, S, d], dt)], [_t([BS, S, d], dt)],
            [_t([d, d], dt)] * 4,
            {"batch_size": BS, "seq_len": S, "hidden_dim": d,
             "num_heads": num_heads, "head_dim": d // max(num_heads, 1)})

        # LayerNorm 1
        add(OperatorType.LAYERNORM, layer, f"layer_{layer}_ln1",
            [_t([BS, S, d], dt)], [_t([BS, S, d], dt)], [],
            {"batch_size": BS, "seq_len": S, "hidden_dim": d})

        # FFN
        add(OperatorType.FFN, layer, f"layer_{layer}_ffn",
            [_t([BS, S, d], dt)], [_t([BS, S, d], dt)],
            [_t([d, ffn_dim], dt), _t([ffn_dim, d], dt)],
            {"batch_size": BS, "seq_len": S, "hidden_dim": d, "ffn_dim": ffn_dim})

        # LayerNorm 2
        add(OperatorType.LAYERNORM, layer, f"layer_{layer}_ln2",
            [_t([BS, S, d], dt)], [_t([BS, S, d], dt)], [],
            {"batch_size": BS, "seq_len": S, "hidden_dim": d})

    return graph


def _build_transformer_decoder(cfg: Dict, BS: int, S: int, dt: str) -> ComputationGraph:
    """Build transformer decoder (GPT-style). Same as encoder but with causal attention."""
    arch  = cfg["architecture"]
    dec   = arch.get("decoder", arch.get("encoder", {}))
    emb   = arch.get("embedding", {})

    d          = dec.get("hidden_dim", emb.get("hidden_dim", 1600))
    num_layers = dec.get("num_layers", 48)
    num_heads  = dec.get("num_heads", 25)
    ffn_dim    = dec.get("ffn_dim", 4 * d)
    vocab_size = emb.get("vocab_size", 50257)

    graph = ComputationGraph(
        model_name=cfg.get("name", "Transformer Decoder"),
        model_type="transformer",
        batch_size=BS, dtype=dt
    )
    op_id = 0

    def add(op_type, layer, name, in_shapes, out_shapes, wt_shapes, params):
        nonlocal op_id
        node = OperatorNode(
            op_id=f"op_{op_id}", op_type=op_type,
            layer_index=layer, layer_name=name,
            input_shapes=in_shapes, output_shapes=out_shapes,
            weight_shapes=wt_shapes, params=params
        )
        op_id += 1
        graph.add_operator(node)

    # Embedding
    add(OperatorType.EMBEDDING, 0, "embedding",
        [], [_t([BS, S, d], dt)], [_t([vocab_size, d], dt)],
        {"vocab_size": vocab_size, "hidden_dim": d})

    for layer in range(1, num_layers + 1):
        # Pre-LN (GPT-2 style)
        add(OperatorType.LAYERNORM, layer, f"layer_{layer}_ln1",
            [_t([BS, S, d], dt)], [_t([BS, S, d], dt)], [],
            {"batch_size": BS, "seq_len": S, "hidden_dim": d})

        add(OperatorType.ATTENTION, layer, f"layer_{layer}_attn",
            [_t([BS, S, d], dt)], [_t([BS, S, d], dt)],
            [_t([d, 3 * d], dt), _t([d, d], dt)],
            {"batch_size": BS, "seq_len": S, "hidden_dim": d,
             "num_heads": num_heads, "head_dim": d // max(num_heads, 1)})

        add(OperatorType.LAYERNORM, layer, f"layer_{layer}_ln2",
            [_t([BS, S, d], dt)], [_t([BS, S, d], dt)], [],
            {"batch_size": BS, "seq_len": S, "hidden_dim": d})

        add(OperatorType.FFN, layer, f"layer_{layer}_ffn",
            [_t([BS, S, d], dt)], [_t([BS, S, d], dt)],
            [_t([d, ffn_dim], dt), _t([ffn_dim, d], dt)],
            {"batch_size": BS, "seq_len": S, "hidden_dim": d, "ffn_dim": ffn_dim})

    return graph


def _build_resnet(cfg: Dict, BS: int, S: int, dt: str) -> ComputationGraph:
    """Build ResNet from config stages."""
    arch   = cfg["architecture"]
    stem   = arch.get("stem", {})
    stages = arch.get("stages", [])
    head   = arch.get("head", {})

    graph = ComputationGraph(
        model_name=cfg.get("name", "ResNet"),
        model_type="cnn",
        batch_size=BS, dtype=dt
    )
    op_id = 0

    def add_conv(layer, name, Cin, Cout, H, W, Kh=3, Kw=3, stride=1, padding=1):
        nonlocal op_id
        Hout = (H - Kh + 2 * padding) // stride + 1
        Wout = (W - Kw + 2 * padding) // stride + 1
        node = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.CONV2D,
            layer_index=layer, layer_name=name,
            input_shapes=[_t([BS, Cin, H, W], dt)],
            output_shapes=[_t([BS, Cout, Hout, Wout], dt)],
            weight_shapes=[_t([Cout, Cin, Kh, Kw], dt)],
            params={"batch_size": BS, "in_channels": Cin, "out_channels": Cout,
                    "input_height": H, "input_width": W, "kernel_h": Kh,
                    "kernel_w": Kw, "stride": stride, "padding": padding}
        )
        op_id += 1
        graph.add_operator(node)
        return Hout, Wout

    def add_bn(layer, name, C, H, W):
        nonlocal op_id
        node = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.BATCHNORM,
            layer_index=layer, layer_name=name,
            input_shapes=[_t([BS, C, H, W], dt)],
            output_shapes=[_t([BS, C, H, W], dt)],
            params={"batch_size": BS, "num_channels": C, "height": H, "width": W}
        )
        op_id += 1
        graph.add_operator(node)

    def add_relu(layer, name, C, H, W):
        nonlocal op_id
        node = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.RELU,
            layer_index=layer, layer_name=name,
            input_shapes=[_t([BS, C, H, W], dt)],
            output_shapes=[_t([BS, C, H, W], dt)],
            params={"batch_size": BS, "seq_len": H * W, "hidden_dim": C}
        )
        op_id += 1
        graph.add_operator(node)

    # Stem
    in_c = stem.get("in_channels", 3)
    out_c = stem.get("out_channels", 64)
    H = stem.get("input_h", 224)
    W = stem.get("input_w", 224)
    Kh = stem.get("kernel", 7)
    stride = stem.get("stride", 2)
    pad = stem.get("padding", 3)
    H, W = add_conv(0, "conv1", in_c, out_c, H, W, Kh, Kh, stride, pad)
    add_bn(0, "bn1", out_c, H, W)
    add_relu(0, "relu1", out_c, H, W)
    # MaxPool
    mp = arch.get("maxpool", {"kernel": 3, "stride": 2, "padding": 1})
    H = (H - mp["kernel"] + 2 * mp.get("padding", 1)) // mp["stride"] + 1
    W = (W - mp["kernel"] + 2 * mp.get("padding", 1)) // mp["stride"] + 1

    # Stages
    prev_out = out_c
    for stage_idx, stage in enumerate(stages, 1):
        nb  = stage.get("num_blocks", 3)
        mid = stage.get("mid_channels", 64)
        oc  = stage.get("out_channels", 256)
        h   = stage.get("height", H)
        w   = stage.get("width", W)
        H, W = h, w
        for block in range(nb):
            in_c_block = prev_out if block == 0 else oc
            # Bottleneck: 1x1 → 3x3 → 1x1
            add_conv(stage_idx, f"s{stage_idx}_b{block}_c1", in_c_block, mid, H, W, 1, 1, 1, 0)
            add_bn(stage_idx, f"s{stage_idx}_b{block}_bn1", mid, H, W)
            add_relu(stage_idx, f"s{stage_idx}_b{block}_r1", mid, H, W)
            add_conv(stage_idx, f"s{stage_idx}_b{block}_c2", mid, mid, H, W, 3, 3, 1, 1)
            add_bn(stage_idx, f"s{stage_idx}_b{block}_bn2", mid, H, W)
            add_relu(stage_idx, f"s{stage_idx}_b{block}_r2", mid, H, W)
            add_conv(stage_idx, f"s{stage_idx}_b{block}_c3", mid, oc, H, W, 1, 1, 1, 0)
            add_bn(stage_idx, f"s{stage_idx}_b{block}_bn3", oc, H, W)
            add_relu(stage_idx, f"s{stage_idx}_b{block}_r3", oc, H, W)
        prev_out = oc

    # Head (FC)
    if head:
        in_f = head.get("in_features", 2048)
        out_f = head.get("out_features", 1000)
        node = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LINEAR,
            layer_index=stage_idx + 1, layer_name="fc",
            input_shapes=[_t([BS, in_f], dt)],
            output_shapes=[_t([BS, out_f], dt)],
            weight_shapes=[_t([out_f, in_f], dt)],
            params={"batch_size": BS, "seq_len": 1,
                    "in_features": in_f, "out_features": out_f}
        )
        graph.add_operator(node)

    return graph


def _build_mlp(cfg: Dict, BS: int, S: int, dt: str) -> ComputationGraph:
    """Build a fully-connected MLP from config."""
    arch        = cfg["architecture"]
    input_dim   = arch.get("input_dim", 512)
    hidden_dims = arch.get("hidden_dims", [1024, 512])
    output_dim  = arch.get("output_dim", 10)
    activation  = arch.get("activation", "relu")

    act_type = OperatorType.RELU if activation == "relu" else OperatorType.GELU

    graph = ComputationGraph(
        model_name=cfg.get("name", "MLP"),
        model_type="mlp",
        batch_size=BS, dtype=dt
    )
    op_id = 0
    dims = [input_dim] + hidden_dims + [output_dim]

    for i in range(len(dims) - 1):
        in_f, out_f = dims[i], dims[i + 1]
        node = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LINEAR,
            layer_index=i, layer_name=f"fc_{i}",
            input_shapes=[_t([BS, in_f], dt)],
            output_shapes=[_t([BS, out_f], dt)],
            weight_shapes=[_t([out_f, in_f], dt)],
            params={"batch_size": BS, "seq_len": 1,
                    "in_features": in_f, "out_features": out_f}
        )
        op_id += 1
        graph.add_operator(node)

        # Activation between layers (not after last)
        if i < len(dims) - 2:
            act = OperatorNode(
                op_id=f"op_{op_id}", op_type=act_type,
                layer_index=i, layer_name=f"act_{i}",
                input_shapes=[_t([BS, out_f], dt)],
                output_shapes=[_t([BS, out_f], dt)],
                params={"batch_size": BS, "seq_len": 1, "hidden_dim": out_f}
            )
            op_id += 1
            graph.add_operator(act)

    return graph


def _build_custom(cfg: Dict, BS: int, S: int, dt: str) -> ComputationGraph:
    """
    Build a graph from an explicit operator list in the YAML.
    Most flexible option — user specifies every operator directly.
    """
    arch = cfg["architecture"]
    operators_cfg = arch.get("operators", [])

    graph = ComputationGraph(
        model_name=cfg.get("name", "Custom Model"),
        model_type=cfg.get("model_type", "custom"),
        batch_size=BS, dtype=dt
    )

    op_type_map = {t.value: t for t in OperatorType}

    for i, op_cfg in enumerate(operators_cfg):
        op_type_str = op_cfg.get("type", "other").lower()
        op_type = op_type_map.get(op_type_str, OperatorType.OTHER)
        params = op_cfg.get("params", {})

        # Inject batch_size if not explicitly set
        if "batch_size" not in params:
            params["batch_size"] = BS

        node = OperatorNode(
            op_id=op_cfg.get("id", f"op_{i}"),
            op_type=op_type,
            layer_index=i,
            layer_name=op_cfg.get("id", f"op_{i}"),
            params=params
        )
        graph.add_operator(node)

    return graph


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_model(
    model_id: str,
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    dtype: Optional[str] = None,
) -> ComputationGraph:
    """
    Main entry point for loading any model.
    Reads from configs/models/{model_id}.yaml automatically.
    """
    return build_graph_from_config(model_id, batch_size, seq_len, dtype)


def get_training_config(model_id: str) -> Dict:
    """
    Return the training configuration for a model.
    Used by other modules to get optimizer, backward_ratio, etc.
    """
    cfg = load_model_config(model_id)
    return cfg.get("training", {})