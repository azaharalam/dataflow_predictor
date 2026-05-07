"""
models/model_zoo.py

Pre-built computation graphs for all benchmark DNN models.
Each function returns a fully annotated ComputationGraph
ready for metric computation.

Supported models:
  - ResNet-50 v1.5
  - BERT-Large
  - GPT-2 1.5B
  - U-Net
  - CANDLE-UNO
  - BraggNN
"""

from .operator_graph import (
    ComputationGraph, OperatorNode, OperatorType, TensorShape
)
from .flops_counter import compute_flops


def _make_tensor(dims: list, dtype: str = "fp16") -> TensorShape:
    return TensorShape(dims=dims, dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# RESNET-50 v1.5
# ─────────────────────────────────────────────────────────────────────────────

def build_resnet50(batch_size: int = 32, dtype: str = "fp16") -> ComputationGraph:
    """
    ResNet-50 v1.5 computation graph.
    Architecture: conv1 → 4 stages of residual blocks → avgpool → fc
    Stages: [3, 4, 6, 3] blocks with channels [64, 128, 256, 512]
    """
    graph = ComputationGraph(
        model_name="ResNet-50 v1.5",
        model_type="cnn",
        batch_size=batch_size,
        dtype=dtype
    )
    op_id = 0

    def add_conv(layer_idx, name, Cin, Cout, H, W, Kh=3, Kw=3,
                 stride=1, padding=1, groups=1):
        nonlocal op_id
        Hout = (H - Kh + 2 * padding) // stride + 1
        Wout = (W - Kw + 2 * padding) // stride + 1
        op = OperatorNode(
            op_id=f"op_{op_id}",
            op_type=OperatorType.CONV2D,
            layer_index=layer_idx,
            layer_name=name,
            input_shapes=[_make_tensor([batch_size, Cin, H, W], dtype)],
            output_shapes=[_make_tensor([batch_size, Cout, Hout, Wout], dtype)],
            weight_shapes=[_make_tensor([Cout, Cin // groups, Kh, Kw], dtype)],
            params={"batch_size": batch_size, "in_channels": Cin,
                    "out_channels": Cout, "input_height": H, "input_width": W,
                    "kernel_h": Kh, "kernel_w": Kw, "stride": stride,
                    "padding": padding, "groups": groups}
        )
        op_id += 1
        graph.add_operator(op)
        return Hout, Wout

    def add_bn(layer_idx, name, C, H, W):
        nonlocal op_id
        op = OperatorNode(
            op_id=f"op_{op_id}",
            op_type=OperatorType.BATCHNORM,
            layer_index=layer_idx,
            layer_name=name,
            input_shapes=[_make_tensor([batch_size, C, H, W], dtype)],
            output_shapes=[_make_tensor([batch_size, C, H, W], dtype)],
            params={"batch_size": batch_size, "num_channels": C,
                    "height": H, "width": W}
        )
        op_id += 1
        graph.add_operator(op)

    def add_relu(layer_idx, name, C, H, W):
        nonlocal op_id
        op = OperatorNode(
            op_id=f"op_{op_id}",
            op_type=OperatorType.RELU,
            layer_index=layer_idx,
            layer_name=name,
            input_shapes=[_make_tensor([batch_size, C, H, W], dtype)],
            output_shapes=[_make_tensor([batch_size, C, H, W], dtype)],
            params={"batch_size": batch_size, "seq_len": H * W, "hidden_dim": C}
        )
        op_id += 1
        graph.add_operator(op)

    def add_add(layer_idx, name, C, H, W):
        nonlocal op_id
        op = OperatorNode(
            op_id=f"op_{op_id}",
            op_type=OperatorType.ADD,
            layer_index=layer_idx,
            layer_name=name,
            input_shapes=[_make_tensor([batch_size, C, H, W], dtype),
                          _make_tensor([batch_size, C, H, W], dtype)],
            output_shapes=[_make_tensor([batch_size, C, H, W], dtype)],
            params={"batch_size": batch_size, "seq_len": H * W, "hidden_dim": C}
        )
        op_id += 1
        graph.add_operator(op)

    # Initial conv + bn + relu + maxpool
    H, W = add_conv(0, "conv1", 3, 64, 224, 224, Kh=7, Kw=7, stride=2, padding=3)
    add_bn(0, "bn1", 64, H, W)
    add_relu(0, "relu1", 64, H, W)
    # MaxPool 3x3 stride 2
    H, W = (H - 3) // 2 + 1, (W - 3) // 2 + 1   # 56x56

    # Stage 1: 3 bottleneck blocks, 64→256 channels, 56x56
    stage_configs = [
        (3, 64,  256,  56,  56),
        (4, 128, 512,  28,  28),
        (6, 256, 1024, 14,  14),
        (3, 512, 2048, 7,   7),
    ]
    layer_idx = 1
    for num_blocks, mid_c, out_c, h, w in stage_configs:
        for block in range(num_blocks):
            in_c = out_c if block > 0 else (out_c // 4 if layer_idx > 1 else 64)
            # Bottleneck: 1x1 → 3x3 → 1x1
            add_conv(layer_idx, f"stage{layer_idx}_b{block}_conv1",
                     in_c, mid_c, h, w, Kh=1, Kw=1, padding=0)
            add_bn(layer_idx, f"stage{layer_idx}_b{block}_bn1", mid_c, h, w)
            add_relu(layer_idx, f"stage{layer_idx}_b{block}_relu1", mid_c, h, w)
            add_conv(layer_idx, f"stage{layer_idx}_b{block}_conv2",
                     mid_c, mid_c, h, w, Kh=3, Kw=3, padding=1)
            add_bn(layer_idx, f"stage{layer_idx}_b{block}_bn2", mid_c, h, w)
            add_relu(layer_idx, f"stage{layer_idx}_b{block}_relu2", mid_c, h, w)
            add_conv(layer_idx, f"stage{layer_idx}_b{block}_conv3",
                     mid_c, out_c, h, w, Kh=1, Kw=1, padding=0)
            add_bn(layer_idx, f"stage{layer_idx}_b{block}_bn3", out_c, h, w)
            add_add(layer_idx, f"stage{layer_idx}_b{block}_add", out_c, h, w)
            add_relu(layer_idx, f"stage{layer_idx}_b{block}_relu3", out_c, h, w)
        layer_idx += 1

    # AvgPool + FC
    fc_idx = layer_idx
    op = OperatorNode(
        op_id=f"op_{op_id}",
        op_type=OperatorType.LINEAR,
        layer_index=fc_idx,
        layer_name="fc",
        input_shapes=[_make_tensor([batch_size, 2048], dtype)],
        output_shapes=[_make_tensor([batch_size, 1000], dtype)],
        weight_shapes=[_make_tensor([1000, 2048], dtype)],
        params={"batch_size": batch_size, "seq_len": 1,
                "in_features": 2048, "out_features": 1000}
    )
    graph.add_operator(op)

    return compute_flops(graph)


# ─────────────────────────────────────────────────────────────────────────────
# BERT-LARGE
# ─────────────────────────────────────────────────────────────────────────────

def build_bert_large(batch_size: int = 8, seq_len: int = 512,
                     dtype: str = "fp16") -> ComputationGraph:
    """
    BERT-Large: 24 transformer encoder layers.
    Hidden dim: 1024, Heads: 16, FFN dim: 4096
    """
    graph = ComputationGraph(
        model_name="BERT-Large",
        model_type="transformer",
        batch_size=batch_size,
        dtype=dtype
    )
    op_id = 0
    d = 1024
    h = 16
    ffn_dim = 4096

    # Embedding
    emb = OperatorNode(
        op_id=f"op_{op_id}",
        op_type=OperatorType.EMBEDDING,
        layer_index=0,
        layer_name="embedding",
        output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
        weight_shapes=[_make_tensor([30522, d], dtype)],
        params={"vocab_size": 30522, "hidden_dim": d}
    )
    op_id += 1
    graph.add_operator(emb)

    for layer_idx in range(1, 25):
        # Self-attention
        attn = OperatorNode(
            op_id=f"op_{op_id}",
            op_type=OperatorType.ATTENTION,
            layer_index=layer_idx,
            layer_name=f"layer_{layer_idx}_attention",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            weight_shapes=[
                _make_tensor([d, d], dtype),   # Q
                _make_tensor([d, d], dtype),   # K
                _make_tensor([d, d], dtype),   # V
                _make_tensor([d, d], dtype),   # O
            ],
            params={"batch_size": batch_size, "seq_len": seq_len,
                    "hidden_dim": d, "num_heads": h, "head_dim": d // h}
        )
        op_id += 1
        graph.add_operator(attn)

        # LayerNorm after attention
        ln1 = OperatorNode(
            op_id=f"op_{op_id}",
            op_type=OperatorType.LAYERNORM,
            layer_index=layer_idx,
            layer_name=f"layer_{layer_idx}_ln1",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len, "hidden_dim": d}
        )
        op_id += 1
        graph.add_operator(ln1)

        # FFN
        ffn = OperatorNode(
            op_id=f"op_{op_id}",
            op_type=OperatorType.FFN,
            layer_index=layer_idx,
            layer_name=f"layer_{layer_idx}_ffn",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            weight_shapes=[
                _make_tensor([d, ffn_dim], dtype),
                _make_tensor([ffn_dim, d], dtype),
            ],
            params={"batch_size": batch_size, "seq_len": seq_len,
                    "hidden_dim": d, "ffn_dim": ffn_dim}
        )
        op_id += 1
        graph.add_operator(ffn)

        # LayerNorm after FFN
        ln2 = OperatorNode(
            op_id=f"op_{op_id}",
            op_type=OperatorType.LAYERNORM,
            layer_index=layer_idx,
            layer_name=f"layer_{layer_idx}_ln2",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len, "hidden_dim": d}
        )
        op_id += 1
        graph.add_operator(ln2)

    return compute_flops(graph)


# ─────────────────────────────────────────────────────────────────────────────
# GPT-2 1.5B
# ─────────────────────────────────────────────────────────────────────────────

def build_gpt2_1_5b(batch_size: int = 4, seq_len: int = 1024,
                    dtype: str = "fp16") -> ComputationGraph:
    """
    GPT-2 1.5B: 48 transformer decoder layers.
    Hidden dim: 1600, Heads: 25, FFN dim: 6400
    """
    graph = ComputationGraph(
        model_name="GPT-2 1.5B",
        model_type="transformer",
        batch_size=batch_size,
        dtype=dtype
    )
    op_id = 0
    d = 1600
    h = 25
    ffn_dim = 6400

    emb = OperatorNode(
        op_id=f"op_{op_id}",
        op_type=OperatorType.EMBEDDING,
        layer_index=0,
        layer_name="embedding",
        output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
        weight_shapes=[_make_tensor([50257, d], dtype)],
        params={"vocab_size": 50257, "hidden_dim": d}
    )
    op_id += 1
    graph.add_operator(emb)

    for layer_idx in range(1, 49):
        # LayerNorm pre-attention (Pre-LN GPT-2)
        ln1 = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LAYERNORM,
            layer_index=layer_idx, layer_name=f"layer_{layer_idx}_ln1",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len, "hidden_dim": d}
        )
        op_id += 1
        graph.add_operator(ln1)

        attn = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.ATTENTION,
            layer_index=layer_idx, layer_name=f"layer_{layer_idx}_attn",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            weight_shapes=[_make_tensor([d, 3 * d], dtype),
                           _make_tensor([d, d], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len,
                    "hidden_dim": d, "num_heads": h, "head_dim": d // h}
        )
        op_id += 1
        graph.add_operator(attn)

        ln2 = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LAYERNORM,
            layer_index=layer_idx, layer_name=f"layer_{layer_idx}_ln2",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len, "hidden_dim": d}
        )
        op_id += 1
        graph.add_operator(ln2)

        ffn = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.FFN,
            layer_index=layer_idx, layer_name=f"layer_{layer_idx}_ffn",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            weight_shapes=[_make_tensor([d, ffn_dim], dtype),
                           _make_tensor([ffn_dim, d], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len,
                    "hidden_dim": d, "ffn_dim": ffn_dim}
        )
        op_id += 1
        graph.add_operator(ffn)

    return compute_flops(graph)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ZOO REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ZOO = {
    "resnet50":    build_resnet50,
    "bert_large":  build_bert_large,
    "gpt2_1_5b":   build_gpt2_1_5b,
}


def get_model(name: str, batch_size: int = 32,
              dtype: str = "fp16", **kwargs) -> ComputationGraph:
    """
    Get a pre-built model from the zoo.

    Args:
        name: Model name. One of MODEL_ZOO keys.
        batch_size: Training batch size.
        dtype: Data type (fp16, bf16, fp32).
        **kwargs: Model-specific params (seq_len, etc.)

    Returns:
        Annotated ComputationGraph
    """
    if name not in MODEL_ZOO:
        raise ValueError(
            f"Model '{name}' not in zoo. Available: {list(MODEL_ZOO.keys())}"
        )
    return MODEL_ZOO[name](batch_size=batch_size, dtype=dtype, **kwargs)
