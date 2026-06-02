"""
models/model_zoo.py

Pre-built computation graphs for all benchmark DNN models.
Each function returns a fully annotated ComputationGraph
ready for metric computation.

Supported models:
  - ResNet-50 v1.5
  - BERT-Large / BERT-Large MSL512
  - GPT-2 1.5B
  - GPT-3 2.7B
  - LLaMA 3-8B
  - T5-3B
  - Mixtral MoE 111M
  - ESM2-650M
  - ViT-Base
  - DiT-Large
"""

from models.operator_graph import (
    ComputationGraph, OperatorNode, OperatorType, TensorShape
)
from models.flops_counter import compute_flops


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
# GPT-3 2.7B
# ─────────────────────────────────────────────────────────────────────────────

def build_gpt3_2p7b(batch_size: int = 4, seq_len: int = 2048,
                    dtype: str = "fp16") -> ComputationGraph:
    """
    GPT-3 2.7B: 32 transformer decoder layers.
    Hidden dim: 2560, Heads: 32, FFN dim: 10240
    Pre-LN, causal attention, GeLU activation.
    """
    graph = ComputationGraph(
        model_name="GPT-3 2.7B",
        model_type="transformer",
        batch_size=batch_size,
        dtype=dtype
    )
    op_id = 0
    d = 2560
    h = 32
    ffn_dim = 10240

    emb = OperatorNode(
        op_id=f"op_{op_id}", op_type=OperatorType.EMBEDDING,
        layer_index=0, layer_name="embedding",
        output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
        weight_shapes=[_make_tensor([50257, d], dtype)],
        params={"vocab_size": 50257, "hidden_dim": d}
    )
    op_id += 1
    graph.add_operator(emb)

    for layer_idx in range(1, 33):
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
# LLaMA 3-8B
# ─────────────────────────────────────────────────────────────────────────────

def build_llama3_8b(batch_size: int = 4, seq_len: int = 4096,
                    dtype: str = "bf16") -> ComputationGraph:
    """
    LLaMA 3-8B: 32 transformer decoder layers with GQA.
    Hidden dim: 4096, Heads: 32, KV heads: 8, FFN dim: 14336 (SwiGLU)
    RMSNorm, RoPE, SwiGLU activation.
    """
    graph = ComputationGraph(
        model_name="LLaMA 3-8B",
        model_type="transformer",
        batch_size=batch_size,
        dtype=dtype
    )
    op_id = 0
    d = 4096
    h = 32
    ffn_dim = 14336   # SwiGLU intermediate

    emb = OperatorNode(
        op_id=f"op_{op_id}", op_type=OperatorType.EMBEDDING,
        layer_index=0, layer_name="embedding",
        output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
        weight_shapes=[_make_tensor([128256, d], dtype)],
        params={"vocab_size": 128256, "hidden_dim": d}
    )
    op_id += 1
    graph.add_operator(emb)

    for layer_idx in range(1, 33):
        # RMSNorm pre-attention
        ln1 = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LAYERNORM,
            layer_index=layer_idx, layer_name=f"layer_{layer_idx}_rmsnorm1",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len, "hidden_dim": d}
        )
        op_id += 1
        graph.add_operator(ln1)

        # GQA attention: Q projects full d, K/V project to kv_dim (num_kv_heads * head_dim)
        kv_dim = 8 * 128  # 8 KV heads × head_dim 128 = 1024
        attn = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.ATTENTION,
            layer_index=layer_idx, layer_name=f"layer_{layer_idx}_gqa",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            weight_shapes=[
                _make_tensor([d, d], dtype),         # Q: d → d
                _make_tensor([d, kv_dim], dtype),    # K: d → kv_dim
                _make_tensor([d, kv_dim], dtype),    # V: d → kv_dim
                _make_tensor([d, d], dtype),         # O projection
            ],
            params={"batch_size": batch_size, "seq_len": seq_len,
                    "hidden_dim": d, "num_heads": h, "head_dim": 128,
                    "num_kv_heads": 8}
        )
        op_id += 1
        graph.add_operator(attn)

        # RMSNorm pre-FFN
        ln2 = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LAYERNORM,
            layer_index=layer_idx, layer_name=f"layer_{layer_idx}_rmsnorm2",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len, "hidden_dim": d}
        )
        op_id += 1
        graph.add_operator(ln2)

        # SwiGLU FFN: gate_proj, up_proj, down_proj (3 matrices)
        ffn = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.FFN,
            layer_index=layer_idx, layer_name=f"layer_{layer_idx}_swiglu_ffn",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            weight_shapes=[
                _make_tensor([d, ffn_dim], dtype),   # gate_proj
                _make_tensor([d, ffn_dim], dtype),   # up_proj
                _make_tensor([ffn_dim, d], dtype),   # down_proj
            ],
            params={"batch_size": batch_size, "seq_len": seq_len,
                    "hidden_dim": d, "ffn_dim": ffn_dim}
        )
        op_id += 1
        graph.add_operator(ffn)

    return compute_flops(graph)


# ─────────────────────────────────────────────────────────────────────────────
# T5-3B
# ─────────────────────────────────────────────────────────────────────────────

def build_t5_3b(batch_size: int = 8, seq_len: int = 512,
                dtype: str = "fp16") -> ComputationGraph:
    """
    T5-3B encoder-decoder: 24 encoder + 24 decoder layers.
    Hidden dim: 1024, Heads: 16, FFN dim: 16384 (large ratio, T5-3B spec).
    RMSNorm, relative position bias, no absolute position embeddings.
    """
    graph = ComputationGraph(
        model_name="T5-3B",
        model_type="transformer",
        batch_size=batch_size,
        dtype=dtype
    )
    op_id = 0
    d = 1024
    h = 16
    ffn_dim = 16384
    dec_seq_len = seq_len  # encoder and decoder same length for training

    # Shared embedding (encoder + decoder share weights)
    emb = OperatorNode(
        op_id=f"op_{op_id}", op_type=OperatorType.EMBEDDING,
        layer_index=0, layer_name="shared_embedding",
        output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
        weight_shapes=[_make_tensor([32128, d], dtype)],
        params={"vocab_size": 32128, "hidden_dim": d}
    )
    op_id += 1
    graph.add_operator(emb)

    # ── Encoder ──────────────────────────────────────────────
    for layer_idx in range(1, 25):
        ln1 = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LAYERNORM,
            layer_index=layer_idx, layer_name=f"enc_{layer_idx}_ln1",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len, "hidden_dim": d}
        )
        op_id += 1
        graph.add_operator(ln1)

        attn = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.ATTENTION,
            layer_index=layer_idx, layer_name=f"enc_{layer_idx}_self_attn",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            weight_shapes=[_make_tensor([d, d], dtype)] * 4,  # Q, K, V, O
            params={"batch_size": batch_size, "seq_len": seq_len,
                    "hidden_dim": d, "num_heads": h, "head_dim": d // h}
        )
        op_id += 1
        graph.add_operator(attn)

        ln2 = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LAYERNORM,
            layer_index=layer_idx, layer_name=f"enc_{layer_idx}_ln2",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len, "hidden_dim": d}
        )
        op_id += 1
        graph.add_operator(ln2)

        ffn = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.FFN,
            layer_index=layer_idx, layer_name=f"enc_{layer_idx}_ffn",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            weight_shapes=[_make_tensor([d, ffn_dim], dtype),
                           _make_tensor([ffn_dim, d], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len,
                    "hidden_dim": d, "ffn_dim": ffn_dim}
        )
        op_id += 1
        graph.add_operator(ffn)

    # ── Decoder ──────────────────────────────────────────────
    for layer_idx in range(25, 49):
        ln1 = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LAYERNORM,
            layer_index=layer_idx, layer_name=f"dec_{layer_idx}_ln1",
            input_shapes=[_make_tensor([batch_size, dec_seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, dec_seq_len, d], dtype)],
            params={"batch_size": batch_size, "seq_len": dec_seq_len, "hidden_dim": d}
        )
        op_id += 1
        graph.add_operator(ln1)

        # Causal self-attention
        self_attn = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.ATTENTION,
            layer_index=layer_idx, layer_name=f"dec_{layer_idx}_self_attn",
            input_shapes=[_make_tensor([batch_size, dec_seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, dec_seq_len, d], dtype)],
            weight_shapes=[_make_tensor([d, d], dtype)] * 4,
            params={"batch_size": batch_size, "seq_len": dec_seq_len,
                    "hidden_dim": d, "num_heads": h, "head_dim": d // h}
        )
        op_id += 1
        graph.add_operator(self_attn)

        ln2 = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LAYERNORM,
            layer_index=layer_idx, layer_name=f"dec_{layer_idx}_ln2",
            input_shapes=[_make_tensor([batch_size, dec_seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, dec_seq_len, d], dtype)],
            params={"batch_size": batch_size, "seq_len": dec_seq_len, "hidden_dim": d}
        )
        op_id += 1
        graph.add_operator(ln2)

        # Cross-attention (decoder attends to encoder output)
        cross_attn = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.ATTENTION,
            layer_index=layer_idx, layer_name=f"dec_{layer_idx}_cross_attn",
            input_shapes=[_make_tensor([batch_size, dec_seq_len, d], dtype),
                          _make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, dec_seq_len, d], dtype)],
            weight_shapes=[_make_tensor([d, d], dtype)] * 4,
            params={"batch_size": batch_size, "seq_len": dec_seq_len,
                    "hidden_dim": d, "num_heads": h, "head_dim": d // h,
                    "cross_attention": True}
        )
        op_id += 1
        graph.add_operator(cross_attn)

        ln3 = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LAYERNORM,
            layer_index=layer_idx, layer_name=f"dec_{layer_idx}_ln3",
            input_shapes=[_make_tensor([batch_size, dec_seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, dec_seq_len, d], dtype)],
            params={"batch_size": batch_size, "seq_len": dec_seq_len, "hidden_dim": d}
        )
        op_id += 1
        graph.add_operator(ln3)

        ffn = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.FFN,
            layer_index=layer_idx, layer_name=f"dec_{layer_idx}_ffn",
            input_shapes=[_make_tensor([batch_size, dec_seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, dec_seq_len, d], dtype)],
            weight_shapes=[_make_tensor([d, ffn_dim], dtype),
                           _make_tensor([ffn_dim, d], dtype)],
            params={"batch_size": batch_size, "seq_len": dec_seq_len,
                    "hidden_dim": d, "ffn_dim": ffn_dim}
        )
        op_id += 1
        graph.add_operator(ffn)

    return compute_flops(graph)


# ─────────────────────────────────────────────────────────────────────────────
# Mixtral MoE 111M
# ─────────────────────────────────────────────────────────────────────────────

def build_mixtral_moe_111m(batch_size: int = 8, seq_len: int = 2048,
                            dtype: str = "bf16") -> ComputationGraph:
    """
    Mixtral MoE 111M: 8 decoder layers, 8 experts, top-2 routing.
    Hidden dim: 512, Heads: 8 (GQA: 2 KV heads), Expert FFN dim: 1024.
    FLOPs counted using active experts only (top-2 of 8).
    """
    graph = ComputationGraph(
        model_name="Mixtral MoE 111M",
        model_type="transformer",
        batch_size=batch_size,
        dtype=dtype
    )
    op_id = 0
    d = 512
    h = 8
    kv_heads = 2
    head_dim = 64
    expert_ffn_dim = 1024
    num_active_experts = 2   # top-2 routing

    emb = OperatorNode(
        op_id=f"op_{op_id}", op_type=OperatorType.EMBEDDING,
        layer_index=0, layer_name="embedding",
        output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
        weight_shapes=[_make_tensor([32000, d], dtype)],
        params={"vocab_size": 32000, "hidden_dim": d}
    )
    op_id += 1
    graph.add_operator(emb)

    for layer_idx in range(1, 9):
        ln1 = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LAYERNORM,
            layer_index=layer_idx, layer_name=f"layer_{layer_idx}_rmsnorm1",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len, "hidden_dim": d}
        )
        op_id += 1
        graph.add_operator(ln1)

        kv_dim = kv_heads * head_dim  # 128
        attn = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.ATTENTION,
            layer_index=layer_idx, layer_name=f"layer_{layer_idx}_gqa",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            weight_shapes=[
                _make_tensor([d, d], dtype),
                _make_tensor([d, kv_dim], dtype),
                _make_tensor([d, kv_dim], dtype),
                _make_tensor([d, d], dtype),
            ],
            params={"batch_size": batch_size, "seq_len": seq_len,
                    "hidden_dim": d, "num_heads": h, "head_dim": head_dim,
                    "num_kv_heads": kv_heads}
        )
        op_id += 1
        graph.add_operator(attn)

        ln2 = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LAYERNORM,
            layer_index=layer_idx, layer_name=f"layer_{layer_idx}_rmsnorm2",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len, "hidden_dim": d}
        )
        op_id += 1
        graph.add_operator(ln2)

        # Router (lightweight linear layer selecting top-2 experts)
        router = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LINEAR,
            layer_index=layer_idx, layer_name=f"layer_{layer_idx}_router",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, 8], dtype)],
            weight_shapes=[_make_tensor([d, 8], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len,
                    "in_features": d, "out_features": 8}
        )
        op_id += 1
        graph.add_operator(router)

        # Active expert FFNs — model only runs top-2 of 8 experts per token
        for exp_idx in range(num_active_experts):
            ffn = OperatorNode(
                op_id=f"op_{op_id}", op_type=OperatorType.FFN,
                layer_index=layer_idx,
                layer_name=f"layer_{layer_idx}_expert_{exp_idx}_ffn",
                input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
                output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
                weight_shapes=[
                    _make_tensor([d, expert_ffn_dim], dtype),   # gate_proj
                    _make_tensor([d, expert_ffn_dim], dtype),   # up_proj
                    _make_tensor([expert_ffn_dim, d], dtype),   # down_proj
                ],
                params={"batch_size": batch_size, "seq_len": seq_len,
                        "hidden_dim": d, "ffn_dim": expert_ffn_dim}
            )
            op_id += 1
            graph.add_operator(ffn)

    return compute_flops(graph)


# ─────────────────────────────────────────────────────────────────────────────
# ESM2-650M
# ─────────────────────────────────────────────────────────────────────────────

def build_esm2_650m(batch_size: int = 16, seq_len: int = 1024,
                    dtype: str = "fp16") -> ComputationGraph:
    """
    ESM2-650M protein language model: 33 encoder layers.
    Hidden dim: 1280, Heads: 20, FFN dim: 5120.
    Bidirectional (no causal masking), rotary position embeddings.
    """
    graph = ComputationGraph(
        model_name="ESM2-650M",
        model_type="transformer",
        batch_size=batch_size,
        dtype=dtype
    )
    op_id = 0
    d = 1280
    h = 20
    ffn_dim = 5120

    emb = OperatorNode(
        op_id=f"op_{op_id}", op_type=OperatorType.EMBEDDING,
        layer_index=0, layer_name="embedding",
        output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
        weight_shapes=[_make_tensor([33, d], dtype)],  # 20 AA + special tokens
        params={"vocab_size": 33, "hidden_dim": d}
    )
    op_id += 1
    graph.add_operator(emb)

    for layer_idx in range(1, 34):
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
            weight_shapes=[_make_tensor([d, d], dtype)] * 4,  # Q, K, V, O
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
# ViT-Base/16
# ─────────────────────────────────────────────────────────────────────────────

def build_vit_base(batch_size: int = 256, dtype: str = "fp16", **kwargs) -> ComputationGraph:
    """
    ViT-Base/16: 12 transformer encoder layers.
    Hidden dim: 768, Heads: 12, FFN dim: 3072.
    196 image patches (16x16 from 224x224) + 1 CLS token = 197 sequence length.
    Patch embedding via Conv2d 16x16.
    """
    graph = ComputationGraph(
        model_name="ViT-Base",
        model_type="transformer",
        batch_size=batch_size,
        dtype=dtype
    )
    op_id = 0
    d = 768
    h = 12
    ffn_dim = 3072
    seq_len = 197   # 196 patches + CLS

    # Patch embedding: Conv2d 16x16 stride 16 → linear projection
    patch_emb = OperatorNode(
        op_id=f"op_{op_id}", op_type=OperatorType.CONV2D,
        layer_index=0, layer_name="patch_embedding",
        input_shapes=[_make_tensor([batch_size, 3, 224, 224], dtype)],
        output_shapes=[_make_tensor([batch_size, d, 14, 14], dtype)],
        weight_shapes=[_make_tensor([d, 3, 16, 16], dtype)],
        params={"batch_size": batch_size, "in_channels": 3, "out_channels": d,
                "input_height": 224, "input_width": 224,
                "kernel_h": 16, "kernel_w": 16, "stride": 16, "padding": 0}
    )
    op_id += 1
    graph.add_operator(patch_emb)

    for layer_idx in range(1, 13):
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
            weight_shapes=[_make_tensor([d, d], dtype)] * 4,
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

    # Classification head
    cls_head = OperatorNode(
        op_id=f"op_{op_id}", op_type=OperatorType.LINEAR,
        layer_index=13, layer_name="classifier",
        input_shapes=[_make_tensor([batch_size, d], dtype)],
        output_shapes=[_make_tensor([batch_size, 1000], dtype)],
        weight_shapes=[_make_tensor([1000, d], dtype)],
        params={"batch_size": batch_size, "seq_len": 1,
                "in_features": d, "out_features": 1000}
    )
    op_id += 1
    graph.add_operator(cls_head)

    return compute_flops(graph)


# ─────────────────────────────────────────────────────────────────────────────
# DiT-Large/4
# ─────────────────────────────────────────────────────────────────────────────

def build_dit_large(batch_size: int = 32, dtype: str = "fp16", **kwargs) -> ComputationGraph:
    """
    DiT-Large/4: 24 transformer layers for diffusion image generation.
    Hidden dim: 1024, Heads: 16, FFN dim: 4096.
    Input: 32x32 latent (VAE 8x downsample of 256x256) → 64 patches at patch_size=4.
    adaLN-Zero conditioning on timestep + class label.
    """
    graph = ComputationGraph(
        model_name="DiT-Large",
        model_type="transformer",
        batch_size=batch_size,
        dtype=dtype
    )
    op_id = 0
    d = 1024
    h = 16
    ffn_dim = 4096
    seq_len = 64    # (32/4)^2 patches

    # Patch embedding: 4x4 conv on 32x32 latent
    patch_emb = OperatorNode(
        op_id=f"op_{op_id}", op_type=OperatorType.CONV2D,
        layer_index=0, layer_name="patch_embedding",
        input_shapes=[_make_tensor([batch_size, 4, 32, 32], dtype)],
        output_shapes=[_make_tensor([batch_size, d, 8, 8], dtype)],
        weight_shapes=[_make_tensor([d, 4, 4, 4], dtype)],
        params={"batch_size": batch_size, "in_channels": 4, "out_channels": d,
                "input_height": 32, "input_width": 32,
                "kernel_h": 4, "kernel_w": 4, "stride": 4, "padding": 0}
    )
    op_id += 1
    graph.add_operator(patch_emb)

    # Timestep + class conditioning MLP (adaLN modulation params)
    cond_mlp = OperatorNode(
        op_id=f"op_{op_id}", op_type=OperatorType.LINEAR,
        layer_index=0, layer_name="conditioning_mlp",
        input_shapes=[_make_tensor([batch_size, 256], dtype)],
        output_shapes=[_make_tensor([batch_size, d], dtype)],
        weight_shapes=[_make_tensor([256, d], dtype),
                       _make_tensor([d, d], dtype)],
        params={"batch_size": batch_size, "seq_len": 1,
                "in_features": 256, "out_features": d}
    )
    op_id += 1
    graph.add_operator(cond_mlp)

    for layer_idx in range(1, 25):
        # adaLN modulation (lightweight per-layer conditioning)
        adaln = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LAYERNORM,
            layer_index=layer_idx, layer_name=f"layer_{layer_idx}_adaln",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len, "hidden_dim": d}
        )
        op_id += 1
        graph.add_operator(adaln)

        attn = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.ATTENTION,
            layer_index=layer_idx, layer_name=f"layer_{layer_idx}_attn",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            weight_shapes=[_make_tensor([d, d], dtype)] * 4,
            params={"batch_size": batch_size, "seq_len": seq_len,
                    "hidden_dim": d, "num_heads": h, "head_dim": d // h}
        )
        op_id += 1
        graph.add_operator(attn)

        adaln2 = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LAYERNORM,
            layer_index=layer_idx, layer_name=f"layer_{layer_idx}_adaln2",
            input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
            params={"batch_size": batch_size, "seq_len": seq_len, "hidden_dim": d}
        )
        op_id += 1
        graph.add_operator(adaln2)

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

    # Final layer: linear decoder to predict noise
    final_ln = OperatorNode(
        op_id=f"op_{op_id}", op_type=OperatorType.LAYERNORM,
        layer_index=25, layer_name="final_ln",
        input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
        output_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
        params={"batch_size": batch_size, "seq_len": seq_len, "hidden_dim": d}
    )
    op_id += 1
    graph.add_operator(final_ln)

    final_linear = OperatorNode(
        op_id=f"op_{op_id}", op_type=OperatorType.LINEAR,
        layer_index=25, layer_name="final_linear",
        input_shapes=[_make_tensor([batch_size, seq_len, d], dtype)],
        output_shapes=[_make_tensor([batch_size, seq_len, 4 * 16], dtype)],  # 4 ch × 4×4 patch
        weight_shapes=[_make_tensor([d, 4 * 16], dtype)],
        params={"batch_size": batch_size, "seq_len": seq_len,
                "in_features": d, "out_features": 64}
    )
    op_id += 1
    graph.add_operator(final_linear)

    return compute_flops(graph)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ZOO REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# DINOv2-Large (ViT-L/14)
# ─────────────────────────────────────────────────────────────────────────────

def build_dinov2_large(batch_size: int = 2048, dtype: str = "fp16",
                       **kwargs) -> ComputationGraph:
    """
    DINOv2-Large: ViT-Large/14 self-supervised vision transformer.
    24 encoder layers, hidden=1024, heads=16, ffn=4096.
    patch_size=14 → 256 patches from 224×224 image + 1 CLS = 257 seq_len.
    Self-supervised objective: DINO student-teacher distillation.
    FLOPs counted for student network only (teacher is EMA, no backward).
    """
    graph = ComputationGraph(
        model_name="DINOv2-Large",
        model_type="transformer",
        batch_size=batch_size,
        dtype=dtype
    )
    op_id = 0
    d = 1024
    h = 16
    ffn_dim = 4096
    seq_len = 257   # 256 patches + CLS

    # Patch embedding: Conv2d 14×14 stride 14
    patch_emb = OperatorNode(
        op_id=f"op_{op_id}", op_type=OperatorType.CONV2D,
        layer_index=0, layer_name="patch_embedding",
        input_shapes=[_make_tensor([batch_size, 3, 224, 224], dtype)],
        output_shapes=[_make_tensor([batch_size, d, 16, 16], dtype)],
        weight_shapes=[_make_tensor([d, 3, 14, 14], dtype)],
        params={"batch_size": batch_size, "in_channels": 3, "out_channels": d,
                "input_height": 224, "input_width": 224,
                "kernel_h": 14, "kernel_w": 14, "stride": 14, "padding": 0}
    )
    op_id += 1
    graph.add_operator(patch_emb)

    for layer_idx in range(1, 25):
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
            weight_shapes=[_make_tensor([d, d], dtype)] * 4,
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

    # Projection head (3-layer MLP on CLS token)
    for proj_idx, (in_f, out_f) in enumerate([(d, 2048), (2048, 2048), (2048, 65536)]):
        proj = OperatorNode(
            op_id=f"op_{op_id}", op_type=OperatorType.LINEAR,
            layer_index=25, layer_name=f"proj_head_{proj_idx}",
            input_shapes=[_make_tensor([batch_size, in_f], dtype)],
            output_shapes=[_make_tensor([batch_size, out_f], dtype)],
            weight_shapes=[_make_tensor([in_f, out_f], dtype)],
            params={"batch_size": batch_size, "seq_len": 1,
                    "in_features": in_f, "out_features": out_f}
        )
        op_id += 1
        graph.add_operator(proj)

    return compute_flops(graph)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ZOO REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ZOO = {
    "resnet50":          build_resnet50,
    "bert_large":        build_bert_large,
    "bert_large_msl512": build_bert_large,
    "gpt2_1_5b":         build_gpt2_1_5b,
    "gpt3_2p7b":         build_gpt3_2p7b,
    "llama3_8b":         build_llama3_8b,
    "t5_3b":             build_t5_3b,
    "mixtral_moe_111m":  build_mixtral_moe_111m,
    "esm2_650m":         build_esm2_650m,
    "vit_base":          build_vit_base,
    "dit_large":         build_dit_large,
    "dinov2_large":      build_dinov2_large,
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