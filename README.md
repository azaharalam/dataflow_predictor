# Dataflow Accelerator Performance Predictor

Analytically predicts 9 performance metrics for DNN training on
dataflow AI accelerators: **Cerebras WSE-2**, **SambaNova SN30 RDU**, **Graphcore Bow-IPU**.

## Metrics

| Module | ID | Metric | Description |
|---|---|---|---|
| Compute | C1 | OLET | Operator-Level Execution Time |
| Compute | C2 | MFU | Model FLOPs Utilization |
| Compute | C3 | AI | Arithmetic Intensity per operator |
| Memory | M1 | MBU | Memory Bandwidth Utilization |
| Memory | M2 | SRE | Scratchpad Reuse Efficiency |
| Memory | M3 | OMT | Off-Chip Memory Traffic (OTAF) |
| Communication | Comm1 | CCR | Communication-to-Computation Ratio |
| Communication | Comm2 | CCBU | Collective Communication BW Utilization |
| Communication | Comm3 | SOF | Synchronization Overhead Fraction |

## Metric Dependency

HARDWARE SPEC ──────────────────────────────────────────────────┐
                                                                 │
GRAPH (tensor shapes, FLOPs) ──────────────────────────────────┐│
                                                                ││
                    ┌──────────────────────────────────────────┘│
                    │                                            │
                    ▼                                            │
            ┌─────────────┐                                      │
            │  C3 (AI)    │◄────────────────────────────────────┤
            │ independent │                                      │
            └──────┬──────┘                                      │
                   │                                             │
                   ▼                                             │
            ┌─────────────┐    uses simplified BW               │
            │  C1 (OLET)  │◄────────────────────────────────────┤
            │  T_compute  │    (should use SRE but doesn't yet) │
            └──────┬──────┘                                      │
                   │                                             │
          ┌────────┴────────┐                                    │
          │                 │                                    │
          ▼                 ▼                                    │
   ┌─────────────┐   ┌─────────────┐                            │
   │  C2 (MFU)  │   │  Comm1 CCR  │                            │
   │            │   │  Comm3 SOF  │                            │
   └─────────────┘   └──────┬──────┘                            │
                             │                                   │
          ┌──────────────────┘                                   │
          │                                                      │
          ▼                                                      │
   ┌─────────────┐                                               │
   │  M2 (SRE)   │◄────────────────────────────────────────────┘
   │  (reuse     │
   │  factors    │
   │  hardcoded) │
   └──────┬──────┘
          │
     ┌────┴────┐
     │         │
     ▼         ▼
┌─────────┐ ┌──────┐
│ M1(MBU) │ │M3(OMT│
│uses C1  │ │      │
│+M2(SRE) │ │      │
└─────────┘ └──────┘

## Installation

```bash
pip install pyyaml
```

No other dependencies required — pure Python analytical model.

## Usage

```bash
# Single model + hardware
python -m dataflow_predictor.main --model bert_large --hardware cerebras_wse2 --batch_size 8

# With multi-device
python -m dataflow_predictor.main --model gpt2_1_5b --hardware sambanova_sn30 --devices 4

# Save results to JSON
python -m dataflow_predictor.main --model resnet50 --hardware graphcore_bow_ipu --save json

# Compare all model × hardware combinations
python -m dataflow_predictor.main --compare_all --batch_size 16 --save csv

# List available options
python -m dataflow_predictor.main --list_models
python -m dataflow_predictor.main --list_hardware
```

## Available Models
- `resnet50`   — ResNet-50 v1.5
- `bert_large` — BERT-Large (24 layers, d=1024)
- `gpt2_1_5b`  — GPT-2 1.5B (48 layers, d=1600)

## Available Hardware
- `cerebras_wse2`     — Cerebras WSE-2
- `sambanova_sn30`    — SambaNova SN30 RDU
- `graphcore_bow_ipu` — Graphcore Bow-IPU (GC200)

## Project Structure

```
dataflow_predictor/
├── main.py                    ← CLI entry point
├── configs/
│   ├── hardware_specs.yaml    ← All three platform specs
│   └── model_configs.yaml     ← Model hyperparameters
├── hardware/
│   └── base_hardware.py       ← Hardware spec loader
├── models/
│   ├── operator_graph.py      ← Computation graph data structures
│   ├── flops_counter.py       ← FLOPs(l,j) for all operator types
│   └── model_zoo.py           ← Pre-built benchmark models
├── core/
│   ├── compute_module.py      ← C1 OLET, C2 MFU, C3 AI
│   ├── memory_module.py       ← M1 MBU, M2 SRE, M3 OMT
│   ├── communication_module.py← Comm1 CCR, Comm2 CCBU, Comm3 SOF
│   └── training_time.py       ← 7 training time equations
└── analysis/
    └── reporter.py            ← Tabular output, JSON/CSV export
```

## Adding a New Model

Create a function in `models/model_zoo.py` that builds and returns
a `ComputationGraph` with all operators defined:

```python
def build_vit_large(batch_size=8, seq_len=196, dtype="fp16"):
    graph = ComputationGraph("ViT-Large", "transformer", batch_size, dtype)
    # Add operators...
    return compute_flops(graph)
```

Then register it in `MODEL_ZOO` dict.

## Adding a New Hardware Platform

Add a new entry to `configs/hardware_specs.yaml` following the existing format.
The `execution_model` field controls which formulas are used:
- `"wafer_scale"` — Cerebras-style
- `"dataflow"`    — SambaNova-style
- `"bsp"`         — Graphcore-style
