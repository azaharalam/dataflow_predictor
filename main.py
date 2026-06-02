#!/usr/bin/env python3
"""
main.py — Dataflow Accelerator Performance Predictor

CLI tool to compute and analyze 9 performance metrics for DNN training
on dataflow AI accelerators (Cerebras WSE-2, SambaNova RDU, Graphcore IPU).

Usage:
  python -m dataflow_predictor.main --model bert_large --hardware cerebras_wse2
  python -m dataflow_predictor.main --model resnet50 --hardware sambanova_sn30 --batch_size 32
  python -m dataflow_predictor.main --model gpt2_1_5b --hardware graphcore_bow_ipu --devices 4
  python -m dataflow_predictor.main --compare_all --batch_size 16 --save csv

  # Add a new model — NO CODE CHANGES:
  # 1. Copy configs/models/TEMPLATE_new_model.yaml
  # 2. Fill it in and save as configs/models/my_model.yaml
  # 3. Run: python -m dataflow_predictor.main --model my_model --hardware cerebras_wse2

Available hardware : cerebras_wse2, sambanova_sn30, graphcore_bow_ipu
Available models   : auto-discovered from configs/models/*.yaml
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hardware.base_hardware import load_hardware_spec, SUPPORTED_PLATFORMS
from models.model_loader import (list_available_models, get_training_config)
from models.model_zoo import get_model
from core.compute_module import run_compute_module
from core.memory_module import run_memory_module
from core.communication_module import run_communication_module
from core.training_time import predict_training_time
from analysis.reporter import Reporter
from validation.c3_validator import (
    validate_graphcore, validate_sambanova, validate_cerebras,
    load_measurements_csv, print_collection_guide, C3ValidationReport
)


def run_analysis(
    model_name: str,
    hardware_name: str,
    batch_size: int,
    seq_len: int,
    dtype: str,
    num_devices: int,
    save_format: str,
    output_dir: str,
    verbose: bool,
) -> dict:
    """
    Run the full 9-metric analysis pipeline for one model + hardware combo.

    Pipeline:
      1. Load model YAML → build computation graph (FLOPs annotated)
      2. Load training config from same YAML
      3. Compute Module → OLET, MFU, AI
      4. Memory Module  → MBU, SRE, OMT
      5. Communication Module → CCR, CCBU, SOF
      6. Training Time → 7 equations
      7. Report
    """
    print(f"\n{'='*60}")
    print(f"  Running: {model_name.upper()} on {hardware_name.upper()}")
    print(f"  Batch: {batch_size}  Devices: {num_devices}  DType: {dtype}")
    print(f"{'='*60}")

    # ── 1. Load hardware ────────────────────────────────────────
    hardware = load_hardware_spec(hardware_name)
    if verbose:
        print(hardware)

    # ── 2. Build computation graph from YAML config ─────────────
    # CLI args override YAML defaults when provided
    graph = get_model(
        model_name,
        batch_size=batch_size,
        seq_len=seq_len,
        dtype=dtype
    )

    # ── 2b. Load training config from same YAML ─────────────────
    train_cfg = get_training_config(model_name)
    # Use training config values as fallbacks where CLI didn't specify
    backward_ratio = train_cfg.get("backward_ratio", 2.0)

    if verbose:
        print("\n" + graph.summary())

    # ── 3. Compute Module ───────────────────────────────────────
    print("\n  [1/4] Running Compute Module...")
    compute_metrics = run_compute_module(graph, hardware)

    # ── 4. Memory Module ────────────────────────────────────────
    print("  [2/4] Running Memory Module...")
    memory_metrics = run_memory_module(
        graph, hardware,
        t_compute_s=compute_metrics.total_compute_time_s
    )

    # ── 5. Communication Module ─────────────────────────────────
    print("  [3/4] Running Communication Module...")
    gradient_bytes = graph.total_weight_bytes * 2  # FP16 gradients
    comm_metrics = run_communication_module(
        graph, hardware,
        t_compute_s=compute_metrics.total_compute_time_s,
        gradient_bytes=gradient_bytes,
        num_devices=num_devices,
        parallelism="data"
    )

    # ── 6. Training Time Prediction ─────────────────────────────
    print("  [4/4] Predicting Training Time (7 equations)...")
    pred = predict_training_time(
        graph, hardware,
        compute_metrics, memory_metrics, comm_metrics,
        num_devices=num_devices
    )

    # ── 7. Report ───────────────────────────────────────────────
    reporter = Reporter(output_dir=output_dir)
    reporter.print_full_report(compute_metrics, memory_metrics, comm_metrics, pred)

    if save_format in ("json", "both"):
        reporter.save_json(compute_metrics, memory_metrics, comm_metrics, pred)

    # Return flat dict for CSV comparison
    result = {
        "model":         model_name,
        "hardware":      hardware_name,
        "batch_size":    batch_size,
        "num_devices":   num_devices,
        "dtype":         dtype,
        # Compute
        "C1_compute_time_ms": round(compute_metrics.total_compute_time_s * 1000, 4),
        "C2_mfu_pct":    round(compute_metrics.mfu_predicted * 100, 2),
        "C3_ai":         round(compute_metrics.ai_weighted_average, 4),
        "C3_compute_bound_pct": round(compute_metrics.fraction_compute_bound * 100, 1),
        "spu_pct":       round(compute_metrics.pipeline_fill_efficiency * 100, 2),
        # Memory
        "M1_mbu_pct":    round(memory_metrics.mbu * 100, 2),
        "M2_sre_pct":    round(memory_metrics.sre * 100, 2),
        "M3_otaf":       round(memory_metrics.otaf, 4),
        "M3_offchip_gb": round(memory_metrics.actual_offchip_bytes / 1e9, 4),
        "peak_mem_gb":   round(memory_metrics.peak_memory_demand_gb, 4),
        "rec_batch_size": memory_metrics.recommended_batch_size,
        # Communication
        "Comm1_ccr":     round(comm_metrics.ccr, 6),
        "Comm2_ccbu_pct": round(comm_metrics.ccbu * 100, 2),
        "Comm3_sof_pct": round(comm_metrics.sof * 100, 2),
        "scaling_eff_pct": round(comm_metrics.scaling_efficiency * 100, 2),
        # Training time (ms)
        "T_eq1_ms":      round(pred.eq1_operator_decomposition_ms, 3),
        "T_eq2_ms":      round(pred.eq2_roofline_bounded_ms, 3),
        "T_eq3_ms":      round(pred.eq3_streaming_pipeline_ms, 3),
        "T_eq4_ms":      round(pred.eq4_memory_traffic_driven_ms, 3),
        "T_eq5_ms":      round(pred.eq5_mfu_normalized_ms, 3),
        "T_eq6_ms":      round(pred.eq6_comm_aware_scaling_ms, 3),
        "T_eq7_ms":      round(pred.eq7_hierarchical_bottleneck_ms, 3),
        "bottleneck":    pred.bottleneck,
        # Throughput from best equation
        "throughput_eq7_sps": round(pred.throughput_eq7, 1),
    }
    return result


def run_compare_all(
    batch_size: int, seq_len: int, dtype: str,
    num_devices: int, save_format: str, output_dir: str, verbose: bool
):
    """Run all model × hardware combinations and produce a comparison table."""
    available_models = list_available_models()
    print("\n" + "="*70)
    print("  FULL CROSS-PLATFORM COMPARISON")
    print(f"  Models: {list(available_models.keys())}")
    print("="*70)

    all_results = []
    for model_name in available_models.keys():
        for hardware_name in SUPPORTED_PLATFORMS:
            try:
                result = run_analysis(
                    model_name, hardware_name, batch_size, seq_len,
                    dtype, num_devices, "none", output_dir, verbose=False
                )
                all_results.append(result)
            except Exception as e:
                print(f"  WARNING: {model_name} x {hardware_name} failed: {e}")

    if all_results and save_format in ("csv", "both"):
        reporter = Reporter(output_dir=output_dir)
        reporter.save_csv(all_results, "comparison_all.csv")

    # Print comparison table
    print(f"\n{'='*100}")
    print(f"  {'Model':<18} {'Hardware':<22} {'MFU%':>5} {'SRE%':>5} "
          f"{'MBU%':>5} {'CCR':>7} {'T_eq7(ms)':>10} {'Bottleneck':>12}")
    print(f"  {'-'*98}")
    for r in all_results:
        print(
            f"  {r['model']:<18} {r['hardware']:<22} "
            f"{r['C2_mfu_pct']:>5.1f} {r['M2_sre_pct']:>5.1f} "
            f"{r['M1_mbu_pct']:>5.1f} {r['Comm1_ccr']:>7.4f} "
            f"{r['T_eq7_ms']:>10.3f} {r['bottleneck']:>12}"
        )
    print(f"{'='*100}")


def build_parser() -> argparse.ArgumentParser:
    # Dynamically discover available models from YAML files — no hardcoding
    available_models = list(list_available_models().keys())

    p = argparse.ArgumentParser(
        description="Dataflow Accelerator Performance Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model — accepts ANY model_id that has a YAML in configs/models/
    p.add_argument(
        "--model", "-m",
        default="bert_large",
        help=(
            f"Model to analyze. Available: {available_models}. "
            "To add a new model: copy configs/models/TEMPLATE_new_model.yaml, "
            "fill it in, no code changes needed. (default: bert_large)"
        )
    )

    # Hardware
    p.add_argument(
        "--hardware", "-hw",
        choices=SUPPORTED_PLATFORMS,
        default="cerebras_wse2",
        help="Target hardware platform (default: cerebras_wse2)"
    )

    # Training config — CLI overrides the YAML defaults
    p.add_argument("--batch_size", "-bs", type=int, default=None,
                   help="Batch size override (default: use model YAML value)")
    p.add_argument("--seq_len", "-sl", type=int, default=None,
                   help="Sequence length override for transformers (default: use YAML)")
    p.add_argument("--dtype", "-dt",
                   choices=["fp16", "bf16", "fp32"], default=None,
                   help="Data type override (default: use model YAML value)")
    p.add_argument("--devices", "-d", type=int, default=1,
                   help="Number of accelerator devices (default: 1)")

    # Output
    p.add_argument(
        "--save", "-s",
        choices=["none", "json", "csv", "both"],
        default="none",
        help="Save results to file (default: none)"
    )
    p.add_argument("--output_dir", "-o", default="./results",
                   help="Output directory for saved files (default: ./results)")

    # Modes
    p.add_argument("--compare_all", action="store_true",
                   help="Run all model × hardware combinations")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print detailed hardware specs and graph summary")
    p.add_argument("--list_models", action="store_true",
                   help="List available models and exit")
    p.add_argument("--list_hardware", action="store_true",
                   help="List available hardware platforms and exit")


    # C3 Validation
    p.add_argument("--validate_c3", action="store_true",
                   help="Run C3 classification accuracy validation")
    p.add_argument("--measurements_file", type=str, default="",
                   help="Path to CSV with profiler measurements "
                        "(Graphcore: PopVision export; SambaNova: SambaTune section report)")
    p.add_argument("--flops_utilization", type=float, default=-1.0,
                   help="Cerebras only: flops_utilization from cstorch profiler [0-1]")
    p.add_argument("--c3_guide", action="store_true",
                   help="Print measurement collection guide for the selected platform")


    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.list_models:
        available = list_available_models()
        print("Available models (from configs/models/*.yaml):")
        for mid, mname in available.items():
            print(f"  {mid:<25} {mname}")
        print(f"\nTo add a new model: copy configs/models/TEMPLATE_new_model.yaml")
        print(f"Fill in your model definition — no Python code changes needed.")
        return

    if args.list_hardware:
        print("Available hardware platforms:")
        for hw in SUPPORTED_PLATFORMS:
            print(f"  {hw}")
        return

    # Resolve batch_size and seq_len:
    # If not provided on CLI, the model YAML defaults will be used
    # (handled inside get_model via build_graph_from_config)
    bs  = args.batch_size   # None = use YAML default
    sl  = args.seq_len      # None = use YAML default
    dt  = args.dtype        # None = use YAML default

    #C3- Arithmatic Intensity Validation command
    if args.validate_c3 or args.c3_guide:
        run_c3_validation(
            model_name=args.model,
            hardware_name=args.hardware,
            batch_size=bs or 32,
            seq_len=sl or 512,
            dtype=dt or "fp16",
            measurements_file=args.measurements_file,
            flops_utilization=args.flops_utilization,
            guide=args.c3_guide,
        )
        return


    if args.compare_all:
        run_compare_all(
            batch_size=bs or 32,
            seq_len=sl or 512,
            dtype=dt or "fp16",
            num_devices=args.devices,
            save_format=args.save,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    else:
        run_analysis(
            model_name=args.model,
            hardware_name=args.hardware,
            batch_size=bs or 32,
            seq_len=sl or 512,
            dtype=dt or "fp16",
            num_devices=args.devices,
            save_format=args.save,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )


def run_c3_validation(
    model_name: str,
    hardware_name: str,
    batch_size: int,
    seq_len: int,
    dtype: str,
    measurements_file: str,
    flops_utilization: float,
    guide: bool,
):
    """
    Run C3 classification accuracy validation for a given platform.

    Loads measurements from CSV (Graphcore / SambaNova) or uses a
    single float (Cerebras flops_utilization), compares against
    predicted compute-bound / memory-bound classification, and
    prints the classification accuracy report.
    """
    if guide:
        print_collection_guide(hardware_name, model_name)
        return

    # Build graph and run compute module to get predictions
    hardware = load_hardware_spec(hardware_name)
    graph    = get_model(model_name, batch_size=batch_size,
                         seq_len=seq_len, dtype=dtype)
    compute_metrics = run_compute_module(graph, hardware)

    # Platform-specific validation
    if hardware_name == "graphcore_bow_ipu":
        if not measurements_file:
            print("ERROR: --measurements_file required for Graphcore validation.")
            print("Run with --c3_guide to see collection instructions.")
            return
        measurements = load_measurements_csv(measurements_file)
        report = validate_graphcore(compute_metrics, graph, hardware, measurements)

    elif hardware_name == "sambanova_sn30":
        if not measurements_file:
            print("ERROR: --measurements_file required for SambaNova validation.")
            print("Run with --c3_guide to see collection instructions.")
            return
        sections = load_measurements_csv(measurements_file)
        report   = validate_sambanova(compute_metrics, graph, hardware, sections)

    elif hardware_name == "cerebras_wse2":
        if flops_utilization < 0:
            print("ERROR: --flops_utilization required for Cerebras validation.")
            print("Run with --c3_guide to see collection instructions.")
            return
        report = validate_cerebras(
            compute_metrics, graph, hardware, flops_utilization
        )

    else:
        print(f"ERROR: Unknown hardware '{hardware_name}'")
        return

    report.print_report()


if __name__ == "__main__":
    main()