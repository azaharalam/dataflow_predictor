#!/usr/bin/env python3
"""
main.py — Dataflow Accelerator Performance Predictor

CLI tool to compute and analyze 9 performance metrics for DNN training
on dataflow AI accelerators (Cerebras WSE-2, SambaNova RDU, Graphcore IPU).

Usage:
  python main.py --model bert_large --hardware cerebras_wse2 --batch_size 8
  python main.py --model resnet50 --hardware sambanova_sn30 --batch_size 32 --devices 4
  python main.py --model gpt2_1_5b --hardware graphcore_bow_ipu --batch_size 4 --save json
  python main.py --compare_all --batch_size 16

Available models   : resnet50, bert_large, gpt2_1_5b
Available hardware : cerebras_wse2, sambanova_sn30, graphcore_bow_ipu
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataflow_predictor.hardware.base_hardware import load_hardware_spec, SUPPORTED_PLATFORMS
from dataflow_predictor.models.model_zoo import get_model, MODEL_ZOO
from dataflow_predictor.core.compute_module import run_compute_module
from dataflow_predictor.core.memory_module import run_memory_module
from dataflow_predictor.core.communication_module import run_communication_module
from dataflow_predictor.core.training_time import predict_training_time
from dataflow_predictor.analysis.reporter import Reporter


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
      1. Build computation graph (FLOPs annotated)
      2. Compute Module → OLET, MFU, AI
      3. Memory Module  → MBU, SRE, OMT
      4. Communication Module → CCR, CCBU, SOF
      5. Training Time → 7 equations
      6. Report
    """
    print(f"\n{'='*60}")
    print(f"  Running: {model_name.upper()} on {hardware_name.upper()}")
    print(f"  Batch: {batch_size}  Devices: {num_devices}  DType: {dtype}")
    print(f"{'='*60}")

    # ── 1. Load hardware ────────────────────────────────────────
    hardware = load_hardware_spec(hardware_name)
    if verbose:
        print(hardware)

    # ── 2. Build computation graph ──────────────────────────────
    kwargs = {}
    if "bert" in model_name or "gpt" in model_name:
        kwargs["seq_len"] = seq_len

    graph = get_model(model_name, batch_size=batch_size, dtype=dtype, **kwargs)

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
    print("\n" + "="*70)
    print("  FULL CROSS-PLATFORM COMPARISON")
    print("="*70)

    all_results = []
    for model_name in MODEL_ZOO.keys():
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
    print(f"  {'Model':<14} {'Hardware':<22} {'MFU%':>5} {'SRE%':>5} "
          f"{'MBU%':>5} {'CCR':>7} {'T_eq7(ms)':>10} {'Bottleneck':>12}")
    print(f"  {'-'*98}")
    for r in all_results:
        print(
            f"  {r['model']:<14} {r['hardware']:<22} "
            f"{r['C2_mfu_pct']:>5.1f} {r['M2_sre_pct']:>5.1f} "
            f"{r['M1_mbu_pct']:>5.1f} {r['Comm1_ccr']:>7.4f} "
            f"{r['T_eq7_ms']:>10.3f} {r['bottleneck']:>12}"
        )
    print(f"{'='*100}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dataflow Accelerator Performance Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model
    p.add_argument(
        "--model", "-m",
        choices=list(MODEL_ZOO.keys()),
        default="bert_large",
        help="DNN model to analyze (default: bert_large)"
    )

    # Hardware
    p.add_argument(
        "--hardware", "-hw",
        choices=SUPPORTED_PLATFORMS,
        default="cerebras_wse2",
        help="Target hardware platform (default: cerebras_wse2)"
    )

    # Training config
    p.add_argument("--batch_size", "-bs", type=int, default=8,
                   help="Training batch size (default: 8)")
    p.add_argument("--seq_len", "-sl", type=int, default=512,
                   help="Sequence length for transformers (default: 512)")
    p.add_argument("--dtype", "-dt",
                   choices=["fp16", "bf16", "fp32"], default="fp16",
                   help="Data type (default: fp16)")
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

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for m in MODEL_ZOO.keys():
            print(f"  {m}")
        return

    if args.list_hardware:
        print("Available hardware platforms:")
        for hw in SUPPORTED_PLATFORMS:
            print(f"  {hw}")
        return

    if args.compare_all:
        run_compare_all(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            dtype=args.dtype,
            num_devices=args.devices,
            save_format=args.save,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    else:
        run_analysis(
            model_name=args.model,
            hardware_name=args.hardware,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            dtype=args.dtype,
            num_devices=args.devices,
            save_format=args.save,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
