#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cerebras.streaming_model import CS3System, ModelCfg, predict
from cerebras.validate import validate


def main():
    ap = argparse.ArgumentParser(description="CS-3 weight-streaming predictor")
    ap.add_argument("--model", default="configs/models/llama3_200b.yaml")
    ap.add_argument("--hardware", default="configs/hardware/cerebras_cs3_alcf.yaml")
    ap.add_argument("--raw", help="raw artifact dir (enables validation)")
    ap.add_argument("--held-out", action="store_true",
                    help="mark this run as held-out (not the calibration run)")
    args = ap.parse_args()

    if args.raw:
        print(validate(args.model, args.hardware, args.raw,
                       calibration_point=not args.held_out))
        return

    m, hw = ModelCfg.from_yaml(args.model), CS3System.from_yaml(args.hardware)
    p = predict(m, hw)
    print(f"{m.name} on {hw.name}")
    print(f"  params        : {p.breakdown['params']:,}")
    print(f"  traffic/step  : {p.traffic_bytes/1e9:,.0f} GB")
    print(f"  T_stream      : {p.t_stream_s:.2f} s")
    print(f"  T_cmp (lower) : {p.t_cmp_lower_s:.2f} s")
    print(f"  regime        : {p.regime}")
    print(f"  step time     : {p.t_step_s:.2f} s   throughput {p.throughput:.4f} samples/s")


if __name__ == "__main__":
    main()