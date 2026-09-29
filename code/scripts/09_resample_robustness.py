"""Metric 8: resample robustness of the trained TCN. SPEC sections 6.3 / 8.

The FIR measurement of section 6.3 showed the cubic-spline round trip costs
4..19 nm for sources up to 4 ms -- but that was a *linear* kernel and the spec
says so explicitly. This script re-measures it on the network: the input CNC is
degraded to a coarser / jittered / lossy sampling, restored to the standard grid
with a cubic spline, and pushed through the model. The target never moves.

    python scripts/09_resample_robustness.py --run tcn_r1
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from config import add_config_args, configure

# (label, decimate, jitter, drop_frac) -- decimate 1 = 0.5 ms source
CASES = [
    ("0.5 ms uniform (reference)", 1, 0.00, 0.00),
    ("0.5 ms, jitter +-25 %", 1, 0.25, 0.00),
    ("0.5 ms, jitter +-50 %", 1, 0.50, 0.00),
    ("0.5 ms, 5 % samples dropped", 1, 0.00, 0.05),
    ("1.0 ms uniform", 2, 0.00, 0.00),
    ("1.0 ms, jitter +-50 %", 2, 0.50, 0.00),
    ("2.0 ms uniform", 4, 0.00, 0.00),
    ("2.0 ms, jitter +-50 %", 4, 0.50, 0.00),
    ("4.0 ms uniform", 8, 0.00, 0.00),
    ("4.0 ms, jitter +-50 %", 8, 0.50, 0.00),
    ("8.0 ms uniform", 16, 0.00, 0.00),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--max-programs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    add_config_args(ap)
    args = ap.parse_args()
    cfg = configure(args)

    from tcn_cnc2tcp import dataset as ds, features, resample
    from tcn_cnc2tcp.trainer import load_checkpoint

    run_dir = cfg.paths.runs_out_dir / args.run
    split = ds.load_split(run_dir / "split.json")
    names = split[args.split][:args.max_programs or None]
    model, ck, device = load_checkpoint(run_dir / args.checkpoint, cfg)
    print(f"{args.run} @ epoch {ck.get('epoch')}, {len(names)} programs\n")

    programs = ds.load_programs(cfg, names, verbose=False)
    warm = cfg.receptive_field
    rows = []
    for label, dec, jit, drop in CASES:
        se = np.zeros(3)
        n = 0.0
        rt = []
        for prog in programs:
            rng = np.random.default_rng(args.seed)
            cnc = resample.corrupt_and_restore(
                prog.cnc, cfg.data.dt, rng, jitter=jit, decimate=dec,
                drop_frac=drop, kind=cfg.augment.interp)
            rt.append(np.abs(cnc - prog.cnc).max() * 1e3)         # um, round trip
            shaken = ds.Program(
                name=prog.name, cnc=cnc, e=prog.e,
                u=features.build_inputs(cnc, prog.g00, cfg.increment_scale),
                g00=prog.g00, valid=prog.valid, block=prog.block, dt=prog.dt)
            pred = ds.predict_program(model, cfg, shaken, device)
            m = prog.valid.copy()
            m[:warm] = False
            r = (prog.e[m] - pred[m]).astype(np.float64)
            se += (r ** 2).sum(0)
            n += m.sum()
        rmse = float(np.sqrt(se.sum() / (3 * n)))
        rows.append({"case": label, "source_dt_ms": dec * cfg.data.dt * 1e3,
                     "jitter": jit, "drop": drop,
                     "cnc_roundtrip_max_um": float(np.max(rt)),
                     "e_rmse_um": rmse})
        print(f"  {label:32s} round-trip {rows[-1]['cnc_roundtrip_max_um']:8.4f} um   "
              f"e RMSE {rmse:8.4f} um")

    df = pd.DataFrame(rows)
    ref = df.loc[0, "e_rmse_um"]
    df["ratio_vs_reference"] = df["e_rmse_um"] / ref
    out_dir = cfg.paths.report_dir / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metric8_resample_robustness.csv", index=False,
              lineterminator="\n")
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print(f"\nwrote {out_dir / 'metric8_resample_robustness.csv'}")
    print("Note: 2.7 ms is the coarsest standard grid that still resolves the "
          "vibration; a source coarser than that is a different question from "
          "a standard grid coarser than that (SPEC 6.4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
