"""Identify the 6-input 3-output FIR model for a trained run. Metric 9.

    python scripts/10_fir_baseline.py --run tcn_r1               # taps = TCN RF
    python scripts/10_fir_baseline.py --run tcn_r1 --taps 200    # the slides' 200

Ordinary least squares, normal equations, Cholesky. No training loop, no
constraints -- the model is identified, not learned. It reads the run's
``split.json`` so it sees the same training programs and the same six increment
channels as the network, and it writes ``fir_baseline.npz`` next to the
checkpoint; 06_evaluate.py finds it there and draws it into the existing
figures, plus a metric-9 table comparing the two models row by row.

Two knobs exist only because this dataset behaves differently from the one the
slides were written against:

* ``--taps``. The slides use 200 (100 ms). Measured here on clean programs with
  plain least squares, held-out G01 RMSE is 3.2 um at 200 taps, 0.18 um at 509
  and 0.005 um at 1021 -- the truncation tail decays with tau ~ 50-70 ms, which
  is what the spec's own 100/200/300 ms table implies, not the ~12 ms of the
  dominant 37 Hz mode. Default is therefore ``eval.fir_taps`` = 1021 (510 ms).
  ``--taps 200`` reproduces the slides.
* ``--dc-penalty``. Leave it at 0. With enough taps the slides are right and
  sum(g) comes out at 5e-3 on its own; the +-10 cancelling DC gains reported
  earlier were an artefact of a truncated fit, the same artefact that produced
  a fake 100 ms kernel tail. Forcing sum(g) = 0 at 509 taps makes the held-out
  error 20x worse, because the true DC gain is small but not zero.

SPEC section 0 is explicit that FIR is neither the deliverable nor the pass
criterion. This exists to answer one question for the report: what did the
non-linearity buy?
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from config import add_config_args, configure


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="training run under work/runs")
    ap.add_argument("--taps", type=int, default=0,
                    help="kernel length; 0 = eval.fir_taps (1021 = 510 ms). "
                         "200 reproduces the slides and is far too short here.")
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="fix the ridge instead of selecting it on the val split "
                         "(fraction of trace(A)/p)")
    ap.add_argument("--dc-penalty", type=float, default=0.0,
                    help="strength of the sum(g)=0 constraint (SPEC 2); 0 turns "
                         "it off, which lets collinear G00 channels park a large "
                         "cancelling DC gain that only cancels on diagonal rapids")
    ap.add_argument("--weighted", action="store_true",
                    help="fit under the NETWORK's objective instead (G00 weighted "
                         "by loss.g00_weight). The slides weight every observed "
                         "error equally, which is the default here.")
    ap.add_argument("--chunk", type=int, default=4096)
    add_config_args(ap)
    args = ap.parse_args()
    cfg = configure(args)

    from tcn_cnc2tcp import dataset as ds, fir, metrics

    run_dir = cfg.paths.runs_out_dir / args.run
    split = ds.load_split(run_dir / "split.json")
    taps = args.taps or cfg.eval.fir_taps
    print(f"{args.run}: FIR with {taps} taps "
          f"({taps * cfg.data.dt * 1e3:.1f} ms), "
          f"{cfg.model.in_channels * taps:,} unknowns per axis, "
          f"{cfg.model.in_channels * taps * 3:,} in total")
    print("objective: " + ("the network's (G00 down-weighted)" if args.weighted
                           else "every observed error equally (as the slides do)"))
    print(f"train programs: {len(split['train'])}\n")

    train = ds.load_programs(cfg, split["train"], verbose=False)
    val = ds.load_programs(cfg, split["val"], verbose=False)
    grid = (args.ridge,) if args.ridge > 0 else fir.RIDGE_GRID
    info = fir.fit(cfg, train, val_programs=val, taps=taps, ridge_grid=grid,
                   chunk=args.chunk, use_weights=args.weighted,
                   dc_penalty=args.dc_penalty)
    path = fir.save(fir.default_path(cfg, args.run), info)
    h = info["h"]

    print(f"\nselected ridge {info['ridge_rel']:.1e} (lambda {info['lam']:.3e}) "
          f"on the {info['selected_on']} split")
    pd.DataFrame(info["sweep"]).to_csv(
        cfg.paths.report_dir / args.run / "metric9_fir_ridge_sweep.csv",
        index=False, lineterminator="\n")
    print("DC gain -- SPEC section 2 says sum(g) ~ 0: a constant velocity "
          "must leave no error")
    for a, ax in enumerate("xyz"):
        print(f"  out {ax}: " + "  ".join(
            f"{v:+.4f}" for v in np.asarray(info["dc_gain"])[a]))

    summ = pd.DataFrame(fir.impulse_response_summary(h, cfg.data.dt))
    summ.to_csv(cfg.paths.report_dir / args.run / "metric9_fir_kernels.csv",
                index=False, lineterminator="\n")
    print("\nkernel summary (energy-weighted centre, tail beyond 100 ms)")
    print(summ.sort_values("energy", ascending=False).head(9)
          .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    # quick look on the test split so the file is useful on its own
    warm = cfg.receptive_field
    rows = []
    for name in split["test"]:
        prog = ds.load_program(cfg, name)
        pred = fir.predict(h, prog.u)
        t = metrics.error_table(prog.e, pred, prog.valid, prog.g00, warm)
        t.insert(0, "program", name)
        rows.append(t)
    pooled = metrics.summarise(rows)
    print("\ntest split, FIR alone")
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    (cfg.paths.report_dir / args.run).mkdir(parents=True, exist_ok=True)
    pooled.to_csv(cfg.paths.report_dir / args.run / "metric9_fir_only.csv",
                  index=False, lineterminator="\n")
    (cfg.paths.report_dir / args.run / "metric9_fir_fit.json").write_text(
        json.dumps({k: v for k, v in info.items() if k != "h"}, indent=2),
        encoding="utf-8")

    print(f"\nwrote {path}")
    print(f"next: python scripts/06_evaluate.py --run {args.run}   "
          f"(the FIR now appears in the figures and in metric 9)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
