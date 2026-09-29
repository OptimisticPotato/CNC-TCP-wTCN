"""Overfit test. SPEC section 7 step 2.

Two or three programs, augmentation off, no validation: the training error must
collapse towards zero. If it does not, the problem is capacity or optimisation,
and there is no point starting the real run.

    python scripts/04_overfit_test.py
    python scripts/04_overfit_test.py --n 3 --epochs 40
"""
from __future__ import annotations

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from config import add_config_args, configure


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", default=None)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    add_config_args(ap)
    args = ap.parse_args()
    cfg = configure(args)

    from tcn_cnc2tcp import dataset as ds
    from tcn_cnc2tcp.runs_io import list_runs
    from tcn_cnc2tcp.trainer import train_model, validate

    if args.runs:
        names = args.runs
    elif args.selection:
        path = cfg.paths.meta_dir / f"selection_{args.selection}.txt"
        names = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
    else:
        names = list_runs(cfg)
    names = names[:args.n]
    print("overfitting on:", names)

    programs = ds.load_programs(cfg, names)
    out = cfg.paths.runs_out_dir / "overfit"
    res = train_model(cfg, programs, [], out, epochs=args.epochs, augment=False,
                      tag="overfit")

    stats = validate(res["model"], cfg, programs, res["device"])

    # baseline = what "predict zero" would score, on exactly the same samples
    se = np.zeros(3)
    se_g01 = np.zeros(3)
    n = n_g01 = 0.0
    for p in programs:
        m = p.valid.copy()
        m[:cfg.receptive_field] = False
        se += (p.e[m].astype(np.float64) ** 2).sum(0)
        n += m.sum()
        mc = m & ~p.g00
        se_g01 += (p.e[mc].astype(np.float64) ** 2).sum(0)
        n_g01 += mc.sum()
    base = float(np.sqrt(se.sum() / (3 * n)))
    base_g01 = float(np.sqrt(se_g01.sum() / (3 * n_g01))) if n_g01 else float("nan")

    def pct(got, ref):
        return 100 * (1 - got / ref) if ref else float("nan")

    got = stats.get("val_rmse_um", float("nan"))
    got_g01 = stats.get("val_rmse_g01_um", float("nan"))
    print("\n---- result -------------------------------------------------------")
    print(f"{'':14s} {'fitted':>10s} {'predict 0':>10s} {'explained':>10s}")
    print(f"{'all samples':14s} {got:10.4f} {base:10.4f} {pct(got, base):9.2f} %")
    print(f"{'G01 only':14s} {got_g01:10.4f} {base_g01:10.4f} "
          f"{pct(got_g01, base_g01):9.2f} %")
    print(f"{'per axis':14s} x {stats.get('val_rmse_x', float('nan')):.4f}  "
          f"y {stats.get('val_rmse_y', float('nan')):.4f}  "
          f"z {stats.get('val_rmse_z', float('nan')):.4f}   [um]")
    print(f"\ntraining down-weighted G00 by loss.g00_weight={cfg.loss.g00_weight}, "
          f"so the 'all samples' row scores samples the objective de-emphasised.")
    print("If neither percentage is close to 100, fix capacity/optimisation "
          "before running 05_train.py (SPEC section 7 step 2). Cheapest knobs, "
          "in order: --epochs, model.linear_skip=true, model.channels, "
          "loss.lambda_stft=0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
