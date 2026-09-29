"""Metric 6: hold-out error vs number of training programs. SPEC sections 3.3 / 8.

Trains the same architecture on nested subsets of the training programs while
val/test stay fixed, so the curve answers one question only: at how many DOE
programs does the error stop falling?

    python scripts/07_sweep_ndoe.py --selection r2
    python scripts/07_sweep_ndoe.py --selection r2 --set eval.ndoe_sweep=4,8,16,32
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import _bootstrap  # noqa: F401
from config import add_config_args, configure


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", default="r1")
    ap.add_argument("--epochs", type=int, default=0, help="0 = cfg.train.epochs")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = configure(args)

    from tcn_cnc2tcp import dataset as ds, plots
    from tcn_cnc2tcp.trainer import train_model, validate

    sel = cfg.paths.meta_dir / f"selection_{args.selection}.txt"
    names = [l.strip() for l in sel.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    split = ds.split_programs(cfg, names)
    print(f"{len(names)} programs: train {len(split['train'])}, "
          f"val {len(split['val'])}, test {len(split['test'])}")

    val_programs = ds.load_programs(cfg, split["val"], verbose=False)
    test_programs = ds.load_programs(cfg, split["test"], verbose=False)

    rows = []
    base = cfg.paths.runs_out_dir / f"sweep_ndoe_{args.selection}"
    for n in cfg.eval.ndoe_sweep:
        n = int(n)
        if n > len(split["train"]):
            print(f"skipping n={n}: only {len(split['train'])} training programs")
            continue
        subset = split["train"][:n]                 # nested by construction
        programs = ds.load_programs(cfg, subset, verbose=False)
        out = base / f"n{n:03d}"
        print(f"\n=== training on {n} programs -> {out}")
        res = train_model(cfg, programs, val_programs, out,
                          epochs=args.epochs or cfg.train.epochs, tag=f"n{n}")
        from tcn_cnc2tcp.trainer import load_checkpoint
        model, _, device = load_checkpoint(res["best_path"], cfg)
        v = validate(model, cfg, val_programs, device)
        t = validate(model, cfg, test_programs, device)
        rows.append({
            "n_programs": n,
            "n_samples": sum(len(p) for p in programs),
            "val_rmse_um": v.get("val_rmse_um"),
            "val_rmse_g01_um": v.get("val_rmse_g01_um"),
            "test_rmse_um": t.get("val_rmse_um"),
            "test_rmse_g01_um": t.get("val_rmse_g01_um"),
            "best_epoch": res["history"][-1]["epoch"],
        })
        print(f"  -> val {rows[-1]['val_rmse_um']:.4f} um | "
              f"test {rows[-1]['test_rmse_um']:.4f} um")

    df = pd.DataFrame(rows)
    out_dir = cfg.paths.report_dir / "sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"metric6_ndoe_{args.selection}.csv", index=False,
              lineterminator="\n")
    if len(df) > 1:
        plots.curve(out_dir / f"metric6_ndoe_{args.selection}.png",
                    df["n_programs"].tolist(),
                    {"val": df["val_rmse_um"].tolist(),
                     "test": df["test_rmse_um"].tolist()},
                    "training programs", "e RMSE [um]",
                    "hold-out error vs number of DOE programs", logx=True)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print(f"\nwrote {out_dir}")
    print("Stop adding programs where this curve flattens (SPEC 3.3 step 3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
