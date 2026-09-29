"""Metric 7: hold-out error vs receptive-field length. SPEC sections 5.1 / 8.

The FIR work measured 100 ms -> 30 nm, 200 ms -> 4 nm, 300 ms -> 2 nm of
truncation error, i.e. longer is better with diminishing returns. This sweep
records what the TCN actually gains.

    python scripts/08_sweep_rf.py --selection r1
    python scripts/08_sweep_rf.py --selection r1 --set eval.rf_sweep_ms=100,200,300,500
"""
from __future__ import annotations

import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from config import add_config_args, configure


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", default="r1")
    ap.add_argument("--epochs", type=int, default=0)
    add_config_args(ap)
    args = ap.parse_args()
    cfg = configure(args)

    from tcn_cnc2tcp import dataset as ds, plots
    from tcn_cnc2tcp.model import dilations_for_ms
    from tcn_cnc2tcp.trainer import load_checkpoint, train_model, validate

    sel = cfg.paths.meta_dir / f"selection_{args.selection}.txt"
    names = [l.strip() for l in sel.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    split = ds.split_programs(cfg, names)
    train_programs = ds.load_programs(cfg, split["train"], verbose=False)
    val_programs = ds.load_programs(cfg, split["val"], verbose=False)
    test_programs = ds.load_programs(cfg, split["test"], verbose=False)

    base_window = cfg.train.window
    rows = []
    for ms in cfg.eval.rf_sweep_ms:
        cfg.model.dilations = dilations_for_ms(float(ms), cfg.data.dt,
                                               cfg.model.kernel_size)
        # the crop has to be comfortably longer than the warm-up it throws away
        cfg.train.window = max(base_window, int(3 * cfg.receptive_field))
        out = cfg.paths.runs_out_dir / f"sweep_rf_{args.selection}" / f"rf{int(ms):04d}ms"
        print(f"\n=== RF target {ms:.0f} ms -> dilations {cfg.model.dilations} "
              f"= {cfg.receptive_field} samples ({cfg.receptive_field_ms:.1f} ms), "
              f"window {cfg.train.window}")
        res = train_model(cfg, train_programs, val_programs, out,
                          epochs=args.epochs or cfg.train.epochs, tag=f"rf{ms}")
        model, _, device = load_checkpoint(res["best_path"], cfg)
        v = validate(model, cfg, val_programs, device)
        t = validate(model, cfg, test_programs, device)
        rows.append({
            "target_ms": float(ms),
            "receptive_field_ms": cfg.receptive_field_ms,
            "receptive_field_samples": cfg.receptive_field,
            "dilations": "-".join(str(d) for d in cfg.model.dilations),
            "parameters": model.num_parameters(),
            "val_rmse_um": v.get("val_rmse_um"),
            "test_rmse_um": t.get("val_rmse_um"),
            "test_rmse_g01_um": t.get("val_rmse_g01_um"),
        })
        print(f"  -> val {rows[-1]['val_rmse_um']:.4f} um | "
              f"test {rows[-1]['test_rmse_um']:.4f} um")

    df = pd.DataFrame(rows)
    out_dir = cfg.paths.report_dir / "sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"metric7_rf_{args.selection}.csv", index=False,
              lineterminator="\n")
    if len(df) > 1:
        plots.curve(out_dir / f"metric7_rf_{args.selection}.png",
                    df["receptive_field_ms"].tolist(),
                    {"val": df["val_rmse_um"].tolist(),
                     "test": df["test_rmse_um"].tolist()},
                    "receptive field [ms]", "e RMSE [um]",
                    "hold-out error vs receptive field")
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
