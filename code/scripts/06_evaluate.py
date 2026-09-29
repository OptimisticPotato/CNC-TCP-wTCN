"""Evaluation report. SPEC section 8, metrics 1-5 and all three figures.

    python scripts/06_evaluate.py --run tcn_r1
    python scripts/06_evaluate.py --run tcn_r1 --split val

Metrics 6, 7 and 8 have their own scripts (07/08/09) because each needs more
than one trained model.

Metric 5 (final form error) is produced as far as this repository can go: the
normal component of the servo error, plus a per-program CSV of predicted TCP so
the real form-error pipeline can consume it. The CSV is the deliverable; the
normal-deviation number is a proxy and is labelled as one.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from config import add_config_args, configure


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="training run name under work/runs")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--max-programs", type=int, default=0)
    add_config_args(ap)
    args = ap.parse_args()
    cfg = configure(args)

    from tcn_cnc2tcp import dataset as ds, metrics, plots
    from tcn_cnc2tcp.trainer import load_checkpoint

    run_dir = cfg.paths.runs_out_dir / args.run
    split = ds.load_split(run_dir / "split.json")
    names = split[args.split]
    if args.max_programs:
        names = names[:args.max_programs]
    if not names:
        raise SystemExit(f"split '{args.split}' is empty")

    model, ck, device = load_checkpoint(run_dir / args.checkpoint, cfg)
    print(f"{args.run}: checkpoint epoch {ck.get('epoch')} "
          f"(score {ck.get('score', float('nan')):.4f} um), "
          f"RF {cfg.receptive_field} samples = {cfg.receptive_field_ms:.1f} ms")
    print(f"evaluating {len(names)} '{args.split}' programs on {device}\n")

    rep = cfg.paths.report_dir / args.run / args.split
    rep.mkdir(parents=True, exist_ok=True)
    fs = 1.0 / cfg.data.dt
    warm = cfg.receptive_field

    err_tables, spec_tables, form_tables, blocks = [], [], [], []
    for i, name in enumerate(names):
        prog = ds.load_program(cfg, name)
        pred = ds.predict_program(model, cfg, prog, device)
        t = np.arange(len(prog)) * cfg.data.dt

        et = metrics.error_table(prog.e, pred, prog.valid, prog.g00, warm)
        et.insert(0, "program", name)
        err_tables.append(et)

        st = metrics.spectral_table(prog.e, pred, prog.valid, warm, fs,
                                    cfg.eval.nperseg, cfg.eval.psd_band_hz)
        st.insert(0, "program", name)
        spec_tables.append(st)

        ft = metrics.form_error_table(prog.cnc, prog.e, pred, prog.valid, prog.g00, warm)
        if len(ft):
            ft.insert(0, "program", name)
            form_tables.append(ft)

        bt = metrics.block_table(prog.block, prog.e, pred, prog.valid, prog.g00, warm)
        if len(bt):
            bt.insert(0, "program", name)
            blocks.append(bt)

        if cfg.eval.export_tcp_csv:
            metrics.export_tcp(rep / "tcp" / f"{name}.csv", t, prog.cnc, pred,
                               prog.e, prog.block)

        row = et[et["segment"] == "all"].iloc[0]
        print(f"  [{i + 1}/{len(names)}] {name[:44]:46s} "
              f"RMSE x/y/z = {row['rmse_x_um']:.3f} / {row['rmse_y_um']:.3f} / "
              f"{row['rmse_z_um']:.3f} um   (true RMS y {row['true_rms_y_um']:.2f})")

        if i < 3:                                   # figures for the first few
            plots.timeseries(rep / f"fig1_timeseries_{name}.png", t, prog.e, pred,
                             prog.valid, warm, title=f"{name}: e truth vs TCN")
            plots.residual_psd(rep / f"fig2_psd_{name}.png", prog.e, pred,
                               prog.valid, warm, fs, cfg.eval.nperseg,
                               title=f"{name}: residual spectrum")
            plots.reversal_alignment(rep / f"fig3_reversal_{name}.png", prog.cnc,
                                     prog.e, pred, prog.valid, warm, cfg.data.dt,
                                     title=f"{name}: residual at direction reversals")

    per_prog = pd.concat(err_tables, ignore_index=True)
    per_prog.to_csv(rep / "metric1_2_per_program.csv", index=False, lineterminator="\n")
    pooled = metrics.summarise(err_tables)
    pooled.to_csv(rep / "metric1_2_pooled.csv", index=False, lineterminator="\n")

    blk = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    if len(blk):
        blk.sort_values("max_err_um", ascending=False).head(200).to_csv(
            rep / "metric3_worst_blocks.csv", index=False, lineterminator="\n")

    spec = pd.concat(spec_tables, ignore_index=True)
    spec.to_csv(rep / "metric4_spectral.csv", index=False, lineterminator="\n")

    if form_tables:
        form = pd.concat(form_tables, ignore_index=True)
        form.to_csv(rep / "metric5_normal_deviation.csv", index=False,
                    lineterminator="\n")
    else:
        form = pd.DataFrame()

    print("\n==== metric 1/2: e RMSE and worst case, pooled ====================")
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    if len(blk):
        print("\n==== metric 3: worst blocks ======================================")
        print(blk.sort_values("max_err_um", ascending=False)
              .head(8)[["program", "block", "g00", "n", "max_err_um",
                        "max_true_x", "max_true_y", "max_true_z"]]
              .to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    print("\n==== metric 4: spectral agreement ================================")
    print(spec.groupby("axis")[["coh_mean", "coh_pw_mean", "psd_ratio",
                                "psd_logdev_db"]].mean()
          .to_string(float_format=lambda v: f"{v:8.3f}"))
    if len(form):
        print("\n==== metric 5: normal deviation (proxy; G01 only) ================")
        print(form[["program", "normal_dev_rms_true_um", "normal_dev_rms_pred_um",
                    "normal_dev_rms_residual_um", "normal_dev_max_true_um"]]
              .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
        print("\nPredicted TCP for the real form-error pipeline: "
              f"{rep / 'tcp'}")

    summary = {
        "run": args.run, "split": args.split, "checkpoint": args.checkpoint,
        "epoch": ck.get("epoch"), "programs": names,
        "receptive_field_ms": cfg.receptive_field_ms,
        "pooled": json.loads(pooled.to_json(orient="records")),
        "spectral_mean": json.loads(
            spec.groupby("axis")[["coh_mean", "coh_pw_mean", "psd_ratio"]]
            .mean().reset_index().to_json(orient="records")),
    }
    (rep / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nreport -> {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
