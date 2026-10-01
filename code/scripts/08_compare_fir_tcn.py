"""Compare the trained TCN against FIR models identified from 1 and from N
programs, on the SAME held-out programs. Answers three questions with numbers:

    1. how well does the trained network predict a held-out operation
    2. how well does a filter identified on ONE operation predict ANOTHER
    3. does either reproduce the vibration (time domain + spectrum)

    python scripts/08_compare_fir_tcn.py --run tcn_r1
    python scripts/08_compare_fir_tcn.py --run tcn_r1 --taps 509   # match the RF

The network's predictions are read from the ``tcp/*.csv`` that 07_evaluate.py
exports, so this script needs no GPU and no torch -- run 07_evaluate.py first.
Those exports are rounded to 1 nm, which is a floor of ~0.001 um on any error
computed from them; the FIR is evaluated on the raw CSVs at full precision
instead, because a well-identified FIR lands at that same 0.001 um.

Everything is measured on the run's own ``split.json`` test programs, and the
FIR sees only ``split['train']`` -- the same programs the network trained on.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from config import add_config_args, configure

RAW_COLS = ["time", "tcode", "feed", "spindlespeed", "block", "cam_x", "cam_y",
            "cam_z", "cnc_x", "cnc_y", "cnc_z", "tcp_x", "tcp_y", "tcp_z",
            "force_x", "force_y", "force_z", "chipthickness", "_pad"]


@dataclass
class Prog:
    """The minimal view fir.fit needs, built without touching torch."""
    name: str
    u: np.ndarray
    e: np.ndarray
    g00: np.ndarray
    valid: np.ndarray


def load(cfg, name: str) -> Prog:
    from tcn_cnc2tcp import runs_io as rio, features as F
    r = rio.load_run(cfg, name)
    g00 = F.g00_mask(r, cfg)
    return Prog(name, F.build_inputs(r.cnc, g00, cfg.increment_scale),
                F.build_target(r.cnc, r.tcp), g00, np.asarray(r.valid, bool))


def load_tcn_export(path: Path) -> dict:
    """cnc / tcp_pred / tcp_true from 06_evaluate's export -> e in um."""
    d = pd.read_csv(path)
    cnc = d[["cnc_x", "cnc_y", "cnc_z"]].to_numpy(np.float64)
    return {
        "t": d["time"].to_numpy(np.float64),
        "e_pred": (d[["tcp_pred_x", "tcp_pred_y", "tcp_pred_z"]].to_numpy(np.float64) - cnc) * 1e3,
        "e_true": (d[["tcp_true_x", "tcp_true_y", "tcp_true_z"]].to_numpy(np.float64) - cnc) * 1e3,
    }


def rmse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-axis RMSE over the masked samples."""
    d = (pred - true)[mask]
    return np.sqrt((d ** 2).mean(axis=0))


def band_power(x: np.ndarray, dt: float, lo: float, hi: float) -> float:
    from scipy.signal import welch
    f, p = welch(x, fs=1.0 / dt, nperseg=min(2048, len(x)))
    s = (f >= lo) & (f < hi)
    return float(p[s].sum())


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def vibration_window(e: np.ndarray, g00: np.ndarray, dt: float,
                     span_ms: float = 300.0) -> slice:
    """Locate the most strongly oscillating G01 stretch, for the zoom panel.

    High-passes each axis by subtracting a 20 ms moving average, then takes the
    window whose summed high-frequency energy is largest. Picking the window by
    |e| instead would just find the largest following error, which is usually a
    smooth ramp and shows no vibration at all.
    """
    w = max(4, int(round(0.020 / dt)))
    k = np.ones(w) / w
    hp = np.zeros_like(e)
    for a in range(e.shape[1]):
        hp[:, a] = e[:, a] - np.convolve(e[:, a], k, mode="same")
    energy = (hp ** 2).sum(axis=1) * (~g00)
    span = int(round(span_ms * 1e-3 / dt))
    cum = np.concatenate([[0.0], np.cumsum(energy)])
    if len(cum) <= span:
        return slice(0, len(e))
    tot = cum[span:] - cum[:-span]
    i = int(np.argmax(tot))
    return slice(i, i + span)


def figure(name: str, t: np.ndarray, e_true: np.ndarray, series: dict,
           g00: np.ndarray, dt: float, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    sl = vibration_window(e_true, g00, dt)
    fig, ax = plt.subplots(3, 4, figsize=(21, 9.5))
    colors = {"TCN": "tab:red", "FIR (1 prog)": "tab:orange", "FIR (N progs)": "tab:blue"}

    for a, axis in enumerate("XYZ"):
        # --- full trace -------------------------------------------------- #
        p = ax[a, 0]
        p.plot(t, e_true[:, a], color="k", lw=0.7, label="truth")
        for k, v in series.items():
            p.plot(t, v[:, a], color=colors.get(k, "tab:green"), lw=0.6, alpha=0.8, label=k)
        p.axvspan(t[sl.start], t[min(sl.stop, len(t) - 1)], color="gold", alpha=0.35, zorder=0)
        p.set_ylabel(f"e_{axis.lower()}  [um]")
        p.grid(alpha=0.3)
        if a == 0:
            p.set_title("full program  (shaded = zoom)")
            p.legend(fontsize=7, ncol=2, loc="upper right")
        if a == 2:
            p.set_xlabel("time [s]")

        # --- zoom: vibration -------------------------------------------- #
        p = ax[a, 1]
        p.plot(t[sl], e_true[sl, a], color="k", lw=1.4, label="truth")
        for k, v in series.items():
            p.plot(t[sl], v[sl, a], color=colors.get(k, "tab:green"), lw=1.1,
                   alpha=0.9, ls="--", label=k)
        p.grid(alpha=0.3)
        if a == 0:
            p.set_title("strongest G01 oscillation  (all curves overlap)")
        if a == 2:
            p.set_xlabel("time [s]")

        # --- residual: the only panel where the models separate ---------- #
        # At +-600 um of following error a 0.5 um difference is invisible, so
        # the panel above proves only that everything tracks the vibration.
        # This one is what the comparison actually rests on.
        p = ax[a, 2]
        for k, v in series.items():
            p.plot(t[sl], (v[sl, a] - e_true[sl, a]),
                   color=colors.get(k, "tab:green"), lw=1.0, alpha=0.9, label=k)
        p.axhline(0, color="k", lw=0.8)
        p.set_ylabel("pred - truth  [um]")
        p.grid(alpha=0.3)
        if a == 0:
            p.set_title("residual over the same stretch")
            p.legend(fontsize=7, loc="upper right")
        if a == 2:
            p.set_xlabel("time [s]")

        # --- spectrum: truth vs each model's residual -------------------- #
        p = ax[a, 3]
        m = ~g00
        f, pt = welch(e_true[m, a], fs=1.0 / dt, nperseg=2048)
        p.semilogy(f, pt, color="k", lw=1.4, label="truth")
        for k, v in series.items():
            _, pr = welch((v - e_true)[m, a], fs=1.0 / dt, nperseg=2048)
            p.semilogy(f, pr, color=colors.get(k, "tab:green"), lw=0.9,
                       alpha=0.85, label=f"{k} resid")
        p.set_xlim(0, 150)
        p.set_ylabel("PSD [um^2/Hz]")
        p.grid(alpha=0.3, which="both")
        if a == 0:
            p.set_title("truth vs residual spectra (gap = accuracy)")
            p.legend(fontsize=7, loc="upper right")
        if a == 2:
            p.set_xlabel("frequency [Hz]")

    fig.suptitle(f"{name}   -   e = TCP - CNC", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--taps", type=int, default=0, help="0 = eval.fir_taps")
    ap.add_argument("--single", default="",
                    help="program for the 1-program FIR; default = first train program")
    ap.add_argument("--figures", type=int, default=3, help="how many test programs to plot")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = configure(args)

    from tcn_cnc2tcp import fir

    run_dir = cfg.paths.runs_out_dir / args.run
    split = json.loads((run_dir / "split.json").read_text())
    taps = args.taps or cfg.eval.fir_taps
    dt = cfg.data.dt
    export_dir = cfg.paths.report_dir / args.run / "test" / "tcp"
    out_dir = cfg.paths.report_dir / args.run / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"run {args.run}:  train {len(split['train'])}  val {len(split['val'])}"
          f"  test {len(split['test'])} programs")
    print(f"FIR taps {taps} ({taps * dt * 1e3:.0f} ms), "
          f"{cfg.model.in_channels * taps:,} unknowns per axis\n")

    single = args.single or split["train"][0]
    print(f"loading train programs ({len(split['train'])}) ...", flush=True)
    train = [load(cfg, n) for n in split["train"]]
    val = [load(cfg, n) for n in split["val"]]

    print(f"\nFIR A: identified on ONE program  ({single})")
    fir_1 = fir.fit(cfg, [p for p in train if p.name == single], val_programs=val,
                    taps=taps, log=lambda *a: None)
    print(f"   ridge {fir_1['ridge_rel']:.1e}   |DC|max "
          f"{np.abs(fir_1['h'].sum(axis=2)).max():.4f}")
    print(f"FIR B: identified on all {len(train)} train programs")
    fir_n = fir.fit(cfg, train, val_programs=val, taps=taps, log=lambda *a: None)
    print(f"   ridge {fir_n['ridge_rel']:.1e}   |DC|max "
          f"{np.abs(fir_n['h'].sum(axis=2)).max():.4f}\n")

    rows: List[dict] = []
    figs: List[Path] = []
    for i, name in enumerate(split["test"]):
        p = load(cfg, name)
        m01 = p.valid & (~p.g00)
        m00 = p.valid & p.g00
        pred = {
            "FIR (1 prog)": fir.predict(fir_1["h"], p.u),
            "FIR (N progs)": fir.predict(fir_n["h"], p.u),
        }

        exp = export_dir / f"{name}.csv"
        e_true_tcn = None
        if exp.exists():
            d = load_tcn_export(exp)
            n = min(len(d["e_pred"]), len(p.e))
            tcn = np.full_like(p.e, np.nan, dtype=np.float64)
            tcn[:n] = d["e_pred"][:n]
            pred["TCN"] = tcn
            e_true_tcn = d["e_true"][:n]
        else:
            print(f"  ! no TCN export for {name} (run 07_evaluate.py first)")

        for model, v in pred.items():
            ok01 = m01 & np.isfinite(v).all(axis=1)
            ok00 = m00 & np.isfinite(v).all(axis=1)
            # the TCN's e_true comes from its own export (1 nm rounding); the
            # FIR is scored against the raw target, which is exact
            truth = e_true_tcn if (model == "TCN" and e_true_tcn is not None) else p.e
            truth = np.asarray(truth, np.float64)[: len(v)]
            r01 = rmse(v, truth, ok01)
            r00 = rmse(v, truth, ok00) if ok00.any() else np.full(3, np.nan)
            rows.append(dict(
                program=name, model=model,
                g01_x=r01[0], g01_y=r01[1], g01_z=r01[2], g01_mean=r01.mean(),
                g00_mean=np.nanmean(r00),
                psd_ratio_10_50=np.mean([
                    band_power(v[ok01, a], dt, 10, 50)
                    / max(band_power(truth[ok01, a], dt, 10, 50), 1e-30)
                    for a in range(3)]),
            ))

        if i < args.figures:
            t = np.arange(len(p.e)) * dt
            figs.append(figure(name, t, p.e.astype(np.float64), pred, p.g00, dt,
                               out_dir / f"fig_vibration_{name}.png"))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison_per_program.csv", index=False)

    piv = df.pivot_table(index="model", values=["g01_mean", "g00_mean", "psd_ratio_10_50"],
                         aggfunc="median").reindex(
        [m for m in ("TCN", "FIR (1 prog)", "FIR (N progs)") if m in set(df.model)])
    print("=== median over the held-out test programs ===")
    print(f"{'model':<16}{'G01 RMSE [um]':>15}{'G00 RMSE [um]':>15}{'PSD ratio 10-50Hz':>20}")
    for model, r in piv.iterrows():
        print(f"{model:<16}{r['g01_mean']:>15.4f}{r['g00_mean']:>15.4f}"
              f"{r['psd_ratio_10_50']:>20.3f}")

    print("\n=== per program, G01 RMSE [um] ===")
    wide = df.pivot_table(index="program", columns="model", values="g01_mean")
    print(wide.to_string(float_format=lambda v: f"{v:10.4f}"))

    # ----------------------------------------------------------------- #
    # coverage: is the error about HOW MANY programs, or WHICH ones?
    # ----------------------------------------------------------------- #
    from tcn_cnc2tcp import runs_io as rio
    te_fam = {rio.family_of(n) for n in split["test"]}
    same = [n for n in rio.list_runs(cfg)
            if rio.family_of(n) in te_fam and n not in split["test"]]
    same = list(np.random.default_rng(4).permutation(same))
    if same:
        print(f"\n=== coverage: FIR identified on programs OF THE TEST FAMILIES "
              f"({len(same)} available) ===")
        print(f"{'FIR identified on':<46}{'n':>3}{'test G01 med':>14}{'max':>10}")
        cov = [(f"{len(split['train'])} train programs (other families)", split["train"])]
        cov += [(f"{k} program(s) from the test families", same[:k])
                for k in (1, 3, 8) if len(same) >= k]
        cov_rows = []
        for tag, names in cov:
            k = fir.fit(cfg, [load(cfg, n) for n in names], taps=taps,
                        ridge_grid=(0.0,), log=lambda *a: None)
            v = []
            for n in split["test"]:
                p = load(cfg, n)
                m = p.valid & (~p.g00)
                d = fir.predict(k["h"], p.u)[m] - p.e[m]
                v.append(float(np.sqrt((d ** 2).mean())))
            v = np.array(v)
            print(f"{tag:<46}{len(names):>3}{np.median(v):>14.4f}{v.max():>10.4f}")
            cov_rows.append(dict(trained_on=tag, n_programs=len(names),
                                 test_g01_median=np.median(v), test_g01_max=v.max()))
        pd.DataFrame(cov_rows).to_csv(out_dir / "coverage.csv", index=False)

    print(f"\nwrote {out_dir / 'comparison_per_program.csv'}")
    for f in figs:
        print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
