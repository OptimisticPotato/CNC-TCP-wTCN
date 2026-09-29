"""Loader validation. SPEC section 7 step 1 -- do this before training anything.

Checks, per program:

  * time grid is uniform at the standard dt
  * no empty cnc/tcp row is ever used as a target
  * the G00/G01 channel identity  u[0:3] + u[3:6] == dCNC / scale
  * G00 share, and (when both are available) agreement between the exact feed
    flag and the speed-threshold fallback
  * e = TCP - CNC statistics and where its energy sits in frequency
  * that float32 would have destroyed the target, and float64 did not

    python scripts/03_validate_loader.py --selection r1
    python scripts/03_validate_loader.py --runs DOE_0001_... DOE_0042_...
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import signal

import _bootstrap  # noqa: F401
from config import add_config_args, configure


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", default=None, help="tag of a selection file")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=12,
                    help="only applies when neither --runs nor --selection is given")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = configure(args)

    from tcn_cnc2tcp import features
    from tcn_cnc2tcp.runs_io import list_runs, load_run

    if args.runs:
        names = args.runs
    elif args.selection:
        path = cfg.paths.meta_dir / f"selection_{args.selection}.txt"
        names = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
    else:
        names = list_runs(cfg)[:args.limit or None]
    print(f"validating {len(names)} programs\n")

    fs = 1.0 / cfg.data.dt
    rows = []
    for name in names:
        run = load_run(cfg, name)

        d = np.diff(run.t)
        dt_err = float(np.max(np.abs(d - cfg.data.dt)) / cfg.data.dt)

        g00 = features.g00_mask(run, cfg)
        u = features.build_inputs(run.cnc, g00, cfg.increment_scale)
        ident = features.check_identity(u, run.cnc, cfg.increment_scale)

        # cross-check the exact flag against the fallback heuristic
        agree = np.nan
        if cfg.data.g00_source == "feed":
            saved = cfg.data.g00_source
            cfg.data.g00_source = "speed"
            try:
                agree = float((features.g00_mask(run, cfg) == g00).mean())
            finally:
                cfg.data.g00_source = saved

        e = features.build_target(run.cnc, run.tcp)
        m = run.valid.copy()
        ev = e[m]

        # where does the energy of e live?
        f, p = signal.welch(ev[:, 1], fs, nperseg=min(4096, max(ev.shape[0], 32)))
        cum = np.cumsum(p) / max(p.sum(), 1e-300)
        f50 = float(f[np.searchsorted(cum, 0.50)])
        f99 = float(f[np.searchsorted(cum, 0.99)])
        above100 = float(p[f > 100].sum() / max(p.sum(), 1e-300))

        # what float32 would have done to the absolute coordinate
        c32 = run.cnc.astype(np.float32).astype(np.float64)
        f32_loss = float(np.abs(c32 - run.cnc).max() * 1e3)      # um

        rows.append(dict(
            run=name, n=len(run), dt_err=dt_err, resampled=int(run.resampled),
            invalid=int((~run.valid).sum()), g00_frac=float(g00.mean()),
            g00_agree=agree, ident_err=ident,
            e_rms_x=float(np.sqrt((ev[:, 0] ** 2).mean())),
            e_rms_y=float(np.sqrt((ev[:, 1] ** 2).mean())),
            e_rms_z=float(np.sqrt((ev[:, 2] ** 2).mean())),
            e_absmax=float(np.abs(ev).max()),
            u_absmax=float(np.abs(u).max()),
            f50_hz=f50, f99_hz=f99, frac_above_100hz=above100,
            float32_coord_loss_um=f32_loss,
        ))
        print(f"  {name[:46]:48s} n={len(run):6d} dt_err={dt_err:.1e} "
              f"g00={g00.mean():.2f} ident={ident:.1e} "
              f"e_rms=({rows[-1]['e_rms_x']:.2f},{rows[-1]['e_rms_y']:.2f},"
              f"{rows[-1]['e_rms_z']:.2f})um")

    df = pd.DataFrame(rows)
    out = cfg.paths.report_dir / "loader_validation.csv"
    df.to_csv(out, index=False, lineterminator="\n")

    print("\n---- aggregate ----------------------------------------------------")
    print(f"dt error (max)            : {df['dt_err'].max():.2e}  (must be << 1)")
    print(f"channel identity (max)    : {df['ident_err'].max():.2e}  "
          f"(float32 rounding of |u|<=1, must be ~1e-7)")
    print(f"empty rows total          : {int(df['invalid'].sum())} (never used as targets)")
    print(f"G00 share                 : {df['g00_frac'].min():.2f} .. {df['g00_frac'].max():.2f}")
    if df["g00_agree"].notna().any():
        print(f"exact vs speed heuristic  : {df['g00_agree'].min():.4f} .. "
              f"{df['g00_agree'].max():.4f} sample agreement")
    print(f"|u| max                   : {df['u_absmax'].max():.3f}  "
          f"(1.0 = one rapid step per sample)")
    print(f"e RMS [um]                : X {df['e_rms_x'].median():.2f} | "
          f"Y {df['e_rms_y'].median():.2f} | Z {df['e_rms_z'].median():.2f}  (median)")
    print(f"e |max| [um]              : {df['e_absmax'].max():.1f}")
    print(f"e energy: median f50      : {df['f50_hz'].median():.1f} Hz, "
          f"f99 {df['f99_hz'].median():.1f} Hz, "
          f"share above 100 Hz {df['frac_above_100hz'].median():.2e}")
    print(f"float32 coordinate loss   : up to {df['float32_coord_loss_um'].max():.4f} um "
          f"of quantisation on the absolute position alone, against an e RMS of "
          f"{df['e_rms_y'].median():.2f} um -- hence float64 coordinates (SPEC 4.1)")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
