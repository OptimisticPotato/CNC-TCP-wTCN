"""Inference pipeline. SPEC section 10, item 3.

    arbitrary timestamps  ->  cubic spline to the standard grid  ->  TCN
                          ->  e on the standard grid
                          ->  interpolate back to the requested timestamps
                          ->  TCP = CNC + e

Usage
-----
    python infer.py --run tcn_r1 --csv path/to/whatever_out.csv --out pred.csv
    python infer.py --run tcn_r1 --run-name DOE_0001_... --out pred.csv

The input CSV needs ``time`` and ``cnc_x/y/z``; ``feed`` is used for the
G00/G01 channels when present (without it the model is fed as if everything
were G01, which the spec's numbers say is worse -- a warning is printed).
``tcp_*`` is optional and only used to report the error if it is there.

As a library:

    from infer import Predictor
    p = Predictor("tcn_r1")
    tcp = p.predict(t, cnc, feed)          # any timestamps, any dt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import CFG, add_config_args, configure


class Predictor:
    """Trained TCN + the grid handling around it."""

    def __init__(self, run: str, checkpoint: str = "best.pt", cfg=None, device=None):
        from tcn_cnc2tcp.trainer import load_checkpoint
        self.cfg = cfg or CFG
        run_dir = Path(self.cfg.paths.runs_out_dir) / run
        self.model, self.ck, self.device = load_checkpoint(
            run_dir / checkpoint, self.cfg, device)
        self.run_dir = run_dir

    # ------------------------------------------------------------------ #
    def predict_error(self, t: np.ndarray, cnc: np.ndarray,
                      feed: Optional[np.ndarray] = None) -> np.ndarray:
        """e = TCP - CNC in micrometres, on the timestamps you passed in."""
        import torch

        from tcn_cnc2tcp import features, resample

        cfg = self.cfg
        t = np.asarray(t, dtype=np.float64)
        cnc = np.asarray(cnc, dtype=np.float64)
        if cnc.shape[0] != t.shape[0] or cnc.shape[1] != 3:
            raise ValueError("cnc must be (N,3) and match t")

        grid = resample.standard_grid(float(t[0]), float(t[-1]), cfg.data.dt)
        cnc_g = resample.interp_to(t, cnc, grid, cfg.augment.interp)

        if feed is None:
            g00_g = np.zeros(grid.size, dtype=bool)
        else:
            feed_g = resample.step_to(t, np.asarray(feed, float), grid)
            g00_g = feed_g >= cfg.data.rapid_feed_mm_min * cfg.data.rapid_feed_tol

        u = features.build_inputs(cnc_g, g00_g, cfg.increment_scale)
        x = torch.from_numpy(np.ascontiguousarray(u.T))[None].to(self.device)
        with torch.no_grad():
            e_g = self.model(x).float()[0].T.cpu().numpy()

        # back to the caller's timestamps
        return resample.interp_to(grid, e_g.astype(np.float64), t, cfg.augment.interp)

    def predict(self, t: np.ndarray, cnc: np.ndarray,
                feed: Optional[np.ndarray] = None) -> np.ndarray:
        """TCP [mm] on the timestamps you passed in."""
        e = self.predict_error(t, cnc, feed)
        return np.asarray(cnc, np.float64) + e / 1e3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="training run under work/runs")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--csv", default=None, help="any simulator-style csv")
    ap.add_argument("--run-name", default=None, help="a run folder under runs_dir")
    ap.add_argument("--out", default=None)
    add_config_args(ap)
    args = ap.parse_args()
    cfg = configure(args)

    from tcn_cnc2tcp.runs_io import read_csv, run_csv_path

    if args.run_name:
        path = run_csv_path(cfg, args.run_name)
    elif args.csv:
        path = Path(args.csv)
    else:
        raise SystemExit("pass --csv or --run-name")

    df = read_csv(path)
    t = df["time"].to_numpy(np.float64)
    cnc = df[["cnc_x", "cnc_y", "cnc_z"]].to_numpy(np.float64)
    feed = df["feed"].to_numpy(np.float64) if "feed" in df else None
    if feed is None:
        print("WARNING: no feed column -- every block is treated as G01. "
              "The spec measured 3.1 um vs 0.8 um for exactly this shortcut.")

    ok = np.isfinite(t) & np.isfinite(cnc).all(1)
    pred = Predictor(args.run, args.checkpoint, cfg)
    print(f"model {args.run} @ epoch {pred.ck.get('epoch')}, "
          f"{ok.sum()} samples, dt = {np.median(np.diff(t[ok])) * 1e3:.4f} ms")

    e = np.full_like(cnc, np.nan)
    e[ok] = pred.predict_error(t[ok], cnc[ok], feed[ok] if feed is not None else None)
    tcp = cnc + e / 1e3

    out = pd.DataFrame({
        "time": t,
        "cnc_x": cnc[:, 0], "cnc_y": cnc[:, 1], "cnc_z": cnc[:, 2],
        "e_x_um": e[:, 0], "e_y_um": e[:, 1], "e_z_um": e[:, 2],
        "tcp_x": tcp[:, 0], "tcp_y": tcp[:, 1], "tcp_z": tcp[:, 2],
    })
    if {"tcp_x", "tcp_y", "tcp_z"}.issubset(df.columns):
        truth = df[["tcp_x", "tcp_y", "tcp_z"]].to_numpy(np.float64)
        e_true = (truth - cnc) * 1e3
        m = ok & np.isfinite(e_true).all(1)
        m[:cfg.receptive_field] = False
        if m.any():
            r = e_true[m] - e[m]
            print("e RMSE [um]  x %.3f  y %.3f  z %.3f   (true RMS  x %.3f  y %.3f  z %.3f)"
                  % tuple(list(np.sqrt((r ** 2).mean(0)))
                          + list(np.sqrt((e_true[m] ** 2).mean(0)))))
        for i, a in enumerate("xyz"):
            out[f"e_true_{a}_um"] = e_true[:, i]

    dest = Path(args.out or (cfg.paths.report_dir / f"infer_{path.stem}.csv"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False, float_format="%.9g", lineterminator="\n")
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
