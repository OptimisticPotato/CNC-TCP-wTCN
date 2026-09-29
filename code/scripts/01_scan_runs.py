"""Scan every run folder once and write work/meta/index.csv.

The index feeds the DOE selector (SPEC section 3.3) and records the things that
must be checked before a run is trusted: dt uniformity, empty rows, G00 share,
block-length and corner-angle distributions, workspace position, error size.

    python scripts/01_scan_runs.py                 # all runs, 8 workers
    python scripts/01_scan_runs.py --limit 40      # quick look
    python scripts/01_scan_runs.py --workers 4
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import pandas as pd

import _bootstrap  # noqa: F401
from config import add_config_args, apply_override, CFG


def _worker(job: Tuple[str, List[str]]):
    name, overrides = job
    from config import CFG as cfg, apply_override as ov
    for spec in overrides:
        ov(cfg, spec)
    from tcn_cnc2tcp.runs_io import run_stats
    try:
        return run_stats(cfg, name)
    except Exception as exc:                       # noqa: BLE001
        return {"run": name, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-excluded", action="store_true",
                    help="also scan the runs listed in excluded_runs.txt")
    add_config_args(ap)
    args = ap.parse_args()
    for spec in args.overrides:
        apply_override(CFG, spec)
    cfg = CFG
    for d in (cfg.paths.work_dir, cfg.paths.meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    from tcn_cnc2tcp.runs_io import list_runs, family_of, feed_of

    names = list_runs(cfg, include_excluded=args.include_excluded)
    if args.limit:
        names = names[:args.limit]
    print(f"scanning {len(names)} runs from {cfg.paths.runs_dir}")

    t0 = time.time()
    rows = []
    jobs = [(n, list(args.overrides)) for n in names]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(_worker, jobs, chunksize=2), 1):
                rows.append(r)
                if i % 50 == 0:
                    print(f"  {i}/{len(names)}  ({time.time() - t0:.0f}s)", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            rows.append(_worker(j))
            if i % 10 == 0:
                print(f"  {i}/{len(names)}", flush=True)

    df = pd.DataFrame(rows)
    bad = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    df = df[df.get("error").isna()] if "error" in df else df
    df["family"] = df["run"].map(family_of)
    df["feed_name"] = df["run"].map(feed_of)

    # measure the rapid-traverse feed instead of trusting the configured 40000
    if len(df) and cfg.data.rapid_feed_auto:
        from tcn_cnc2tcp.runs_io import rapid_feed, save_machine
        used = rapid_feed(cfg)                      # what this pass classified with
        levels = sorted({float(v) for v in df["feed_max"].dropna()}
                        | {float(v) for v in df["feed_cut_max"].dropna()})
        info = save_machine(cfg, df["feed_max"].tolist(), levels)
        how = ("measured" if info["detected"] else
               "NOT separable, using config.data.rapid_feed_mm_min")
        print(f"\nrapid feed      : {info['rapid_feed_mm_min']:.0f} mm/min "
              f"({how}: {info['reason']})")
        print(f"feed levels     : {info['feed_levels']}")
        print(f"                  -> {cfg.paths.meta_dir / 'machine.json'}; "
              f"check the G00 share above is plausible (0.00 or 0.9+ means "
              f"this number is wrong -- override with "
              f"--set data.rapid_feed_mm_min=...)")
        if abs(info["rapid_feed_mm_min"] - used) > 1e-6:
            print(f"\n*** the scan classified G00 with {used:.0f} mm/min but the "
                  f"dataset says {info['rapid_feed_mm_min']:.0f}. Re-run this "
                  f"script so g00_frac / feed_cut_* are recomputed. ***")

    out = cfg.paths.meta_dir / "index.csv"
    df.to_csv(out, index=False, lineterminator="\n")
    print(f"\nwrote {out}  ({len(df)} runs, {time.time() - t0:.0f}s)")

    bad_out = cfg.paths.meta_dir / "index_errors.csv"
    if len(bad):
        bad.to_csv(bad_out, index=False, lineterminator="\n")
        print(f"{len(bad)} runs failed -> {bad_out}")
        for _, r in bad.head(10).iterrows():
            print("   ", r["run"], r["error"])
    elif bad_out.exists():
        bad_out.unlink()                            # do not leave a stale list

    if len(df):
        print("\nsummary")
        print(f"  samples      : {df['n'].sum():,} rows, "
              f"{df['duration_s'].sum() / 60:.1f} min of machining")
        print(f"  dt native    : {sorted(df['dt_native_ms'].round(4).unique())} ms")
        print(f"  resampled    : {int(df['resampled'].sum())} runs")
        print(f"  empty rows   : {int(df['n_invalid'].sum())}")
        print(f"  G00 share    : {df['g00_frac'].min():.2f} .. {df['g00_frac'].max():.2f}"
              f" (median {df['g00_frac'].median():.2f})")
        print(f"  cutting feed : {sorted(df['feed_cut_max'].round().unique().tolist())}")
        print(f"  e RMS [um]   : X {df['e_rms_x'].median():.2f} | "
              f"Y {df['e_rms_y'].median():.2f} | Z {df['e_rms_z'].median():.2f} (median)")
        print(f"  e |max| [um] : {df['e_absmax'].max():.1f}")
        print(f"  families     : {df['family'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
