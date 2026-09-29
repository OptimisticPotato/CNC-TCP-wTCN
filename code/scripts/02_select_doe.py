"""Pick the programs to train on. SPEC section 3.3.

The CNC->TCP relation is fixed, so what matters is persistent excitation, not
volume: cover the axes of variation, then stop when the hold-out error stops
falling (that is metric 6 / script 07).

Selection is coverage-first, because a plain farthest-point search in feature
space just collects outliers:

    1. one stratum per cutting-feed level (the spec wants 5-7, log spaced)
    2. the budget is split evenly over the strata
    3. inside a stratum, pick round-robin over path families -- always the
       candidate farthest from everything picked so far (block length, corner
       angle, G00 share, speed, workspace position, error size)

    python scripts/02_select_doe.py --set select.n_programs=32
    python scripts/02_select_doe.py --set select.n_programs=64 --set select.tag=r2

Same seed -> round 2 is a superset of round 1, so the saturation curve compares
nested subsets instead of unrelated draws.
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from config import add_config_args, configure

FEATURES = [
    "speed_p95", "block_len_med_mm", "block_len_p10_mm",
    "corner_ang_med_deg", "corner_ang_p90_deg", "g00_frac",
    "pos_x", "pos_y", "pos_z", "span_mm", "duration_s",
    "e_rms_x", "e_rms_y", "e_rms_z",
]


def rank_all(x: np.ndarray, strata: dict, families: np.ndarray, feeds: list,
             rng) -> List[int]:
    """Rank every candidate, best first.

    One pass = one pick from each feed stratum, and inside a stratum the next
    pick is the least-used family's candidate that lies farthest from
    everything picked so far. Nothing here depends on how many programs are
    wanted, so truncating the ranking at any k gives a feed-stratified,
    family-balanced selection, and selection(k) is a subset of selection(k').
    """
    left = {f: list(pool) for f, pool in strata.items()}
    used = {fam: 0 for fam in set(families)}
    d = np.full(len(x), np.inf)
    order = sorted(feeds)
    ranking: List[int] = []
    while any(left.values()):
        moved = False
        for f in order:
            pool = left[f]
            if not pool:
                continue
            lo = min(used[families[i]] for i in pool)
            cand = [i for i in pool if used[families[i]] == lo]
            best = max(cand, key=lambda i: (d[i], rng.random()))
            ranking.append(best)
            pool.remove(best)
            used[families[best]] += 1
            d = np.minimum(d, np.linalg.norm(x - x[best], axis=1))
            moved = True
        if not moved:
            break
    return ranking


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=None, help="default: work/meta/index.csv")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = configure(args)

    index = args.index or (cfg.paths.meta_dir / "index.csv")
    df = pd.read_csv(index)
    if cfg.select.exclude_substrings:
        df = df[~df["run"].apply(
            lambda r: any(s in r for s in cfg.select.exclude_substrings))]
    df = df.reset_index(drop=True)
    k = min(int(cfg.select.n_programs), len(df))
    rng = np.random.default_rng(cfg.select.selection_seed)

    x = np.nan_to_num(df[FEATURES].to_numpy(float), nan=0.0)
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-9)
    families = df["family"].to_numpy()

    feeds = sorted(df["feed_cut_max"].round().unique())
    print(f"{len(df)} candidates, {len(feeds)} feed levels {feeds}, "
          f"{df['family'].nunique()} families -> selecting {k}")

    strata = {f: df.index[df["feed_cut_max"].round() == f].tolist() for f in feeds}
    if k < len(feeds) * cfg.select.min_per_feed_bucket:
        print(f"  note: n_programs={k} cannot give {cfg.select.min_per_feed_bucket} "
              f"programs to each of the {len(feeds)} feed levels")

    # A single global ranking, built by cycling over the feed strata, is what
    # makes every selection nested: selection(k) is always the first k entries.
    ranking = rank_all(x, strata, families, feeds, rng)
    df.loc[ranking, "rank"] = np.arange(1, len(ranking) + 1)
    df["rank"] = df["rank"].fillna(len(ranking) + 1)
    df.sort_values("rank").to_csv(cfg.paths.meta_dir / "ranking.csv", index=False,
                                  lineterminator="\n")

    chosen = ranking[:k]
    for f in feeds:
        got = sum(1 for i in chosen if round(df.loc[i, "feed_cut_max"]) == f)
        print(f"  feed {f:>7.0f}: {len(strata[f]):4d} available -> {got} picked")

    sel = df.loc[sorted(chosen)].copy()
    out = cfg.paths.meta_dir / f"selection_{cfg.select.tag}.txt"
    header = [
        f"# DOE selection '{cfg.select.tag}'  n={len(sel)}",
        f"# seed={cfg.select.selection_seed}  index={index}",
        "# feed-stratified, family round-robin, farthest-point inside strata",
    ]
    out.write_text("\n".join(header + sorted(sel["run"].tolist())) + "\n",
                   encoding="utf-8")
    sel.to_csv(cfg.paths.meta_dir / f"selection_{cfg.select.tag}.csv",
               index=False, lineterminator="\n")

    print(f"\nwrote {out}")
    print("\ncoverage (SPEC 3.3 asks for all of these to be spread)")
    print("  feed levels     :", sorted(sel["feed_cut_max"].round().unique().tolist()))
    print(f"  families        : {sel['family'].nunique()} distinct "
          f"({', '.join(sorted(sel['family'].unique())[:6])}...)")
    for col, unit in (("block_len_med_mm", "mm"), ("corner_ang_med_deg", "deg"),
                      ("g00_frac", ""), ("speed_p95", "mm/min"),
                      ("duration_s", "s"), ("e_rms_y", "um")):
        q = sel[col].quantile([0, .25, .5, .75, 1]).to_numpy()
        print(f"  {col:16s}: {q[0]:9.3f} {q[1]:9.3f} {q[2]:9.3f} {q[3]:9.3f} "
              f"{q[4]:9.3f}  [min q25 med q75 max] {unit}")
    print(f"  workspace spread: x {sel['pos_x'].min():.0f}..{sel['pos_x'].max():.0f} "
          f"y {sel['pos_y'].min():.0f}..{sel['pos_y'].max():.0f} "
          f"z {sel['pos_z'].min():.0f}..{sel['pos_z'].max():.0f} mm")
    print(f"  total machining : {sel['duration_s'].sum() / 60:.1f} min, "
          f"{sel['n'].sum():,} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
