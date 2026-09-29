"""Pick the programs to identify/train on. SPEC section 3.3.

The CNC->TCP relation is fixed, so what matters is persistent excitation, not
volume. Two criteria are available:

    --criterion excitation   (default) pick programs that pin down the kernel
    --criterion stratified   the earlier feature-space coverage heuristic

    python scripts/02_select_doe.py --set select.n_programs=32
    python scripts/02_select_doe.py --criterion excitation --target 'DOE_1'
    python scripts/02_select_doe.py --criterion stratified --set select.tag=r2

Both produce a single global ranking and truncate it, so selection(k) is always
a subset of selection(k') -- the saturation curve (script 07) compares nested
subsets instead of unrelated draws.

Why the default changed
-----------------------
The stratified criterion spreads the selection over feed levels, path families
and 14 geometric features (block length, corner angle, G00 share, workspace
position, error size). Measured on this dataset, that is not what decides the
result.

A 1021-tap 6-channel kernel has 6126 free coefficients per axis, and least
squares determines it only in the directions the inputs actually excite. The 22
programs the stratified criterion picked left 6126 - 44 directions essentially
free (lambda/lambda_max < 1e-3), because they are all synthetic excitation paths
-- random walks, sub-mm jitter, single-axis moves, circles. Held out on
machining paths (corner sequences, pocket clearing, arc transitions, helix
entries) the kernel gave 0.435 um; eight machining programs of *any* kind gave
0.0012 um on the same five test programs, a factor of 360. Two programs can look
identical in all 14 features and still constrain completely different
directions, so the features cannot see this.

The excitation criterion optimises the thing that was actually wrong. For a
candidate set S and a target set T it minimises

    tr( A_T (A_S + lambda I)^-1 ),     A = X^T X

which is the expected prediction variance on T -- textbook V-optimality. A
program scores well when it constrains directions T uses and S has not pinned
down yet, and a second program of the same kind scores near zero afterwards.

The criterion runs on a shorter kernel than the fit (``--design-taps``). It has
to rank candidates, not estimate coefficients, and the correlation structure
that separates them is visible well before 1021 taps.
"""
from __future__ import annotations

import argparse
import re
from typing import List, Sequence

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


# --------------------------------------------------------------------------- #
# criterion 1: feature-space coverage (the earlier default)
# --------------------------------------------------------------------------- #
def rank_stratified(x: np.ndarray, strata: dict, families: np.ndarray,
                    feeds: list, rng) -> List[int]:
    """One pass = one pick per feed stratum; inside a stratum, the least-used
    family's candidate farthest from everything picked so far."""
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


# --------------------------------------------------------------------------- #
# criterion 2: excitation coverage
# --------------------------------------------------------------------------- #
def _load_light(cfg, name: str):
    """A program with just the fields the correlations need, without torch."""
    from dataclasses import dataclass
    from tcn_cnc2tcp import runs_io as rio, features as F

    @dataclass
    class P:
        name: str
        u: np.ndarray
        valid: np.ndarray

    r = rio.load_run(cfg, name)
    g00 = F.g00_mask(r, cfg)
    return P(name, F.build_inputs(r.cnc, g00, cfg.increment_scale),
             np.asarray(r.valid, bool))


def correlations_for(cfg, names: Sequence[str], taps: int, label: str) -> dict:
    from tcn_cnc2tcp import fir

    out = {}
    for i, n in enumerate(names, 1):
        try:
            out[n] = fir.input_correlations(cfg, _load_light(cfg, n), taps)
        except Exception as exc:                       # noqa: BLE001
            print(f"  ! skipping {n}: {exc}")
        if i % 20 == 0 or i == len(names):
            print(f"  {label}: {i}/{len(names)} correlated", flush=True)
    return out


def variance_score(a_target: np.ndarray, a_sel: np.ndarray, lam: float) -> float:
    """tr(A_T (A_S + lambda I)^-1) -- expected prediction variance on T."""
    from scipy.linalg import cho_factor, cho_solve

    m = a_sel + lam * np.eye(a_sel.shape[0])
    try:
        c = cho_factor(m, lower=True, check_finite=False)
        return float(np.trace(cho_solve(c, a_target, check_finite=False)))
    except np.linalg.LinAlgError:
        return float("inf")


def rank_excitation(cfg, cand_names: Sequence[str], target_names: Sequence[str],
                    taps: int, depth: int, ridge_rel: float,
                    feed_of: dict = None, floor: int = 0, budget: int = 0) -> tuple:
    """Greedy V-optimal ranking, optionally with a per-feed-level floor.

    Left alone the criterion concentrates on the fastest programs, and it is
    right to: the map is feed-independent here, and a high-feed program excites
    strictly more bandwidth per second than the same path run slowly, so it
    reduces variance more per pick. Measured on this dataset an unconstrained
    selection of 22 took 16 at F15000 and nothing below F3000.

    That is optimal for the estimate and awkward for a report -- SPEC 3.3 asks
    for the feed range to be spread, and nothing verifies low-feed behaviour if
    no low-feed program is ever fitted. ``floor`` reserves that many picks for
    each feed level: once the remaining budget is only just enough to meet the
    unmet floors, candidates are restricted to the levels that still need one.
    """
    from tcn_cnc2tcp import fir

    print(f"design kernel: {taps} taps, {cfg.model.in_channels * taps} coefficients")
    r_cand = correlations_for(cfg, cand_names, taps, "candidates")
    r_targ = correlations_for(cfg, target_names, taps, "target")
    if not r_cand or not r_targ:
        raise SystemExit("no usable candidate or target programs")

    a_target = fir.gram_from_correlations(
        sum(r_targ.values()) / len(r_targ), taps)
    p = a_target.shape[0]
    lam = ridge_rel * np.trace(a_target) / p
    print(f"target built from {len(r_targ)} programs; ridge {lam:.4e}")

    names = list(r_cand)
    a_sel = np.zeros((p, p))
    chosen: List[str] = []
    scores: List[float] = []
    depth = min(depth, len(names))
    base = variance_score(a_target, a_sel, lam)
    print(f"\nstarting variance {base:.6e}")
    feed_of = feed_of or {}
    need = {}
    if floor and feed_of:
        levels = {feed_of[n] for n in names if n in feed_of}
        need = {f: floor for f in levels}
        if budget and floor * len(levels) > budget:
            print(f"  note: floor {floor} x {len(levels)} feed levels > budget "
                  f"{budget}; floors will be met in feed order until it runs out")
    for step in range(depth):
        unmet = [f for f, c in need.items() if c > 0]
        left_budget = (budget or depth) - step
        pool_names = names
        if unmet and left_budget <= len(unmet):
            # only just enough picks remain to satisfy the floors
            pool_names = [n for n in names if feed_of.get(n) in set(unmet)]
        best, best_s, best_a = None, np.inf, None
        for n in pool_names:
            if n in chosen:
                continue
            a_try = a_sel + fir.gram_from_correlations(r_cand[n], taps)
            s = variance_score(a_target, a_try, lam)
            if s < best_s:
                best, best_s, best_a = n, s, a_try
        if best is None:
            break
        if feed_of.get(best) in need and need[feed_of[best]] > 0:
            need[feed_of[best]] -= 1
        chosen.append(best)
        scores.append(best_s)
        a_sel = best_a
        print(f"  {step + 1:3d}. {best:<50} variance {best_s:.6e}"
              f"  ({100 * (1 - best_s / base):5.1f}% down)", flush=True)

    # everything not greedily ranked: order by what it would contribute alone
    rest = [n for n in names if n not in chosen]
    if rest:
        solo = {n: variance_score(a_target,
                                  fir.gram_from_correlations(r_cand[n], taps), lam)
                for n in rest}
        rest.sort(key=lambda n: solo[n])
    return chosen + rest, scores, a_sel, a_target, lam


def coverage_report(a_sel: np.ndarray, a_target: np.ndarray) -> None:
    """How much of the target's excitation the selection leaves undetermined."""
    lam, v = np.linalg.eigh(a_sel)
    lam = np.maximum(lam, 0.0)
    d = np.maximum(np.einsum("ij,ji->i", v.T, a_target @ v), 0.0)
    tot = d.sum()
    print("\n  target excitation landing in weakly constrained directions")
    print(f"    {'selection lambda/lambda_max below':>36}{'share':>10}")
    for t in (1e-12, 1e-9, 1e-6, 1e-3):
        s = lam <= t * lam.max()
        print(f"    {t:>36.0e}{100 * d[s].sum() / tot:>9.3f}%")
    for t in (1e-3, 1e-6):
        print(f"    directions with lambda/lambda_max > {t:.0e}: "
              f"{int((lam > t * lam.max()).sum())} / {len(lam)}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=None, help="default: work/meta/index.csv")
    ap.add_argument("--criterion", choices=("excitation", "stratified"),
                    default="excitation")
    ap.add_argument("--target", default="",
                    help="regex on run names for what the model must predict "
                         "well; default = the whole index")
    ap.add_argument("--target-sample", type=int, default=32,
                    help="programs averaged into the target Gram")
    ap.add_argument("--max-candidates", type=int, default=160,
                    help="candidate pool, pre-thinned by feed x family so the "
                         "greedy search stays affordable")
    ap.add_argument("--design-taps", type=int, default=96,
                    help="kernel length for the criterion only; the fit still "
                         "uses eval.fir_taps")
    ap.add_argument("--rank-depth", type=int, default=64,
                    help="how many picks are ranked greedily; beyond this the "
                         "order is by standalone score")
    ap.add_argument("--ridge-rel", type=float, default=1e-8)
    ap.add_argument("--feed-floor", type=int, default=1,
                    help="minimum programs per cutting-feed level; 0 lets the "
                         "criterion run free, which concentrates on high feed")
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

    feeds = sorted(df["feed_cut_max"].round().unique())
    print(f"{len(df)} candidates, {len(feeds)} feed levels, "
          f"{df['family'].nunique()} families -> selecting {k} "
          f"[{args.criterion}]")

    if args.criterion == "stratified":
        x = np.nan_to_num(df[FEATURES].to_numpy(float), nan=0.0)
        x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-9)
        strata = {f: df.index[df["feed_cut_max"].round() == f].tolist()
                  for f in feeds}
        order_idx = rank_stratified(x, strata, df["family"].to_numpy(), feeds, rng)
        ranking = df.loc[order_idx, "run"].tolist()
        note = "feed-stratified, family round-robin, farthest-point inside strata"
        a_sel = a_target = None
    else:
        # thin the pool: round-robin over (family, feed) so every kind still has
        # a representative, then let the criterion decide which ones matter
        df["_stratum"] = df["family"].astype(str) + "|" + df["feed_cut_max"].round().astype(int).astype(str)
        pool: List[str] = []
        groups = {s: list(rng.permutation(g["run"].tolist()))
                  for s, g in df.groupby("_stratum")}
        i = 0
        while len(pool) < min(args.max_candidates, len(df)):
            added = False
            for s in sorted(groups):
                if i < len(groups[s]) and len(pool) < args.max_candidates:
                    pool.append(groups[s][i])
                    added = True
            if not added:
                break
            i += 1
        targets = df["run"].tolist()
        if args.target:
            pat = re.compile(args.target)
            targets = [n for n in targets if pat.search(n)]
            if not targets:
                raise SystemExit(f"--target {args.target!r} matched no run")
        targets = list(rng.permutation(targets))[:args.target_sample]
        print(f"candidate pool {len(pool)} of {len(df)}; "
              f"target '{args.target or 'all'}' -> {len(targets)} programs")
        feed_of = dict(zip(df["run"], df["feed_cut_max"].round()))
        ranking, _, a_sel, a_target, _ = rank_excitation(
            cfg, pool, targets, args.design_taps, args.rank_depth, args.ridge_rel,
            feed_of=feed_of, floor=args.feed_floor, budget=k)
        ranking += [n for n in df["run"] if n not in set(ranking)]
        note = (f"V-optimal excitation coverage, {args.design_taps} design taps, "
                f"target '{args.target or 'all'}', feed floor {args.feed_floor}")

    pos = {n: i for i, n in enumerate(ranking)}
    df["rank"] = df["run"].map(lambda n: pos.get(n, len(ranking)) + 1)
    df.drop(columns=[c for c in ("_stratum",) if c in df], errors="ignore") \
      .sort_values("rank").to_csv(cfg.paths.meta_dir / "ranking.csv",
                                  index=False, lineterminator="\n")

    chosen = ranking[:k]
    sel = df[df["run"].isin(chosen)].copy()
    for f in feeds:
        got = int((sel["feed_cut_max"].round() == f).sum())
        print(f"  feed {f:>7.0f}: "
              f"{int((df['feed_cut_max'].round() == f).sum()):4d} available -> {got} picked")

    out = cfg.paths.meta_dir / f"selection_{cfg.select.tag}.txt"
    header = [
        f"# DOE selection '{cfg.select.tag}'  n={len(sel)}",
        f"# seed={cfg.select.selection_seed}  index={index}",
        f"# {note}",
    ]
    out.write_text("\n".join(header + sorted(sel["run"].tolist())) + "\n",
                   encoding="utf-8")
    sel.drop(columns=[c for c in ("_stratum",) if c in sel], errors="ignore") \
       .to_csv(cfg.paths.meta_dir / f"selection_{cfg.select.tag}.csv",
               index=False, lineterminator="\n")

    print(f"\nwrote {out}")
    if a_sel is not None:
        coverage_report(a_sel, a_target)
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
    print(f"  total machining : {sel['duration_s'].sum() / 60:.1f} min, "
          f"{sel['n'].sum():,} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
