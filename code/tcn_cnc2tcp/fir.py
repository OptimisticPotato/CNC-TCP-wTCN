"""The 6-input 3-output FIR model, exactly as the project's own slides define it.

    e_a[n] = sum_{b=1..6} sum_{k=0..199} g_{b,a}[k] u_b[n-k],   a in {X, Y, Z}
    u_{1..3}[n] = dCNC_{x,y,z}[n] * 1_{G01}
    u_{4..6}[n] = dCNC_{x,y,z}[n] * 1_{G00}
    TCP[n] = CNC[n] + e[n]

Three axes x two motion modes = 6 input channels, 200 taps -> 1200 unknowns per
output axis, 3600 in total. Ordinary least squares: build the normal equations
from the command history and solve them with Cholesky. **There is no training
loop and no tuning** -- one over-determined linear system, one factorisation.

Deliberately NOT done here, because the slides do not do it:

* no G00 down-weighting. That belongs to the network's loss (spec 5.4); the
  identification fits every observed error equally. Pass ``use_weights=True``
  to reproduce the network's objective instead, for an apples-to-apples
  comparison of the two function classes under one criterion.
* no ridge by default. ``ridge_grid`` starts at 0.0 = plain least squares; the
  sweep is kept only so the report can state that regularisation was not needed
  rather than assume it.
* no sum(g)=0 constraint. The slides describe that as an observed property of
  the solution (constant velocity -> constant u -> almost no error -> the taps
  must sum to zero), so it is measured and reported, not imposed. Turn it on
  with ``dc_penalty`` if a fit comes back with a large cancelling DC gain.

Still inherited from the spec, because they are properties of the data and not
of the method: the first receptive-field samples of a sequence and rows with
empty cnc/tcp are never used as targets.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.signal import fftconvolve

from . import features


def design_view(u: np.ndarray, taps: int) -> np.ndarray:
    """(N,C) -> a *view* of shape (N, C, taps) with ``[n, c, k] = u[n-k, c]``."""
    n, c = u.shape
    z = np.concatenate([np.zeros((taps - 1, c), u.dtype), u], axis=0)
    win = np.lib.stride_tricks.sliding_window_view(z, taps, axis=0)   # (N, C, taps)
    return win[:, :, ::-1]


#: ridge values tried, as a fraction of trace(A)/p. 0.0 = the slides' plain
#: least squares, and it is first so that a tie is resolved in its favour.
RIDGE_GRID = (0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)


def normal_equations_fft(cfg, programs: Sequence, taps: int, log=print):
    """The same normal equations, from cross-correlations instead of a gemm.

    The design matrix is a convolution matrix, so its Gram matrix is block
    Toeplitz::

        (X^T X)[(b,k),(c,l)] = sum_n u_b[n-k] u_c[n-l] = R_bc[k-l]
        (X^T y)[(b,k),a]     = sum_n u_b[n-k] e_a[n]   = R_ue[b,a][k]

    which means the whole p x p matrix is determined by 36 cross-correlations
    of the input channels and 18 between input and output. By FFT that is
    O(N log N) instead of the O(N p^2) of accumulating X^T X directly -- at
    1021 taps and 253k samples, 9.5e12 flops become ~2e7, and what is left is
    the Cholesky.

    X^T y is exactly the correlation. X^T X is *not* quite Toeplitz: the
    correlation runs the input off both ends, while the design matrix only
    zero-pads the front. The difference is exactly the Gram of the L-1 rows
    that would follow the last sample, so::

        X^T X = toeplitz(R) - T^T T ,   T = the L-1 run-out rows

    and that correction is a thin rank-(L-1) update, not another O(N p^2) pass.
    With it the result is exact, not an approximation.

    Two conditions make the front padding legitimate:

    * the zero-padded warm-up rows are *correct*, not filler: every program
      starts from rest with TCP = CNC, so zero history is the real initial
      condition.
    * rows whose targets are empty must be dropped, which would break the
      structure -- so each program is split into maximal runs of valid rows and
      each run is treated as its own sequence. In this dataset that is one
      trailing row per program, so the split is a no-op.

    Weighted fits (``use_weights=True``) have no such structure and go through
    ``normal_equations`` instead.
    """
    from scipy.linalg import toeplitz
    from scipy.signal import correlate

    n_in = cfg.model.in_channels
    p = n_in * taps
    r_uu = np.zeros((n_in, n_in, 2 * taps - 1))     # R_bc[tau], tau = -(L-1)..L-1
    r_ue = np.zeros((n_in, taps, 3))                # R_ue[b][k][a], k = 0..L-1
    tail = np.zeros((p, p))                         # sum of T^T T over segments
    n_used = 0

    for i, prog in enumerate(programs, 1):
        u_all = prog.u.astype(np.float64)
        e_all = prog.e.astype(np.float64)
        ok = np.asarray(prog.valid, bool)
        edges = np.flatnonzero(np.diff(ok.astype(np.int8)))
        bounds = np.concatenate([[0], edges + 1, [ok.size]])
        used = 0
        for s, t in zip(bounds[:-1], bounds[1:]):
            if not ok[s] or t - s < taps + 1:
                continue
            u = u_all[s:t]
            e = e_all[s:t]
            used += t - s
            lag0 = len(u) - 1
            for b in range(n_in):
                for c in range(b, n_in):
                    full = correlate(u[:, c], u[:, b], mode="full", method="fft")
                    seg = full[lag0 - (taps - 1): lag0 + taps]
                    r_uu[b, c] += seg
                    if c != b:
                        r_uu[c, b] += seg[::-1]     # R_cb[tau] = R_bc[-tau]
                for a in range(3):
                    full = correlate(e[:, a], u[:, b], mode="full", method="fft")
                    r_ue[b, :, a] += full[lag0: lag0 + taps]
            if taps > 1:                            # the run-out rows
                pad = np.concatenate([u[-(taps - 1):], np.zeros((taps - 1, n_in))])
                t_rows = design_view(pad, taps)[taps - 1:].reshape(taps - 1, p)
                tail += t_rows.T @ t_rows
        n_used += used
        log(f"  [{i}/{len(programs)}] correlated {prog.name} ({used:,} samples)")

    a_mat = np.empty((p, p), dtype=np.float64)
    for b in range(n_in):
        for c in range(n_in):
            r = r_uu[b, c]
            # block[k,l] = R_bc[k-l]: first column tau = 0..L-1, first row tau = 0..-(L-1)
            a_mat[b * taps:(b + 1) * taps, c * taps:(c + 1) * taps] = toeplitz(
                r[taps - 1:], r[taps - 1::-1])
    a_mat -= tail
    # int(): the segment bounds are numpy integers, so n_used comes out int64
    # and would otherwise reach json.dumps in save() as a non-serialisable type
    return a_mat, r_ue.reshape(p, 3), int(n_used)


# --------------------------------------------------------------------------- #
# excitation summaries, for choosing WHICH programs to identify on (script 02)
# --------------------------------------------------------------------------- #
def input_correlations(cfg, prog, taps: int) -> np.ndarray:
    """R_bc[tau] for one program, tau = -(taps-1)..taps-1, inputs only.

    This is the left half of ``normal_equations_fft`` without the target: it is
    everything a program contributes to X^T X, in 6*6*(2L-1) numbers instead of
    a (6L)^2 matrix. That compression is what makes it affordable to score
    every candidate program in the dataset and to re-score them against a
    growing selection.

    The run-out correction is deliberately dropped here. It is a rank-(L-1)
    change out of ~20,000 rows per program and cannot reorder candidates; the
    fit itself still uses the exact path.
    """
    from scipy.signal import correlate

    n_in = cfg.model.in_channels
    r_uu = np.zeros((n_in, n_in, 2 * taps - 1))
    u_all = prog.u.astype(np.float64)
    ok = np.asarray(prog.valid, bool)
    edges = np.flatnonzero(np.diff(ok.astype(np.int8)))
    bounds = np.concatenate([[0], edges + 1, [ok.size]])
    for s, t in zip(bounds[:-1], bounds[1:]):
        if not ok[s] or t - s < taps + 1:
            continue
        u = u_all[s:t]
        lag0 = len(u) - 1
        for b in range(n_in):
            for c in range(b, n_in):
                full = correlate(u[:, c], u[:, b], mode="full", method="fft")
                seg = full[lag0 - (taps - 1): lag0 + taps]
                r_uu[b, c] += seg
                if c != b:
                    r_uu[c, b] += seg[::-1]
    return r_uu


def gram_from_correlations(r_uu: np.ndarray, taps: int) -> np.ndarray:
    """(n_in, n_in, 2L-1) correlations -> the (n_in*L)^2 block-Toeplitz Gram."""
    from scipy.linalg import toeplitz

    n_in = r_uu.shape[0]
    p = n_in * taps
    a = np.empty((p, p), dtype=np.float64)
    for b in range(n_in):
        for c in range(n_in):
            r = r_uu[b, c]
            a[b * taps:(b + 1) * taps, c * taps:(c + 1) * taps] = toeplitz(
                r[taps - 1:], r[taps - 1::-1])
    return a


def normal_equations(cfg, programs: Sequence, taps: int, chunk: int = 4096,
                     use_weights: bool = False, log=print):
    """Accumulate the weighted normal equations once, for every ridge value."""
    n_in = cfg.model.in_channels
    p = n_in * taps
    a_mat = np.zeros((p, p), dtype=np.float64)
    b_mat = np.zeros((p, 3), dtype=np.float64)
    n_used = 0
    for i, prog in enumerate(programs, 1):
        w = (features.loss_weight(prog.valid, prog.g00, cfg.receptive_field,
                                  cfg.loss.g00_weight) if use_weights
             else prog.valid.astype(np.float64))
        sw = np.sqrt(w).astype(np.float64)
        view = design_view(prog.u.astype(np.float64), taps)
        e = prog.e.astype(np.float64)
        for start in range(0, len(prog), chunk):
            sl = slice(start, min(len(prog), start + chunk))
            s = sw[sl]
            if not s.any():
                continue
            x = view[sl].reshape(sl.stop - sl.start, p) * s[:, None]
            a_mat += x.T @ x
            b_mat += x.T @ (e[sl] * s[:, None])
        n_used += int((w > 0).sum())
        log(f"  [{i}/{len(programs)}] accumulated {prog.name} "
            f"({int((w > 0).sum()):,} samples)")
    return a_mat, b_mat, n_used


def solve_ridge(a_mat, b_mat, ridge_rel: float, taps: int, n_in: int,
                dc_penalty: float = 0.0):
    """Normal equations solved by Cholesky, as the slides specify.

    ``ridge_rel = 0`` is the plain least-squares solution. ``dc_penalty`` is
    off by default: the slides treat sum(g) = 0 as a property the solution
    exhibits, not a constraint to impose, so it is measured and reported. It is
    available because the property can fail where the data does not constrain
    it -- if every training rapid runs diagonally, the G00 x and y channels are
    collinear and the fit may park +10 on one and -10 on the other, which
    cancels on diagonal moves and injects an offset on an x-only rapid.
    """
    p = a_mat.shape[0]
    scale = np.trace(a_mat) / p
    m = a_mat + (ridge_rel * scale) * np.eye(p) if ridge_rel else a_mat.copy()
    if dc_penalty > 0:
        for b in range(n_in):
            s = slice(b * taps, (b + 1) * taps)
            m[s, s] += dc_penalty * scale          # = mu * c_b c_b^T
    try:                                           # the slides' Cholesky
        h_flat = cho_solve(cho_factor(m, lower=True, check_finite=False), b_mat,
                           check_finite=False)
    except LinAlgError:                            # not positive definite
        h_flat = np.linalg.lstsq(m, b_mat, rcond=None)[0]
    return h_flat.reshape(n_in, taps, 3).transpose(2, 0, 1), float(ridge_rel * scale)


def _rmse_g01(cfg, h, programs) -> float:
    se = np.zeros(3)
    n = 0.0
    for prog in programs:
        pred = predict(h, prog.u)
        m = prog.valid.copy()
        m[:cfg.receptive_field] = False
        m &= ~prog.g00
        if not m.any():
            continue
        r = (prog.e[m] - pred[m]).astype(np.float64)
        se += (r ** 2).sum(0)
        n += m.sum()
    return float(np.sqrt(se.sum() / (3 * n))) if n else float("inf")


def fit(cfg, programs: Sequence, val_programs: Optional[Sequence] = None,
        taps: Optional[int] = None, ridge_grid: Sequence[float] = RIDGE_GRID,
        chunk: int = 4096, use_weights: bool = False, dc_penalty: float = 0.0,
        method: str = "auto", log=print) -> Dict[str, object]:
    """Least squares for h, shape (3, 6, taps). One pass, one factorisation.

    ``method`` picks how X^T X is built: "fft" uses the block-Toeplitz
    structure (see ``normal_equations_fft``), "direct" accumulates the gemm,
    "auto" takes the fast path whenever the fit is unweighted. The two agree to
    machine precision on X^T X, and their predictions to ~0.2 nm; at 1021 taps
    the build goes from 53 s to 3.4 s.

    ``ridge_grid`` defaults to 0.0 first, i.e. the slides' plain normal
    equations. The remaining values are tried only so the report can say
    regularisation was measured and not needed; since the normal equations are
    accumulated once, each extra value costs one Cholesky. If a ridge is
    selected over 0.0, that is a finding about the data, not a design choice.
    """
    taps = int(taps or cfg.eval.fir_taps)
    n_in = cfg.model.in_channels
    use_fft = method == "fft" or (method == "auto" and not use_weights)
    log(f"building normal equations: {n_in * taps:,} coefficients per axis"
        f" ({'correlation / FFT' if use_fft else 'direct accumulation'})")
    if use_fft:
        a_mat, b_mat, n_used = normal_equations_fft(cfg, programs, taps, log)
    else:
        a_mat, b_mat, n_used = normal_equations(cfg, programs, taps, chunk,
                                                use_weights, log)

    sweep = []
    best = None
    for r in ridge_grid:
        h, lam = solve_ridge(a_mat, b_mat, r, taps, n_in, dc_penalty)
        row = {
            "ridge_rel": float(r), "lam": lam,
            "train_rmse_g01_um": _rmse_g01(cfg, h, programs),
            "dc_gain_absmax": float(np.abs(h.sum(axis=2)).max()),
            "coef_absmax": float(np.abs(h).max()),
        }
        if val_programs:
            row["val_rmse_g01_um"] = _rmse_g01(cfg, h, val_programs)
        score = row.get("val_rmse_g01_um", row["train_rmse_g01_um"])
        row["score"] = score
        sweep.append(row)
        log(f"  ridge {r:8.1e}: train {row['train_rmse_g01_um']:8.4f} um"
            + (f"   val {row['val_rmse_g01_um']:8.4f} um" if val_programs else "")
            + f"   |DC|max {row['dc_gain_absmax']:8.4f}")
        if best is None or score < best[0]:
            best = (score, r, h, lam)

    _, ridge_rel, h, lam = best
    return {
        "h": h, "taps": taps, "ridge_rel": float(ridge_rel), "lam": lam,
        "dc_penalty": float(dc_penalty),
        "selected_on": "val" if val_programs else "train",
        "sweep": sweep,
        "n_samples": n_used, "n_programs": len(programs),
        "programs": [p.name for p in programs],
        "val_programs": [p.name for p in (val_programs or [])],
        "weighted": bool(use_weights),
        "dc_gain": h.sum(axis=2).tolist(),
    }


def predict(h: np.ndarray, u: np.ndarray) -> np.ndarray:
    """(N,6) increments -> (N,3) predicted e [um]. Causal, zero history."""
    n = u.shape[0]
    out = np.zeros((n, h.shape[0]), dtype=np.float64)
    for a in range(h.shape[0]):
        for b in range(h.shape[1]):
            if not np.any(h[a, b]):
                continue
            out[:, a] += fftconvolve(u[:, b].astype(np.float64), h[a, b],
                                     mode="full")[:n]
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
def default_path(cfg, run: str) -> Path:
    return Path(cfg.paths.runs_out_dir) / run / "fir_baseline.npz"


def _jsonable(o):
    """numpy scalars/arrays -> plain python, for the metadata blob.

    Counts and indices picked up from numpy operations arrive here as int64 /
    float32, which json.dumps rejects. Converting on the way out keeps the
    metadata readable by any reader and costs nothing.
    """
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def save(path: Path, info: Dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in info.items() if k != "h"}
    np.savez_compressed(path, h=info["h"],
                        meta=json.dumps(meta, default=_jsonable))
    return path


def load(path: Path) -> Dict[str, object]:
    with np.load(Path(path), allow_pickle=False) as z:
        info = json.loads(str(z["meta"]))
        info["h"] = z["h"]
    return info


def impulse_response_summary(h: np.ndarray, dt: float) -> List[dict]:
    """Per (output, input) pair: DC gain, centre of mass, energy spread.

    Useful for the report -- this is the part of the model you can actually
    read, and the TCN has no equivalent.
    """
    rows = []
    axes = "xyz"
    chan = ["dx_G01", "dy_G01", "dz_G01", "dx_G00", "dy_G00", "dz_G00"]
    k = np.arange(h.shape[2])
    for a in range(h.shape[0]):
        for b in range(h.shape[1]):
            g = h[a, b]
            energy = float((g ** 2).sum())
            if energy <= 0:
                continue
            centre = float((k * g ** 2).sum() / energy)
            rows.append({
                "out": axes[a], "in": chan[b],
                "dc_gain": float(g.sum()),
                "peak_abs": float(np.abs(g).max()),
                "energy": energy,
                "centre_ms": centre * dt * 1e3,
                "tail_frac_after_100ms": float(
                    (g[int(0.1 / dt):] ** 2).sum() / energy)
                if g.size > int(0.1 / dt) else 0.0,
            })
    return rows
