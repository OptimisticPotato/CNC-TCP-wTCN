"""Cubic-spline transport between arbitrary timestamps and the standard grid.

SPEC section 6: the TCN only ever sees one grid. Everything else -- a different
``TcpSampleTime``, jitter, dropped samples -- is absorbed here, before the model,
and undone after it.

SPEC section 6.4: cubic spline, never PCHIP (3-5x worse on every derivative
channel). The helpers below refuse ``kind="pchip"`` unless you insist.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator


def _interpolator(t: np.ndarray, y: np.ndarray, kind: str):
    if kind == "cubic":
        return CubicSpline(t, y, axis=0, extrapolate=True)
    if kind == "linear":
        return lambda q: np.stack(
            [np.interp(q, t, y[:, i]) for i in range(y.shape[1])], axis=1
        ) if y.ndim == 2 else np.interp(q, t, y)
    if kind == "pchip":       # kept only so section 6.4 can be re-measured
        return PchipInterpolator(t, y, axis=0, extrapolate=True)
    raise ValueError(f"unknown interpolation kind: {kind}")


def interp_to(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray,
              kind: str = "cubic") -> np.ndarray:
    """Interpolate ``y_src(t_src)`` onto ``t_dst``. Works for (N,) and (N,C)."""
    y = np.asarray(y_src, dtype=np.float64)
    squeeze = y.ndim == 1
    if squeeze:
        y = y[:, None]
    good = np.isfinite(y).all(1)
    if good.sum() < 4:
        out = np.full((t_dst.size, y.shape[1]), np.nan)
        return out[:, 0] if squeeze else out
    f = _interpolator(np.asarray(t_src, np.float64)[good], y[good], kind)
    out = np.asarray(f(np.asarray(t_dst, np.float64)))
    if out.ndim == 1:
        out = out[:, None]
    return out[:, 0] if squeeze else out


def step_to(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Nearest-previous sampling -- for signals that are piecewise constant
    (``feed``, ``block#``); interpolating those would invent values."""
    idx = np.searchsorted(t_src, t_dst, side="right") - 1
    idx = np.clip(idx, 0, t_src.size - 1)
    return np.asarray(y_src)[idx]


def standard_grid(t0: float, t1: float, dt: float) -> np.ndarray:
    n = int(np.floor((t1 - t0) / dt + 1e-9)) + 1
    return t0 + dt * np.arange(max(n, 1), dtype=np.float64)


def run_to_grid(t, cnc, tcp, feed, block, dt, kind: str = "cubic"):
    """Put a whole run on the standard grid (SPEC section 6.2)."""
    grid = standard_grid(float(t[0]), float(t[-1]), dt)
    cnc_g = interp_to(t, cnc, grid, kind)
    tcp_g = interp_to(t, tcp, grid, kind)
    feed_g = step_to(t, feed, grid)
    block_g = step_to(t, block, grid)

    # do not invent targets across gaps where tcp was empty
    bad = ~np.isfinite(tcp).all(1)
    if bad.any():
        holes = step_to(t, bad.astype(np.float64), grid) > 0.5
        tcp_g[holes] = np.nan
    return grid, cnc_g, tcp_g, feed_g, block_g


# --------------------------------------------------------------------------- #
# augmentation: corrupt the source sampling, then restore the standard grid
# --------------------------------------------------------------------------- #
def corrupt_and_restore(
    x: np.ndarray,
    dt: float,
    rng: np.random.Generator,
    jitter: float = 0.0,
    decimate: int = 1,
    drop_frac: float = 0.0,
    kind: str = "cubic",
) -> np.ndarray:
    """Round-trip ``x`` (N,C) sampled on a uniform ``dt`` grid.

    The engine is pretended to have emitted at ``decimate*dt`` with timestamp
    jitter and dropped samples; a cubic spline brings it back. SPEC section 6.3
    measured this round trip at 4..19 nm for sources up to 4 ms, but that was
    for a *linear* kernel -- hence section 7 step 3 makes the TCN learn it
    instead of assuming it.

    Length and grid of the result are identical to the input.
    """
    n = x.shape[0]
    grid = np.arange(n, dtype=np.float64) * dt

    idx = np.arange(0, n, max(1, int(decimate)))
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)

    if drop_frac > 0 and idx.size > 8:
        keep = rng.random(idx.size) >= drop_frac
        keep[0] = keep[-1] = True
        idx = idx[keep]

    t_src = grid[idx].copy()
    if jitter > 0 and idx.size > 3:
        step = dt * max(1, int(decimate))
        t_src[1:-1] += (rng.random(idx.size - 2) - 0.5) * 2.0 * jitter * step
        t_src = np.maximum.accumulate(t_src)               # keep it monotone
        eps = 1e-4 * step
        for _ in range(2):                                 # de-duplicate ties
            same = np.flatnonzero(np.diff(t_src) <= 0)
            if same.size == 0:
                break
            t_src[same + 1] = t_src[same] + eps

    if idx.size < 8:
        return x.copy()
    x_src = interp_to(grid, x, t_src, kind)                # what the engine "emitted"
    return interp_to(t_src, x_src, grid, kind)             # back to the standard grid


def sample_augmentation(cfg, rng: np.random.Generator) -> dict:
    a = cfg.augment
    if not a.enabled or rng.random() >= a.prob:
        return dict(jitter=0.0, decimate=1, drop_frac=0.0, active=False)
    return dict(
        jitter=float(rng.choice(np.asarray(a.jitter, dtype=float))),
        decimate=int(rng.choice(np.asarray(a.decimate, dtype=int))),
        drop_frac=float(rng.choice(np.asarray(a.dropout_frac, dtype=float))),
        active=True,
    )
