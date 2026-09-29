"""Input / output convention. SPEC section 4 -- this is where it is won or lost.

Output  e = TCP - CNC in micrometres, never absolute coordinates.
Input   six channels of *increments*, split by modal state:

    u[0:3] = dCNC_xyz * 1_{G01}
    u[3:6] = dCNC_xyz * 1_{G00}          (normalised by the rapid step)

The identity ``u[0:3] + u[3:6] == dCNC / scale`` must hold everywhere; the
loader checks it (``check_identity``).

G00/G01 is a *channel mask*, never a dataset split (SPEC section 2): the ringing
that a G00 deceleration leaves behind lands in the first millimetres of the next
G01 cut, and cutting the sequence there would delete the cause of the largest
form error in the program.
"""
from __future__ import annotations

import numpy as np


def increments(cnc: np.ndarray) -> np.ndarray:
    """dCNC[n] = CNC[n] - CNC[n-1], with dCNC[0] = 0. float64 in, float64 out."""
    d = np.zeros_like(cnc, dtype=np.float64)
    d[1:] = np.diff(cnc, axis=0)
    return d


def g00_mask(run, cfg) -> np.ndarray:
    """True where the block is a rapid traverse (G00).

    The rapid feed is not a constant of this code: with
    ``data.rapid_feed_auto`` it comes from work/meta/machine.json, which
    01_scan_runs derives from the dataset itself (the most common per-run
    maximum feed). ``data.rapid_feed_mm_min`` is the fallback and the manual
    override. On a machine whose rapid is, say, 24 m/min, nothing here needs
    editing -- but re-run 01_scan_runs so the number is re-measured.
    """
    mode = cfg.data.g00_source
    if mode == "none":
        return np.zeros(len(run), dtype=bool)

    if mode == "feed":
        from .runs_io import rapid_feed
        thr = rapid_feed(cfg) * cfg.data.rapid_feed_tol
        return run.feed >= thr

    if mode == "speed":
        # per-block decision: a block is G00 if its peak commanded speed is
        # above the threshold. Measured against the feed column on 24 runs:
        # 3616/3653 blocks correct (98.99 %); every miss is a short rapid that
        # never reaches full speed, so it is labelled G01.
        step = np.linalg.norm(increments(run.cnc), axis=1)
        speed = step / run.dt * 60.0
        out = np.zeros(len(run), dtype=bool)
        blk = run.block
        edges = np.flatnonzero(np.diff(blk) != 0) + 1
        for s in np.split(np.arange(blk.size), edges):
            if s.size and speed[s].max() >= cfg.data.g00_speed_threshold:
                out[s] = True
        return out

    raise ValueError(f"unknown data.g00_source: {mode!r}")


def build_inputs(cnc: np.ndarray, g00: np.ndarray, scale: float) -> np.ndarray:
    """(N,3) positions -> (N,6) normalised, modally split increments [float32]."""
    d = increments(cnc) / scale
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    is_g00 = np.asarray(g00, dtype=bool)[:, None]
    u = np.concatenate([d * ~is_g00, d * is_g00], axis=1)
    return u.astype(np.float32)


def build_target(cnc: np.ndarray, tcp: np.ndarray) -> np.ndarray:
    """(N,3) e = TCP - CNC in micrometres [float32].

    float64 subtraction first, then cast: 320 mm +- um is not representable in
    float32, but the 20 um difference is (SPEC section 4.1).
    """
    e = (tcp.astype(np.float64) - cnc.astype(np.float64)) * 1e3
    return np.nan_to_num(e, nan=0.0).astype(np.float32)


def check_identity(u: np.ndarray, cnc: np.ndarray, scale: float,
                   atol: float = 1e-9) -> float:
    """Max violation of ``u[0:3] + u[3:6] == dCNC / scale``. Returns the value
    so callers can report it; raises only when it is clearly broken."""
    d = np.nan_to_num(increments(cnc) / scale, nan=0.0)
    err = np.abs(u[:, :3].astype(np.float64) + u[:, 3:].astype(np.float64) - d).max()
    if not np.isfinite(err) or err > max(atol, 1e-6 * np.abs(d).max()):
        raise AssertionError(f"G00/G01 channel identity violated: max |err| = {err:g}")
    return float(err)


def loss_weight(valid: np.ndarray, g00: np.ndarray, warmup: int,
                g00_weight: float) -> np.ndarray:
    """Per-sample loss weight. SPEC section 5.4.

    * the first ``warmup`` samples of a sequence have no history -> 0
    * rows whose cnc/tcp were empty -> 0
    * G00 stays in the *input* but may be down-weighted in the loss
    """
    w = valid.astype(np.float32).copy()
    if warmup > 0:
        w[:warmup] = 0.0
    if g00_weight != 1.0:
        w[g00] *= float(g00_weight)
    return w
