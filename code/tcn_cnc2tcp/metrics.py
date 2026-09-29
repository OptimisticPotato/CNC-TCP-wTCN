"""Evaluation metrics. SPEC section 8.

No pass/fail verdict is produced anywhere in here -- the numbers are computed
and reported, the judgement belongs to whoever ordered the work.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal

AXES = ("x", "y", "z")


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


# --------------------------------------------------------------------------- #
# 1 / 2 -- RMSE and worst-case error, per axis, G01 and overall
# --------------------------------------------------------------------------- #
def error_table(e_true: np.ndarray, e_pred: np.ndarray, valid: np.ndarray,
                g00: np.ndarray, warmup: int) -> pd.DataFrame:
    m = valid.copy()
    m[:warmup] = False
    rows = []
    for label, sel in (("all", m), ("G01", m & ~g00), ("G00", m & g00)):
        if sel.sum() == 0:
            continue
        r = e_true[sel] - e_pred[sel]
        row: Dict[str, object] = {"segment": label, "n": int(sel.sum())}
        for i, ax in enumerate(AXES):
            row[f"rmse_{ax}_um"] = _rms(r[:, i])
            row[f"maxabs_{ax}_um"] = float(np.abs(r[:, i]).max())
            row[f"true_rms_{ax}_um"] = _rms(e_true[sel][:, i])
        row["rmse_norm_um"] = _rms(np.linalg.norm(r, axis=1))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3 -- worst error per NC block
# --------------------------------------------------------------------------- #
def block_table(block: np.ndarray, e_true: np.ndarray, e_pred: np.ndarray,
                valid: np.ndarray, g00: np.ndarray, warmup: int) -> pd.DataFrame:
    m = valid.copy()
    m[:warmup] = False
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return pd.DataFrame()
    r = np.abs(e_true[idx] - e_pred[idx])
    df = pd.DataFrame({
        "block": block[idx],
        "g00": g00[idx],
        "err_x": r[:, 0], "err_y": r[:, 1], "err_z": r[:, 2],
        "true_x": np.abs(e_true[idx][:, 0]),
        "true_y": np.abs(e_true[idx][:, 1]),
        "true_z": np.abs(e_true[idx][:, 2]),
    })
    g = df.groupby("block")
    out = g.agg(
        n=("err_x", "size"),
        g00=("g00", "max"),
        max_err_x=("err_x", "max"), max_err_y=("err_y", "max"),
        max_err_z=("err_z", "max"),
        max_true_x=("true_x", "max"), max_true_y=("true_y", "max"),
        max_true_z=("true_z", "max"),
    ).reset_index()
    out["max_err_um"] = out[["max_err_x", "max_err_y", "max_err_z"]].max(axis=1)
    return out.sort_values("max_err_um", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4 -- spectral agreement: did the vibration survive?
# --------------------------------------------------------------------------- #
def spectral_table(e_true: np.ndarray, e_pred: np.ndarray, valid: np.ndarray,
                   warmup: int, fs: float, nperseg: int = 2048,
                   band: Tuple[float, float] = (0.0, 200.0)) -> pd.DataFrame:
    m = valid.copy()
    m[:warmup] = False
    rows = []
    for i, ax in enumerate(AXES):
        x = np.where(m, e_true[:, i], 0.0)
        y = np.where(m, e_pred[:, i], 0.0)
        n = min(nperseg, x.size)
        if n < 32:
            continue
        f, pxx = signal.welch(x, fs, nperseg=n)
        _, pyy = signal.welch(y, fs, nperseg=n)
        _, pxy = signal.csd(x, y, fs, nperseg=n)
        coh = np.abs(pxy) ** 2 / np.maximum(pxx * pyy, 1e-300)
        sel = (f >= band[0]) & (f <= band[1])
        wgt = pxx[sel] / max(pxx[sel].sum(), 1e-300)
        rows.append({
            "axis": ax,
            "coh_mean": float(np.mean(coh[sel])),
            "coh_pw_mean": float(np.sum(wgt * coh[sel])),      # power-weighted
            "psd_ratio": float(pyy[sel].sum() / max(pxx[sel].sum(), 1e-300)),
            "psd_logdev_db": float(np.mean(np.abs(
                10 * np.log10(np.maximum(pyy[sel], 1e-300) /
                              np.maximum(pxx[sel], 1e-300))))),
            "peak_hz_true": float(f[sel][np.argmax(pxx[sel])]),
            "peak_hz_pred": float(f[sel][np.argmax(pyy[sel])]),
        })
    return pd.DataFrame(rows)


def psd(x: np.ndarray, fs: float, nperseg: int = 2048):
    n = min(nperseg, x.size)
    return signal.welch(x, fs, nperseg=max(n, 32))


# --------------------------------------------------------------------------- #
# 5 -- form error
# --------------------------------------------------------------------------- #
def normal_deviation(cnc: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Component of e perpendicular to the commanded direction of travel [um].

    This is a *proxy*: the part of the servo error that actually lands on the
    workpiece surface, since the along-path component only shifts the tool in
    time. It is NOT the form-error pipeline of SPEC section 8 metric 5 -- that
    one lives outside this repository and is fed by ``export_tcp``.
    """
    d = np.zeros_like(cnc)
    d[1:] = np.diff(cnc, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    t = np.divide(d, n, out=np.zeros_like(d), where=n > 1e-12)
    along = np.sum(e * t, axis=1, keepdims=True) * t
    return np.linalg.norm(e - along, axis=1)


def form_error_table(cnc: np.ndarray, e_true: np.ndarray, e_pred: np.ndarray,
                     valid: np.ndarray, g00: np.ndarray, warmup: int) -> pd.DataFrame:
    m = valid.copy()
    m[:warmup] = False
    sel = m & ~g00
    if sel.sum() == 0:
        return pd.DataFrame()
    nt = normal_deviation(cnc, e_true)[sel]
    npd = normal_deviation(cnc, e_pred)[sel]
    return pd.DataFrame([{
        "segment": "G01",
        "normal_dev_rms_true_um": _rms(nt),
        "normal_dev_rms_pred_um": _rms(npd),
        "normal_dev_max_true_um": float(nt.max()),
        "normal_dev_max_pred_um": float(npd.max()),
        "normal_dev_rms_residual_um": _rms(nt - npd),
    }])


def export_tcp(path: Path, t: np.ndarray, cnc: np.ndarray, e_pred: np.ndarray,
               e_true: np.ndarray, block: np.ndarray) -> Path:
    """Predicted TCP as CSV, ready for an external form-error pipeline.

    TCP = CNC + e/1000, written with enough digits to keep the micrometres.
    """
    tcp_pred = cnc + e_pred.astype(np.float64) / 1e3
    tcp_true = cnc + e_true.astype(np.float64) / 1e3
    df = pd.DataFrame({
        "time": t, "block#": block,
        "cnc_x": cnc[:, 0], "cnc_y": cnc[:, 1], "cnc_z": cnc[:, 2],
        "tcp_pred_x": tcp_pred[:, 0], "tcp_pred_y": tcp_pred[:, 1],
        "tcp_pred_z": tcp_pred[:, 2],
        "tcp_true_x": tcp_true[:, 0], "tcp_true_y": tcp_true[:, 1],
        "tcp_true_z": tcp_true[:, 2],
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.9g", lineterminator="\n")
    return Path(path)


# --------------------------------------------------------------------------- #
# aggregation helper
# --------------------------------------------------------------------------- #
def summarise(per_program: List[pd.DataFrame]) -> pd.DataFrame:
    """Pool per-program error tables into one row per segment (n-weighted)."""
    if not per_program:
        return pd.DataFrame()
    df = pd.concat(per_program, ignore_index=True)
    rows = []
    for seg, g in df.groupby("segment"):
        n = g["n"].to_numpy(float)
        row = {"segment": seg, "n": int(n.sum()), "programs": len(g)}
        for ax in AXES:
            row[f"rmse_{ax}_um"] = float(np.sqrt(np.sum(n * g[f"rmse_{ax}_um"] ** 2) / n.sum()))
            row[f"maxabs_{ax}_um"] = float(g[f"maxabs_{ax}_um"].max())
            row[f"true_rms_{ax}_um"] = float(np.sqrt(np.sum(n * g[f"true_rms_{ax}_um"] ** 2) / n.sum()))
        rows.append(row)
    return pd.DataFrame(rows)
