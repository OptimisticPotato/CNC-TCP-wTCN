"""Figures for the report. SPEC section 8.

    1. predicted e vs true e, per axis, with a zoom on the worst corner
    2. PSD of the residual (and of the truth, for scale)
    3. residual aligned on direction-reversal points -- the picture that shows
       whether the net caught the non-linear part (quadrant glitch, friction
       reversal, backlash) or only the linear dynamics
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import psd  # noqa: E402

AXES = ("X", "Y", "Z")


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


REF_COLOR = "#2e86ab"


def timeseries(path: Path, t: np.ndarray, e_true: np.ndarray, e_pred: np.ndarray,
               valid: np.ndarray, warmup: int, title: str = "",
               span_s: float = 1.0, zoom_s: float = 0.06,
               e_ref: Optional[np.ndarray] = None,
               ref_label: str = "FIR") -> Path:
    """Overlay of true and predicted e, plus a zoom on the largest error.

    ``e_ref`` draws a second prediction (the linear FIR baseline) in the same
    axes, so the comparison is one picture rather than two.
    """
    m = valid.copy()
    m[:warmup] = False
    idx = np.flatnonzero(m)
    dt = float(np.median(np.diff(t))) if t.size > 2 else 5e-4
    n_span = max(int(span_s / dt), 64)

    worst = idx[np.argmax(np.abs(e_true[idx] - e_pred[idx]).max(axis=1))]
    lo = max(idx[0], worst - n_span // 2)
    hi = min(idx[-1] + 1, lo + n_span)
    zl = max(idx[0], worst - int(zoom_s / dt) // 2)
    zh = min(idx[-1] + 1, zl + max(int(zoom_s / dt), 32))

    fig, axes = plt.subplots(3, 2, figsize=(13, 7.5), sharex="col")
    for i in range(3):
        for j, (a, b) in enumerate(((lo, hi), (zl, zh))):
            ax = axes[i, j]
            ax.plot(t[a:b], e_true[a:b, i], lw=1.4, label="true", color="#222222")
            if e_ref is not None:
                ax.plot(t[a:b], e_ref[a:b, i], lw=1.0, label=ref_label,
                        color=REF_COLOR, ls="--", alpha=0.9)
            ax.plot(t[a:b], e_pred[a:b, i], lw=1.0, label="TCN", color="#d1495b")
            ax.set_ylabel(f"e_{AXES[i]} [um]")
            ax.grid(alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(loc="upper right", fontsize=8)
            if i == 0:
                ax.set_title("worst-error region" if j == 0 else "zoom", fontsize=10)
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 1].set_xlabel("time [s]")
    fig.suptitle(title or "e = TCP - CNC : truth vs TCN", fontsize=11)
    return _save(fig, path)


def residual_psd(path: Path, e_true: np.ndarray, e_pred: np.ndarray,
                 valid: np.ndarray, warmup: int, fs: float,
                 nperseg: int = 2048, title: str = "",
                 e_ref: Optional[np.ndarray] = None,
                 ref_label: str = "FIR") -> Path:
    m = valid.copy()
    m[:warmup] = False
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for i, ax in enumerate(axes):
        x = np.where(m, e_true[:, i], 0.0)
        r = np.where(m, e_true[:, i] - e_pred[:, i], 0.0)
        f, pt = psd(x, fs, nperseg)
        _, pr = psd(r, fs, nperseg)
        ax.semilogy(f, pt, lw=1.2, color="#222222", label="true e")
        if e_ref is not None:
            rr = np.where(m, e_true[:, i] - e_ref[:, i], 0.0)
            _, prr = psd(rr, fs, nperseg)
            ax.semilogy(f, prr, lw=1.0, color=REF_COLOR, ls="--",
                        label=f"residual {ref_label}")
        ax.semilogy(f, pr, lw=1.0, color="#d1495b", label="residual TCN")
        ax.set_xlim(0, min(300.0, fs / 2))
        ax.set_xlabel("frequency [Hz]")
        ax.set_title(f"axis {AXES[i]}", fontsize=10)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("PSD [um^2/Hz]")
    axes[0].legend(fontsize=8)
    fig.suptitle(title or "residual spectrum", fontsize=11)
    return _save(fig, path)


def reversal_alignment(path: Path, cnc: np.ndarray, e_true: np.ndarray,
                       e_pred: np.ndarray, valid: np.ndarray, warmup: int,
                       dt: float, pre_ms: float = 20.0, post_ms: float = 60.0,
                       min_speed: float = 1e-5, title: str = "",
                       e_ref: Optional[np.ndarray] = None,
                       ref_label: str = "FIR") -> Path:
    """Average residual around the instants where an axis reverses direction.

    A purely linear model leaves a signature here (quadrant glitch / friction
    reversal are not LTI); a flat line means the net absorbed it.
    """
    n_pre = max(int(pre_ms * 1e-3 / dt), 4)
    n_post = max(int(post_ms * 1e-3 / dt), 8)
    m = valid.copy()
    m[:warmup] = False

    v = np.zeros_like(cnc)
    v[1:] = np.diff(cnc, axis=0) / dt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    tau = (np.arange(-n_pre, n_post) * dt) * 1e3
    for i, ax in enumerate(axes):
        s = np.sign(v[:, i])
        s[np.abs(v[:, i]) < min_speed] = 0
        prev = np.zeros_like(s)
        last = 0.0
        for k in range(s.size):                       # last non-zero sign
            prev[k] = last
            if s[k] != 0:
                last = s[k]
        hits = np.flatnonzero((s != 0) & (prev != 0) & (s != prev))
        hits = hits[(hits > n_pre + warmup) & (hits < s.size - n_post)]
        hits = hits[m[hits]]
        if hits.size == 0:
            ax.text(0.5, 0.5, "no reversal found", ha="center", transform=ax.transAxes)
            ax.set_title(f"axis {AXES[i]}", fontsize=10)
            continue
        seg_t = np.stack([e_true[h - n_pre:h + n_post, i] for h in hits])
        seg_p = np.stack([e_pred[h - n_pre:h + n_post, i] for h in hits])
        sign = np.sign(s[hits])[:, None]              # fold both directions together
        res = (seg_t - seg_p) * sign
        ax.plot(tau, (seg_t * sign).mean(0), color="#222222", lw=1.4, label="true e")
        ax.plot(tau, (seg_p * sign).mean(0), color="#d1495b", lw=1.0, label="TCN")
        ax.plot(tau, res.mean(0), color="#d1495b", lw=1.2, ls=":",
                label="residual TCN")
        ax.fill_between(tau, res.mean(0) - res.std(0), res.mean(0) + res.std(0),
                        color="#d1495b", alpha=0.12)
        if e_ref is not None:
            seg_r = np.stack([e_ref[h - n_pre:h + n_post, i] for h in hits])
            res_r = (seg_t - seg_r) * sign
            ax.plot(tau, (seg_r * sign).mean(0), color=REF_COLOR, lw=1.0, ls="--",
                    label=ref_label)
            ax.plot(tau, res_r.mean(0), color=REF_COLOR, lw=1.2, ls=":",
                    label=f"residual {ref_label}")
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("time from reversal [ms]")
        ax.set_title(f"axis {AXES[i]}  (n={hits.size})", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("e, direction-folded [um]")
    axes[0].legend(fontsize=8)
    fig.suptitle(title or "residual at direction reversals", fontsize=11)
    return _save(fig, path)


def curve(path: Path, x: Sequence[float], ys: dict, xlabel: str, ylabel: str,
          title: str = "", logx: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for label, y in ys.items():
        ax.plot(x, y, marker="o", lw=1.4, label=label)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=11)
    return _save(fig, path)


def training_curve(path: Path, history: Sequence[dict]) -> Optional[Path]:
    if not history:
        return None
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.semilogy(ep, [h["train_loss"] for h in history], label="train")
    if any("val_rmse_um" in h for h in history):
        ax.semilogy(ep, [h.get("val_rmse_um", np.nan) for h in history],
                    label="val RMSE [um]")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / RMSE")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    return _save(fig, path)
