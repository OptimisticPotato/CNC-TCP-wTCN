"""Training loop shared by the overfit test, the main run and the sweeps."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from . import dataset as ds
from .losses import TCNLoss
from .model import build_model


def pick_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_grad_scaler(enabled: bool):
    """``torch.amp.GradScaler`` exists from 2.4; fall back for older builds."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):             # pragma: no cover
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def validate(model, cfg, programs: Sequence[ds.Program], device,
             amp: bool = False) -> Dict[str, float]:
    """Masked RMSE over whole validation programs (never over random windows)."""
    model.eval()
    se = np.zeros(3)
    n = 0.0
    se_g01 = np.zeros(3)
    n_g01 = 0.0
    for p in programs:
        pred = ds.predict_program(model, cfg, p, device, amp=amp)
        m = p.valid.copy()
        m[:cfg.receptive_field] = False
        if m.sum() == 0:
            continue
        r = (p.e[m] - pred[m]).astype(np.float64)
        se += (r ** 2).sum(axis=0)
        n += m.sum()
        mc = m & ~p.g00
        if mc.sum():
            rc = (p.e[mc] - pred[mc]).astype(np.float64)
            se_g01 += (rc ** 2).sum(axis=0)
            n_g01 += mc.sum()
    out = {}
    if n:
        rms = np.sqrt(se / n)
        out.update(val_rmse_x=float(rms[0]), val_rmse_y=float(rms[1]),
                   val_rmse_z=float(rms[2]),
                   val_rmse_um=float(np.sqrt(se.sum() / (3 * n))))
    if n_g01:
        out["val_rmse_g01_um"] = float(np.sqrt(se_g01.sum() / (3 * n_g01)))
    return out


def train_model(
    cfg,
    train_programs: Sequence[ds.Program],
    val_programs: Sequence[ds.Program],
    out_dir: Path,
    epochs: Optional[int] = None,
    augment: Optional[bool] = None,
    log: Callable[[str], None] = print,
    tag: str = "",
) -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(epochs if epochs is not None else cfg.train.epochs)
    augment = cfg.augment.enabled if augment is None else augment

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    device = pick_device()
    model = build_model(cfg).to(device)
    zero = model.check_zero_input(device=device)
    leak = model.check_causality(device=device)
    log(f"model: {model.num_parameters():,} parameters | RF {cfg.receptive_field} "
        f"samples ({cfg.receptive_field_ms:.1f} ms) | device {device}")
    log(f"structure checks: zero-input output = {zero:.3e} (must be 0), "
        f"future leak = {leak:.3e} (must be 0)")
    log(f"best.pt selected on: {cfg.train.select_metric}"
        + ("" if val_programs else "  (no val programs -> train loss)"))
    if zero != 0.0 or leak != 0.0:
        raise RuntimeError("model violates the zero-input or causality guarantee")

    criterion = TCNLoss(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    use_amp = bool(cfg.train.amp and device.type == "cuda")
    scaler = make_grad_scaler(use_amp)

    loader = ds.make_loader(cfg, train_programs, augment=augment, shuffle=False)

    history: List[dict] = []
    best = float("inf")
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        loader.dataset.seed = cfg.train.seed + epoch * 7919   # new crops each epoch
        model.train()
        t0 = time.time()
        acc: Dict[str, float] = {}
        steps = 0
        for u, e, w in loader:
            u = u.to(device, non_blocking=True)
            e = e.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                pred = model(u)
            # the loss is amplitude-sensitive (um) and has an STFT in it, so it
            # is always evaluated in float32, outside the autocast region
            loss, parts = criterion(pred.float(), e, w)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v
            steps += 1
        sched.step()

        rec = {"epoch": epoch, "lr": opt.param_groups[0]["lr"],
               "train_loss": acc.get("total", 0.0) / max(steps, 1),
               "train_mse": acc.get("mse", 0.0) / max(steps, 1),
               "sec": time.time() - t0}
        if val_programs:
            rec.update(validate(model, cfg, val_programs, device, amp=use_amp))
        history.append(rec)

        msg = (f"epoch {epoch:3d}/{epochs}  loss {rec['train_loss']:.4f}  "
               f"mse {rec['train_mse']:.4f}")
        if "val_rmse_um" in rec:
            msg += (f"  val RMSE {rec['val_rmse_um']:.3f} um"
                    f"  (G01 {rec.get('val_rmse_g01_um', float('nan')):.3f})")
        msg += f"  {rec['sec']:.1f}s"
        log(msg)

        torch.save({"model": model.state_dict(), "cfg": _cfg_dict(cfg),
                    "epoch": epoch, "history": history}, last_path)
        score = rec.get(cfg.train.select_metric)
        if score is None:
            score = rec.get("val_rmse_um", rec["train_loss"])
        if score < best - 1e-9:
            best = score
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "cfg": _cfg_dict(cfg),
                        "epoch": epoch, "score": best, "history": history}, best_path)
        else:
            bad_epochs += 1
            if cfg.train.patience and bad_epochs >= cfg.train.patience:
                log(f"early stop: no improvement for {bad_epochs} epochs")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"history": history, "best": best, "best_path": best_path,
            "last_path": last_path, "model": model, "device": device, "tag": tag}


def _cfg_dict(cfg) -> dict:
    from config import as_dict
    return as_dict(cfg)


#: data settings that define the input convention: a checkpoint is only
#: meaningful together with these, so they are restored from it
_INPUT_CONVENTION = ("dt", "rapid_feed_mm_min", "rapid_feed_tol", "g00_source")


def load_checkpoint(path: Path, cfg, device=None, log=print):
    """Rebuild the model from a checkpoint.

    Both the architecture and the input convention (standard grid, increment
    scaling, G00 rule) come from the checkpoint rather than from the current
    config: conv tap k means "k * dt seconds ago", so a model run on a
    different dt or a different input scaling is quietly wrong rather than
    loudly broken.
    """
    device = device or pick_device()
    ck = torch.load(Path(path), map_location=device, weights_only=False)
    saved = ck.get("cfg", {}).get("model")
    if saved:
        for k, v in saved.items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, tuple(v) if isinstance(v, list) else v)
    saved_data = ck.get("cfg", {}).get("data") or {}
    for k in _INPUT_CONVENTION:
        if k in saved_data and hasattr(cfg.data, k):
            old = getattr(cfg.data, k)
            if old != saved_data[k]:
                log(f"note: {k} = {saved_data[k]!r} taken from the checkpoint "
                    f"(current config says {old!r})")
            setattr(cfg.data, k, saved_data[k])
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck, device
