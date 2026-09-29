"""Loss. SPEC section 5.3.

    L = MSE(e_hat, e)                       [um]
      + l1 * MSE(diff(e_hat), diff(e))      high-frequency weighting
      + l2 * multi-resolution STFT magnitude

Plain MSE alone averages the ringing away -- that is exactly how the earlier
NC->TCP attempt ended up predicting a smooth curve through the vibration. The
second term punishes a wrong slope, the third punishes a wrong amplitude
spectrum; the third is the standard fix from neural vocoders.

Every term is masked: the first ``receptive_field`` samples of a crop have no
history, empty rows have no target, and G00 may be down-weighted.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               w: torch.Tensor) -> torch.Tensor:
    """pred/target: (B,C,T), w: (B,1,T) or (B,T)."""
    if w.dim() == 2:
        w = w.unsqueeze(1)
    num = (w * (pred - target) ** 2).sum()
    den = w.sum() * pred.shape[1]
    return num / den.clamp_min(1.0)


def masked_diff_mse(pred: torch.Tensor, target: torch.Tensor,
                    w: torch.Tensor) -> torch.Tensor:
    if w.dim() == 2:
        w = w.unsqueeze(1)
    dp = pred[..., 1:] - pred[..., :-1]
    dt = target[..., 1:] - target[..., :-1]
    wd = torch.minimum(w[..., 1:], w[..., :-1])
    num = (wd * (dp - dt) ** 2).sum()
    den = wd.sum() * pred.shape[1]
    return num / den.clamp_min(1.0)


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral-convergence + log-magnitude loss over several resolutions."""

    def __init__(self, fft_sizes: Sequence[int] = (256, 512, 1024),
                 hop_sizes: Sequence[int] = (64, 128, 256)):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes)
        self.fft_sizes = tuple(int(n) for n in fft_sizes)
        self.hop_sizes = tuple(int(n) for n in hop_sizes)
        for n in self.fft_sizes:
            self.register_buffer(f"win{n}", torch.hann_window(n), persistent=False)

    def _stft_mag(self, x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
        win = getattr(self, f"win{n_fft}").to(x.dtype)
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                          window=win, center=True, return_complex=True)
        return spec.abs().clamp_min(1e-7)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred/target: (B,C,T) already masked (invalid samples zeroed)."""
        b, c, t = pred.shape
        p = pred.reshape(b * c, t).float()
        q = target.reshape(b * c, t).float()
        total = p.new_zeros(())
        used = 0
        for n_fft, hop in zip(self.fft_sizes, self.hop_sizes):
            if t < n_fft:
                continue
            mp = self._stft_mag(p, n_fft, hop)
            mq = self._stft_mag(q, n_fft, hop)
            sc = torch.norm(mq - mp, p="fro") / torch.norm(mq, p="fro").clamp_min(1e-7)
            mag = (torch.log(mq) - torch.log(mp)).abs().mean()
            total = total + sc + mag
            used += 1
        return total / max(used, 1)


class TCNLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.l_diff = float(cfg.loss.lambda_diff)
        self.l_stft = float(cfg.loss.lambda_stft)
        self.stft = (
            MultiResolutionSTFTLoss(cfg.loss.stft_ffts, cfg.loss.stft_hops)
            if self.l_stft > 0 else None
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                w: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        mse = masked_mse(pred, target, w)
        loss = mse
        parts = {"mse": float(mse.detach())}

        if self.l_diff > 0:
            d = masked_diff_mse(pred, target, w)
            loss = loss + self.l_diff * d
            parts["diff"] = float(d.detach())

        if self.stft is not None:
            m = (w.unsqueeze(1) if w.dim() == 2 else w) > 0
            s = self.stft(pred * m, target * m)
            loss = loss + self.l_stft * s
            parts["stft"] = float(s.detach())

        parts["total"] = float(loss.detach())
        return loss, parts
