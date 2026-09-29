"""Causal, bias-free, dilated TCN. SPEC section 5.

Three structural properties are built in rather than learned:

* **causal** -- left padding only, no future sample ever reaches the output
* **zero in -> zero out** -- no bias anywhere, activations with f(0)=0, no
  BatchNorm. If nothing moved during the last receptive field, the predicted
  error is exactly 0, which is what the physics says.
* **no DC response** -- follows from the two above plus increment inputs
  (a constant velocity is a constant input whose weighted sum the net is free
  to send to zero, and the FIR work showed sum(g) ~ 0).

Everything is one model for all three axes, so cross-axis coupling is available.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":       # f(0)=0 as well, but unbounded derivative at 0
        return nn.GELU()
    raise ValueError(f"activation must satisfy f(0)=0; got {name!r}")


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels, affine-free so that f(0)=0 survives.

    Off by default (``model.norm = "none"``). BatchNorm is forbidden by the
    spec because the amplitude of e *is* the answer.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, C, T)
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.eps)


class CausalConv1d(nn.Conv1d):
    """Conv1d that only looks backwards. Never has a bias."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__(in_ch, out_ch, kernel_size, dilation=dilation, bias=False)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_pad:
            x = F.pad(x, (self.left_pad, 0))
        return super().forward(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int,
                 activation: str, norm: str, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.act1 = _activation(activation)
        self.act2 = _activation(activation)
        self.norm1 = ChannelLayerNorm(out_ch) if norm == "layernorm" else nn.Identity()
        self.norm2 = ChannelLayerNorm(out_ch) if norm == "layernorm" else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # 1x1 projection for the skip path, still bias-free
        self.proj = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act1(self.norm1(self.conv1(x))))
        y = self.drop(self.act2(self.norm2(self.conv2(y))))
        s = x if self.proj is None else self.proj(x)
        return y + s


class TCN(nn.Module):
    """6 increment channels in, 3 error channels (um) out."""

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        channels: int = 48,
        kernel_size: int = 5,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        activation: str = "relu",
        norm: str = "none",
        dropout: float = 0.0,
        linear_skip: bool = False,
        linear_skip_taps: int = 0,
    ):
        super().__init__()
        if norm.lower() == "batchnorm":
            raise ValueError("BatchNorm is forbidden (SPEC section 9)")
        self.kernel_size = kernel_size
        self.dilations = tuple(int(d) for d in dilations)
        self.receptive_field = 1 + (kernel_size - 1) * sum(self.dilations)

        blocks = []
        ch_in = in_channels
        for d in self.dilations:
            blocks.append(ResidualBlock(ch_in, channels, kernel_size, d,
                                        activation, norm.lower(), dropout))
            ch_in = channels
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv1d(channels, out_channels, 1, bias=False)

        self.linear = None
        if linear_skip:
            taps = int(linear_skip_taps) or self.receptive_field
            self.linear = CausalConv1d(in_channels, out_channels, taps, 1)
            nn.init.zeros_(self.linear.weight)     # start as a pure TCN

        for m in self.modules():                   # paranoia: the spec forbids bias
            if isinstance(m, nn.Conv1d) and m.bias is not None:
                raise RuntimeError("a conv ended up with a bias term")

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, in_channels, T) -> e_hat: (B, out_channels, T), micrometres."""
        x = u
        for b in self.blocks:
            x = b(x)
        y = self.head(x)
        if self.linear is not None:
            y = y + self.linear(u)
        return y

    # ------------------------------------------------------------------ #
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def check_zero_input(self, length: int = 512, device=None) -> float:
        """Max |output| for an all-zero input. Must be exactly 0."""
        dev = device or next(self.parameters()).device
        z = torch.zeros(1, self.blocks[0].conv1.in_channels, length, device=dev)
        return float(self(z).abs().max())

    @torch.no_grad()
    def check_homogeneity(self, length: int = 512, alpha: float = 2.0,
                          device=None) -> float:
        """Max relative violation of ``f(alpha*u) == alpha*f(u)`` for alpha > 0.

        With no bias anywhere and an activation satisfying f(0)=0, a ReLU net is
        *positively homogeneous of degree 1*: doubling the commanded motion
        doubles the predicted error, exactly. That is a consequence of the
        spec's bias ban, and it has a cost -- amplitude-dependent saturation
        (an axis hitting its acceleration limit) cannot be represented, no
        matter how the loss is weighted. Direction-dependent effects (quadrant
        glitch, friction reversal) and cross-axis interaction still can.

        Returns ~0 for ReLU, clearly non-zero for tanh, which trades this
        property for the ability to saturate.
        """
        dev = device or next(self.parameters()).device
        c = self.blocks[0].conv1.in_channels
        u = torch.randn(1, c, length, device=dev)
        a = self(u) * alpha
        b = self(u * alpha)
        return float((a - b).abs().max() / a.abs().max().clamp_min(1e-12))

    @torch.no_grad()
    def check_causality(self, length: int = 512, device=None) -> float:
        """Perturb the last input sample; anything but the last output moving
        would mean the net peeks into the future. Returns the max leak."""
        dev = device or next(self.parameters()).device
        c = self.blocks[0].conv1.in_channels
        a = torch.randn(1, c, length, device=dev)
        b = a.clone()
        b[..., -1] += 1.0
        d = (self(a) - self(b)).abs()
        return float(d[..., :-1].max())


def build_model(cfg) -> TCN:
    m = cfg.model
    return TCN(
        in_channels=m.in_channels,
        out_channels=m.out_channels,
        channels=m.channels,
        kernel_size=m.kernel_size,
        dilations=m.dilations,
        activation=m.activation,
        norm=m.norm,
        dropout=m.dropout,
        linear_skip=m.linear_skip,
        linear_skip_taps=m.linear_skip_taps,
    )


def dilations_for_ms(target_ms: float, dt: float, kernel_size: int = 5) -> tuple:
    """Smallest power-of-two dilation stack whose receptive field covers
    ``target_ms``. Used by the receptive-field sweep (SPEC section 8, metric 7)."""
    need = target_ms * 1e-3 / dt
    dil, total = [], 0
    d = 1
    while True:
        dil.append(d)
        total += d
        if 1 + (kernel_size - 1) * total >= need:
            return tuple(dil)
        d *= 2
        if d > 2 ** 16:
            raise ValueError(f"cannot reach {target_ms} ms with kernel {kernel_size}")
