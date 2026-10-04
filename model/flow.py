#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The shared bijection `f` — a Glow-style convolutional normalizing flow.

`f` maps A's representation to B's and back on a 256x28x28 feature map, with no
flatten and no FC layer: 1.84 M parameters against the 51 M an equivalent dense
vector flow would need.  Two properties the rest of the model depends on:

  * `f` is the EXACT identity at initialisation.  SpatialActNorm starts at
    log_scale = bias = 0, and SpatialCoupling's last conv is zero-init, so
    f(z) = z and f^-1(u) = u to machine precision.  This is what lets a
    warm-started FlowCycle reproduce its plain-CycleGAN host bit-for-bit
    (train.py --check_init).
  * `f^-1(f(z)) = z` holds by construction rather than being penalised into
    place, so the two cross directions are one set of weights, not two.

Extracted from the multimodal flow-VAE line of the project; the VAE wrappers
that used to live alongside these blocks are gone with the GMM prior.
"""
import torch
import torch.nn as nn


class SpatialActNorm(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        y = x * torch.exp(self.log_scale) + self.bias
        ld = self.log_scale.sum() * x.shape[2] * x.shape[3]
        return y, ld.expand(x.shape[0])

    def inverse(self, y):
        x = (y - self.bias) * torch.exp(-self.log_scale)
        return x, (-self.log_scale.sum() * y.shape[2] * y.shape[3]).expand(y.shape[0])


class SpatialCoupling(nn.Module):
    """Channel-split affine coupling; zero-init -> starts as the identity."""

    def __init__(self, ch=128, hidden=128, swap=False):
        super().__init__()
        self.swap = swap; self.half = ch // 2
        self.net = nn.Sequential(
            nn.Conv2d(self.half, hidden, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 1), nn.SiLU(),
            nn.Conv2d(hidden, (ch - self.half) * 2, 3, 1, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def _split(self, x):
        return (x[:, self.half:], x[:, :self.half]) if self.swap else (x[:, :self.half], x[:, self.half:])

    def _merge(self, a, b):
        return torch.cat([b, a], 1) if self.swap else torch.cat([a, b], 1)

    def forward(self, x):
        x1, x2 = self._split(x)
        s, t = self.net(x1).chunk(2, dim=1)
        s = torch.tanh(s)
        return self._merge(x1, x2 * torch.exp(s) + t), s.flatten(1).sum(1)

    def inverse(self, y):
        y1, y2 = self._split(y)
        s, t = self.net(y1).chunk(2, dim=1)
        s = torch.tanh(s)
        return self._merge(y1, (y2 - t) * torch.exp(-s)), -s.flatten(1).sum(1)


class SpatialFlow(nn.Module):
    def __init__(self, n_layers=4, ch=128, hidden=128):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(SpatialActNorm(ch))
            self.layers.append(SpatialCoupling(ch, hidden, swap=(i % 2 == 1)))

    def forward(self, x):
        ld = torch.zeros(x.shape[0], device=x.device)
        for l in self.layers:
            x, d = l(x); ld = ld + d
        return x, ld

    def inverse(self, u):
        ld = torch.zeros(u.shape[0], device=u.device)
        for l in reversed(self.layers):
            u, d = l.inverse(u); ld = ld + d
        return u, ld


