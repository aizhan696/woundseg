"""WoundFormer-GL (global + local tiles): one shared MiT-B5 encoder looks at the full image (global branch) and at a
fixed 2 x 2 grid of overlapping, magnified tiles (local branch); the global features are fused into every tile
(cross-view fusion), each branch has a WoundFormer head, and the class probabilities are averaged per pixel.

  global : F^g = MiT-B5(I);                logits_g = Head_g(F^g)
  local  : T_k = grid_sample(I, box_k)     (4 tiles, each 60% of the side, resampled to 224: ~1.7x magnification)
           F^t_k = MiT-B5(T_k)             (same encoder weights; all views in one batched call)
  fusion : F^t_k,i <- F^t_k,i + BN(Conv1x1([F^t_k,i, warp(F^g_i, box_k)]))   at every scale i (BN starts at 0)
           logits_t_k = Head_t(F^t_k)
  output : p = (p_g + sum_k W_k^-1(p_t_k)) / (1 + sum_k m_k)      (m_k = 1 inside tile k)
  loss   : CE(output) + aux_weight * (CE(global) + CE(tiles))

The views do not depend on a prediction, so inference is one batched encoder call of 5 images per input image.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mit import mit_b5
from .woundformer import MultiScaleAggregationHead, load_mit_b5_imagenet


def _theta(box: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Affine matrices for boxes (cx, cy, half_w, half_h) in normalised [-1, 1] coordinates.
    forward: output (crop) coordinate u -> image coordinate c + s*u;  inverse: image u -> crop (u - c)/s."""
    cx, cy, sx, sy = box.unbind(1)
    z = torch.zeros_like(cx)
    if inverse:
        return torch.stack([torch.stack([1 / sx, z, -cx / sx], 1), torch.stack([z, 1 / sy, -cy / sy], 1)], 1)
    return torch.stack([torch.stack([sx, z, cx], 1), torch.stack([z, sy, cy], 1)], 1)


def warp(t: torch.Tensor, box: torch.Tensor, size, inverse: bool = False, mode: str = "bilinear") -> torch.Tensor:
    grid = F.affine_grid(_theta(box, inverse), (t.shape[0], t.shape[1], *size), align_corners=False)
    return F.grid_sample(t, grid, mode=mode, padding_mode="zeros", align_corners=False)


def tile_boxes(n: int = 2, overlap: float = 0.1) -> torch.Tensor:
    """n x n fixed square tiles covering the image, each (1/n + overlap) of the side: (n*n, 4) boxes (cx, cy, h, h)."""
    half = 1.0 / n + overlap  # half-size in [-1, 1] units equals the side fraction
    centers = torch.linspace(-1 + half, 1 - half, n)
    return torch.tensor([[cx, cy, half, half] for cy in centers.tolist() for cx in centers.tolist()])


class CrossViewFusion(nn.Module):
    """F^t <- F^t + BN(Conv1x1([F^t, G])); BN weight starts at 0 (identity at initialisation)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        nn.init.zeros_(self.bn.weight)

    def forward(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return z + self.bn(self.conv(torch.cat([z, g], dim=1)))


class WoundFormerGL(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, drop_path_rate: float = 0.1, channels: int = 128,
                 grid: int = 2, overlap: float = 0.1, size: int = 224, fuse_features: bool = True):
        super().__init__()
        self.encoder = mit_b5(drop_path_rate=drop_path_rate)
        if pretrained:
            load_mit_b5_imagenet(self.encoder)
        enc_ch = (64, 128, 320, 512)
        self.head_g = MultiScaleAggregationHead(enc_ch, channels, num_classes)
        self.head_t = MultiScaleAggregationHead(enc_ch, channels, num_classes)
        self.fusion = nn.ModuleList(CrossViewFusion(c) for c in enc_ch) if fuse_features else None
        self.register_buffer("boxes", tile_boxes(grid, overlap), persistent=False)
        self.size = size

    def views(self, x: torch.Tensor):
        B, T = x.shape[0], self.boxes.shape[0]
        box = self.boxes.repeat(B, 1)                                  # (B*T, 4), tile-major within each image
        tiles = warp(x.repeat_interleave(T, 0), box, (self.size, self.size))
        feats = self.encoder(torch.cat([x, tiles], 0))                 # one batched pass: B globals + B*T tiles
        fg, ft = [f[:B] for f in feats], [f[B:] for f in feats]
        if self.fusion is not None:
            ft = [fuse(f, warp(g.repeat_interleave(T, 0), box, f.shape[-2:])) for fuse, f, g in zip(self.fusion, ft, fg)]
        return self.head_g(fg), self.head_t(ft), box

    def combine(self, logits_g, logits_t, box, size):
        B = logits_g.shape[0]
        p = F.interpolate(logits_g, size=size, mode="bilinear", align_corners=False).softmax(1)
        pt = F.interpolate(logits_t, size=(self.size, self.size), mode="bilinear", align_corners=False).softmax(1)
        pt_i = warp(pt, box, size, inverse=True)                      # (B*T, K, H, W), 0 outside each tile
        m = warp(torch.ones_like(pt[:, :1]), box, size, inverse=True)
        T = pt.shape[0] // B
        num = p + pt_i.view(B, T, *pt_i.shape[1:]).sum(1)
        den = 1 + m.view(B, T, *m.shape[1:]).sum(1)
        return torch.log(num / den + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Log-probabilities at the input size (cross-entropy on them equals the usual CE)."""
        logits_g, logits_t, box = self.views(x)
        return self.combine(logits_g, logits_t, box, x.shape[-2:])

    def training_loss(self, x: torch.Tensor, y: torch.Tensor, criterion, aux_weight: float = 0.4) -> torch.Tensor:
        logits_g, logits_t, box = self.views(x)
        loss = criterion(self.combine(logits_g, logits_t, box, y.shape[-2:]), y)
        loss_g = criterion(F.interpolate(logits_g, size=y.shape[-2:], mode="bilinear", align_corners=False), y)
        T = box.shape[0] // x.shape[0]
        yt = warp(y.repeat_interleave(T, 0)[:, None].float() + 1, box, (self.size, self.size), mode="nearest")[:, 0].long() - 1
        lt = F.interpolate(logits_t, size=(self.size, self.size), mode="bilinear", align_corners=False)
        valid = yt >= 0
        loss_t = F.cross_entropy(lt.permute(0, 2, 3, 1)[valid], yt[valid]) if valid.any() else lt.sum() * 0
        return loss + aux_weight * (loss_g + loss_t)
