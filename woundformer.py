"""WoundFormer (Kabir & Dulal, arXiv:2605.19868): MiT-B5 encoder + spatially-preserving multi-scale aggregation head.

    F_i      : MiT-B5 stage outputs, strides 4/8/16/32, C_i = 64/128/320/512
    F^_i     = ReLU(BN(Conv1x1(F_i)))  -> 128 channels                       (channel alignment)
    X        = F^_4;  for i = 3,2,1:  X <- ReLU(BN(Conv1x1([F^_i, Up(X)])))  (coarse-to-fine fusion)
    X_fused  = Conv3x3(X)                                                    (spatial refinement)
    M        = Conv1x1(X_fused)  in R^{H/4 x W/4 x N_cls}                     (prediction layer)
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mit import mit_b5


class ConvNormAct(nn.Sequential):
    """ReLU(BN(Conv_kxk(x)))."""

    def __init__(self, cin: int, cout: int, kernel: int = 1):
        layers: List[nn.Module] = [nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
                                   nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        super().__init__(*layers)


class MultiScaleAggregationHead(nn.Module):
    """Spatially-preserving multi-scale aggregation head (Sec. 2.2 of the paper)."""

    def __init__(self, in_channels: Sequence[int] = (64, 128, 320, 512), channels: int = 128, num_classes: int = 4):
        super().__init__()
        self.align = nn.ModuleList([ConvNormAct(c, channels, 1) for c in in_channels])
        self.fuse = nn.ModuleList([ConvNormAct(2 * channels, channels, 1) for _ in range(len(in_channels) - 1)])
        self.refine = nn.Conv2d(channels, channels, 3, padding=1)  # plain 3x3 conv, as in the paper's equation
        self.cls = nn.Conv2d(channels, num_classes, kernel_size=1)

    def features(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        f = [align(x) for align, x in zip(self.align, feats)]
        x = f[-1]  # X = F^_4
        for stage, fuse in zip(range(len(f) - 2, -1, -1), self.fuse):  # i = 3, 2, 1
            x = F.interpolate(x, size=f[stage].shape[-2:], mode="bilinear", align_corners=False)
            x = fuse(torch.cat([f[stage], x], dim=1))
        return self.refine(x)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.cls(self.features(feats))  # H/4 x W/4 x N_cls


class WoundFormer(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, drop_path_rate: float = 0.1):
        super().__init__()
        self.encoder = mit_b5(drop_path_rate=drop_path_rate)
        if pretrained:
            load_mit_b5_imagenet(self.encoder)
        self.decode_head = MultiScaleAggregationHead((64, 128, 320, 512), 128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits at H/4 x W/4 (train.py upsamples them to the label size)."""
        return self.decode_head(self.encoder(x))


def default_mit_b5_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", "mit_b5.pth")


def load_mit_b5_imagenet(encoder: nn.Module, path: Optional[str] = None) -> None:
    """Official ImageNet-1K MiT-B5 weights (SegFormer release). Uses `path` / the torch-hub cache file mit_b5.pth if it
    exists, otherwise downloads the same weights through segmentation_models_pytorch (HuggingFace smp-hub)."""
    path = path or default_mit_b5_path()
    if os.path.exists(path):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
    else:
        import segmentation_models_pytorch as smp

        print("[weights] mit_b5.pth not found, downloading MiT-B5 ImageNet weights via segmentation_models_pytorch",
              flush=True)
        sd = smp.encoders.get_encoder("mit_b5", in_channels=3, depth=5, weights="imagenet").state_dict()
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}  # drop the ImageNet classifier
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"MiT-B5 weight mismatch. missing={missing} unexpected={unexpected}")


def upsample_logits(logits: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
