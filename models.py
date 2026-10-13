"""Model registry: woundformer | dfutissuesegnet | woundformer_gl."""
from __future__ import annotations

import torch
import torch.nn as nn

MODELS = ["woundformer", "dfutissuesegnet", "woundformer_gl"]


def build_model(name: str, num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    if name == "woundformer":
        from .woundformer import WoundFormer
        return WoundFormer(num_classes, pretrained)
    if name == "dfutissuesegnet":
        from .dfutissuesegnet import DFUTissueSegNet
        return DFUTissueSegNet(num_classes, pretrained)
    if name == "woundformer_gl":
        from .woundformer_gl import WoundFormerGL
        return WoundFormerGL(num_classes, pretrained)
    raise ValueError(f"unknown model {name}; choose from {MODELS}")


class CheckpointedStage(nn.Module):
    """Runs an MiT stage (nn.Sequential of blocks) with activation checkpointing during training. Saves GPU memory
    only; the computation (and, with preserved RNG state, stochastic depth) is unchanged."""

    def __init__(self, blocks: nn.Sequential, segments: int = 2):
        super().__init__()
        self.blocks, self.segments = blocks, segments

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint_sequential
            return checkpoint_sequential(self.blocks, min(self.segments, len(self.blocks)), x, use_reentrant=False)
        return self.blocks(x)


def enable_grad_checkpointing(model: nn.Module) -> nn.Module:
    """Activation checkpointing in the four MiT stages (needed for WoundFormer-GL on an 8 GB GPU)."""
    enc = model.encoder
    for i in (1, 2, 3, 4):
        name = f"block{i}"
        if hasattr(enc, name):
            setattr(enc, name, CheckpointedStage(getattr(enc, name), segments=4 if i == 3 else 2))
    return model


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    decoder = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("encoder."))
    return total, decoder
