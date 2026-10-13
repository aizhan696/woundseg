"""DFUTissueSegNet (Dhar et al., arXiv:2406.16012; github.com/uwm-bigdata/DFUTissueSegNet), re-implemented on top of
segmentation_models_pytorch 0.5 from the authors' released code (Codes/supervised_training.ipynb and their modified
smp package):

    model = smp.Unet(encoder_name="mit_b3", encoder_weights="imagenet", classes=4, decoder_attention_type="pscse")

"pscse" is the authors' parallel scSE decoder block (their `decoders/unet/decoder.py`, active `DecoderBlock`):

    x  = Upsample_x2(x)  (nearest);  x = [x, skip]
    x1 = scSE_maxout(x)    = max(x * cSE(x), x * sSE(x))
    x2 = scSE_addition(x)  = x * cSE(x) + x * sSE(x)
    x  = x1 + x2           (block without skip:  x = x + scSE_addition(x))
    out = Conv3x3-BN-ReLU(x)          (their conv2 is created but never used; kept here for the parameter count)

    cSE = Sigmoid(Conv1x1(ReLU(Conv1x1(AvgPool(x), C -> C/16)), C/16 -> C)),  sSE = Sigmoid(Conv1x1(x, C -> 1))

As in the WoundFormer paper's comparison, the model outputs logits and is trained with the same unified protocol as
WoundFormer (train.py). The authors' own recipe (sigmoid + Dice + Focal loss, 256 px, batch 16, patience 50) is not
used here.
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSE(nn.Module):
    """Concurrent spatial and channel squeeze & excitation with an 'addition' or 'maxout' combination."""

    def __init__(self, in_channels: int, strategy: str, reduction: int = 16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, in_channels // reduction, 1),
                                 nn.ReLU(inplace=True), nn.Conv2d(in_channels // reduction, in_channels, 1), nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())
        self.strategy = strategy

    def forward(self, x):
        xc, xs = x * self.cSE(x), x * self.sSE(x)
        return torch.maximum(xc, xs) if self.strategy == "maxout" else xc + xs


class Conv2dReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class PSCSEDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        c = in_channels + skip_channels
        self.conv1 = Conv2dReLU(c, out_channels)
        self.attention1 = SCSE(c, "maxout")
        self.conv2 = Conv2dReLU(out_channels, out_channels)  # unused in the authors' forward (kept for parity)
        self.attention2 = SCSE(c, "addition")

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x) + self.attention2(x)
        else:
            x = x + self.attention2(x)
        return self.conv1(x)


class PSCSEUnetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()
        enc = list(encoder_channels)[1:][::-1]          # drop the input, deepest first: 512, 320, 128, 64, 0
        ins = [enc[0]] + list(decoder_channels)[:-1]
        skips = enc[1:] + [0]
        self.blocks = nn.ModuleList(PSCSEDecoderBlock(i, s, o) for i, s, o in zip(ins, skips, decoder_channels))

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        features = features[1:][::-1]
        x, skips = features[0], features[1:]
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i] if i < len(skips) else None)
        return x


class DFUTissueSegNet(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True, encoder: str = "mit_b3"):
        super().__init__()
        import segmentation_models_pytorch as smp
        from segmentation_models_pytorch.base import initialization as init

        local = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", f"{encoder}.pth")
        use_local = pretrained and os.path.exists(local)
        self.encoder = smp.encoders.get_encoder(encoder, in_channels=3, depth=5,
                                                weights=("imagenet" if pretrained and not use_local else None))
        if use_local:  # official SegFormer ImageNet-1K release file, same weights as smp's 'imagenet'
            sd = torch.load(local, map_location="cpu", weights_only=False)
            sd = {k: v for k, v in sd.get("state_dict", sd).items() if not k.startswith("head.")}
            self.encoder.load_state_dict(sd)
        self.decoder = PSCSEUnetDecoder(self.encoder.out_channels)
        self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=3, padding=1)  # smp SegmentationHead
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits at the input resolution."""
        return self.segmentation_head(self.decoder(self.encoder(x)))
