"""DFUTissue data: official split, 224 x 224 resizing, ImageNet normalisation, and the DFUTissueSegNet augmentation.

Expected layout (DFUTissue release, github.com/uwm-bigdata/DFUTissueSegNet, folder DFUTissue):

  <DATA_ROOT>/Labeled/labeled_train_names.txt   (78 names)
  <DATA_ROOT>/Labeled/labeled_val_names.txt     (16 names)
  <DATA_ROOT>/Labeled/test_names.txt            (16 names)
  <DATA_ROOT>/Labeled/Original/Images/{TrainVal,Test}/<name>.png
  <DATA_ROOT>/Labeled/Original/Annotations/{TrainVal,Test}/<name>.png   (class index masks: 0 bg, 1 fibrin,
                                                                          2 granulation, 3 callus)
  (and the same under Labeled/Padded/ for the 256 x 256 padded release)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as albu  # noqa: E402

cv2.setNumThreads(0)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class DatasetSpec:
    name: str
    class_names: List[str]          # index -> name, index 0 = background
    report_classes: List[str]       # classes (and order) averaged in the paper's "Avg." column
    palette: Dict[str, Tuple[int, int, int]]


DFUTISSUE = DatasetSpec(
    name="dfutissue",
    class_names=["Background", "Fibrin", "Granulation", "Callus"],
    # Table 3 of the WoundFormer paper: Gran, Fibrin, Callus, Back -> Avg. over all 4 classes (incl. background)
    report_classes=["Granulation", "Fibrin", "Callus", "Background"],
    palette={"Background": (0, 0, 0), "Fibrin": (255, 0, 0), "Granulation": (0, 255, 0), "Callus": (0, 0, 255)},
)


def dfutissuesegnet_augmentation(seed: Optional[int] = None) -> albu.Compose:
    """`get_training_augmentation()` of DFUTissueSegNet (written for albumentations 1.x). All albumentations-2.x
    defaults used here equal the 1.x defaults except GaussNoise, which is pinned to the 1.x variance range 10-50."""
    gauss_std = (math.sqrt(10.0) / 255.0, math.sqrt(50.0) / 255.0)
    train_transform = [
        albu.OneOf([albu.HorizontalFlip(p=0.5), albu.VerticalFlip(p=0.5)], p=0.8),
        albu.OneOf(
            [
                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0, p=0.1, border_mode=0),
                albu.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0, p=0.1, border_mode=0),
                albu.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=0.6, border_mode=0),
                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=0.2, border_mode=0),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.Perspective(p=0.2),
                albu.GaussNoise(std_range=gauss_std, p=0.2),
                albu.Sharpen(p=0.2),
                albu.Blur(blur_limit=3, p=0.2),
                albu.MotionBlur(blur_limit=3, p=0.2),
            ],
            p=0.5,
        ),
        albu.OneOf(
            [
                albu.CLAHE(p=0.25),
                albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
                albu.RandomGamma(p=0.25),
                albu.HueSaturationValue(p=0.25),
            ],
            p=0.3,
        ),
    ]
    return albu.Compose(train_transform, p=0.9, seed=seed)


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_index_mask(path: str) -> np.ndarray:
    """Single-channel class-index mask (Padded annotations are stored as 3 identical channels)."""
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        if not (np.array_equal(m[..., 0], m[..., 1]) and np.array_equal(m[..., 0], m[..., 2])):
            raise ValueError(f"{path}: multi-channel index mask with differing channels")
        m = m[..., 0]
    return m.astype(np.int64)


@dataclass
class Sample:
    name: str
    image: str
    label: str


def dfutissue_samples(root: str, variant: str = "Original") -> Dict[str, List[Sample]]:
    """Official DFUTissue split (78 train / 16 val / 16 test) from Labeled/*_names.txt."""
    lab = os.path.join(root, "Labeled")

    def names(f):
        with open(os.path.join(lab, f)) as fh:
            return [l.strip() for l in fh if l.strip()]

    def mk(ids, sub):
        return [Sample(i, os.path.join(lab, variant, "Images", sub, i + ".png"),
                       os.path.join(lab, variant, "Annotations", sub, i + ".png")) for i in ids]

    return {"train": mk(names("labeled_train_names.txt"), "TrainVal"),
            "val": mk(names("labeled_val_names.txt"), "TrainVal"),
            "test": mk(names("test_names.txt"), "Test")}


class SegDataset(Dataset):
    """Loads every sample once (the dataset is tiny), resizes to `size` (image bilinear, mask nearest) and keeps it in
    RAM. Train: optional augmentation after resizing, then ImageNet normalisation. The native mask is kept too."""

    def __init__(self, samples: Sequence[Sample], spec: DatasetSpec, size: int = 224,
                 augment: Optional[albu.Compose] = None):
        self.samples, self.spec, self.size, self.augment = list(samples), spec, size, augment
        self.images, self.masks, self.native_masks = [], [], []
        for s in self.samples:
            img, mask = read_rgb(s.image), read_index_mask(s.label)
            if mask.shape[:2] != img.shape[:2]:
                raise ValueError(f"{s.name}: image {img.shape[:2]} vs mask {mask.shape[:2]}")
            if mask.max() >= len(spec.class_names):
                raise ValueError(f"{s.name}: label value {mask.max()} out of range")
            self.native_masks.append(mask.astype(np.uint8))
            self.images.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR))
            self.masks.append(cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img, mask = self.images[i], self.masks[i]
        if self.augment is not None:
            out = self.augment(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(mask.astype(np.int64)), i
