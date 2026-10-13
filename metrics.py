"""Dice similarity coefficient (DSC) and the two-sided Wilcoxon signed-rank test (Sec. 3).

The paper reports per-class DSC (%) and their average, but does not state whether DSC is
accumulated over the whole test set or averaged per image.  Both are computed:

* ``dataset``: TP/FP/FN of each class summed over all test images, DSC = 2TP / (2TP+FP+FN).
* ``image``:   DSC per image and class, averaged over the images in which the class occurs in
               the ground truth or the prediction (images where it is absent from both are skipped).

The per-image scores are also what the paired Wilcoxon test needs.

Besides the paper's "Avg", two summaries are added for the architecture study:
* ``Tissue``: mean DSC of the report classes without Background (DFUTissue: fibrin, granulation, callus).
* ``FG``:     binary foreground DSC, all tissue classes merged vs. Background (wound-region Dice).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats


class DiceMeter:
    def __init__(self, class_names: Sequence[str], report_classes: Sequence[str]):
        self.class_names = list(class_names)
        self.report_idx = [self.class_names.index(c) for c in report_classes]
        self.report_classes = list(report_classes)
        k = len(self.class_names)
        self.tp = np.zeros(k, np.int64)
        self.fp = np.zeros(k, np.int64)
        self.fn = np.zeros(k, np.int64)
        self.fg = np.zeros(3, np.int64)  # binary foreground TP, FP, FN (class 0 = Background)
        self.tissue = [c for c in self.report_classes if c != "Background"]
        self.per_image: List[Dict] = []

    def update(self, pred: np.ndarray, gt: np.ndarray, name: str = "") -> None:
        k = len(self.class_names)
        cm = np.bincount(gt.reshape(-1) * k + pred.reshape(-1), minlength=k * k).reshape(k, k)
        tp = np.diag(cm)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        self.tp += tp
        self.fp += fp
        self.fn += fn
        fg = np.array([cm[1:, 1:].sum(), cm[0, 1:].sum(), cm[1:, 0].sum()])
        self.fg += fg
        dice = {}
        for c in range(k):
            denom = 2 * tp[c] + fp[c] + fn[c]
            dice[self.class_names[c]] = float(2 * tp[c] / denom) if denom > 0 else float("nan")
        denom = 2 * fg[0] + fg[1] + fg[2]
        dice["FG"] = float(2 * fg[0] / denom) if denom > 0 else float("nan")
        present = [dice[c] for c in self.report_classes if not math.isnan(dice[c])]
        self.per_image.append({"name": name, "dice": dice,
                               "mean_dice": float(np.mean(present)) if present else float("nan")})

    def summary(self) -> Dict:
        out = {"dataset": {}, "image": {}}
        for c in range(len(self.class_names)):
            name = self.class_names[c]
            denom = 2 * self.tp[c] + self.fp[c] + self.fn[c]
            out["dataset"][name] = float(2 * self.tp[c] / denom) if denom > 0 else float("nan")
            vals = [d["dice"][name] for d in self.per_image if not math.isnan(d["dice"][name])]
            out["image"][name] = float(np.mean(vals)) if vals else float("nan")
        denom = 2 * self.fg[0] + self.fg[1] + self.fg[2]
        out["dataset"]["FG"] = float(2 * self.fg[0] / denom) if denom > 0 else float("nan")
        vals = [d["dice"]["FG"] for d in self.per_image if not math.isnan(d["dice"]["FG"])]
        out["image"]["FG"] = float(np.mean(vals)) if vals else float("nan")
        for mode in ("dataset", "image"):
            vals = [out[mode][c] for c in self.report_classes]
            out[mode]["Avg"] = float(np.nanmean(vals))
            out[mode]["Tissue"] = float(np.nanmean([out[mode][c] for c in self.tissue]))
        # mean IoU over ALL classes (incl. background), accumulated over the set: checkpoint criterion
        union = self.tp + self.fp + self.fn
        out["iou"] = {n: float(self.tp[c] / union[c]) if union[c] > 0 else float("nan")
                      for c, n in enumerate(self.class_names)}
        out["miou"] = float(np.nanmean(list(out["iou"].values())))
        out["per_image"] = self.per_image
        return out


def wilcoxon_test(a: Sequence[float], b: Sequence[float]) -> Dict:
    """Two-sided Wilcoxon signed-rank test on paired per-image scores, with effect size
    r = |Z| / sqrt(N) (N = number of pairs), as reported in the paper (p, r)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    res = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", method="approx")
    z = float(res.zstatistic)
    return {"n": int(len(a)), "statistic": float(res.statistic), "p": float(res.pvalue),
            "z": z, "r": abs(z) / math.sqrt(len(a)),
            "mean_a": float(a.mean()), "mean_b": float(b.mean())}
