"""Figures from finished runs (uses the predictions train.py saves in <run>/test_pred/index).

  qualitative  input | ground truth | one column per run, for chosen test images
  curves       validation Dice and train / validation loss per epoch
  bars         per-class test Dice (mean ± SD over seeds) of your models next to the paper's numbers

  python visualize.py qualitative --data-root /path/DFUTissue --runs runs/woundformer_noaug_s0 runs/woundformer_gl_noaug_s0 \
         --labels WoundFormer WoundFormer-GL --out figures/qualitative.png
  python visualize.py curves --runs runs/woundformer_noaug_s0 runs/woundformer_gl_noaug_s0 --out figures/curves.png
  python visualize.py bars --runs-root runs --setting noaug --seeds 0 1 2 3 4 --out figures/bars_noaug.png
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import List, Optional, Sequence

import cv2
import matplotlib
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from woundseg.data import DFUTISSUE, dfutissue_samples, read_index_mask, read_rgb  # noqa: E402

PAL = np.array([DFUTISSUE.palette[c] for c in DFUTISSUE.class_names], np.uint8)
DEFAULT_IMAGES = ["0914", "0927", "0961", "0994"]  # three with fibrin, one hard callus case


def qualitative(runs: Sequence[str], labels: Optional[Sequence[str]], data_root: str, names: Sequence[str] = None,
                variant: str = "Original", out: Optional[str] = None):
    test = {s.name: s for s in dfutissue_samples(data_root, variant)["test"]}
    names = list(names or [n for n in DEFAULT_IMAGES if n in test] or list(test)[:4])
    labels = list(labels or [os.path.basename(r.rstrip("/\\")) for r in runs])
    cols = 2 + len(runs)
    fig, axes = plt.subplots(len(names), cols, figsize=(2.3 * cols, 2.3 * len(names) + 0.6))
    axes = np.atleast_2d(axes)
    for r, n in enumerate(names):
        s = test[n]
        panels = [read_rgb(s.image), PAL[read_index_mask(s.label)]]
        for run in runs:
            pred = cv2.imread(os.path.join(run, "test_pred", "index", n + ".png"), cv2.IMREAD_GRAYSCALE)
            panels.append(PAL[pred] if pred is not None else np.full_like(panels[0], 255))
        for c, im in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(["Input", "Ground truth"][c] if c < 2 else labels[c - 2], fontsize=9)
            if c == 0:
                ax.set_ylabel(n, fontsize=8)
    handles = [Patch(facecolor=np.array(DFUTISSUE.palette[c]) / 255, edgecolor="k", label=c) for c in DFUTISSUE.class_names]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out)
    return fig


def _read_log(run: str):
    with open(os.path.join(run, "log.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def curves(runs: Sequence[str], labels: Optional[Sequence[str]] = None, out: Optional[str] = None):
    labels = list(labels or [os.path.basename(r.rstrip("/\\")) for r in runs])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for run, lab in zip(runs, labels):
        L = _read_log(run)
        if not L:
            continue
        line, = a1.plot(L["epoch"], 100 * L["val_dice_avg"], lw=2, label=lab)
        a2.plot(L["epoch"], L["train_loss"], lw=2, color=line.get_color(), label=f"{lab} train")
        a2.plot(L["epoch"], L["val_loss"], lw=2, ls="--", color=line.get_color(), label=f"{lab} val")
    a1.set(xlabel="epoch", ylabel="validation Dice, Avg (4) %", title="Validation Dice")
    a2.set(xlabel="epoch", ylabel="cross-entropy", title="Loss (solid train, dashed validation)")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out)
    return fig


def bars(runs_root: str = "runs", setting: str = "noaug", seeds=(0, 1, 2, 3, 4),
         models=("woundformer", "dfutissuesegnet", "woundformer_gl"), out: Optional[str] = None):
    from compare_with_paper import summary_rows

    rows = summary_rows(runs_root, models, [setting], seeds)
    metrics = ["Granulation", "Fibrin", "Callus", "Background", "Avg", "Tissue"]
    groups = list(dict.fromkeys(r["model"] for r in rows))
    colour = {"WoundFormer": "#2a78d6", "DFUTissueSegNet": "#eb6834", "WoundFormer-GL": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(11, 4))
    w = 0.8 / max(1, len(groups))
    lo = 100.0
    for k, g in enumerate(groups):
        vals = {c: np.array([r["dice"] for r in rows if r["model"] == g and r["metric"] == c]) for c in metrics}
        m = [float(vals[c].mean()) for c in metrics]
        sd = [float(vals[c].std(ddof=1)) if len(vals[c]) > 1 else 0.0 for c in metrics]
        lo = min(lo, min(m))
        x = np.arange(len(metrics)) + (k - (len(groups) - 1) / 2) * w
        paper = "paper" in g
        ax.bar(x, m, width=w * 0.9, yerr=sd, capsize=2, label=g, color=colour.get(g.replace(" (paper)", ""), "#898781"),
               hatch="//" if paper else None, alpha=0.45 if paper else 1.0, edgecolor="white" if paper else None)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(["Gran", "Fibrin", "Callus", "Back", "Avg (4)", "Tissue (3)"])
    ax.set_ylim(max(0, 10 * int((lo - 5) // 10)), 100)
    ax.set_ylabel("test Dice %")
    ax.set_title(f"DFUTissue, {'with' if setting == 'aug' else 'without'} augmentation (mean ± SD over seeds; "
                 f"hatched = paper)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out)
    return fig


def _save(fig, out):
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out, dpi=150)
        print("saved", out)


def main():
    matplotlib.use("Agg")  # command line: write files only (notebooks keep their inline backend)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qualitative")
    q.add_argument("--runs", nargs="+", required=True)
    q.add_argument("--labels", nargs="+")
    q.add_argument("--data-root", required=True)
    q.add_argument("--variant", default="Original")
    q.add_argument("--images", nargs="+", help="test image names (default 0914 0927 0961 0994)")
    q.add_argument("--out", default="figures/qualitative.png")
    c = sub.add_parser("curves")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--labels", nargs="+")
    c.add_argument("--out", default="figures/curves.png")
    b = sub.add_parser("bars")
    b.add_argument("--runs-root", default="runs")
    b.add_argument("--setting", default="noaug", choices=["noaug", "aug"])
    b.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    b.add_argument("--out", default="figures/bars.png")
    a = p.parse_args()
    if a.cmd == "qualitative":
        qualitative(a.runs, a.labels, a.data_root, a.images, a.variant, a.out)
    elif a.cmd == "curves":
        curves(a.runs, a.labels, a.out)
    else:
        bars(a.runs_root, a.setting, a.seeds, out=a.out)


if __name__ == "__main__":
    main()
