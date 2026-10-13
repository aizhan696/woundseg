"""Compare your runs with the paper's DFUTissue scores (WoundFormer paper, arXiv:2605.19868, Table 3).

For every setting (without / with augmentation) it prints one table: the paper's WoundFormer and DFUTissueSegNet rows,
then for each of your models the best single run (the seed with the highest Avg (4)) and the mean ± SD over the seeds
found, and finally the seed-paired difference of each model against your WoundFormer (seeds improved / seeds).

Columns: Gran, Fibrin, Callus, Back (per-class Dice %), Avg (4) = mean of the four (the paper's "Avg."),
Tissue (3) = mean of fibrin, granulation and callus, FG = wound vs background Dice (all tissues merged).
Your numbers are test Dice at 224 x 224, accumulated over the whole test set (dataset-level), as in train.py.

  python compare_with_paper.py --runs-root runs --seeds 0 1 2 3 4 --out runs/comparison.md
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

COLS = ["Granulation", "Fibrin", "Callus", "Background", "Avg", "Tissue", "FG"]
HEAD = ["Gran", "Fibrin", "Callus", "Back", "Avg (4)", "Tissue (3)", "FG"]
SETTINGS = {"noaug": "without augmentation", "aug": "with augmentation"}
LABELS = {"woundformer": "WoundFormer", "dfutissuesegnet": "DFUTissueSegNet", "woundformer_gl": "WoundFormer-GL"}
# Table 3 of the paper (single runs): Gran, Fibrin, Callus, Back, Avg
PAPER = {
    "noaug": {"WoundFormer": [82.45, 69.32, 86.71, 74.88, 78.34], "DFUTissueSegNet": [90.75, 68.40, 82.31, 76.50, 79.49]},
    "aug": {"WoundFormer": [88.27, 70.75, 87.42, 95.13, 85.40], "DFUTissueSegNet": [91.99, 74.52, 85.91, 95.90, 87.08]},
}


def paper_row(v: List[float]) -> Dict[str, float]:
    return {"Granulation": v[0], "Fibrin": v[1], "Callus": v[2], "Background": v[3], "Avg": v[4],
            "Tissue": float(np.mean(v[:3])), "FG": float("nan")}


def load_runs(runs_root: str, model: str, setting: str, seeds, level: str = "dataset", res: str = "test_224"):
    """{seed: {column: Dice %}} for the finished runs <runs_root>/<model>_<setting>_s<seed>/results.json."""
    out = {}
    for s in seeds:
        f = os.path.join(runs_root, f"{model}_{setting}_s{s}", "results.json")
        if os.path.exists(f):
            d = json.load(open(f))[res][level]
            out[s] = {c: 100 * d[c] for c in COLS}
    return out


def fmt_row(name: str, runs: str, vals, sds=None) -> str:
    cells = []
    for i, c in enumerate(COLS):
        v = vals[c] if isinstance(vals, dict) else vals[i]
        if v != v:  # nan
            cells.append("–")
        elif sds is not None:
            cells.append(f"{v:.2f} ± {sds[i]:.2f}")
        else:
            cells.append(f"{v:.2f}")
    return f"| {name} | {runs} | " + " | ".join(cells) + " |"


def compare(runs_root: str = "runs", models=("woundformer", "dfutissuesegnet", "woundformer_gl"),
            settings=("noaug", "aug"), seeds=(0, 1, 2, 3, 4), level: str = "dataset", res: str = "test_224") -> str:
    lines = [f"Test Dice (%), {level}-level, {res}. Paper rows = single runs of Table 3; best run = seed with the "
             f"highest Avg (4); mean ± SD over the seeds found."]
    for setting in settings:
        lines += [f"\n### DFUTissue, {SETTINGS[setting]}\n", "| Model | runs | " + " | ".join(HEAD) + " |",
                  "|" + "---|" * (len(HEAD) + 2)]
        for name, v in PAPER[setting].items():
            lines.append(fmt_row(f"{name} (paper)", "1", paper_row(v)))
        found = {}
        for m in models:
            R = load_runs(runs_root, m, setting, seeds, level, res)
            if not R:
                continue
            found[m] = R
            A = np.array([[R[s][c] for c in COLS] for s in sorted(R)])
            best = max(R, key=lambda s: R[s]["Avg"])
            lines.append(fmt_row(f"{LABELS[m]} (ours), best run (seed {best})", "1", R[best]))
            if len(R) > 1:
                lines.append(fmt_row(f"{LABELS[m]} (ours), seeds {','.join(map(str, sorted(R)))}", str(len(R)),
                                     A.mean(0), A.std(0, ddof=1)))
        ref = found.get("woundformer")
        others = [m for m in found if m != "woundformer"]
        if ref and others:
            lines += ["", f"Seed-paired difference vs your WoundFormer ({SETTINGS[setting]}): mean Δ (seeds improved)",
                      "", "| Model | shared seeds | " + " | ".join(HEAD) + " |", "|" + "---|" * (len(HEAD) + 2)]
            for m in others:
                shared = sorted(set(found[m]) & set(ref))
                if not shared:
                    continue
                d = np.array([[found[m][s][c] - ref[s][c] for c in COLS] for s in shared])
                cells = [f"{x.mean():+.2f} ({(x > 0).sum()}/{len(x)})" for x in d.T]
                lines.append(f"| {LABELS[m]} | {','.join(map(str, shared))} | " + " | ".join(cells) + " |")
        for m in found:  # gap to the paper's numbers of the same model, where the paper has them
            if LABELS[m] in PAPER[setting]:
                p = paper_row(PAPER[setting][LABELS[m]])
                A = np.array([[found[m][s][c] for c in COLS] for s in sorted(found[m])]).mean(0)
                lines.append(f"\n{LABELS[m]}: your mean minus the paper's run: " + ", ".join(
                    f"{h} {a - p[c]:+.2f}" for h, c, a in zip(HEAD, COLS, A) if p[c] == p[c]))
    return "\n".join(lines) + "\n"


def summary_rows(runs_root: str = "runs", models=("woundformer", "dfutissuesegnet", "woundformer_gl"),
                 settings=("noaug", "aug"), seeds=(0, 1, 2, 3, 4)) -> List[dict]:
    """Long-format rows {setting, model, seed, metric, dice} incl. the paper rows (seed = 'paper'). Plain Python, no
    pandas (pandas 3 + PyArrow can crash next to PyTorch on Windows)."""
    rows = []
    for setting in settings:
        for name, v in PAPER[setting].items():
            for c, x in paper_row(v).items():
                rows.append({"setting": setting, "model": f"{name} (paper)", "seed": "paper", "metric": c, "dice": x})
        for m in models:
            for s, d in load_runs(runs_root, m, setting, seeds).items():
                for c, x in d.items():
                    rows.append({"setting": setting, "model": LABELS[m], "seed": s, "metric": c, "dice": x})
    return rows


def per_seed_markdown(runs_root: str = "runs", models=("woundformer", "dfutissuesegnet", "woundformer_gl"),
                      settings=("noaug", "aug"), seeds=(0, 1, 2, 3, 4)) -> str:
    """One row per finished run: setting, model, seed and the test Dice columns."""
    lines = ["| setting | model | seed | " + " | ".join(HEAD) + " |", "|" + "---|" * (len(HEAD) + 3)]
    for setting in settings:
        for m in models:
            for s, d in sorted(load_runs(runs_root, m, setting, seeds).items()):
                lines.append(f"| {setting} | {LABELS[m]} | {s} | " + " | ".join(f"{d[c]:.2f}" for c in COLS) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--models", nargs="+", default=["woundformer", "dfutissuesegnet", "woundformer_gl"])
    p.add_argument("--settings", nargs="+", default=["noaug", "aug"], choices=list(SETTINGS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--level", default="dataset", choices=["dataset", "image"])
    p.add_argument("--res", default="test_224", choices=["test_224", "test_native"])
    p.add_argument("--out", default=None, help="also write the tables to this markdown file")
    a = p.parse_args()
    txt = compare(a.runs_root, a.models, a.settings, a.seeds, a.level, a.res)
    print(txt)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(txt)


if __name__ == "__main__":
    main()
