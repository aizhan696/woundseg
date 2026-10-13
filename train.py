"""Train, validate and test one model on DFUTissue (one seed) with the WoundFormer paper's unified protocol.

  * inputs resized to 224 x 224, ImageNet normalisation
  * Adam, lr 1e-4, batch size 8, cross-entropy loss
  * ReduceLROnPlateau(factor 0.1, patience 5) and early stopping (patience 15), both on the validation loss
  * at most 200 epochs; the checkpoint with the highest validation mIoU is kept and evaluated on the test set
  * optional DFUTissueSegNet augmentation (--aug)

Outputs in --out: log.csv (per epoch), best.pth, results.json (validation and test Dice, dataset-level and per image,
at 224 and at native resolution), test_pred/{index,color}/<name>.png.

  python train.py --model woundformer      --data-root /path/DFUTissue --seed 0
  python train.py --model dfutissuesegnet  --data-root /path/DFUTissue --seed 0 --aug
  python train.py --model woundformer_gl   --data-root /path/DFUTissue --seed 0      (activation checkpointing on)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from woundseg.data import DFUTISSUE, SegDataset, dfutissue_samples, dfutissuesegnet_augmentation  # noqa: E402
from woundseg.metrics import DiceMeter  # noqa: E402
from woundseg.models import MODELS, build_model, count_parameters, enable_grad_checkpointing  # noqa: E402
from woundseg.woundformer import upsample_logits  # noqa: E402


def run_name(model: str, aug: bool, seed: int) -> str:
    return f"{model}_{'aug' if aug else 'noaug'}_s{seed}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--data-root", required=True, help="DFUTissue folder that contains Labeled/")
    p.add_argument("--variant", default="Original", choices=["Original", "Padded"],
                   help="DFUTissue release: Original crops (default) or the 256 x 256 Padded images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aug", action="store_true", help="DFUTissueSegNet augmentation")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--out", default=None, help="run folder (default: <runs-root>/<model>_<aug|noaug>_s<seed>)")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-factor", type=float, default=0.1)
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--es-patience", type=int, default=15)
    p.add_argument("--aux-weight", type=float, default=0.4, help="WoundFormer-GL branch loss weight")
    p.add_argument("--grad-ckpt", choices=["auto", "on", "off"], default="auto",
                   help="activation checkpointing in the MiT stages (auto: on for woundformer_gl)")
    p.add_argument("--no-pretrained", action="store_true", help="random encoder initialisation (testing only)")
    p.add_argument("--no-ckpt", action="store_true", help="delete best.pth after testing (saves ~330 MB per run)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-deterministic", action="store_true")
    return p.parse_args(argv)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    tot, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        tot += criterion(upsample_logits(model(x), y.shape[-2:]), y).item() * x.size(0)
        n += x.size(0)
    return tot / n


@torch.no_grad()
def predict_and_score(model, ds: SegDataset, device, save_dir=None):
    """DiceMeter summaries at the network resolution (224) and at native label resolution."""
    spec = ds.spec
    model.eval()
    m224 = DiceMeter(spec.class_names, spec.report_classes)
    mnat = DiceMeter(spec.class_names, spec.report_classes)
    pal = np.array([spec.palette[c] for c in spec.class_names], np.uint8)
    for i in range(len(ds)):
        x, y, _ = ds[i]
        logits = model(x[None].to(device))
        p224 = upsample_logits(logits, y.shape[-2:]).argmax(1)[0].cpu().numpy()
        native = ds.native_masks[i]
        pnat = upsample_logits(logits, native.shape[:2]).argmax(1)[0].cpu().numpy()
        name = ds.samples[i].name
        m224.update(p224, y.numpy(), name)
        mnat.update(pnat, native.astype(np.int64), name)
        if save_dir:
            os.makedirs(os.path.join(save_dir, "index"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "color"), exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, "index", name + ".png"), pnat.astype(np.uint8))
            cv2.imwrite(os.path.join(save_dir, "color", name + ".png"), cv2.cvtColor(pal[pnat], cv2.COLOR_RGB2BGR))
    return m224.summary(), mnat.summary()


def main(argv=None):
    args = parse_args(argv)
    args.out = args.out or os.path.join(args.runs_root, run_name(args.model, args.aug, args.seed))
    os.makedirs(args.out, exist_ok=True)
    seed_everything(args.seed, not args.no_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = DFUTISSUE

    splits = dfutissue_samples(args.data_root, args.variant)
    aug = dfutissuesegnet_augmentation(seed=args.seed) if args.aug else None
    train_ds = SegDataset(splits["train"], spec, args.size, aug)
    val_ds = SegDataset(splits["val"], spec, args.size, None)
    test_ds = SegDataset(splits["test"], spec, args.size, None)
    print(f"[data] train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}  ({args.variant})", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model, len(spec.class_names), pretrained=not args.no_pretrained)
    if args.grad_ckpt == "on" or (args.grad_ckpt == "auto" and args.model == "woundformer_gl"):
        model = enable_grad_checkpointing(model)
    model = model.to(device)
    n_par, n_dec = count_parameters(model)
    print(f"[model] {args.model}: {n_par/1e6:.2f}M parameters (decoder/heads {n_dec/1e6:.2f}M)  device {device}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor,
                                                           patience=args.lr_patience)
    best_loss, wait = float("inf"), 0          # early stopping on the validation loss
    best_score, best_epoch = -float("inf"), -1  # checkpoint: highest validation mIoU
    ckpt = os.path.join(args.out, "best.pth")
    log_f = open(os.path.join(args.out, "log.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["epoch", "lr", "train_loss", "val_loss", "val_miou", "val_dice_avg", "val_dice_tissue", "seconds"])
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if hasattr(model, "training_loss"):  # WoundFormer-GL: fused + global + tile losses
                loss = model.training_loss(x, y, criterion, args.aux_weight)
            else:
                loss = criterion(upsample_logits(model(x), y.shape[-2:]), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = tot / n
        val_loss = evaluate_loss(model, val_dl, criterion, device)
        val_summary, _ = predict_and_score(model, val_ds, device)
        val_miou = val_summary["miou"]
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, wait = val_loss, 0
        else:
            wait += 1
        improved = val_miou > best_score
        if improved:
            best_score, best_epoch = val_miou, epoch
            torch.save(model.state_dict(), ckpt)
        dt = time.time() - t0
        log.writerow([epoch, lr, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_miou:.4f}",
                      f"{val_summary['dataset']['Avg']:.4f}", f"{val_summary['dataset']['Tissue']:.4f}", f"{dt:.1f}"])
        log_f.flush()
        print(f"ep {epoch:3d} lr {lr:.0e} train {train_loss:.4f} val {val_loss:.4f} mIoU {100*val_miou:.2f} "
              f"valDice {100*val_summary['dataset']['Avg']:.2f} {'*' if improved else ''} ({dt:.1f}s)", flush=True)
        if wait >= args.es_patience:
            print(f"[early stop] epoch {epoch}, best epoch {best_epoch}", flush=True)
            break
    log_f.close()
    train_time = time.time() - t_start

    model.load_state_dict(torch.load(ckpt, map_location=device))
    v224, vnat = predict_and_score(model, val_ds, device)
    s224, snat = predict_and_score(model, test_ds, device, os.path.join(args.out, "test_pred"))
    strip = lambda s: {k: v for k, v in s.items() if k != "per_image"}  # noqa: E731
    res = {"args": vars(args), "model": args.model, "aug": args.aug, "seed": args.seed,
           "best_epoch": best_epoch, "epochs_run": epoch, "best_val_loss": best_loss, "best_val_miou": best_score,
           "train_time_s": train_time, "params": n_par, "decoder_params": n_dec,
           "val_224": strip(v224), "val_native": strip(vnat), "test_224": s224, "test_native": snat}
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    if args.no_ckpt:
        os.remove(ckpt)

    def fmt(s):
        return ("  ".join(f"{c[:4]} {100*s[c]:.2f}" for c in spec.report_classes)
                + f"  | Avg {100*s['Avg']:.2f}  Tissue {100*s['Tissue']:.2f}  FG {100*s['FG']:.2f}")
    print("[val  224, dataset-level ] " + fmt(v224["dataset"]))
    print("[test 224, dataset-level ] " + fmt(s224["dataset"]))
    print("[test 224, per-image mean] " + fmt(s224["image"]))
    print("[test native, dataset    ] " + fmt(snat["dataset"]))
    return res


if __name__ == "__main__":
    main()
