# WoundTissueSeg: WoundFormer, DFUTissueSegNet and WoundFormer-GL on DFUTissue

Code to train, validate, test and visualise three wound-tissue segmentation models on the DFUTissue dataset
(4 classes: background, fibrin, granulation, callus), and to compare your results with the WoundFormer paper.

| Model | What it is | Params | Encoder passes / image |
|---|---|---|---|
| `woundformer` | **WoundFormer** (Kabir & Dulal, arXiv:2605.19868): MiT-B5 + spatially-preserving multi-scale aggregation head | 81.8M | 1 |
| `dfutissuesegnet` | **DFUTissueSegNet** (Dhar et al., arXiv:2406.16012): U-Net with MiT-B3 encoder and parallel scSE ("P-scSE") decoder blocks | 47.6M | 1 |
| `woundformer_gl` | **WoundFormer-GL** (ours): WoundFormer on the full image (global) + a 2 × 2 grid of magnified tiles (local), global features fused into the tiles, probabilities averaged; one shared encoder | 83.0M | 5 |

## 1. Setup

```bash
# Python 3.11; install PyTorch for your CUDA first, e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Data.** Download the DFUTissue dataset (folder `DFUTissue` of github.com/uwm-bigdata/DFUTissueSegNet). The code needs:

```
<DATA_ROOT>/Labeled/labeled_train_names.txt     78 names
<DATA_ROOT>/Labeled/labeled_val_names.txt       16 names
<DATA_ROOT>/Labeled/test_names.txt              16 names
<DATA_ROOT>/Labeled/Original/Images/{TrainVal,Test}/<name>.png
<DATA_ROOT>/Labeled/Original/Annotations/{TrainVal,Test}/<name>.png    class-index masks 0..3
```

**Pretrained weights** (ImageNet-1K, official SegFormer release) are downloaded automatically on first use through
`segmentation_models_pytorch` (internet needed once). If you already have `mit_b5.pth` / `mit_b3.pth` from the SegFormer
release, put them in `~/.cache/torch/hub/checkpoints/` and they are used instead.

## 2. Run

**Notebooks** (open from the `notebooks/` folder; set `DATA_ROOT` in the first code cell):

* `notebooks/01_single_seed.ipynb`: all three models, one seed (`SEED`), without and/or with augmentation; results
  table vs the paper, training curves, qualitative figure, per-class bar chart.
* `notebooks/02_multi_seed.ipynb`: the same for a list of seeds (`SEEDS = [0, 1, 2, 3, 4]`, edit as you like):
  mean ± SD, best run, seed-paired differences vs WoundFormer, bar charts. Finished runs are skipped, so you can stop,
  resume or add seeds.

Set `EPOCHS = 1` first to check that everything runs (a few minutes), then `EPOCHS = 200` (the protocol; early
stopping ends runs much earlier). Start Jupyter from the Python environment where you installed the requirements (or
select that environment's kernel); the notebooks run `train.py` with that same interpreter. The settings can also be
given as environment variables (`WOUNDSEG_DATA`, `WOUNDSEG_SEED(S)`, `WOUNDSEG_SETTINGS`, `WOUNDSEG_EPOCHS`,
`WOUNDSEG_RUNS`), e.g. for running a notebook head-less with `jupyter nbconvert --execute`.

**Command line:**

```bash
python train.py --model woundformer     --data-root /path/DFUTissue --seed 0            # without augmentation
python train.py --model dfutissuesegnet --data-root /path/DFUTissue --seed 0 --aug      # with augmentation
python train.py --model woundformer_gl  --data-root /path/DFUTissue --seed 0            # -> runs/woundformer_gl_noaug_s0

python compare_with_paper.py --runs-root runs --seeds 0 1 2 3 4 --out runs/comparison.md
python visualize.py qualitative --data-root /path/DFUTissue --runs runs/woundformer_noaug_s0 runs/woundformer_gl_noaug_s0 --labels WoundFormer WoundFormer-GL
python visualize.py curves --runs runs/woundformer_noaug_s0 runs/dfutissuesegnet_noaug_s0
python visualize.py bars --runs-root runs --setting noaug --seeds 0 1 2 3 4
```

Each run folder `runs/<model>_<noaug|aug>_s<seed>/` contains `log.csv` (per epoch), `best.pth`, `results.json` and
`test_pred/{index,color}/` (test predictions at native resolution).

Time per run on an 8 GB RTX 4060: WoundFormer 2–3 min, DFUTissueSegNet 2–4 min, WoundFormer-GL 15–25 min
(activation checkpointing is switched on automatically for WoundFormer-GL so it fits in 8 GB).

## 3. Protocol (identical for all models: the WoundFormer paper's unified protocol)

* official DFUTissue split, Original crops, resized to 224 × 224 (image bilinear, mask nearest), ImageNet normalisation
* Adam, lr 1e-4, batch size 8, cross-entropy loss, at most 200 epochs
* ReduceLROnPlateau (factor 0.1, patience 5) and early stopping (patience 15), both on the validation loss
* the checkpoint with the highest validation mIoU is evaluated on the test set
* `--aug`: DFUTissueSegNet's training augmentation (flips; scale / rotate / shift; perspective, noise, sharpen, blur;
  CLAHE, brightness/contrast, gamma, hue/saturation), applied with probability 0.9
* fixed seeds and deterministic cuDNN: a seed gives the same run on the same machine (small differences across GPUs /
  library versions are normal)

## 4. Metrics (`woundseg/metrics.py`)

Dice = 2TP / (2TP + FP + FN) per class, accumulated over all test pixels (dataset-level; per-image means are stored
too), at 224 × 224 (also at native resolution in `results.json`).

* **Avg (4)**: mean of background, fibrin, granulation, callus Dice = the paper's "Avg." column
* **Tissue (3)**: mean of fibrin, granulation, callus Dice
* **FG**: wound vs background Dice with all tissue classes merged

## 5. Models

**WoundFormer** (`woundseg/woundformer.py`), following Sec. 2 of the paper: MiT-B5 encoder; each stage projected to 128
channels by 1×1 conv + BN + ReLU; coarse-to-fine fusion (upsample, concatenate, 1×1 conv + BN + ReLU) from the 1/32 to
the 1/4 stage; 3×3 refinement conv; 1×1 prediction layer at 1/4 resolution.

**DFUTissueSegNet** (`woundseg/dfutissuesegnet.py`), re-implemented from the authors' released code
(`smp.Unet(encoder_name="mit_b3", decoder_attention_type="pscse")` with their modified decoder): each decoder block
upsamples ×2, concatenates the skip connection, applies a max-out scSE and an additive scSE in parallel (sum), then one
3×3 conv + BN + ReLU. As in the WoundFormer paper's comparison, it is trained with the unified protocol above (not the
authors' own recipe of sigmoid outputs, Dice + Focal loss, 256 px, batch 16). The WoundFormer paper lists 51.1M
parameters for DFUTissueSegNet; this implementation has 47.6M (the same paper also lists ~3M more than the architecture
has for WoundFormer).

**WoundFormer-GL** (`woundseg/woundformer_gl.py`): the same MiT-B5 encoder looks at the full image and at four
overlapping tiles (each 60% of the side, resampled to 224, about 1.7× magnification) in one batched call. The global
features are warped into each tile and fused at every scale (1×1 conv + BN, initialised to zero). A WoundFormer head
predicts the full image and another one each tile; the tile probabilities are mapped back and averaged with the global
ones per pixel. Loss = CE(fused output) + 0.4 × (CE(global) + CE(tiles)). It needs no wound localisation and no extra
labels; inference is one forward call.

## 6. Reference results (our runs, test Dice %, mean ± SD over seeds 0–4)

| Without augmentation | Gran | Fibrin | Callus | Back | Avg (4) | Tissue (3) |
|---|---|---|---|---|---|---|
| WoundFormer, paper (1 run) | 82.45 | 69.32 | 86.71 | 74.88 | 78.34 | 79.49 |
| DFUTissueSegNet, paper (1 run) | 90.75 | 68.40 | 82.31 | 76.50 | 79.49 | 80.49 |
| WoundFormer, this code | 90.18 ± 1.43 | 58.91 ± 13.85 | 73.96 ± 0.78 | 93.81 ± 0.28 | 79.21 ± 3.56 | 74.35 ± 4.69 |
| WoundFormer-GL, this code | 91.71 ± 1.35 | 73.35 ± 4.41 | 77.16 ± 1.57 | 94.61 ± 0.52 | 84.21 ± 1.89 | 80.74 ± 2.36 |

| With augmentation | Gran | Fibrin | Callus | Back | Avg (4) | Tissue (3) |
|---|---|---|---|---|---|---|
| WoundFormer, paper (1 run) | 88.27 | 70.75 | 87.42 | 95.13 | 85.40 | 82.15 |
| DFUTissueSegNet, paper (1 run) | 91.99 | 74.52 | 85.91 | 95.90 | 87.08 | 84.14 |
| WoundFormer, this code | 90.78 ± 1.41 | 77.70 ± 3.82 | 74.76 ± 3.47 | 94.09 ± 0.51 | 84.33 ± 1.70 | 81.08 ± 2.12 |
| WoundFormer-GL, this code | 91.88 ± 0.93 | 81.97 ± 1.69 | 77.41 ± 1.11 | 94.57 ± 0.35 | 86.46 ± 0.82 | 83.75 ± 0.99 |

DFUTissueSegNet has not been trained with this pipeline yet; your runs give its first numbers under the unified
protocol. Fibrin occurs in only 6 of the 16 test images, so single-seed fibrin (and Tissue (3)) varies a lot between
seeds; compare models over several seeds. The paper's callus and background-without-augmentation values could not be
reached by any model under this evaluation.

## 7. Files

```
train.py               train + validate + test one model / one seed
compare_with_paper.py  tables of your runs vs the paper (best run, mean ± SD, paired differences)
visualize.py           qualitative figure, training curves, per-class bar chart
notebooks/             01_single_seed.ipynb, 02_multi_seed.ipynb
woundseg/
  woundformer.py       WoundFormer (+ MiT-B5 weight loading)
  dfutissuesegnet.py   DFUTissueSegNet
  woundformer_gl.py    WoundFormer-GL
  models.py            model registry, activation checkpointing
  data.py              DFUTissue loading, resizing, augmentation
  metrics.py           Dice / IoU (dataset-level and per image), Wilcoxon test
  mit.py               Mix Transformer encoder (NVIDIA SegFormer code, vendored from segmentation_models_pytorch 0.5.0)
```

Credits: MiT encoder © NVIDIA (SegFormer, NVIDIA Source Code License); DFUTissueSegNet architecture and augmentation
from github.com/uwm-bigdata/DFUTissueSegNet; WoundFormer from arXiv:2605.19868.
