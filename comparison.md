Test Dice (%), dataset-level, test_224. Paper rows = single runs of Table 3; best run = seed with the highest Avg (4); mean ± SD over the seeds found.

### DFUTissue, without augmentation

| Model | runs | Gran | Fibrin | Callus | Back | Avg (4) | Tissue (3) | FG |
|---|---|---|---|---|---|---|---|---|
| WoundFormer (paper) | 1 | 82.45 | 69.32 | 86.71 | 74.88 | 78.34 | 79.49 | – |
| DFUTissueSegNet (paper) | 1 | 90.75 | 68.40 | 82.31 | 76.50 | 79.49 | 80.49 | – |
| WoundFormer (ours), best run (seed 0) | 1 | 91.33 | 81.10 | 72.42 | 94.07 | 84.73 | 81.62 | 86.29 |
| WoundFormer (ours), seeds 0,1,2,3,4 | 5 | 89.93 ± 1.62 | 65.45 ± 9.62 | 73.39 ± 1.34 | 93.76 ± 0.21 | 80.63 ± 2.53 | 76.26 ± 3.33 | 85.63 ± 0.39 |
| DFUTissueSegNet (ours), best run (seed 1) | 1 | 90.89 | 74.46 | 74.17 | 94.48 | 83.50 | 79.84 | 87.44 |
| DFUTissueSegNet (ours), seeds 0,1,2,3,4 | 5 | 88.67 ± 1.74 | 65.53 ± 5.13 | 73.56 ± 2.22 | 93.93 ± 0.49 | 80.42 ± 1.77 | 75.92 ± 2.25 | 86.27 ± 1.16 |
| WoundFormer-GL (ours), best run (seed 3) | 1 | 92.31 | 78.98 | 75.78 | 94.44 | 85.38 | 82.36 | 86.92 |
| WoundFormer-GL (ours), seeds 0,1,2,3,4 | 5 | 92.29 ± 0.46 | 74.76 ± 6.68 | 76.65 ± 1.65 | 94.58 ± 0.28 | 84.57 ± 1.29 | 81.23 ± 1.79 | 87.76 ± 0.96 |

Seed-paired difference vs your WoundFormer (without augmentation): mean Δ (seeds improved)

| Model | shared seeds | Gran | Fibrin | Callus | Back | Avg (4) | Tissue (3) | FG |
|---|---|---|---|---|---|---|---|---|
| DFUTissueSegNet | 0,1,2,3,4 | -1.26 (1/5) | +0.08 (3/5) | +0.17 (3/5) | +0.17 (3/5) | -0.21 (3/5) | -0.34 (3/5) | +0.64 (3/5) |
| WoundFormer-GL | 0,1,2,3,4 | +2.36 (5/5) | +9.31 (4/5) | +3.26 (5/5) | +0.82 (5/5) | +3.94 (4/5) | +4.98 (4/5) | +2.13 (5/5) |

WoundFormer: your mean minus the paper's run: Gran +7.48, Fibrin -3.87, Callus -13.32, Back +18.88, Avg (4) +2.29, Tissue (3) -3.23

DFUTissueSegNet: your mean minus the paper's run: Gran -2.08, Fibrin -2.87, Callus -8.75, Back +17.43, Avg (4) +0.93, Tissue (3) -4.57

### DFUTissue, with augmentation

| Model | runs | Gran | Fibrin | Callus | Back | Avg (4) | Tissue (3) | FG |
|---|---|---|---|---|---|---|---|---|
| WoundFormer (paper) | 1 | 88.27 | 70.75 | 87.42 | 95.13 | 85.40 | 82.15 | – |
| DFUTissueSegNet (paper) | 1 | 91.99 | 74.52 | 85.91 | 95.90 | 87.08 | 84.14 | – |
| WoundFormer (ours), best run (seed 0) | 1 | 92.81 | 84.88 | 75.70 | 94.49 | 86.97 | 84.46 | 87.77 |
| WoundFormer (ours), seeds 0,1,2,3,4 | 5 | 91.43 ± 0.86 | 77.85 ± 4.52 | 75.46 ± 1.22 | 94.16 ± 0.61 | 84.72 ± 1.53 | 81.58 ± 1.90 | 87.24 ± 1.13 |
| DFUTissueSegNet (ours), best run (seed 2) | 1 | 92.47 | 82.36 | 77.48 | 94.91 | 86.80 | 84.10 | 88.80 |
| DFUTissueSegNet (ours), seeds 0,1,2,3,4 | 5 | 91.09 ± 1.01 | 76.76 ± 3.38 | 76.03 ± 1.57 | 94.43 ± 0.34 | 84.58 ± 1.34 | 81.29 ± 1.68 | 87.57 ± 0.94 |
| WoundFormer-GL (ours), best run (seed 3) | 1 | 93.19 | 83.89 | 77.29 | 94.32 | 87.17 | 84.79 | 87.78 |
| WoundFormer-GL (ours), seeds 0,1,2,3,4 | 5 | 92.25 ± 1.09 | 81.71 ± 2.37 | 77.31 ± 1.51 | 94.49 ± 0.46 | 86.44 ± 0.79 | 83.76 ± 1.01 | 87.83 ± 0.98 |

Seed-paired difference vs your WoundFormer (with augmentation): mean Δ (seeds improved)

| Model | shared seeds | Gran | Fibrin | Callus | Back | Avg (4) | Tissue (3) | FG |
|---|---|---|---|---|---|---|---|---|
| DFUTissueSegNet | 0,1,2,3,4 | -0.34 (2/5) | -1.09 (3/5) | +0.57 (2/5) | +0.27 (3/5) | -0.15 (2/5) | -0.29 (2/5) | +0.33 (3/5) |
| WoundFormer-GL | 0,1,2,3,4 | +0.82 (5/5) | +3.86 (4/5) | +1.85 (5/5) | +0.33 (4/5) | +1.72 (4/5) | +2.18 (4/5) | +0.59 (4/5) |

WoundFormer: your mean minus the paper's run: Gran +3.16, Fibrin +7.10, Callus -11.96, Back -0.97, Avg (4) -0.68, Tissue (3) -0.57

DFUTissueSegNet: your mean minus the paper's run: Gran -0.90, Fibrin +2.24, Callus -9.88, Back -1.47, Avg (4) -2.50, Tissue (3) -2.85
