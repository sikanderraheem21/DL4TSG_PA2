# AI651 Assignment 2 — Deep Learning for Space, Time and Graphs

Muhammad Sikander Raheem (25280017), LUMS, Fall 2026.

## Contents

| Path | What it is |
|---|---|
| `report/Q1_report.tex` | Question 1: river forecasting proposal (LaTeX, compiles on Overleaf) |
| `predictions/` | Q2.5 and Q2.11 predictions, each committed before the experiment it predicts |
| `q2/` | Question 2 code: data, transforms, models, training, evaluation, figures |
| `results/tables/` | Tables 1–6 as CSV/JSON |
| `results/figures/` | Figures 1–6 |
| `results/gifs/` | Animated GIFs for Figure 3 and Figure 6(a) |
| `checkpoints/` | Trained model weights (seeds 0 and 1) |

## Reproducing Question 2

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

| Output | Command | Hardware |
|---|---|---|
| Table 1 (transform and window checks) | `python -m q2.checks` | CPU |
| Figure 1 (optical flow of training data) | `python -m q2.optical_flow` | CPU |
| Train one model and seed | `python -m q2.train --model A --seed 0` | GPU ([colab/train_q2.ipynb](colab/train_q2.ipynb)) |
| Table 2, Figure 2 (model A vs Blank, Copy-last) | `python -m q2.baseline` | CPU or GPU |

Training saves a resumable checkpoint after every epoch; rerunning the same
command continues from the last finished epoch.
