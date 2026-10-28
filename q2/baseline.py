"""Table 2 and Figure 2: model A against Blank and Copy-last on the test split
(1x, t = 1: input frames 1-5, target frames 6-10).

Run after model A is trained: python -m q2.baseline
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from q2.data import SPLITS, load_videos
from q2.evaluate import error_summary, per_video_horizon_mse, predict_windows
from q2.models import Blank, CopyLast, n_params
from q2.results import SEEDS, fmt, fmt_seeds, load_model, pick_device

TAB = Path("results/tables")
FIG = Path("results/figures")


def overlap_scores(videos):
    """Large when the two digits overlap in frames 1-10.

    Overlapping digits cover fewer bright pixels than the same digits apart, so
    the drop of a frame's bright-pixel count below the video's maximum flags it.
    """
    bright = (videos[:, :10] > 51).flatten(2).sum(-1).float()  # (N, 10)
    return 1 - bright.min(dim=1).values / bright.max(dim=1).values


def main():
    device = pick_device()
    test = load_videos()[SPLITS["test"].start:SPLITS["test"].stop]

    rows, preds_s0 = {}, None
    per_seed = []
    for seed in SEEDS:
        model, _ = load_model(f"A_s{seed}", device)
        _, y, mask, pred = predict_windows(model, test, device=device)
        per_seed.append(error_summary(per_video_horizon_mse(pred, y, mask)))
        if seed == 0:
            preds_s0, params = pred, n_params(model)
    keys = list(per_seed[0])
    rows["Model A"] = {k: fmt_seeds([s[k] for s in per_seed]) for k in keys} | {"params": f"{params:,}"}
    for name, ref in [("Blank", Blank()), ("Copy-last", CopyLast())]:
        _, y, mask, pred = predict_windows(ref, test, device=device)
        rows[name] = {k: fmt(v) for k, v in error_summary(per_video_horizon_mse(pred, y, mask)).items()} | {"params": "0"}

    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "table2_baseline.json").write_text(json.dumps(
        {"formatted": rows, "raw_model_A": per_seed}, indent=2))
    for name, r in rows.items():
        print(f"{name:<10}", "  ".join(f"{k}={v}" for k, v in r.items() if k != "E2"))

    # Figure 2: three test videos, the first chosen for overlapping digits.
    scores = overlap_scores(test)
    chosen = [int(scores.argmax())] + [i for i in range(len(test)) if scores[i] < 0.05][:2]
    x, y, _, _ = predict_windows(CopyLast(), test[chosen], device=device)
    pred = preds_s0[chosen]
    fig, axes = plt.subplots(9, 5, figsize=(7.5, 13.5), layout="constrained")
    for v, idx in enumerate(chosen):
        for r, (label, frames, first) in enumerate([("input", x[v], 1), ("truth", y[v], 6),
                                                    ("model A", pred[v], 6)]):
            for c in range(5):
                ax = axes[3 * v + r, c]
                ax.imshow(frames[c, 0], cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([]), ax.set_yticks([])
                ax.set_title(f"frame {first + c}", fontsize=7)
                if c == 0:
                    ax.set_ylabel(f"test {9000 + idx}\n{label}", fontsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "figure2_baseline_examples.png", dpi=150)
    fig.savefig(FIG / "figure2_baseline_examples.pdf")
    print("Figure 2 videos (dataset index):", [9000 + i for i in chosen],
          "overlap scores:", [round(float(scores[i]), 3) for i in chosen])


if __name__ == "__main__":
    main()
