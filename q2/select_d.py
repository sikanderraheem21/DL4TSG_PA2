"""Choose model D on the VALIDATION split (videos 8,000-8,999) only.

Each candidate is trained like model C (speed augmentation, seed 0). We compare
val E(5) at 1x and val E(2) at 1x, 2x, 3x, and pick the candidate with the
lowest mean E(2) over the three speeds: it must handle fast motion without
giving up accuracy at normal speed. Model C (seed 0) is the reference with
model A's reach.

Run: python -m q2.select_d
"""
import json
from pathlib import Path

import torch

from q2.data import SPLITS, load_videos
from q2.evaluate import error_summary, per_video_horizon_mse, predict_windows
from q2.models import ARCHITECTURES, D_CANDIDATES, n_params, reach_pixels
from q2.results import fmt, load_model

TAB = Path("results/tables")


def main():
    device = torch.device("cpu")
    val = load_videos()[SPLITS["val"].start:SPLITS["val"].stop]
    rows = {}
    for name in ("C",) + D_CANDIDATES:
        model, log = load_model(f"{name}_s0", device)
        row = {"architecture": log["architecture"], "params": n_params(model),
               "reach_px": reach_pixels(**ARCHITECTURES[log["architecture"]]),
               "sec_per_epoch": log["mean_epoch_seconds"]}
        for k in (1, 2, 3):
            _, y, mask, pred = predict_windows(model, val, k=k, device=device)
            s = error_summary(per_video_horizon_mse(pred, y, mask))
            row[f"val_E2_{k}x"] = s["E2"]
            if k == 1:
                row["val_E5_1x"] = s["E5"]
        row["val_mean_E2"] = sum(row[f"val_E2_{k}x"] for k in (1, 2, 3)) / 3
        rows[name] = row
        print(f"{name:8s}", "  ".join(f"{k}={fmt(v) if isinstance(v, float) else v}" for k, v in row.items()))
    best = min(D_CANDIDATES, key=lambda n: rows[n]["val_mean_E2"])
    print(f"\nChosen model D (lowest mean validation E(2) over 1x, 2x, 3x): {best}")
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "model_d_selection_val.json").write_text(json.dumps({"rows": rows, "chosen": best}, indent=2))


if __name__ == "__main__":
    main()
