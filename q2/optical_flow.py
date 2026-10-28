"""Figure 1: motion statistics of the training data (dense optical flow).

First 50 training videos, all 19 consecutive frame pairs each (950 pairs).
Farneback flow with the handout's exact settings; keep pixels brighter than
0.2 in both frames with flow magnitude above 0.5 px.
Run: python -m q2.optical_flow
"""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from q2.data import SPLITS, load_videos, to_float

FIG = Path("results/figures")
TAB = Path("results/tables")
N_VIDEOS = 50


def pooled_flow(videos):
    """Return (dx, dy) of kept pixels pooled over all frame pairs."""
    dxs, dys = [], []
    for v in videos:
        frames = (to_float(v[:, 0]).numpy() * 255).round().astype(np.uint8)  # (20, 64, 64)
        for s in range(frames.shape[0] - 1):
            prev, nxt = frames[s], frames[s + 1]
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx, dy = flow[..., 0], flow[..., 1]
            keep = (prev > 51) & (nxt > 51) & (np.hypot(dx, dy) > 0.5)
            dxs.append(dx[keep])
            dys.append(dy[keep])
    return np.concatenate(dxs), np.concatenate(dys)


def main():
    videos = load_videos()[list(SPLITS["train"])[:N_VIDEOS]]
    dx, dy = pooled_flow(videos)
    n = dx.size
    theta = np.degrees(np.arctan2(-dy, dx)) % 360.0  # 0 = right, 90 = up on screen
    speed = np.hypot(dx, dy)
    median = float(np.median(speed))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    ax1.hist(theta, bins=np.arange(0, 361, 20), weights=np.full(n, 1 / n),
             color="#4774b5", edgecolor="white")
    ax1.axhline(1 / 18, color="black", ls="--", lw=1, label="uniform (1/18)")
    ax1.set(xlabel="flow direction (degrees; 0 = right, 90 = up)", ylabel="fraction of kept pixels",
            xticks=range(0, 361, 45), xlim=(0, 360), title="(a) direction")
    ax1.legend(frameon=False)

    upper = np.percentile(speed, 99.5)
    ax2.hist(speed, bins=np.linspace(0.5, upper, 40), weights=np.full(n, 1 / n),
             color="#228b73", edgecolor="white")
    ax2.axvline(median, color="black", ls="--", lw=1, label=f"median = {median:.3g} px/frame")
    ax2.set(xlabel="flow speed (pixels per frame)", ylabel="fraction of kept pixels",
            title="(b) speed")
    ax2.legend(frameon=False)

    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "figure1_optical_flow.png", dpi=200)
    fig.savefig(FIG / "figure1_optical_flow.pdf")

    direction_hist, _ = np.histogram(theta, bins=np.arange(0, 361, 20))
    stats = {
        "pixels_kept": int(n),
        "frame_pairs": N_VIDEOS * 19,
        "median_speed_px_per_frame": median,
        "mean_speed_px_per_frame": float(speed.mean()),
        "speed_percentiles": {str(p): float(np.percentile(speed, p)) for p in (10, 25, 75, 90, 99)},
        "direction_fraction_per_20deg_bin": (direction_hist / n).round(4).tolist(),
    }
    (TAB / "figure1_flow_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
