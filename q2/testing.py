"""Testing the Baseline: Tables 3-4, Figures 3-5 and the Figure 3 GIFs.

    Table 3   E(5), C_M, E_M, C_R, E_R        model A (2 seeds), Blank, Copy-last
    Figure 3  top-2 videos by per-video C_M and by C_R (seed-0 model A), 4 rows each:
                (1) X    | f(X)        (2) T(X) | T(Y)
                (3) T(X) | f(T(X))     (4) T(X) | T(f(X))
    Figure 4  histograms of per-video C_M and C_R, Figure 3 videos marked
    Table 4   E(2) at 1x, 2x, 3x and E(5) at 1x, 2x     model A, Copy-last
    Figure 5  one non-overlapping test video at 1x, 2x, 3x

Run: python -m q2.testing   (after predictions/Q2.5_predictions.md is committed)
"""
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from q2.data import SPLITS, apply_g, load_videos, make_windows, window_frames
from q2.evaluate import consistency, error_summary, per_video_horizon_mse, predict_windows
from q2.models import Blank, CopyLast
from q2.results import SEEDS, fmt, fmt_seeds, load_model

TAB = Path("results/tables")
FIG = Path("results/figures")
GIF = Path("results/gifs")
TRANSFORMS = {"M": (1, 0), "R": (0, 1)}
ROW_LABELS = {
    "M": ["X | f(X)", "M(X) | M(Y)", "M(X) | f(M(X))", "M(X) | M(f(X))"],
    "R": ["X | f(X)", "R(X) | R(Y)", "R(X) | f(R(X))", "R(X) | R(f(X))"],
}


def symmetry_row(model, test, device, mean=True):
    """E(5) at 1x plus C_T, E_T for T in {M, R}; per-video values if mean=False."""
    _, y, mask, pred = predict_windows(model, test, device=device)
    out = {"E5": per_video_horizon_mse(pred, y, mask).mean(dim=1)}
    for name, g in TRANSFORMS.items():
        out[f"C_{name}"], out[f"E_{name}"] = consistency(model, test, g, device=device)
    return {k: float(v.mean()) for k, v in out.items()} if mean else out


def panel_rows(model, x, y, g):
    """The four (left, right) rows of a Figure 3 panel for one video."""
    with torch.no_grad():
        fx = model(x)
        ftx = model(apply_g(x, g))
    tx = apply_g(x, g)
    return [(x, fx), (tx, apply_g(y, g)), (tx, ftx), (tx, apply_g(fx, g))]


def draw_panel(subfig, rows, labels, title):
    axes = subfig.subplots(4, 10)
    subfig.suptitle(title, fontsize=9, weight="bold", x=0.02, ha="left")
    for r, ((left, right), label) in enumerate(zip(rows, labels)):
        frames = torch.cat([left[0], right[0]])  # (10, 1, 64, 64)
        for c in range(10):
            ax = axes[r, c]
            ax.imshow(frames[c, 0].clamp(0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(f"in {c + 1}" if c < 5 else f"h={c - 4}", fontsize=7)
            if c == 0:
                ax.set_ylabel(label, fontsize=7)


def save_gif(rows, labels, title, path):
    """One GIF frame per forecast horizon; each frame keeps the same four rows
    (last input frame on the left, horizon h on the right)."""
    images = []
    for h in range(5):
        fig, axes = plt.subplots(4, 2, figsize=(3.4, 6.4), layout="constrained")
        for r, ((left, right), label) in enumerate(zip(rows, labels)):
            for c, (img, sub) in enumerate([(left[0, 4], "input 5"), (right[0, h], f"horizon {h + 1}")]):
                axes[r, c].imshow(img[0].clamp(0, 1), cmap="gray", vmin=0, vmax=1)
                axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
                if r == 0:
                    axes[r, c].set_title(sub, fontsize=8)
            axes[r, 0].set_ylabel(label, fontsize=7)
        fig.suptitle(title, fontsize=8)
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(path, images, duration=0.8, loop=0)


def two_separate_digits(frame, min_area=15):
    """True when the frame's bright pixels form at least two sizeable connected
    components, i.e. the two digits are apart (touching or overlapping digits
    merge into one component)."""
    mask = (frame.numpy() > 51).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return int((stats[1:, cv2.CC_STAT_AREA] >= min_area).sum()) >= 2


def non_overlapping(videos, frames):
    """Indices of videos whose digits are apart in every listed frame (1-based)."""
    return [i for i in range(len(videos))
            if all(two_separate_digits(videos[i, f - 1, 0]) for f in frames)]


def main():
    device = torch.device("cpu")
    test = load_videos()[SPLITS["test"].start:SPLITS["test"].stop]
    models = {seed: load_model(f"A_s{seed}", device)[0] for seed in SEEDS}
    TAB.mkdir(parents=True, exist_ok=True), FIG.mkdir(parents=True, exist_ok=True), GIF.mkdir(parents=True, exist_ok=True)

    # ---- Table 3 -------------------------------------------------------------
    per_video_s0 = symmetry_row(models[0], test, device, mean=False)
    seeds = [{k: float(v.mean()) for k, v in per_video_s0.items()},
             symmetry_row(models[1], test, device)]
    cols = ["E5", "C_M", "E_M", "C_R", "E_R"]
    table3 = {"Model A": {c: fmt_seeds([s[c] for s in seeds]) for c in cols}}
    for name, ref in [("Blank", Blank()), ("Copy-last", CopyLast())]:
        table3[name] = {c: fmt(v) for c, v in symmetry_row(ref, test, device).items()}
    (TAB / "table3_symmetry.json").write_text(json.dumps({"formatted": table3, "raw_model_A": seeds}, indent=2))
    print("Table 3")
    for name, r in table3.items():
        print(f"  {name:<10}", "  ".join(f"{k}={v}" for k, v in r.items()))

    # ---- Figure 3 + GIFs -----------------------------------------------------
    chosen = []
    for name in TRANSFORMS:
        top = per_video_s0[f"C_{name}"].argsort(descending=True)[:2].tolist()
        chosen += [(name, i) for i in top]
    fig = plt.figure(figsize=(10, 18), layout="constrained")
    subfigs = fig.subfigures(4, 1, hspace=0.02)
    gif_paths = []
    for p, (name, i) in enumerate(chosen):
        batch = test[i:i + 1].float().div(255.0)
        x, y, _ = make_windows(batch, torch.tensor([1]), torch.tensor([1]))
        rows = panel_rows(models[0], x, y, TRANSFORMS[name])
        c_value = float(per_video_s0[f"C_{name}"][i])
        title = f"({'abcd'[p]}) test video {9000 + i}, T = {name}, C_{name} = {fmt(c_value)}"
        draw_panel(subfigs[p], rows, ROW_LABELS[name], title)
        path = GIF / f"figure3{'abcd'[p]}_{name}_video{9000 + i}.gif"
        save_gif(rows, ROW_LABELS[name], title, path)
        gif_paths.append(str(path))
    fig.savefig(FIG / "figure3_symmetry_worst.png", dpi=150)
    fig.savefig(FIG / "figure3_symmetry_worst.pdf")
    plt.close(fig)
    print("Figure 3 videos:", [(n, 9000 + i) for n, i in chosen])
    print("GIFs:", gif_paths)

    # ---- Figure 4 ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    for ax, name in zip(axes, TRANSFORMS):
        values = per_video_s0[f"C_{name}"].numpy()
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 40)
        ax.hist(values, bins=bins, color="#4774b5", edgecolor="white")
        ax.set_xscale("log")
        marked = [i for n, i in chosen if n == name]
        for rank, i in enumerate(marked):
            ax.axvline(values[i], color="#c44963", ls="--", lw=1.2)
            ax.annotate(f"video {9000 + i}", (values[i], 0.92 - 0.12 * rank), xycoords=("data", "axes fraction"),
                        xytext=(-4, 0), textcoords="offset points", fontsize=7, ha="right", color="#c44963")
        ax.axvline(np.median(values), color="black", lw=1, label=f"median {fmt(float(np.median(values)))}")
        ax.set(xlabel=f"per-video C_{name} (log scale)", ylabel="number of test videos",
               title=f"({'ab'[list(TRANSFORMS).index(name)]}) consistency gap under {'mirror' if name == 'M' else 'rotation'}")
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "figure4_consistency_hist.png", dpi=200)
    fig.savefig(FIG / "figure4_consistency_hist.pdf")
    plt.close(fig)

    # ---- Table 4 -------------------------------------------------------------
    def speed_errors(model):
        out = {}
        for k in (1, 2, 3):
            _, y, mask, pred = predict_windows(model, test, t=1, k=k, device=device)
            s = error_summary(per_video_horizon_mse(pred, y, mask))
            out[f"E2_{k}x"] = s["E2"]
            if k < 3:
                out[f"E5_{k}x"] = s["E5"]
        return out

    speed_seeds = [speed_errors(models[s]) for s in SEEDS]
    copy = speed_errors(CopyLast())
    table4 = {"Model A": {c: fmt_seeds([s[c] for s in speed_seeds]) for c in speed_seeds[0]},
              "Copy-last": {c: fmt(v) for c, v in copy.items()}}
    (TAB / "table4_speed.json").write_text(json.dumps(
        {"formatted": table4, "raw_model_A": speed_seeds, "raw_copy_last": copy}, indent=2))
    print("Table 4")
    for name, r in table4.items():
        print(f"  {name:<10}", "  ".join(f"{k}={v}" for k, v in r.items()))

    # ---- Figure 5 ------------------------------------------------------------
    shown_true = sorted({f for k in (1, 2, 3) for f in window_frames(1, k)[1][:2]})
    i = non_overlapping(test, shown_true)[0]
    fig, axes = plt.subplots(3, 9, figsize=(11, 4.2), layout="constrained")
    for r, k in enumerate((1, 2, 3)):
        batch = test[i:i + 1].float().div(255.0)
        x, y, _ = make_windows(batch, torch.tensor([1]), torch.tensor([k]))
        with torch.no_grad():
            pred = models[0](x)
        inputs, targets = window_frames(1, k)
        frames = [(x[0, j], f"in: frame {inputs[j]}") for j in range(5)]
        frames += [(y[0, h], f"true: frame {targets[h]}") for h in range(2)]
        frames += [(pred[0, h], f"pred: h={h + 1}") for h in range(2)]
        for c, (img, label) in enumerate(frames):
            axes[r, c].imshow(img[0].clamp(0, 1), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
            axes[r, c].set_title(label, fontsize=7)
        axes[r, 0].set_ylabel(f"{k}x", fontsize=10)
    fig.savefig(FIG / "figure5_speed.png", dpi=170)
    fig.savefig(FIG / "figure5_speed.pdf")
    plt.close(fig)
    print("Figure 5 video:", 9000 + i, "(first test video with no overlap in frames", shown_true, ")")

    (TAB / "phase3_selection.json").write_text(json.dumps(
        {"figure3": [{"transform": n, "video": 9000 + i_} for n, i_ in chosen],
         "figure3_gifs": gif_paths, "figure5_video": 9000 + i}, indent=2))


if __name__ == "__main__":
    main()
