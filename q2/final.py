"""What Can Data Fix? and The Final Model: Tables 5-6, Figure 6 (+ GIFs).

    Table 5   models A-D:  E(5) 1x; C_M, E_M, C_R, E_R; E(2) at 1x, 2x, 3x;
              params; training s/epoch (Colab T4 logs); inference time
    Table 6   the same columns for symmetrised B and symmetrised D (final model)
    Figure 6  (a) Figure 3 repeated with the final model on the same videos (+ GIFs)
              (b) Figure 5 repeated with the final model on the same video

Inference time = wall-clock to predict all 1,000 test videos (1x, t=1) at batch
size 32 under eval() and no_grad(), counted only after each batch is on the
device; for symmetrised models it includes all 8 transforms, forward passes,
inverse transforms and the averaging.

Run: python -m q2.final
"""
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from q2.data import SPLITS, apply_g, load_videos, make_windows, window_frames
from q2.evaluate import Symmetrised, per_video_horizon_mse
from q2.models import n_params
from q2.results import SEEDS, fmt, fmt_seeds, load_model
from q2.testing import GIF, ROW_LABELS, TRANSFORMS, draw_panel, panel_rows, save_gif

TAB = Path("results/tables")
FIG = Path("results/figures")
COLUMNS = ["E5_1x", "C_M", "E_M", "C_R", "E_R", "E2_1x", "E2_2x", "E2_3x"]


@torch.no_grad()
def evaluate_all(model, videos, batch_size=32):
    """Every Table 5 error column for one model, reusing f(X) across metrics."""
    model.eval()
    sums = {c: 0.0 for c in COLUMNS}
    n = len(videos)
    for start in range(0, n, batch_size):
        batch = videos[start:start + batch_size].float().div_(255.0)
        b = batch.shape[0]
        ones = torch.ones(b, dtype=torch.long)
        x, y, mask = make_windows(batch, ones, ones)
        fx = model(x)
        per_h = per_video_horizon_mse(fx, y, mask)
        sums["E5_1x"] += float(per_h.mean(dim=1).sum())
        sums["E2_1x"] += float(per_h[:, :2].mean(dim=1).sum())
        for name, g in TRANSFORMS.items():
            ftx = model(apply_g(x, g))
            sums[f"C_{name}"] += float((ftx - apply_g(fx, g)).pow(2).mean(dim=(1, 2, 3, 4)).sum())
            sums[f"E_{name}"] += float((ftx - apply_g(y, g)).pow(2).mean(dim=(1, 2, 3, 4)).sum())
        for k in (2, 3):
            xk, yk, mk = make_windows(batch, ones, ones * k)
            sums[f"E2_{k}x"] += float(per_video_horizon_mse(model(xk), yk, mk)[:, :2].mean(dim=1).sum())
    return {c: v / n for c, v in sums.items()}


@torch.no_grad()
def inference_seconds(model, videos, batch_size=32):
    model.eval()
    batches = []
    for start in range(0, len(videos), batch_size):
        batch = videos[start:start + batch_size].float().div_(255.0)
        ones = torch.ones(batch.shape[0], dtype=torch.long)
        batches.append(make_windows(batch, ones, ones)[0])
    model(batches[0])  # warm-up, not timed
    total = 0.0
    for x in batches:
        began = time.perf_counter()
        model(x)
        total += time.perf_counter() - began
    return total


def table(rows_by_model, test):
    """rows_by_model: {label: (tags, wrap)} -> formatted rows and raw values."""
    formatted, raw = {}, {}
    for label, (tags, wrap) in rows_by_model.items():
        per_seed = []
        for tag in tags:
            model, log = load_model(tag)
            model = wrap(model)
            r = evaluate_all(model, test)
            r["infer_s"] = inference_seconds(model, test)
            r["train_s_per_epoch"] = log["mean_epoch_seconds"]
            r["params"] = n_params(model)
            per_seed.append(r)
            print(f"  {tag}{' (symmetrised)' if wrap is Symmetrised else ''} done", flush=True)
        raw[label] = per_seed
        formatted[label] = {c: fmt_seeds([s[c] for s in per_seed])
                            for c in COLUMNS + ["train_s_per_epoch", "infer_s"]}
        formatted[label]["params"] = f"{per_seed[0]['params']:,}"
        print(f"{label:<16}", "  ".join(f"{k}={v}" for k, v in formatted[label].items()), flush=True)
    return formatted, raw


def figure6(test):
    final = Symmetrised(load_model("D_s0")[0])
    selection = json.loads((TAB / "phase3_selection.json").read_text())
    fig = plt.figure(figsize=(10, 18), layout="constrained")
    subfigs = fig.subfigures(4, 1, hspace=0.02)
    gif_paths = []
    for p, entry in enumerate(selection["figure3"]):
        name, video = entry["transform"], entry["video"]
        i = video - 9000
        batch = test[i:i + 1].float().div(255.0)
        x, y, _ = make_windows(batch, torch.tensor([1]), torch.tensor([1]))
        rows = panel_rows(final, x, y, TRANSFORMS[name])
        with torch.no_grad():
            c = float((rows[2][1] - rows[3][1]).pow(2).mean())
        title = f"({'abcd'[p]}) test video {video}, T = {name}, final model, C_{name} = {fmt(c)}"
        draw_panel(subfigs[p], rows, ROW_LABELS[name], title)
        path = GIF / f"figure6a-{'abcd'[p]}_{name}_video{video}_final.gif"
        save_gif(rows, ROW_LABELS[name], title, path)
        gif_paths.append(str(path))
    fig.savefig(FIG / "figure6a_final_symmetry.png", dpi=150)
    fig.savefig(FIG / "figure6a_final_symmetry.pdf")
    plt.close(fig)

    i = selection["figure5_video"] - 9000
    fig, axes = plt.subplots(3, 9, figsize=(11, 4.2), layout="constrained")
    for r, k in enumerate((1, 2, 3)):
        batch = test[i:i + 1].float().div(255.0)
        x, y, _ = make_windows(batch, torch.tensor([1]), torch.tensor([k]))
        with torch.no_grad():
            pred = final(x)
        inputs, targets = window_frames(1, k)
        frames = [(x[0, j], f"in: frame {inputs[j]}") for j in range(5)]
        frames += [(y[0, h], f"true: frame {targets[h]}") for h in range(2)]
        frames += [(pred[0, h], f"final: h={h + 1}") for h in range(2)]
        for c, (img, label) in enumerate(frames):
            axes[r, c].imshow(img[0].clamp(0, 1), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
            axes[r, c].set_title(label, fontsize=7)
        axes[r, 0].set_ylabel(f"{k}x", fontsize=10)
    fig.savefig(FIG / "figure6b_final_speed.png", dpi=170)
    fig.savefig(FIG / "figure6b_final_speed.pdf")
    plt.close(fig)
    return gif_paths


def main():
    torch.manual_seed(0)
    test = load_videos()[SPLITS["test"].start:SPLITS["test"].stop]
    hardware = {"training": "Tesla T4 (Google Colab)",
                "evaluation_and_inference": f"{platform.processor() or platform.machine()} CPU "
                                            f"({torch.get_num_threads()} threads), torch {torch.__version__}"}
    print("Table 5")
    t5, raw5 = table({m: ([f"{m}_s{s}" for s in SEEDS], lambda f: f) for m in "ABCD"}, test)
    (TAB / "table5_data_fix.json").write_text(json.dumps({"formatted": t5, "raw": raw5, "hardware": hardware}, indent=2))
    print("Table 6")
    t6, raw6 = table({f"symmetrised {m}": ([f"{m}_s{s}" for s in SEEDS], Symmetrised) for m in "BD"}, test)
    (TAB / "table6_final.json").write_text(json.dumps({"formatted": t6, "raw": raw6, "hardware": hardware,
                                                       "final_model": "symmetrised D"}, indent=2))
    gifs = figure6(test)
    print("Figure 6 GIFs:", gifs)


if __name__ == "__main__":
    main()
