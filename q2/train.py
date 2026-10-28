"""Train one Q2 model with one seed, following "Training and evaluation".

    Adam 1e-3, no schedule, batch 32, 30 epochs, final-epoch weights.
    One epoch = the 8,000 training videos in shuffled order, one window each.
    Loss = squared error averaged over pixels and AVAILABLE target frames per
    video, then over the videos in the batch.
    The seed controls initialisation, shuffling, window and augmentation sampling.

Per-epoch resumable checkpoints make Colab disconnects cheap: rerunning the same
command continues from the last finished epoch.

Run: python -m q2.train --model A --seed 0
"""
import argparse
import json
import platform
import time
from pathlib import Path

import torch

from q2.data import GROUP, SPLITS, apply_g, load_videos, make_windows, sample_windows
from q2.evaluate import error_summary, per_video_horizon_mse, predict_windows
from q2.models import build, n_params

# model -> (architecture, window regime, symmetry augmentation)
EXPERIMENTS = {
    "A": ("A", "normal", False),
    "B": ("A", "normal", True),
    "C": ("A", "speed", False),
    "D": ("D", "speed", False),
    "DB": ("D", "speed", True),
}


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"{platform.processor() or platform.machine()} ({device.type})"


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def masked_video_loss(pred, target, mask):
    """Mean over pixels and available target frames per video, then over videos."""
    per_frame = (pred - target).pow(2).mean(dim=(2, 3, 4))  # (B, 5)
    per_video = (per_frame * mask).sum(dim=1) / mask.sum(dim=1)
    return per_video.mean()


def augment(x, y, g_index):
    """Apply the same sampled symmetry g to the input and the target of each window."""
    for j, g in enumerate(GROUP):
        rows = g_index == j
        if rows.any():
            x[rows] = apply_g(x[rows], g)
            y[rows] = apply_g(y[rows], g)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=EXPERIMENTS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="results/train_logs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=None, help="smoke test only")
    args = parser.parse_args()

    architecture, regime, symmetric = EXPERIMENTS[args.model]
    device = pick_device(args.device)
    tag = f"{args.model}_s{args.seed}"
    ckpt_dir, log_dir = Path(args.ckpt_dir), Path(args.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / f"{tag}_last.pt"

    videos = load_videos(args.data_root)
    train = videos[SPLITS["train"].start:SPLITS["train"].stop].to(device)
    val = videos[SPLITS["val"].start:SPLITS["val"].stop]

    torch.manual_seed(args.seed)
    model = build(architecture).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    log = {"model": args.model, "architecture": architecture, "regime": regime,
           "symmetry_augmentation": symmetric, "seed": args.seed, "params": n_params(model),
           "hardware": hardware(device), "torch": torch.__version__, "epochs": []}
    start_epoch = 0

    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        generator.set_state(saved["generator"])
        log = saved["log"]
        start_epoch = saved["epoch"] + 1
        print(f"Resuming {tag} from epoch {start_epoch + 1}")

    print(f"{tag}: {architecture=} {regime=} {symmetric=} params={log['params']:,} on {log['hardware']}")
    n_train = train.shape[0]
    for epoch in range(start_epoch, args.epochs):
        order = torch.randperm(n_train, generator=generator)
        t, k = sample_windows(n_train, regime, generator)
        g_index = torch.randint(len(GROUP), (n_train,), generator=generator) if symmetric else None

        model.train()
        sync(device)
        began = time.perf_counter()
        total, count = 0.0, 0
        for b, start in enumerate(range(0, n_train, args.batch_size)):
            if args.max_batches is not None and b >= args.max_batches:
                break
            rows = order[start:start + args.batch_size]
            batch = train[rows.to(device)].float().div_(255.0)
            x, y, mask = make_windows(batch, t[rows], k[rows])
            if symmetric:
                x, y = augment(x, y, g_index[rows].to(device))
            loss = masked_video_loss(model(x), y, mask)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(rows)
            count += len(rows)
        sync(device)
        seconds = time.perf_counter() - began

        # Monitoring only: validation E(5) at 1x, t=1 (never used to pick epochs).
        _, y_val, m_val, p_val = predict_windows(model, val, device=device)
        val_e5 = error_summary(per_video_horizon_mse(p_val, y_val, m_val))["E5"]
        log["epochs"].append({"epoch": epoch + 1, "train_loss": total / count,
                              "val_E5": val_e5, "seconds": seconds})
        print(f"{tag} epoch {epoch + 1:2d}/{args.epochs}  train {total / count:.5f}  "
              f"val E(5) {val_e5:.5f}  {seconds:.1f}s", flush=True)

        torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                    "generator": generator.get_state(), "epoch": epoch, "log": log}, resume_path)

    seconds = [e["seconds"] for e in log["epochs"]]
    log["mean_epoch_seconds"] = sum(seconds) / len(seconds)
    torch.save({"model": model.state_dict(), "architecture": architecture, "log": log},
               ckpt_dir / f"{tag}.pt")
    (log_dir / f"{tag}.json").write_text(json.dumps(log, indent=2))
    print(f"{tag} done: mean {log['mean_epoch_seconds']:.1f}s/epoch -> {ckpt_dir / f'{tag}.pt'}")


if __name__ == "__main__":
    main()
