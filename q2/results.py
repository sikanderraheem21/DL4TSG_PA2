"""Shared helpers for building the report tables: loading trained models and
formatting numbers the way the handout asks (3 significant figures, scientific
notation below 1e-3, learned models as "mean (seed 0, seed 1)")."""
from pathlib import Path

import torch

from q2.models import ConvLSTMForecaster, ARCHITECTURES

SEEDS = (0, 1)
CKPT = Path("checkpoints")


def load_model(tag, device="cpu", ckpt_dir=CKPT):
    """tag like 'A_s0' -> trained model in eval mode."""
    saved = torch.load(Path(ckpt_dir) / f"{tag}.pt", map_location=device, weights_only=False)
    model = ConvLSTMForecaster(**ARCHITECTURES[saved["architecture"]]).to(device)
    model.load_state_dict(saved["model"])
    return model.eval(), saved["log"]


def fmt(value):
    if value == 0:
        return "0"
    return f"{value:.2e}" if abs(value) < 1e-3 else f"{value:.3g}"


def fmt_seeds(values):
    """[seed0, seed1] -> 'mean (seed0, seed1)'."""
    mean = sum(values) / len(values)
    return f"{fmt(mean)} ({', '.join(fmt(v) for v in values)})"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
