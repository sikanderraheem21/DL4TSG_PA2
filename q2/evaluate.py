"""Metrics for Q2: per-horizon MSE, E(5), E(2), consistency gap C_T, transformed
error E_T, and the symmetrised forecaster.

    C_T = MSE( f(T(X)), T(f(X)) )      consistency: does the diagram commute?
    E_T = MSE( f(T(X)), T(Y) )         correctness on the transformed clip

Every error is averaged over the 64x64 pixels of each available horizon, then
over horizons per video, then over videos.
"""
import torch

from q2.data import GROUP, apply_g, apply_g_inverse, make_windows


class Symmetrised(torch.nn.Module):
    """f_bar(X) = 1/8 sum_g g^-1( f( g(X) ) )."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, n_out=5):
        total = 0
        for g in GROUP:
            total = total + apply_g_inverse(self.model(apply_g(x, g), n_out), g)
        return total / len(GROUP)


def per_video_horizon_mse(pred, target, mask):
    """(B, 5, 1, H, W) x2, mask (B, 5) -> (B, 5) MSE per horizon, NaN where missing."""
    mse = (pred - target).pow(2).mean(dim=(2, 3, 4))
    return mse.masked_fill(~mask, float("nan"))


@torch.no_grad()
def predict_windows(model, videos, t=1, k=1, batch_size=32, device="cpu", transform=None):
    """Run `model` on every video's window (t, k).

    videos: uint8 (N, 20, 1, 64, 64). transform: optional g applied to X and Y.
    Returns (X, Y, mask, Y_hat) on CPU, float32.
    """
    model.eval()
    xs, ys, masks, preds = [], [], [], []
    for start in range(0, len(videos), batch_size):
        batch = videos[start:start + batch_size].to(device).float().div_(255.0)
        n = batch.shape[0]
        x, y, mask = make_windows(batch, torch.full((n,), t), torch.full((n,), k))
        if transform is not None:
            x, y = apply_g(x, transform), apply_g(y, transform)
        preds.append(model(x).cpu())
        xs.append(x.cpu()), ys.append(y.cpu()), masks.append(mask.cpu())
    return torch.cat(xs), torch.cat(ys), torch.cat(masks), torch.cat(preds)


def error_summary(per_horizon):
    """(N, 5) per-video per-horizon MSE -> dict of MSE_h, E(5), E(2) averaged over videos."""
    out = {f"MSE_{h + 1}": float(per_horizon[:, h].nanmean()) for h in range(per_horizon.shape[1])}
    available = ~per_horizon.isnan()
    if available.all():
        out["E5"] = float(per_horizon.mean(dim=1).mean())
    out["E2"] = float(per_horizon[:, :2].mean(dim=1).mean())
    return out


@torch.no_grad()
def consistency(model, videos, g, batch_size=32, device="cpu"):
    """Per-video C_T and E_T at 1x, t=1, averaged over pixels and the five horizons."""
    model.eval()
    gaps, errors = [], []
    for start in range(0, len(videos), batch_size):
        batch = videos[start:start + batch_size].to(device).float().div_(255.0)
        n = batch.shape[0]
        x, y, _ = make_windows(batch, torch.ones(n, dtype=torch.long), torch.ones(n, dtype=torch.long))
        f_tx = model(apply_g(x, g))
        t_fx = apply_g(model(x), g)
        gaps.append((f_tx - t_fx).pow(2).mean(dim=(1, 2, 3, 4)).cpu())
        errors.append((f_tx - apply_g(y, g)).pow(2).mean(dim=(1, 2, 3, 4)).cpu())
    return torch.cat(gaps), torch.cat(errors)
