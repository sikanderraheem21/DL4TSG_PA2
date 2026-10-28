"""Moving MNIST data, symmetry transforms and forecasting windows (Q2 Setup).

Frames are numbered 1-20 as in the handout; tensors index them 0-19.

    video (20, 1, 64, 64)
        |
        |  window(t, k):  X = frames t, t+k, ..., t+4k
        |                 Y = frames t+5k, ..., t+9k  (only those <= 20)
        v
    X (5, 1, 64, 64), Y (5, 1, 64, 64) zero-padded, mask (5,) True where Y exists
"""
import torch
from torchvision.datasets import MovingMNIST

N_FRAMES = 20
N_IN = 5
N_OUT = 5

SPLITS = {
    "train": range(0, 8000),
    "val": range(8000, 9000),
    "test": range(9000, 10000),
}

# Allowed start frames t (1-based) per training regime and speed k.
NORMAL_STARTS = {1: list(range(1, 12))}  # models A, B
AUGMENTED_STARTS = {1: list(range(1, 15)), 2: list(range(1, 9)), 3: [1, 2]}  # models C, D


def load_videos(root="data"):
    """All 10,000 videos as uint8, shape (10000, 20, 1, 64, 64).

    split=None keeps every frame of every video. The built-in train/test
    options split frames, not videos, so we split by index instead (SPLITS).
    """
    dataset = MovingMNIST(root=root, split=None, download=True)
    return dataset.data


def to_float(videos):
    """uint8 -> float32 in [0, 1]."""
    return videos.float().div_(255.0)


# ---------------------------------------------------------------------------
# Symmetries of the square: G = {R^a, M R^a : a = 0..3}
# ---------------------------------------------------------------------------

def mirror(x):
    """(Mx)[i, j] = x[i, 63 - j]: flip columns."""
    return x.flip(-1)


def rotate(x, a=1):
    """R^a: rotate a * 90 degrees anticlockwise."""
    return torch.rot90(x, k=a, dims=(-2, -1))


# Group elements as (m, a): g(x) = M(R^a(x)) if m else R^a(x).
GROUP = [(m, a) for m in (0, 1) for a in range(4)]


def apply_g(x, g):
    m, a = g
    x = rotate(x, a)
    return mirror(x) if m else x


def apply_g_inverse(x, g):
    """R^a has inverse R^-a; each M R^a is its own inverse."""
    m, a = g
    return apply_g(x, g) if m else rotate(x, -a)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def window_frames(t, k):
    """1-based input and available target frame numbers for start t, speed k."""
    inputs = [t + j * k for j in range(N_IN)]
    targets = [t + (N_IN + j) * k for j in range(N_OUT)]
    return inputs, [f for f in targets if f <= N_FRAMES]


def make_windows(videos, t, k):
    """Batched windows.

    videos: (B, 20, C, H, W); t, k: (B,) long tensors (t 1-based).
    Returns X (B, 5, C, H, W), Y (B, 5, C, H, W) with missing targets zeroed,
    and mask (B, 5) bool marking targets that exist.
    """
    device = videos.device
    t = t.to(device)
    k = k.to(device)
    steps = torch.arange(N_IN + N_OUT, device=device)
    frames = t[:, None] + steps[None, :] * k[:, None]  # (B, 10), 1-based
    exists = frames <= N_FRAMES
    index = (frames.clamp(max=N_FRAMES) - 1)  # 0-based, clamped for gather
    batch = torch.arange(videos.shape[0], device=device)[:, None]
    seq = videos[batch, index]  # (B, 10, C, H, W)
    seq = seq * exists[:, :, None, None, None].to(seq.dtype)
    return seq[:, :N_IN], seq[:, N_IN:], exists[:, N_IN:]


def sample_windows(n, regime, generator, speeds=(1, 2, 3)):
    """Sample (t, k) for n videos.

    regime "normal": k = 1, t uniform on 1..11.
    regime "speed":  k uniform on speeds, then t uniform on its allowed starts.
    """
    if regime == "normal":
        k = torch.ones(n, dtype=torch.long)
        t = torch.randint(1, 12, (n,), generator=generator)
        return t, k
    k = torch.tensor(speeds)[torch.randint(len(speeds), (n,), generator=generator)]
    t = torch.empty(n, dtype=torch.long)
    for speed in speeds:
        rows = k == speed
        starts = torch.tensor(AUGMENTED_STARTS[speed])
        t[rows] = starts[torch.randint(len(starts), (int(rows.sum()),), generator=generator)]
    return t, k
