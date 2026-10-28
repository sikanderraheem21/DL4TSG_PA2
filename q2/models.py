"""Forecasters for Q2: ConvLSTM (model A, and the configurable base for model D),
plus the two untrained references Blank and Copy-last.

Model A, one rollout step tau (shapes for one video):

    u_tau (1x64x64) --pixel_unshuffle(4)--> 16x16x16
        --> ConvLSTM 1 (3x3, 64 ch) --h1--> ConvLSTM 2 (3x3, 64 ch) --h2-->
        --> 1x1 conv 64->16 --> pixel_shuffle(4) --> sigmoid --> x_hat_{tau+1} (1x64x64)

Rollout (no teacher forcing): steps 1-5 read the five input frames, steps 6-9
read the network's own previous prediction; the outputs of steps 5-9 are Y_hat.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """[i, f, o, g] = Conv([u, h_prev]);  c = s(f) c_prev + s(i) tanh(g);  h = s(o) tanh(c).

    No peephole terms. One convolution produces all four gates.
    """

    def __init__(self, in_channels, hidden, kernel=3, dilation=1):
        super().__init__()
        self.hidden = hidden
        self.conv = nn.Conv2d(in_channels + hidden, 4 * hidden, kernel,
                              padding=dilation * (kernel // 2), dilation=dilation, bias=True)

    def forward(self, u, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([u, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """Patch-folded stacked ConvLSTM with a 9-step free-running rollout.

    Defaults are model A. `kernels` and `dilations` give one entry per layer,
    so model D can change reach without touching anything else.
    """

    def __init__(self, patch=4, hidden=64, kernels=(3, 3), dilations=(1, 1)):
        super().__init__()
        self.patch = patch
        folded = patch * patch
        self.cells = nn.ModuleList()
        in_channels = folded
        for kernel, dilation in zip(kernels, dilations):
            self.cells.append(ConvLSTMCell(in_channels, hidden, kernel, dilation))
            in_channels = hidden
        self.readout = nn.Conv2d(hidden, folded, 1, bias=True)

    def step(self, frame, states):
        u = F.pixel_unshuffle(frame.contiguous(), self.patch)
        new_states = []
        for cell, state in zip(self.cells, states):
            h, c = cell(u, state)
            new_states.append((h, c))
            u = h
        return torch.sigmoid(F.pixel_shuffle(self.readout(u), self.patch)), new_states

    def forward(self, x, n_out=5):
        """x: (B, 5, 1, 64, 64) -> Y_hat: (B, n_out, 1, 64, 64)."""
        b, n_in, _, height, width = x.shape
        size = (b, self.cells[0].hidden, height // self.patch, width // self.patch)
        states = [(x.new_zeros(size), x.new_zeros(size)) for _ in self.cells]
        outputs = []
        for tau in range(n_in + n_out - 1):
            frame = x[:, tau] if tau < n_in else prediction
            prediction, states = self.step(frame, states)
            if tau >= n_in - 1:
                outputs.append(prediction)
        return torch.stack(outputs, dim=1)


class Blank(nn.Module):
    def forward(self, x, n_out=5):
        return x.new_zeros(x.shape[0], n_out, *x.shape[2:])


class CopyLast(nn.Module):
    def forward(self, x, n_out=5):
        return x[:, -1:].expand(-1, n_out, -1, -1, -1).clone()


# Model D candidates are added here once chosen on the validation split (Q2.12).
ARCHITECTURES = {
    "A": dict(patch=4, hidden=64, kernels=(3, 3), dilations=(1, 1)),
}


def build(name):
    return ConvLSTMForecaster(**ARCHITECTURES[name])


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
