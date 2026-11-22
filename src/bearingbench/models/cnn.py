"""Compact 1D-CNN and Matryoshka Representation Learning (MRL) architectures."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

NESTED_DIMS: tuple[int, ...] = (8, 16, 32, 64, 128)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class OrderCNN(nn.Module):
    """Single-granularity multi-band 1D-CNN operating on envelope spectra [B, C, L]."""

    def __init__(
        self,
        in_channels: int = 4,
        n_bins: int = 400,
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_bins = n_bins
        self.n_classes = n_classes
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=7, padding=3),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        reduced = n_bins // 8
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(32 * reduced, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.head(self.features(x)))


class MatryoshkaOrderCNN(nn.Module):
    """Multi-band 1D-CNN with nested Matryoshka Representation Learning (Kusupati et al., 2022).

    Trains a single 128-d representation jointly across nested prefix dimensions
    D = (8, 16, 32, 64, 128) with weight-tied classifier slices, enabling elastic
    edge deployment without retraining.
    """

    def __init__(
        self,
        in_channels: int = 4,
        n_bins: int = 400,
        n_classes: int = 5,
        nested_dims: tuple[int, ...] = NESTED_DIMS,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_bins = n_bins
        self.n_classes = n_classes
        self.nested_dims = nested_dims
        self.max_dim = nested_dims[-1]

        self.conv_backbone = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=7, padding=3),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(4),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4, self.max_dim),
            nn.ReLU(),
        )
        self.drop = nn.Dropout(0.25)
        self.classifier = nn.Linear(self.max_dim, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.proj(self.conv_backbone(x)))

    def logits_at_dim(self, z: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute logits using only the top-d nested embedding prefix."""
        w = self.classifier.weight[:, :dim]
        b = self.classifier.bias
        return torch.nn.functional.linear(z[:, :dim], w, b)

    def forward_nested(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        z = self.drop(self.encode(x))
        return {d: self.logits_at_dim(z, d) for d in self.nested_dims}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.drop(self.encode(x))
        return cast(torch.Tensor, self.classifier(z))


class SlicedMatryoshkaCNN(nn.Module):
    """Physically sliced sub-model for exporting an arbitrary prefix width d to ONNX."""

    def __init__(self, parent: MatryoshkaOrderCNN, dim: int) -> None:
        super().__init__()
        if dim <= 0 or dim > parent.max_dim:
            raise ValueError(f"dim={dim} out of range (1..{parent.max_dim})")
        self.in_channels = parent.in_channels
        self.n_bins = parent.n_bins
        self.n_classes = parent.n_classes
        self.dim = dim
        self.conv_backbone = parent.conv_backbone
        # Slice projection & classifier weights physically so exported ONNX size/MACs shrink.
        orig_linear = cast(nn.Linear, parent.proj[1])
        self.proj_sliced = nn.Linear(orig_linear.in_features, dim)
        with torch.no_grad():
            self.proj_sliced.weight.copy_(orig_linear.weight[:dim])
            self.proj_sliced.bias.copy_(orig_linear.bias[:dim])
        self.relu = nn.ReLU()
        self.head_sliced = nn.Linear(dim, parent.n_classes)
        with torch.no_grad():
            self.head_sliced.weight.copy_(parent.classifier.weight[:, :dim])
            self.head_sliced.bias.copy_(parent.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = torch.flatten(self.conv_backbone(x), start_dim=1)
        z = self.relu(self.proj_sliced(feat))
        return cast(torch.Tensor, self.head_sliced(z))


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 2.5e-3
    weight_decay: float = 2e-3
    seed: int = 42
    n_threads: int = 4


def train_order_cnn(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    *,
    n_classes: int,
    cfg: TrainConfig | None = None,
) -> OrderCNN:
    """Train standard single-granularity OrderCNN using direct tensor slicing."""
    conf = cfg or TrainConfig()
    set_seed(conf.seed)
    torch.set_num_threads(conf.n_threads)
    model = OrderCNN(in_channels=x_train.shape[1], n_bins=x_train.shape[2], n_classes=n_classes)
    xt = torch.from_numpy(x_train).float()
    yt = torch.from_numpy(y_train).long()
    n = len(xt)
    opt = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=conf.epochs)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(conf.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, conf.batch_size):
            idx = perm[i : i + conf.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def train_matryoshka_cnn(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    *,
    n_classes: int,
    cfg: TrainConfig | None = None,
    nested_dims: tuple[int, ...] = NESTED_DIMS,
) -> MatryoshkaOrderCNN:
    """Train MatryoshkaOrderCNN optimizing multi-granularity nested cross-entropy."""
    conf = cfg or TrainConfig()
    set_seed(conf.seed)
    torch.set_num_threads(conf.n_threads)
    model = MatryoshkaOrderCNN(
        in_channels=x_train.shape[1],
        n_bins=x_train.shape[2],
        n_classes=n_classes,
        nested_dims=nested_dims,
    )
    xt = torch.from_numpy(x_train).float()
    yt = torch.from_numpy(y_train).long()
    n = len(xt)
    opt = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=conf.epochs)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(conf.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, conf.batch_size):
            idx = perm[i : i + conf.batch_size]
            opt.zero_grad()
            logits_map = model.forward_nested(xt[idx])
            loss = sum(loss_fn(lg, yt[idx]) for lg in logits_map.values()) / float(len(logits_map))
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def predict_logits_torch(model: nn.Module, x: NDArray[np.float32]) -> NDArray[np.float64]:
    """Return raw logits from a PyTorch classifier."""
    model.eval()
    xt = torch.from_numpy(x).float()
    return np.asarray(model(xt).cpu().numpy(), dtype=np.float64)


@torch.no_grad()
def predict_nested_logits_torch(
    model: MatryoshkaOrderCNN, x: NDArray[np.float32]
) -> dict[int, NDArray[np.float64]]:
    """Return logits at every nested prefix width d in model.nested_dims."""
    model.eval()
    xt = torch.from_numpy(x).float()
    out = model.forward_nested(xt)
    return {d: np.asarray(v.cpu().numpy(), dtype=np.float64) for d, v in out.items()}
