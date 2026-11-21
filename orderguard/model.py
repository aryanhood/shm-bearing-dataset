"""Multi-band order-domain 1D-CNN, classical baselines, and ONNX export."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import CLASSES
from .order_tracking import CARRIER_BANDS_HZ, N_ORDER_BINS

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class OrderCNN(nn.Module):
    """Multi-band order-aware 1D-CNN operating on Order-SES [B, C, 400].

    Reads C=4 carrier envelope spectra simultaneously and preserves coarse
    order-axis coordinates (0.5 orders/bin before the MLP head) so the network
    distinguishes BPFO (3.57x), 2*BSF (4.65x), and BPFI (5.43x) under slip.
    """

    def __init__(
        self,
        in_channels: int = len(CARRIER_BANDS_HZ),
        n_bins: int = N_ORDER_BINS,
        n_classes: int = len(CLASSES),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_bins = n_bins
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


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 2e-3
    seed: int = SEED


def train_order_cnn(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    cfg: TrainConfig | None = None,
) -> OrderCNN:
    """Train OrderCNN on multi-band Order-SES inputs of shape (N, C, 400)."""
    conf = cfg or TrainConfig()
    set_seed(conf.seed)
    model = OrderCNN(
        in_channels=x_train.shape[1],
        n_bins=x_train.shape[2],
        n_classes=len(CLASSES),
    )

    xt = torch.from_numpy(x_train).float()
    yt = torch.from_numpy(y_train).long()
    loader = DataLoader(TensorDataset(xt, yt), batch_size=conf.batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=conf.epochs)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(conf.epochs):
        for bx, by in loader:
            opt.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    return model


@torch.no_grad()
def predict_proba_torch(model: OrderCNN, x: NDArray[np.float32]) -> NDArray[np.float64]:
    """Run PyTorch inference and return calibrated softmax probabilities."""
    model.eval()
    xt = torch.from_numpy(x).float()
    logits = model(xt)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return np.asarray(probs, dtype=np.float64)


def export_onnx(
    model: OrderCNN,
    onnx_path: Path,
    meta_path: Path | None = None,
) -> None:
    """Export trained OrderCNN to a self-contained single-file ONNX graph."""
    import onnx

    model.eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, model.in_channels, model.n_bins, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        export_params=True,
        opset_version=18,
        input_names=["order_ses"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}},
        dynamo=True,
        external_data=False,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    if meta_path is not None:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "classes": list(CLASSES),
            "in_channels": model.in_channels,
            "n_bins": model.n_bins,
            "onnx_bytes": onnx_path.stat().st_size,
            "params": sum(p.numel() for p in model.parameters()),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def train_random_forest(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    *,
    seed: int = SEED,
) -> Pipeline:
    """StandardScaler -> RandomForestClassifier pipeline for tabular baselines."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    return pipe
