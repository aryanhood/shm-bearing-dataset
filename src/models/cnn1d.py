"""
1D CNN Advanced Model
=====================
Residual 1D CNN: Conv blocks → GAP → FC head.
Guarded imports so the file is importable without PyTorch installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

from ..utils.logger import get_logger
from .base import BaseModel

log = get_logger("models.cnn1d")


def _make_net(channels, kernel_size, dropout, fc_dim, n_classes):
    """Build the network only when torch is available."""

    class _ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch, k, drop):
            super().__init__()
            p = k // 2
            self.net = nn.Sequential(
                nn.Conv1d(in_ch,  out_ch, k, padding=p, bias=False),
                nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
                nn.Dropout(drop),
                nn.Conv1d(out_ch, out_ch, k, padding=p, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            self.skip = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.net(x) + self.skip(x))

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            blocks, in_ch = [], 1
            for out_ch in channels:
                blocks += [_ResBlock(in_ch, out_ch, kernel_size, dropout), nn.MaxPool1d(2)]
                in_ch = out_ch
            self.encoder = nn.Sequential(*blocks)
            self.gap  = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(
                nn.Linear(in_ch, fc_dim), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(fc_dim, n_classes),
            )
        def forward(self, x):
            return self.head(self.gap(self.encoder(x)).squeeze(-1))
        def embed(self, x):
            return self.gap(self.encoder(x)).squeeze(-1)

    return _Net()


class CNN1DModel(BaseModel):
    """Residual 1D CNN classifier — raw signal input (N, W)."""

    def __init__(self, cfg: Dict[str, Any], n_classes: int = 4, seed: int = 42) -> None:
        super().__init__(cfg, seed)
        if not TORCH_OK:
            raise ImportError("PyTorch is not installed: pip install torch")

        if "models" in cfg:
            tr = cfg["training"]
            m = cfg["models"]["advanced"]
        else:
            tr = cfg.get("training", cfg)
            m = cfg.get("advanced", cfg)

        self.n_classes  = n_classes
        self.device     = torch.device(tr.get("device", "cpu"))
        self.epochs     = int(tr.get("epochs",        40))
        self.batch_size = int(tr.get("batch_size",    64))
        self.lr         = float(tr.get("lr",           0.001))
        self.wd         = float(tr.get("weight_decay", 1e-4))
        self.patience   = int(tr.get("patience",       8))

        torch.manual_seed(seed)
        self.net = _make_net(
            channels    = m.get("channels",    [64, 128, 128]),
            kernel_size = m.get("kernel_size", 5),
            dropout     = m.get("dropout",     0.3),
            fc_dim      = m.get("fc_dim",      256),
            n_classes   = n_classes,
        ).to(self.device)

        self.train_history: List[Dict] = []

    def fit(
        self,
        X:     np.ndarray,
        y:     np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "CNN1DModel":
        from torch.utils.data import DataLoader, TensorDataset

        X_t = torch.from_numpy(X[:, None, :]).to(self.device)
        y_t = torch.from_numpy(y.astype(np.int64)).to(self.device)

        counts  = np.bincount(y, minlength=self.n_classes).astype(np.float32)
        weights = torch.tensor(1.0 / (counts + 1e-6), device=self.device)
        weights /= weights.sum()

        criterion = nn.CrossEntropyLoss(weight=weights)
        optim     = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched     = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.epochs, eta_min=1e-5)
        dl        = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        best_val, no_imp = float("inf"), 0

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            ep_loss = 0.0
            for xb, yb in dl:
                optim.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optim.step()
                ep_loss += loss.item() * len(xb)
            ep_loss /= len(y)
            sched.step()

            row: Dict = {"epoch": epoch, "train_loss": ep_loss}
            if X_val is not None and y_val is not None:
                vl, va = self._eval(X_val, y_val, criterion)
                row.update(val_loss=vl, val_acc=va)
                if epoch % 5 == 0:
                    log.info(f"[{epoch:3d}/{self.epochs}] loss={ep_loss:.4f}  val_loss={vl:.4f}  val_acc={va:.4f}")
                if vl < best_val - 1e-4:
                    best_val = vl; no_imp = 0
                else:
                    no_imp += 1
                    if no_imp >= self.patience:
                        log.info(f"Early stop at epoch {epoch}")
                        self.train_history.append(row); break
            elif epoch % 5 == 0:
                log.info(f"[{epoch:3d}/{self.epochs}] loss={ep_loss:.4f}")
            self.train_history.append(row)

        self._fitted = True
        return self

    def _eval(self, X, y, criterion) -> Tuple[float, float]:
        self.net.eval()
        X_t = torch.from_numpy(X[:, None, :]).to(self.device)
        y_t = torch.from_numpy(y.astype(np.int64)).to(self.device)
        with torch.no_grad():
            logits = self.net(X_t)
            return criterion(logits, y_t).item(), (logits.argmax(1) == y_t).float().mean().item()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(torch.from_numpy(X[:, None, :]).to(self.device)), dim=1).cpu().numpy()

    def embed(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        self.net.eval()
        with torch.no_grad():
            return self.net.embed(torch.from_numpy(X[:, None, :]).to(self.device)).cpu().numpy()

    def save(self, path: Path | str) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "cfg": self.cfg, "n_classes": self.n_classes}, p)
        log.info(f"CNN1D saved -> {p}")

    def load(self, path: Path | str) -> "CNN1DModel":
        ck = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.net.load_state_dict(ck["state"])
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
