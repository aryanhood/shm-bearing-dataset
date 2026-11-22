"""Classical tabular baselines and compact Matryoshka 1D-CNN architectures."""

from __future__ import annotations

from .classical import predict_proba_sklearn, train_mlp, train_random_forest
from .cnn import (
    NESTED_DIMS,
    MatryoshkaOrderCNN,
    OrderCNN,
    SlicedMatryoshkaCNN,
    TrainConfig,
    predict_logits_torch,
    predict_nested_logits_torch,
    train_matryoshka_cnn,
    train_order_cnn,
)

__all__ = [
    "NESTED_DIMS",
    "MatryoshkaOrderCNN",
    "OrderCNN",
    "SlicedMatryoshkaCNN",
    "TrainConfig",
    "predict_logits_torch",
    "predict_nested_logits_torch",
    "predict_proba_sklearn",
    "train_matryoshka_cnn",
    "train_mlp",
    "train_order_cnn",
    "train_random_forest",
]
