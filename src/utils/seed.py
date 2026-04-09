"""
Reproducibility
===============
Call ``set_all_seeds(42)`` once at program start.
Covers Python, NumPy, and PyTorch (CPU + CUDA).
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_all_seeds(seed: int = 42) -> None:
    """Fix every random source we know about."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
