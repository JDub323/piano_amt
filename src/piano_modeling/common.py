"""Shared runtime utilities."""
from __future__ import annotations

import random
import warnings

import numpy as np
import torch


def setup_runtime(seed: int = 1337, quiet_warnings: bool = True) -> str:
    if quiet_warnings:
        warnings.filterwarnings("ignore")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return get_device()


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = get_device()
