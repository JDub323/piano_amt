"""Losses, batch device movement, and simple frame metrics."""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from .config import Config


def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def bce_logits(logits: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)


def compute_loss(pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: Config):
    losses = {}
    losses["onset"] = bce_logits(pred["onset"], batch["onset"], cfg.onset_pos_weight) * cfg.onset_loss_weight
    losses["offset"] = bce_logits(pred["offset"], batch["offset"], cfg.offset_pos_weight) * cfg.offset_loss_weight
    losses["frame"] = bce_logits(pred["frame"], batch["frame"], cfg.frame_pos_weight) * cfg.frame_loss_weight
    losses["sustain"] = bce_logits(pred["sustain"], batch["sustain"], cfg.sustain_pos_weight) * cfg.sustain_loss_weight

    vel_pred = torch.sigmoid(pred["velocity"])
    vel_mask = batch["onset"].clamp(0, 1)
    losses["velocity"] = (((vel_pred - batch["velocity"]) ** 2) * vel_mask).sum() / vel_mask.sum().clamp_min(1.0)
    losses["velocity"] = losses["velocity"] * cfg.velocity_loss_weight

    total = sum(losses.values())
    losses["total"] = total
    return total, {k: float(v.detach().cpu()) for k, v in losses.items()}


@torch.no_grad()
def binary_f1_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    pred = (torch.sigmoid(logits) >= threshold).float()
    target = (target >= 0.5).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision.cpu()), float(recall.cpu()), float(f1.cpu())
