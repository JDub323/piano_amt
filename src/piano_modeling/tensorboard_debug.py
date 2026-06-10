"""TensorBoard/SummaryWriter debugging helpers.

These helpers deliberately avoid assumptions about the concrete model architecture. They log what
is available through the stable training contract: a model input tensor, an output dictionary, the
batch targets, optional loss scalars, and named model parameters/gradients.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn


OUTPUT_KEYS = ("onset", "offset", "frame", "velocity", "sustain")


def create_summary_writer(log_dir: str | Path | None, enabled: bool = True):
    """Create a TensorBoard SummaryWriter, returning ``None`` when unavailable/disabled."""
    if not enabled or log_dir is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - depends on optional tensorboard install
        print(f"TensorBoard logging disabled: could not import SummaryWriter ({exc})")
        return None
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def _sample_flat(x: torch.Tensor, max_points: int = 200_000) -> torch.Tensor:
    x = x.detach().float().flatten().cpu()
    if x.numel() > max_points:
        idx = torch.linspace(0, x.numel() - 1, max_points).long()
        x = x[idx]
    return x


def _normalize_image(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    lo, hi = torch.nan_to_num(x).min(), torch.nan_to_num(x).max()
    return (x - lo) / (hi - lo).clamp_min(1e-6)


def _first_image_from_tensor(x: torch.Tensor) -> Optional[torch.Tensor]:
    """Return a [1, H, W] image from common model input/output tensor shapes."""
    if not torch.is_tensor(x) or x.numel() == 0:
        return None
    x = x.detach()
    if x.ndim == 4:  # [B, C, F, T]
        img = x[0, 0]
    elif x.ndim == 3:  # [B, F, T] or [B, T, P]
        img = x[0]
    elif x.ndim == 2:
        img = x
    else:
        return None
    return _normalize_image(img).unsqueeze(0)


def log_scalar_dict(writer, prefix: str, values: Mapping[str, Any], step: int) -> None:
    if writer is None:
        return
    for key, value in values.items():
        try:
            if torch.is_tensor(value):
                value = float(value.detach().cpu())
            if isinstance(value, (int, float)):
                writer.add_scalar(f"{prefix}/{key}", value, step)
        except Exception:
            continue


def log_epoch_metrics(writer, train_stats: Mapping[str, float], val_stats: Mapping[str, float], epoch: int) -> None:
    """Log epoch-level metrics under stable namespaces."""
    log_scalar_dict(writer, "epoch/train", train_stats, epoch)
    log_scalar_dict(writer, "epoch/validation", val_stats, epoch)
    if writer is not None:
        writer.flush()


class _GraphTraceWrapper(nn.Module):
    """Trace-friendly wrapper for models that return dictionaries."""

    def __init__(self, model: nn.Module, target_frames: Optional[int]):
        super().__init__()
        self.model = model
        self.target_frames = target_frames

    def forward(self, x: torch.Tensor):
        out = self.model(x, target_frames=self.target_frames)
        return tuple(out[k] for k in OUTPUT_KEYS if k in out)


def try_add_model_graph(writer, model: nn.Module, model_input: torch.Tensor, target_frames: Optional[int]) -> bool:
    """Best-effort graph trace. Returns True when a graph was logged.

    The model's original train/eval mode is restored because this helper can be called from inside
    the training loop.
    """
    if writer is None:
        return False
    was_training = model.training
    try:
        model.eval()
        wrapper = _GraphTraceWrapper(model, target_frames)
        writer.add_graph(wrapper, model_input[:1].detach())
        return True
    except Exception as exc:  # pragma: no cover - tracing may fail on some dynamic architectures
        writer.add_text("debug/graph_trace_warning", str(exc), 0)
        return False
    finally:
        model.train(was_training)


def _threshold_for_key(key: str, cfg) -> float:
    if key == "onset":
        return float(getattr(cfg, "onset_threshold", 0.5))
    if key == "offset":
        return float(getattr(cfg, "offset_threshold", 0.5))
    if key == "frame":
        return float(getattr(cfg, "frame_threshold", 0.5))
    return 0.5


def _add_pr_curve(writer, tag: str, target: torch.Tensor, prob: torch.Tensor, step: int) -> None:
    labels = _sample_flat(target >= 0.5, max_points=50_000).bool()
    preds = _sample_flat(prob, max_points=50_000)
    if labels.numel() == 0 or labels.float().sum() == 0:
        return
    writer.add_pr_curve(tag, labels, preds, step)


def log_model_debug_batch(
    writer,
    model: nn.Module,
    model_input: torch.Tensor,
    batch: Mapping[str, Any],
    pred: Mapping[str, torch.Tensor],
    loss_dict: Optional[Mapping[str, float]],
    cfg,
    step: int,
    *,
    max_parameter_histograms: int = 24,
) -> None:
    """Log architecture-agnostic model debugging signals.

    Important things to watch in TensorBoard:
    - loss components and validation F1 by head;
    - output probability histograms, especially saturated all-zero/all-one heads;
    - target density vs prediction activation rate;
    - PR curves for sparse heads such as onset/offset;
    - parameter and gradient histograms/norms for dead or exploding layers;
    - visual spectrogram/probability images for quick sanity checks.
    """
    if writer is None:
        return

    if loss_dict:
        log_scalar_dict(writer, "batch/loss", loss_dict, step)

    img = _first_image_from_tensor(model_input)
    if img is not None:
        writer.add_image("debug/input_channel0", img, step)

    for key in OUTPUT_KEYS:
        if key not in pred:
            continue
        logits = pred[key].detach()
        prob = torch.sigmoid(logits)
        threshold = _threshold_for_key(key, cfg)
        writer.add_histogram(f"debug_logits/{key}", _sample_flat(logits), step)
        writer.add_histogram(f"debug_probabilities/{key}", _sample_flat(prob), step)
        writer.add_scalar(f"debug_probabilities/{key}_mean", float(prob.mean().cpu()), step)
        writer.add_scalar(f"debug_probabilities/{key}_max", float(prob.max().cpu()), step)
        writer.add_scalar(
            f"debug_activation_rate/{key}_at_threshold",
            float((prob >= threshold).float().mean().cpu()),
            step,
        )

        if key in batch and torch.is_tensor(batch[key]):
            target = batch[key].detach()
            writer.add_scalar(f"debug_target_density/{key}", float((target >= 0.5).float().mean().cpu()), step)
            try:
                _add_pr_curve(writer, f"debug_pr_curve/{key}", target, prob, step)
            except Exception:
                pass
            target_img = _first_image_from_tensor(target)
            if target_img is not None and key in {"onset", "frame", "offset"}:
                writer.add_image(f"debug_targets/{key}", target_img, step)

        prob_img = _first_image_from_tensor(prob)
        if prob_img is not None and key in {"onset", "frame", "offset", "velocity"}:
            writer.add_image(f"debug_predictions/{key}", prob_img, step)

    total_param_norm_sq = 0.0
    total_grad_norm_sq = 0.0
    histograms_logged = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_norm = float(param.detach().float().norm().cpu())
        total_param_norm_sq += param_norm**2
        writer.add_scalar(f"debug_param_norm/{name}", param_norm, step)
        if histograms_logged < max_parameter_histograms:
            writer.add_histogram(f"debug_parameters/{name}", _sample_flat(param), step)
            histograms_logged += 1
        if param.grad is not None:
            grad_norm = float(param.grad.detach().float().norm().cpu())
            total_grad_norm_sq += grad_norm**2
            writer.add_scalar(f"debug_grad_norm/{name}", grad_norm, step)
            if histograms_logged < max_parameter_histograms:
                writer.add_histogram(f"debug_gradients/{name}", _sample_flat(param.grad), step)
                histograms_logged += 1

    writer.add_scalar("debug_global_norm/parameters", total_param_norm_sq**0.5, step)
    writer.add_scalar("debug_global_norm/gradients", total_grad_norm_sq**0.5, step)
    writer.flush()
