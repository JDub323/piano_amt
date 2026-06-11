"""FastAI learning-rate finder for the piano transcription model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn

from .common import DEVICE
from .config import Config
from .datasets import make_sliced_loaders
from .losses import compute_loss, move_batch_to_device
from .training import forward_from_batch, model_input_from_batch


class _FastAIBatchLoader:
    """
    Wrap a PyTorch DataLoader so FastAI sees batches as:

        x = batch_dict
        y = batch_dict

    The target is duplicated as a reference, not copied. This lets the FastAI
    loss function access onset/frame/velocity/etc. targets from the batch dict.
    """

    def __init__(self, loader, device: str):
        self.loader = loader
        self.device = device

        # FastAI sometimes inspects these attributes.
        self.dataset = getattr(loader, "dataset", None)
        self.bs = getattr(loader, "batch_size", None)
        self.n_inp = 1

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            batch = move_batch_to_device(batch, self.device)
            yield batch, batch


class _FastAIPianoModel(nn.Module):
    """
    Adapter so FastAI can call your existing PianoTranscriptionSystem with
    a full batch dictionary.
    """

    def __init__(self, system: nn.Module):
        super().__init__()
        self.system = system

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        model_input = model_input_from_batch(batch)
        return forward_from_batch(self.system, batch, model_input)


class _FastAIPianoLoss(nn.Module):
    """
    Adapter from FastAI's loss signature:

        loss(pred, target)

    to your project loss:

        compute_loss(pred, batch, cfg)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss, _ = compute_loss(pred, batch, self.cfg)
        return loss


def find_optimal_lr_fastai(
    system: nn.Module,
    cfg: Config,
    sliced_meta: Optional[pd.DataFrame] = None,
    *,
    train_loader=None,
    val_loader=None,
    train_samples_per_epoch: int = 2048,
    val_samples: int = 256,
    device: str = DEVICE,
    start_lr: float = 1e-7,
    end_lr: float = 1e-1,
    num_it: int = 100,
    suggestion: str = "valley",
    use_amp: Optional[bool] = None,
    show_plot: bool = True,
    save_plot_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Run FastAI's LR finder on the piano transcription system.

    Parameters
    ----------
    system:
        Your PianoTranscriptionSystem instance.

    cfg:
        Project Config.

    sliced_meta:
        DataFrame used by make_sliced_loaders. Required unless train_loader
        and val_loader are passed directly.

    train_loader, val_loader:
        Optional prebuilt PyTorch loaders. If omitted, this function builds
        sliced loaders from sliced_meta.

    train_samples_per_epoch:
        Number of random training chunks to expose to the LR finder.

    val_samples:
        Number of validation chunks to expose to FastAI.

    device:
        "cuda" or "cpu".

    start_lr, end_lr:
        LR sweep range.

    num_it:
        Number of mini-batches in the LR sweep.

    suggestion:
        Which FastAI suggestion to treat as the recommended LR.
        Valid common values: "valley", "slide", "steep", "minimum".

    use_amp:
        Whether to use FastAI mixed precision. Defaults to cfg.use_amp on CUDA.

    show_plot:
        Whether to show FastAI's LR-vs-loss plot.

    save_plot_path:
        Optional path to save the LR finder plot.

    Returns
    -------
    dict with:
        suggested_lr
        suggestions
        lrs
        losses
        learner
    """

    try:
        from fastai.callback.schedule import minimum, slide, steep, valley
        from fastai.callback.fp16 import MixedPrecision
        from fastai.data.core import DataLoaders
        from fastai.learner import Learner
        from fastai.optimizer import Adam
    except ImportError as exc:
        raise ImportError(
            "fastai is required for find_optimal_lr_fastai. "
            "Install it with: pip install fastai"
        ) from exc

    if train_loader is None or val_loader is None:
        if sliced_meta is None:
            raise ValueError(
                "Pass sliced_meta, or pass both train_loader and val_loader."
            )

        train_loader, val_loader, _, _ = make_sliced_loaders(
            sliced_meta,
            cfg,
            train_samples_per_epoch=train_samples_per_epoch,
            val_samples=val_samples,
            device=device,
        )

    system.to(device)

    wrapped_model = _FastAIPianoModel(system).to(device)
    loss_func = _FastAIPianoLoss(cfg)

    fastai_train_dl = _FastAIBatchLoader(train_loader, device=device)
    fastai_val_dl = _FastAIBatchLoader(val_loader, device=device)

    dls = DataLoaders(fastai_train_dl, fastai_val_dl)
    dls.n_inp = 1

    callbacks = []
    if use_amp is None:
        use_amp = bool(getattr(cfg, "use_amp", False) and device == "cuda")
    if use_amp:
        callbacks.append(MixedPrecision())

    learn = Learner(
        dls,
        wrapped_model,
        loss_func=loss_func,
        opt_func=Adam,
        wd=getattr(cfg, "weight_decay", 0.0),
        cbs=callbacks,
    )

    suggest_funcs = (minimum, steep, valley, slide)

    lr_suggestions = learn.lr_find(
        start_lr=start_lr,
        end_lr=end_lr,
        num_it=num_it,
        show_plot=show_plot,
        suggest_funcs=suggest_funcs,
    )

    suggestions = {
        name: float(value)
        for name, value in lr_suggestions._asdict().items()
        if value is not None
    }

    if suggestion not in suggestions:
        # Prefer valley, then slide, then steep, then minimum.
        for fallback in ("valley", "slide", "steep", "minimum"):
            if fallback in suggestions:
                suggestion = fallback
                break

    suggested_lr = suggestions[suggestion]

    if save_plot_path is not None:
        import matplotlib.pyplot as plt

        save_plot_path = Path(save_plot_path)
        save_plot_path.parent.mkdir(parents=True, exist_ok=True)

        learn.recorder.plot_lr_find()
        plt.savefig(save_plot_path, bbox_inches="tight", dpi=160)

    return {
        "suggested_lr": suggested_lr,
        "suggestion_used": suggestion,
        "suggestions": suggestions,
        "lrs": [float(x) for x in learn.recorder.lrs],
        "losses": [float(x.detach().cpu()) for x in learn.recorder.losses],
        "learner": learn,
    }


def find_learning_rate_range_test(
    system: nn.Module,
    cfg: Config,
    sliced_meta: Optional[pd.DataFrame] = None,
    *,
    train_loader=None,
    train_samples_per_epoch: Optional[int] = None,
    device: str = DEVICE,
    start_lr: Optional[float] = None,
    end_lr: Optional[float] = None,
    num_iters: Optional[int] = None,
    smoothing: Optional[float] = None,
    stop_divergence_factor: float = 4.0,
    restore_state: bool = True,
    save_json_path: Optional[str | Path] = None,
    save_csv_path: Optional[str | Path] = None,
    save_plot_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Dependency-light LR range test with steepest-smoothed-loss derivative selection.

    This is the practical version of "sweeping all learning rates": it exponentially sweeps from
    start_lr to end_lr over a small subset, smooths the loss curve, stops when loss diverges, and
    recommends the LR at the steepest negative derivative of smoothed loss vs log10(LR).
    """
    start_lr = float(start_lr if start_lr is not None else cfg.lr_finder_start_lr)
    end_lr = float(end_lr if end_lr is not None else cfg.lr_finder_end_lr)
    num_iters = int(num_iters if num_iters is not None else cfg.lr_finder_num_iters)
    smoothing = float(smoothing if smoothing is not None else cfg.lr_finder_smoothing)
    train_samples_per_epoch = int(train_samples_per_epoch or cfg.lr_finder_train_samples)
    if train_loader is None:
        if sliced_meta is None:
            raise ValueError("Pass sliced_meta or train_loader to find_learning_rate_range_test().")
        train_loader, _, _, _ = make_sliced_loaders(
            sliced_meta,
            cfg,
            train_samples_per_epoch=train_samples_per_epoch,
            val_samples=1,
            device=device,
        )

    system.to(device)
    model_state = {k: v.detach().cpu().clone() for k, v in system.state_dict().items()} if restore_state else None
    optimizer = torch.optim.AdamW(system.parameters(), lr=start_lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device == "cuda"))
    lrs, losses, smooth_losses = [], [], []
    best_loss = float("inf")
    system.train()
    iterator = iter(train_loader)
    ratio = (end_lr / start_lr) ** (1 / max(1, num_iters - 1))
    lr = start_lr
    try:
        for i in range(num_iters):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = move_batch_to_device(batch, device)
            model_input = model_input_from_batch(batch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and device == "cuda")):
                pred = forward_from_batch(system, batch, model_input)
                loss, _ = compute_loss(pred, batch, cfg)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            loss_value = float(loss.detach().cpu())
            if i == 0:
                smooth = loss_value
            else:
                smooth = smoothing * loss_value + (1.0 - smoothing) * smooth_losses[-1]
            lrs.append(float(lr))
            losses.append(loss_value)
            smooth_losses.append(float(smooth))
            best_loss = min(best_loss, smooth)
            if i > 8 and smooth > best_loss * stop_divergence_factor:
                break
            lr *= ratio
    finally:
        if restore_state and model_state is not None:
            system.load_state_dict(model_state, strict=True)
        del optimizer, scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(lrs) < 3:
        suggested_lr = float(lrs[-1]) if lrs else cfg.lr
        suggestion_index = len(lrs) - 1
    else:
        xs = torch.log10(torch.tensor(lrs, dtype=torch.float32))
        ys = torch.tensor(smooth_losses, dtype=torch.float32)
        slopes = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1]).clamp_min(1e-8)
        # Ignore the first couple of unstable points and the last point near divergence.
        lo = min(2, len(slopes) - 1)
        hi = max(lo + 1, len(slopes) - 1)
        local = slopes[lo:hi]
        suggestion_index = int(torch.argmin(local).item() + lo)
        suggested_lr = float(lrs[suggestion_index])

    result = {
        "suggested_lr": suggested_lr,
        "suggestion_used": "steepest_smoothed_loss_derivative",
        "suggestion_index": int(suggestion_index),
        "start_lr": start_lr,
        "end_lr": end_lr,
        "num_iters_requested": num_iters,
        "num_iters_completed": len(lrs),
        "lrs": lrs,
        "losses": losses,
        "smooth_losses": smooth_losses,
    }
    if save_json_path is not None:
        import json

        p = Path(save_json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2))
    if save_csv_path is not None:
        p = Path(save_csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"lr": lrs, "loss": losses, "smooth_loss": smooth_losses}).to_csv(p, index=False)
    if save_plot_path is not None:
        import matplotlib.pyplot as plt

        p = Path(save_plot_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.semilogx(lrs, smooth_losses, label="smoothed loss")
        plt.semilogx(lrs, losses, alpha=0.35, label="raw loss")
        plt.axvline(suggested_lr, linestyle="--", label=f"suggested {suggested_lr:.2e}")
        plt.xlabel("learning rate")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(p, bbox_inches="tight", dpi=160)
        plt.close()
    return result
