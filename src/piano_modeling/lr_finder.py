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
from .training import model_input_from_batch


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
        target_frames = batch["frame"].shape[1]
        return self.system(model_input, target_frames=target_frames)


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
