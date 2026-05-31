"""Training, validation, checkpointing, and run orchestration."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from tqdm.auto import tqdm

from .common import DEVICE
from .config import Config, default_run_name
from .datasets import make_sliced_loaders
from .losses import compute_loss, move_batch_to_device
from .paths import ProjectPaths


def save_checkpoint(path: Path, system, optimizer, epoch: int, cfg: Config, best_metric: float, extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "cfg": asdict(cfg),
        "model_state": system.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "best_metric": best_metric,
        "extra": extra or {},
    }
    torch.save(payload, path)


def assert_checkpoint_config_compatible(ckpt: dict, cfg: Config) -> None:
    saved = ckpt.get("cfg") or {}
    if not saved:
        print("WARNING: checkpoint has no saved config; cannot verify compatibility.")
        return
    keys = [
        "feature_type", "sample_rate", "hop_length", "n_fft", "n_mels",
        "f_min", "f_max", "segment_seconds", "midi_min", "midi_max",
        "resnet_name", "decoder_channels",
    ]
    mismatches = []
    current = asdict(cfg)
    for k in keys:
        if k in saved and saved[k] != current[k]:
            mismatches.append((k, saved[k], current[k]))
    if mismatches:
        msg = "\n".join([f"  {k}: checkpoint={old!r}, current={new!r}" for k, old, new in mismatches])
        raise ValueError(
            "Checkpoint config is incompatible with the current cfg. "
            "Use the matching config/checkpoint pair or start a new run.\n" + msg
        )


def load_checkpoint(path: Path, system=None, optimizer=None, map_location="cpu", cfg: Optional[Config] = None):
    ckpt = torch.load(path, map_location=map_location)
    if cfg is not None:
        assert_checkpoint_config_compatible(ckpt, cfg)
    if system is not None:
        system.load_state_dict(ckpt["model_state"], strict=True)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def model_input_from_batch(batch: Dict[str, Any]) -> torch.Tensor:
    return batch["spec"] if "spec" in batch else batch["audio"]


def train_one_epoch(system, loader, optimizer, scaler, cfg: Config, epoch: int, device: str = DEVICE):
    system.train()
    loss_sums, n_examples = {}, 0
    ema = {}
    pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for i, batch in enumerate(pbar):
        batch = move_batch_to_device(batch, device)
        model_input = model_input_from_batch(batch)
        if i == 0 and device == "cuda":
            mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_res = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"\n[GPU Memory - Batch 0] Allocated: {mem_alloc:.2f} GB, Reserved: {mem_res:.2f} GB")

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and device == "cuda")):
            pred = system(model_input, target_frames=batch["frame"].shape[1])
            loss, loss_dict = compute_loss(pred, batch, cfg)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(system.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(system.parameters(), cfg.grad_clip)
            optimizer.step()

        bs = int(model_input.shape[0])
        n_examples += bs
        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + v * bs
            ema[k] = ema.get(k, v) * 0.98 + v * 0.02 if k in ema else v
        pbar.set_postfix({k: f"{v:.4f}" for k, v in ema.items() if k in ["total", "onset", "frame", "velocity"]})
    return {k: v / max(1, n_examples) for k, v in loss_sums.items()}


@torch.no_grad()
def binary_counts_from_logits(logits, target, threshold: float = 0.5):
    pred = (torch.sigmoid(logits) >= threshold)
    target = (target >= 0.5)
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    return float(tp), float(fp), float(fn)


def f1_from_counts(tp: float, fp: float, fn: float):
    precision = tp / max(1e-8, tp + fp)
    recall = tp / max(1e-8, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return precision, recall, f1


@torch.no_grad()
def validate_quick(system, loader, cfg: Config, device: str = DEVICE):
    """Validation with epoch-averaged losses and globally accumulated F1 counts."""
    system.eval()
    loss_sums, n_examples = {}, 0
    counts = {
        "onset": [0.0, 0.0, 0.0],
        "offset": [0.0, 0.0, 0.0],
        "frame": [0.0, 0.0, 0.0],
        "sustain": [0.0, 0.0, 0.0],
    }
    for batch in tqdm(loader, desc="validate", leave=False):
        batch = move_batch_to_device(batch, device)
        model_input = model_input_from_batch(batch)
        pred = system(model_input, target_frames=batch["frame"].shape[1])
        _, loss_dict = compute_loss(pred, batch, cfg)
        bs = int(model_input.shape[0])
        n_examples += bs
        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + v * bs
        for name, threshold in [
            ("onset", cfg.onset_threshold),
            ("offset", cfg.offset_threshold),
            ("frame", cfg.frame_threshold),
            ("sustain", 0.5),
        ]:
            tp, fp, fn = binary_counts_from_logits(pred[name], batch[name], threshold)
            counts[name][0] += tp
            counts[name][1] += fp
            counts[name][2] += fn
    out = {k: v / max(1, n_examples) for k, v in loss_sums.items()}
    for name, (tp, fp, fn) in counts.items():
        precision, recall, f1 = f1_from_counts(tp, fp, fn)
        out[f"{name}_precision"] = precision
        out[f"{name}_recall"] = recall
        out[f"{name}_f1"] = f1
    return out


def run_training(
    system,
    cfg: Config,
    paths: ProjectPaths,
    sliced_meta: pd.DataFrame,
    run_name: Optional[str] = None,
    custom_run_suffix: str = "",
    train_samples_per_epoch: int = 20000,
    val_samples: int = 0,
    resume: bool = True,
    device: str = DEVICE,
):
    """Create loaders/optimizer/checkpoints and run the training loop."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    run_name = run_name or default_run_name(cfg)
    if custom_run_suffix.strip():
        run_name = f"{run_name}_{custom_run_suffix.strip()}"

    train_loader, val_loader, train_ds, val_ds = make_sliced_loaders(
        sliced_meta,
        cfg,
        train_samples_per_epoch=train_samples_per_epoch,
        val_samples=val_samples,
        device=device,
    )

    optimizer = torch.optim.AdamW(system.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device == "cuda"))
    run_dir = paths.checkpoint_dir / run_name
    last_ckpt = run_dir / "last.pt"
    best_ckpt = run_dir / "best.pt"
    run_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_metric = -1.0
    history = []
    if resume and last_ckpt.exists():
        ckpt = load_checkpoint(last_ckpt, system, optimizer, map_location=device, cfg=cfg)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", -1.0))
        history = ckpt.get("extra", {}).get("history", []) or []
        print(f"Resumed from {last_ckpt}, starting epoch {start_epoch}, best_metric={best_metric:.4f}")
    elif resume:
        print(f"resume=True, but no checkpoint found at {last_ckpt}. Starting a new run.")

    print("Run directory:", run_dir)
    print("RUN_NAME:", run_name)

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_stats = train_one_epoch(system, train_loader, optimizer, scaler, cfg, epoch, device=device)
        val_stats = validate_quick(system, val_loader, cfg, device=device)
        metric = val_stats.get("onset_f1", 0.0) + val_stats.get("frame_f1", 0.0) + val_stats.get("offset_f1", 0.0)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        save_checkpoint(last_ckpt, system, optimizer, epoch, cfg, best_metric, {"history": history})
        if metric > best_metric:
            best_metric = metric
            save_checkpoint(best_ckpt, system, optimizer, epoch, cfg, best_metric, {"history": history})
            print(f"Epoch {epoch}: new best metric={best_metric:.4f}; saved {best_ckpt}")
        print(row)

    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "last_ckpt": last_ckpt,
        "best_ckpt": best_ckpt,
        "history": history,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_ds": train_ds,
        "val_ds": val_ds,
    }
