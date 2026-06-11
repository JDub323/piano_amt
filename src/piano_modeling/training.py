"""Training, validation, checkpointing, TensorBoard logging, and run orchestration."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from tqdm.auto import tqdm

from .common import DEVICE
from .config import Config, config_to_json, default_run_name
from .datasets import make_sliced_loaders
from .losses import compute_loss, move_batch_to_device
from .paths import ProjectPaths
from .profiling import make_flame_chart_profiler
from .tensorboard_debug import (
    create_summary_writer,
    log_epoch_metrics,
    log_model_debug_batch,
    log_scalar_dict,
    try_add_model_graph,
)


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
        "resnet_name", "decoder_channels", "decoder_type", "model_type", "training_data_format",
        "token_velocity_bins",
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


def forward_from_batch(system, batch: Dict[str, Any], model_input: Optional[torch.Tensor] = None):
    model_input = model_input if model_input is not None else model_input_from_batch(batch)
    if "tokens" in batch:
        return system(model_input, target_tokens=batch["tokens"][:, :-1])
    return system(model_input, target_frames=batch["frame"].shape[1])


def is_token_batch(batch: Dict[str, Any]) -> bool:
    return "tokens" in batch


def validation_selection_metric(val_stats: Dict[str, float], cfg: Config) -> float:
    if "token_ce" in val_stats or str(getattr(cfg, "model_type", "framewise")).lower() in ("token_seq2seq", "seq2seq", "tokenwise"):
        return -float(val_stats.get("total", float("inf")))
    return float(val_stats.get("onset_f1", 0.0) + val_stats.get("frame_f1", 0.0) + val_stats.get("offset_f1", 0.0))


def _maybe_apply_found_training_settings(cfg: Config) -> dict:
    if not getattr(cfg, "use_found_training_settings", False):
        return {}
    path = getattr(cfg, "found_training_settings_path", "")
    if not path:
        print("cfg.use_found_training_settings=True, but cfg.found_training_settings_path is empty; keeping current settings.")
        return {}
    p = Path(path)
    if not p.exists():
        print(f"Found training settings file does not exist: {p}; keeping current settings.")
        return {}
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        ok = df[df.get("status", "") == "ok"] if "status" in df.columns else df
        if len(ok) == 0:
            print(f"No usable rows in found training settings file: {p}")
            return {}
        row = ok.sort_values("chunks_per_sec", ascending=False).iloc[0].to_dict() if "chunks_per_sec" in ok.columns else ok.iloc[0].to_dict()
    else:
        row = json.loads(p.read_text())
        if "best" in row and isinstance(row["best"], dict):
            row = row["best"]
    applied = {}
    for key in ("batch_size", "num_workers", "use_amp"):
        if key in row and pd.notna(row[key]):
            value = bool(row[key]) if key == "use_amp" else int(row[key])
            setattr(cfg, key, value)
            applied[key] = value
    if applied:
        print("Applied found training-speed settings:", applied)
    return applied


def _maybe_apply_found_learning_rate(cfg: Config) -> Optional[float]:
    if not getattr(cfg, "use_found_learning_rate", False):
        return None
    path = getattr(cfg, "found_learning_rate_path", "")
    if not path:
        print("cfg.use_found_learning_rate=True, but cfg.found_learning_rate_path is empty; keeping cfg.lr.")
        return None
    p = Path(path)
    if not p.exists():
        print(f"Found learning-rate file does not exist: {p}; keeping cfg.lr={cfg.lr}.")
        return None
    data = json.loads(p.read_text())
    lr = data.get("suggested_lr", data.get("lr"))
    if lr is None:
        print(f"No suggested_lr in {p}; keeping cfg.lr={cfg.lr}.")
        return None
    cfg.lr = float(lr)
    print(f"Applied found learning rate: cfg.lr={cfg.lr:g}")
    return cfg.lr


def train_one_epoch(
    system,
    loader,
    optimizer,
    scaler,
    cfg: Config,
    epoch: int,
    device: str = DEVICE,
    *,
    writer=None,
    tensorboard_log_every: int = 100,
    tensorboard_log_graph: bool = False,
    profiler_trace_dir: Optional[str | Path] = None,
):
    system.train()
    loss_sums, n_examples = {}, 0
    ema = {}
    graph_logged = False
    pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    with make_flame_chart_profiler(cfg, profiler_trace_dir, enabled=(epoch == 1)) as profiler:
        for i, batch in enumerate(pbar):
            batch = move_batch_to_device(batch, device)
            model_input = model_input_from_batch(batch)
            global_step = max(0, epoch - 1) * max(1, len(loader)) + i
            if i == 0 and device == "cuda":
                mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                mem_res = torch.cuda.memory_reserved() / (1024 ** 3)
                print(f"\n[GPU Memory - Batch 0] Allocated: {mem_alloc:.2f} GB, Reserved: {mem_res:.2f} GB")
                log_scalar_dict(
                    writer,
                    "debug_cuda_memory_gb",
                    {"allocated": mem_alloc, "reserved": mem_res},
                    global_step,
                )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and device == "cuda")):
                pred = forward_from_batch(system, batch, model_input)
                loss, loss_dict = compute_loss(pred, batch, cfg)

            if writer is not None and tensorboard_log_graph and not graph_logged and not is_token_batch(batch):
                graph_logged = try_add_model_graph(writer, system, model_input, int(batch["frame"].shape[1]))

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

            if writer is not None and tensorboard_log_every > 0 and i % tensorboard_log_every == 0 and not is_token_batch(batch):
                log_model_debug_batch(writer, system, model_input, batch, pred, loss_dict, cfg, global_step)

            if profiler is not None:
                profiler.step()

            bs = int(model_input.shape[0])
            n_examples += bs
            for k, v in loss_dict.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v * bs
                ema[k] = ema.get(k, v) * 0.98 + v * 0.02 if k in ema else v
            pbar.set_postfix({k: f"{v:.4f}" for k, v in ema.items() if k in ["total", "onset", "frame", "velocity", "token_ce"]})
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
    """Validation with epoch-averaged losses.

    Framewise runs also report dense-label F1. Token seq2seq runs report token CE and
    teacher-forced token accuracy; checkpoint selection uses negative validation loss.
    """
    system.eval()
    loss_sums, n_examples = {}, 0
    counts = {
        "onset": [0.0, 0.0, 0.0],
        "offset": [0.0, 0.0, 0.0],
        "frame": [0.0, 0.0, 0.0],
        "sustain": [0.0, 0.0, 0.0],
    }
    saw_token_batch = False
    for batch in tqdm(loader, desc="validate", leave=False):
        batch = move_batch_to_device(batch, device)
        model_input = model_input_from_batch(batch)
        pred = forward_from_batch(system, batch, model_input)
        _, loss_dict = compute_loss(pred, batch, cfg)
        bs = int(model_input.shape[0])
        n_examples += bs
        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + v * bs
        if is_token_batch(batch):
            saw_token_batch = True
            continue
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
    if not saw_token_batch:
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
    *,
    enable_tensorboard: bool = True,
    tensorboard_log_dir: Optional[str | Path] = None,
    tensorboard_log_every: int = 100,
    tensorboard_log_graph: bool = True,
    formal_eval_meta: Optional[pd.DataFrame] = None,
    formal_eval_split: str = "validation",
    formal_eval_n_tracks: int = 0,
):
    """Create loaders/optimizer/checkpoints and run the training loop.

    ``validate_quick`` remains the cheap per-epoch dense-label check. When ``formal_eval_meta`` and
    ``formal_eval_n_tracks`` are provided, the run also performs note-level evaluation with the same
    machinery used by the ``piano-evaluate`` CLI whenever a new best checkpoint is found.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    applied_training_settings = _maybe_apply_found_training_settings(cfg)
    applied_lr = _maybe_apply_found_learning_rate(cfg)

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
    interrupted_ckpt = run_dir / "interrupted.pt"
    run_dir.mkdir(parents=True, exist_ok=True)

    writer_dir = Path(tensorboard_log_dir) if tensorboard_log_dir is not None else run_dir / "tensorboard"
    flame_chart_dir = Path(cfg.flame_chart_output_dir) if getattr(cfg, "flame_chart_output_dir", "") else run_dir / "flame_charts"
    writer = create_summary_writer(writer_dir, enabled=enable_tensorboard)
    if writer is not None:
        writer.add_text("config/json", f"```json\n{config_to_json(cfg)}\n```", 0)
        writer.add_text("run/name", run_name, 0)

    start_epoch = 1
    current_epoch = 0
    best_metric = -1.0
    history = []
    if resume and last_ckpt.exists():
        ckpt = load_checkpoint(last_ckpt, system, optimizer, map_location=device, cfg=cfg)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        current_epoch = start_epoch - 1
        best_metric = float(ckpt.get("best_metric", -1.0))
        history = ckpt.get("extra", {}).get("history", []) or []
        print(f"Resumed from {last_ckpt}, starting epoch {start_epoch}, best_metric={best_metric:.4f}")
    elif resume:
        print(f"resume=True, but no checkpoint found at {last_ckpt}. Starting a new run.")

    print("Run directory:", run_dir)
    print("RUN_NAME:", run_name)
    if writer is not None:
        print("TensorBoard log directory:", writer_dir)
    if getattr(cfg, "enable_flame_charts", False):
        print("Flame chart trace directory:", flame_chart_dir)

    interrupted = False
    interrupt_stage = None
    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            current_epoch = epoch
            train_stats = train_one_epoch(
                system,
                train_loader,
                optimizer,
                scaler,
                cfg,
                epoch,
                device=device,
                writer=writer,
                tensorboard_log_every=tensorboard_log_every,
                tensorboard_log_graph=(tensorboard_log_graph and epoch == start_epoch),
                profiler_trace_dir=flame_chart_dir,
            )
            val_stats = validate_quick(system, val_loader, cfg, device=device)
            metric = validation_selection_metric(val_stats, cfg)
            is_best = metric > best_metric
            if is_best:
                best_metric = metric

            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "selection_metric": metric,
                "best_metric": best_metric,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            save_checkpoint(last_ckpt, system, optimizer, epoch, cfg, best_metric, {"history": history})
            if is_best:
                save_checkpoint(best_ckpt, system, optimizer, epoch, cfg, best_metric, {"history": history})
                print(f"Epoch {epoch}: new best metric={best_metric:.4f}; saved {best_ckpt}")
                if formal_eval_meta is not None and formal_eval_n_tracks and formal_eval_n_tracks > 0:
                    try:
                        from .evaluation import evaluate_split_sample, summarize_evaluation_report

                        report = evaluate_split_sample(
                            formal_eval_meta,
                            system,
                            paths.export_dir,
                            run_name,
                            split=formal_eval_split,
                            n_tracks=formal_eval_n_tracks,
                            checkpoint_path=None,
                            cfg=cfg,
                            device=device,
                        )
                        formal_summary = summarize_evaluation_report(report)
                        row.update({f"formal_eval_{k}": v for k, v in formal_summary.items()})
                        log_scalar_dict(writer, "epoch/formal_eval", formal_summary, epoch)
                    except KeyboardInterrupt:
                        interrupted = True
                        interrupt_stage = "formal_eval_inference"
                        print("KeyboardInterrupt during formal-eval inference. Current inference work was discarded; returning partial train_result.")
                        save_checkpoint(interrupted_ckpt, system, optimizer, epoch, cfg, best_metric, {"history": history, "interrupted": True, "interrupt_stage": interrupt_stage})
                        break
            log_epoch_metrics(writer, train_stats, val_stats, epoch)
            print(row)
    except KeyboardInterrupt:
        interrupted = True
        interrupt_stage = "training"
        print("KeyboardInterrupt during training. Current batch/epoch work was discarded; saving interrupted.pt and returning partial train_result.")
        save_checkpoint(interrupted_ckpt, system, optimizer, current_epoch, cfg, best_metric, {"history": history, "interrupted": True, "interrupt_stage": interrupt_stage})
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "last_ckpt": last_ckpt,
        "best_ckpt": best_ckpt,
        "interrupted_ckpt": interrupted_ckpt if interrupted else None,
        "interrupted": interrupted,
        "interrupt_stage": interrupt_stage,
        "history": history,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "tensorboard_dir": writer_dir if enable_tensorboard else None,
        "flame_chart_dir": flame_chart_dir if getattr(cfg, "enable_flame_charts", False) else None,
        "applied_training_settings": applied_training_settings,
        "applied_learning_rate": applied_lr,
    }
