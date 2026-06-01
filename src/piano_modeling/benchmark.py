"""Training-only benchmark helpers for batch size/worker sweeps."""
from __future__ import annotations

import copy
import gc
import itertools
import time
from typing import Iterable

import pandas as pd
import torch

from .common import DEVICE
from .config import CFG, Config
from .datasets import make_sliced_loaders
from .losses import compute_loss, move_batch_to_device
from .models import PianoTranscriptionSystem
from .training import model_input_from_batch


torch.backends.cudnn.benchmark = True


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def describe_runtime(device: str = DEVICE) -> None:
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory: {props.total_memory / 1024**3:.1f} GB")
        print("CUDA:", torch.version.cuda)
    print("PyTorch:", torch.__version__)


def _safe_make_train_loader(sliced_meta: pd.DataFrame, cfg: Config, train_samples: int = 1024, device: str = DEVICE):
    train_loader, _, train_ds, _ = make_sliced_loaders(sliced_meta, cfg, train_samples_per_epoch=train_samples, val_samples=1, device=device)
    return train_loader, train_ds


def _next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def benchmark_training_setting(
    sliced_meta: pd.DataFrame,
    *,
    batch_size: int,
    num_workers: int,
    base_cfg: Config = CFG,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    lr: float | None = None,
    compile_model: bool = True,
    train_samples: int = 1024,
    warmup_steps: int = 2,
    timed_optimizer_steps: int = 8,
    device: str = DEVICE,
):
    cleanup_cuda()
    cfg = copy.deepcopy(base_cfg)
    cfg.batch_size = int(batch_size)
    cfg.num_workers = int(num_workers)
    cfg.use_amp = bool(use_amp)
    if lr is None:
        lr = cfg.lr

    result = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "use_amp": use_amp,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "lr": lr,
        "compile_model": compile_model,
        "status": "not_run",
        "chunks_per_sec": 0.0,
        "optimizer_steps_per_sec": 0.0,
        "sec_per_optimizer_step": None,
        "peak_gpu_gb": None,
    }
    try:
        train_loader, train_ds = _safe_make_train_loader(sliced_meta, cfg, train_samples=max(train_samples, batch_size * grad_accum_steps * 4), device=device)
        bench_model = PianoTranscriptionSystem(cfg).to(device)
        if compile_model:
            if hasattr(torch, "compile"):
                bench_model = torch.compile(bench_model)
            else:
                result["status"] = "ERROR: torch.compile unavailable"
                return result
        optimizer = torch.optim.AdamW(bench_model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device == "cuda"))
        bench_model.train()
        iterator = iter(train_loader)
        nonlocal_iterator = [iterator]

        def run_one_optimizer_step():
            optimizer.zero_grad(set_to_none=True)
            total_chunks = 0
            total_loss_value = 0.0
            for _ in range(grad_accum_steps):
                batch, nonlocal_iterator[0] = _next_batch(train_loader, nonlocal_iterator[0])
                batch = move_batch_to_device(batch, device)
                model_input = model_input_from_batch(batch)
                total_chunks += int(model_input.shape[0])
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and device == "cuda")):
                    pred = bench_model(model_input, target_frames=batch["frame"].shape[1])
                    loss, _ = compute_loss(pred, batch, cfg)
                    loss = loss / grad_accum_steps
                total_loss_value += float(loss.detach().cpu()) * grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            return total_chunks, total_loss_value

        for _ in range(warmup_steps):
            run_one_optimizer_step()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        total_chunks = 0
        t0 = time.perf_counter()
        for _ in range(timed_optimizer_steps):
            chunks, _ = run_one_optimizer_step()
            total_chunks += chunks
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        result.update({
            "status": "ok",
            "chunks_per_sec": total_chunks / elapsed,
            "optimizer_steps_per_sec": timed_optimizer_steps / elapsed,
            "sec_per_optimizer_step": elapsed / timed_optimizer_steps,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None,
        })
    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
    except Exception as e:
        result["status"] = f"ERROR: {type(e).__name__}: {e}"
    finally:
        try:
            del bench_model, optimizer, scaler, train_loader, train_ds
        except Exception:
            pass
        cleanup_cuda()
    return result


def sweep_training_settings(
    sliced_meta: pd.DataFrame,
    base_cfg: Config = CFG,
    batch_sizes: Iterable[int] = (16, 24, 32),
    num_workers_list: Iterable[int] = (2, 4, 8),
    amp_options: Iterable[bool] = (True,),
    device: str = DEVICE,
) -> pd.DataFrame:
    rows = []
    for batch_size, num_workers, use_amp in itertools.product(batch_sizes, num_workers_list, amp_options):
        print(f"\nTesting batch={batch_size}, workers={num_workers}, amp={use_amp}")
        row = benchmark_training_setting(
            sliced_meta,
            batch_size=batch_size,
            num_workers=num_workers,
            use_amp=use_amp,
            grad_accum_steps=1,
            compile_model=True,
            train_samples=1024,
            warmup_steps=2,
            timed_optimizer_steps=8,
            base_cfg=base_cfg,
            device=device,
        )
        print(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by=["status", "chunks_per_sec"], ascending=[True, False]).reset_index(drop=True)


def apply_best_benchmark_config(cfg: Config, bench_df: pd.DataFrame, memory_soft_limit_gb: float = 14.0) -> Config:
    ok = bench_df[bench_df["status"] == "ok"].copy()
    if len(ok) == 0:
        raise RuntimeError("No successful benchmark configs. Try smaller batch sizes.")
    max_speed = ok["chunks_per_sec"].max()
    ok["speed_ratio"] = ok["chunks_per_sec"] / max_speed
    near_best = ok[ok["speed_ratio"] >= 0.97].copy()
    safe = near_best[near_best["peak_gpu_gb"] <= memory_soft_limit_gb].copy()
    if len(safe) == 0:
        safe = near_best.copy()
    best = safe.sort_values(["batch_size", "num_workers"], ascending=[False, True]).iloc[0]
    cfg.batch_size = int(best["batch_size"])
    cfg.num_workers = int(best["num_workers"])
    cfg.use_amp = bool(best["use_amp"])
    print("Chosen config:")
    print("cfg.batch_size =", cfg.batch_size)
    print("cfg.num_workers =", cfg.num_workers)
    print("cfg.use_amp =", cfg.use_amp)
    print(best)
    return cfg
