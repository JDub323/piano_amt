"""Torch profiler helpers that emit Chrome/Perfetto flame-chart traces."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch

from .config import Config


def make_flame_chart_profiler(cfg: Config, output_dir: Optional[str | Path], *, enabled: bool = True):
    if not enabled or not getattr(cfg, "enable_flame_charts", False):
        return nullcontext(None)
    output_dir = Path(output_dir or getattr(cfg, "flame_chart_output_dir", "") or "flame_charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    wait = max(0, int(getattr(cfg, "flame_chart_wait_batches", 1)))
    warmup = max(0, int(getattr(cfg, "flame_chart_warmup_batches", 1)))
    active = max(1, int(getattr(cfg, "flame_chart_batches", 8)))
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
        record_shapes=bool(getattr(cfg, "flame_chart_record_shapes", True)),
        profile_memory=bool(getattr(cfg, "flame_chart_profile_memory", True)),
        with_stack=bool(getattr(cfg, "flame_chart_with_stack", True)),
    )
