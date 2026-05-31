"""Deployment/export helpers for trained models."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from piano_modeling.common import DEVICE
from piano_modeling.config import CFG, Config
from piano_modeling.models import PianoTranscriptionSystem
from piano_modeling.training import load_checkpoint


class TupleOutputWrapper(nn.Module):
    def __init__(self, system: PianoTranscriptionSystem, cfg: Config):
        super().__init__()
        self.system = system
        self.cfg = cfg

    def forward(self, wav: torch.Tensor):
        pred = self.system(wav, target_frames=1 + wav.shape[-1] // self.cfg.hop_length)
        return pred["onset"], pred["offset"], pred["frame"], pred["velocity"], pred["sustain"]


def export_torchscript(
    checkpoint_path: Path,
    out_path: Path,
    system: PianoTranscriptionSystem,
    cfg: Config = CFG,
    device: str = DEVICE,
) -> Path:
    load_checkpoint(checkpoint_path, system, map_location=device, cfg=cfg)
    system.eval()
    wrapper = TupleOutputWrapper(system, cfg).to(device).eval()
    example = torch.zeros(1, int(2.0 * cfg.sample_rate), device=device)
    traced = torch.jit.trace(wrapper, example)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    print("saved:", out_path)
    return out_path
