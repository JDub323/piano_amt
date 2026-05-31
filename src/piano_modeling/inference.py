"""Offline audio-file inference."""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

from .audio import load_audio_segment
from .common import DEVICE
from .config import CFG, Config
from .models import PianoTranscriptionSystem


@torch.no_grad()
def infer_audio_file(
    audio_path: str,
    system: PianoTranscriptionSystem,
    cfg: Config = CFG,
    window_seconds: Optional[float] = None,
    overlap_seconds: float = 2.0,
    batch_size: int = 8,
    device: str = DEVICE,
) -> Dict[str, np.ndarray]:
    """Run chunked inference and stitch probabilities over an audio file."""
    system.eval()
    window_seconds = window_seconds or cfg.segment_seconds
    info = sf.info(audio_path)
    duration = float(info.duration)
    total_frames = int(math.ceil(duration * cfg.fps)) + 1
    P = cfg.n_pitches
    sums = {
        "onset": np.zeros((total_frames, P), dtype=np.float32),
        "offset": np.zeros((total_frames, P), dtype=np.float32),
        "frame": np.zeros((total_frames, P), dtype=np.float32),
        "velocity": np.zeros((total_frames, P), dtype=np.float32),
        "sustain": np.zeros((total_frames,), dtype=np.float32),
    }
    counts_pitch = np.zeros((total_frames, 1), dtype=np.float32)
    counts_sustain = np.zeros((total_frames,), dtype=np.float32)
    step = max(0.25, window_seconds - overlap_seconds)
    starts = np.arange(0, max(0.0, duration - 0.001), step).tolist()
    if len(starts) == 0 or starts[-1] < max(0, duration - window_seconds):
        starts.append(max(0.0, duration - window_seconds))
    starts = sorted(set(round(float(s), 4) for s in starts))

    for i in tqdm(range(0, len(starts), batch_size), desc="inference chunks"):
        batch_starts = starts[i:i + batch_size]
        wavs = [load_audio_segment(audio_path, s, window_seconds, cfg.sample_rate) for s in batch_starts]
        wav = torch.stack(wavs).to(device)
        pred = system(wav, target_frames=1 + int(round(window_seconds * cfg.fps)))
        prob = {k: torch.sigmoid(v).detach().cpu().numpy() for k, v in pred.items()}
        for b, s in enumerate(batch_starts):
            start_f = int(round(s * cfg.fps))
            T = prob["sustain"][b].shape[0]
            end_f = min(total_frames, start_f + T)
            local_T = end_f - start_f
            if local_T <= 0:
                continue
            for key in ["onset", "offset", "frame", "velocity"]:
                sums[key][start_f:end_f] += prob[key][b, :local_T]
            sums["sustain"][start_f:end_f] += prob["sustain"][b, :local_T]
            counts_pitch[start_f:end_f] += 1
            counts_sustain[start_f:end_f] += 1

    counts_pitch = np.maximum(counts_pitch, 1.0)
    counts_sustain = np.maximum(counts_sustain, 1.0)
    out = {k: sums[k] / counts_pitch for k in ["onset", "offset", "frame", "velocity"]}
    out["sustain"] = sums["sustain"] / counts_sustain
    out["duration"] = duration
    return out
