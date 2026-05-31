"""Low-latency/streaming transcription core."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio

from piano_modeling.common import DEVICE
from piano_modeling.config import CFG, Config
from piano_modeling.decoding import decode_notes_from_probs
from piano_modeling.models import PianoTranscriptionSystem


@torch.no_grad()
def benchmark_latency(system: PianoTranscriptionSystem, cfg: Config = CFG, seconds: float = 2.0, repeats: int = 50, device: str = DEVICE):
    system.eval()
    n = int(seconds * cfg.sample_rate)
    x = torch.zeros(1, n, device=device)
    for _ in range(5):
        _ = system(x, target_frames=1 + n // cfg.hop_length)
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = system(x, target_frames=1 + n // cfg.hop_length)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    print(f"Median latency for {seconds:.2f}s window: {np.median(times)*1000:.1f} ms")
    print(f"Real-time factor: {seconds / np.median(times):.2f}x")
    return times


class RealTimePianoTranscriber:
    def __init__(
        self,
        system: PianoTranscriptionSystem,
        cfg: Config = CFG,
        window_seconds: float = 4.0,
        step_seconds: float = 0.25,
        safety_latency_seconds: float = 0.20,
        device: str = DEVICE,
    ):
        self.system = system.eval()
        self.cfg = cfg
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.safety_latency_seconds = safety_latency_seconds
        self.device = device
        self.buffer = torch.zeros(0, dtype=torch.float32)
        self.absolute_start_time = 0.0
        self.last_processed_audio_time = 0.0
        self.last_emit_time = 0.0
        self.emitted_keys = set()

    def reset(self) -> None:
        self.buffer = torch.zeros(0, dtype=torch.float32)
        self.absolute_start_time = 0.0
        self.last_processed_audio_time = 0.0
        self.last_emit_time = 0.0
        self.emitted_keys.clear()

    @torch.no_grad()
    def push(self, audio_chunk: np.ndarray, sr: int) -> pd.DataFrame:
        """Push new audio samples and return newly emitted events."""
        if audio_chunk is None or len(audio_chunk) == 0:
            return pd.DataFrame()
        x = np.asarray(audio_chunk, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        wav = torch.from_numpy(x)
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
        self.buffer = torch.cat([self.buffer, wav.cpu()])
        current_audio_time = self.absolute_start_time + self.buffer.numel() / self.cfg.sample_rate
        if current_audio_time - self.last_processed_audio_time < self.step_seconds:
            return pd.DataFrame()
        self.last_processed_audio_time = current_audio_time

        max_len = int(self.window_seconds * self.cfg.sample_rate)
        if self.buffer.numel() > max_len:
            drop = self.buffer.numel() - max_len
            self.buffer = self.buffer[drop:]
            self.absolute_start_time += drop / self.cfg.sample_rate
        if self.buffer.numel() < int(0.5 * self.cfg.sample_rate):
            return pd.DataFrame()

        model_len = max_len
        model_wav = self.buffer
        if model_wav.numel() < model_len:
            model_wav = F.pad(model_wav, (model_len - model_wav.numel(), 0))
            model_start_time = self.absolute_start_time - (model_len - self.buffer.numel()) / self.cfg.sample_rate
        else:
            model_start_time = self.absolute_start_time

        inp = model_wav.unsqueeze(0).to(self.device)
        pred = self.system(inp, target_frames=1 + model_len // self.cfg.hop_length)
        prob = {k: torch.sigmoid(v).squeeze(0).detach().cpu().numpy() for k, v in pred.items()}
        events = decode_notes_from_probs(prob["onset"], prob["frame"], prob["offset"], prob["velocity"], self.cfg)
        if len(events) == 0:
            return events
        events["onset_sec"] += model_start_time
        events["offset_sec"] += model_start_time
        emit_before = current_audio_time - self.safety_latency_seconds
        new = events[(events["onset_sec"] >= self.last_emit_time) & (events["onset_sec"] <= emit_before)].copy()
        dedup_rows = []
        for _, row in new.iterrows():
            key = (int(row["midi_pitch"]), round(float(row["onset_sec"]), 2))
            if key not in self.emitted_keys:
                self.emitted_keys.add(key)
                dedup_rows.append(row)
        self.last_emit_time = max(self.last_emit_time, emit_before)
        return pd.DataFrame(dedup_rows)
