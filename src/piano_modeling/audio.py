"""Audio loading and augmentation utilities."""
from __future__ import annotations

import random

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .config import Config


def load_audio_segment(path: str, start_sec: float, duration_sec: float, target_sr: int) -> torch.Tensor:
    """Load mono audio segment, resample, pad/truncate to exact duration."""
    info = sf.info(path)
    native_sr = int(info.samplerate)
    start_frame = max(0, int(round(start_sec * native_sr)))
    n_frames = int(round(duration_sec * native_sr))
    with sf.SoundFile(path) as f:
        f.seek(min(start_frame, len(f)))
        audio = f.read(frames=n_frames, dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    wav = torch.from_numpy(audio)
    if native_sr != target_sr and wav.numel() > 0:
        wav = torchaudio.functional.resample(wav, native_sr, target_sr)
    target_len = int(round(duration_sec * target_sr))
    if wav.numel() < target_len:
        wav = F.pad(wav, (0, target_len - wav.numel()))
    elif wav.numel() > target_len:
        wav = wav[:target_len]
    return wav.float().clamp(-1.0, 1.0)


def random_exponential_ir(sr: int, max_seconds: float = 0.35) -> torch.Tensor:
    """Synthetic small-room-ish impulse response for robustness."""
    length = int(sr * random.uniform(0.05, max_seconds))
    t = torch.linspace(0, 1, steps=max(16, length))
    decay = torch.exp(-random.uniform(3.0, 9.0) * t)
    noise = torch.randn_like(decay) * decay
    noise[0] += 1.0
    noise = noise / (noise.abs().sum() + 1e-8)
    return noise.view(1, 1, -1)


class AudioAugmenter:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        x = wav.clone()

        if random.random() < cfg.gain_probability:
            gain_db = random.uniform(-8.0, 5.0)
            x = x * (10 ** (gain_db / 20.0))

        if random.random() < cfg.noise_probability:
            noise = torch.randn_like(x)
            if random.random() < 0.5:
                noise = torch.cumsum(noise, dim=0)
                noise = noise / (noise.std() + 1e-8)
            snr_db = random.uniform(18.0, 38.0)
            sig = x.pow(2).mean().sqrt().clamp_min(1e-6)
            noi = noise.pow(2).mean().sqrt().clamp_min(1e-6)
            x = x + noise * (sig / noi) * (10 ** (-snr_db / 20.0))

        if random.random() < cfg.reverb_probability:
            ir = random_exponential_ir(cfg.sample_rate).to(x.device)
            y = F.conv1d(x.view(1, 1, -1), ir, padding=ir.shape[-1] - 1).view(-1)
            x = y[:x.numel()]

        if random.random() < cfg.eq_probability:
            if random.random() < 0.5:
                cutoff = random.uniform(40, 180)
                x = torchaudio.functional.highpass_biquad(x, cfg.sample_rate, cutoff)
            else:
                cutoff = random.uniform(4500, 7800)
                x = torchaudio.functional.lowpass_biquad(x, cfg.sample_rate, cutoff)

        return x.clamp(-1.0, 1.0)


class SpecAugment(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, C, F, T]
        if not self.training or not self.cfg.enable_spec_augment:
            return feat
        if random.random() > self.cfg.spec_mask_probability:
            return feat
        B, C, Freq, Time = feat.shape
        out = feat.clone()
        for b in range(B):
            for _ in range(random.randint(1, 3)):
                width = random.randint(4, max(5, Freq // 12))
                start = random.randint(0, max(0, Freq - width))
                out[b, :, start:start + width, :] = 0
            for _ in range(random.randint(1, 3)):
                width = random.randint(5, max(6, Time // 12))
                start = random.randint(0, max(0, Time - width))
                out[b, :, :, start:start + width] = 0
        return out
