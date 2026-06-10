"""Spectrogram front-end used for waveform inference and training fallbacks."""
from __future__ import annotations

import torch
import torch.nn as nn
from .audio import SpecAugment
from .config import Config


class SpectrogramFrontend(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.specaugment = SpecAugment(cfg)
        if cfg.feature_type.lower() == "mel":
            try:
                import torchaudio
            except (ImportError, OSError) as exc:
                raise ImportError(
                    "Mel spectrogram frontend requires a working torchaudio install. "
                    "Install the torch/torchaudio build that matches your platform."
                ) from exc
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.sample_rate,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                f_min=cfg.f_min,
                f_max=cfg.f_max,
                n_mels=cfg.n_mels,
                power=2.0,
                center=True,
                norm="slaney",
                mel_scale="slaney",
            )
            self.kind = "mel"
        elif cfg.feature_type.lower() == "cqt":
            from nnAudio.features import CQT1992v2
            self.transform = CQT1992v2(
                sr=cfg.sample_rate,
                hop_length=cfg.hop_length,
                fmin=cfg.f_min,
                n_bins=cfg.cqt_n_bins,
                bins_per_octave=cfg.cqt_bins_per_octave,
                output_format="Magnitude",
                verbose=False,
            )
            self.kind = "cqt"
        else:
            raise ValueError(f"Unsupported feature_type={cfg.feature_type}")

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, N] or [N]
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        x = self.transform(wav)
        if isinstance(x, tuple):
            x = x[0]
        if x.ndim == 4:
            x = x.squeeze(1)
        x = torch.log1p(x.clamp_min(0) * 1000.0)

        d1 = torch.diff(x, dim=-1, prepend=x[..., :1])
        d2 = torch.diff(d1, dim=-1, prepend=d1[..., :1])
        feat = torch.stack([x, d1, d2], dim=1)  # [B, 3, F, T]
        mean = feat.mean(dim=(2, 3), keepdim=True)
        std = feat.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        feat = (feat - mean) / std
        feat = self.specaugment(feat)
        return feat
