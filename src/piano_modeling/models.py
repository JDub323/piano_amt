"""ResNet-based multi-head piano transcription model."""
from __future__ import annotations

from typing import Dict, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .frontend import SpectrogramFrontend


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetPianoTranscriber(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = timm.create_model(
            cfg.resnet_name,
            pretrained=cfg.pretrained,
            features_only=True,
            in_chans=3,
            out_indices=(1, 2, 3, 4),
        )
        enc_ch = self.encoder.feature_info.channels()[-1]
        ch = cfg.decoder_channels
        self.neck = nn.Sequential(
            ConvBNAct(enc_ch, ch, 3, 1, cfg.dropout),
            ConvBNAct(ch, ch, 3, 1, cfg.dropout),
        )
        self.shared = nn.Sequential(
            ConvBNAct(ch, ch, 3, 1, cfg.dropout),
            ConvBNAct(ch, ch // 2, 3, 1, cfg.dropout),
        )
        out_ch = ch // 2
        self.onset_head = nn.Conv2d(out_ch, 1, kernel_size=1)
        self.offset_head = nn.Conv2d(out_ch, 1, kernel_size=1)
        self.frame_head = nn.Conv2d(out_ch, 1, kernel_size=1)
        self.velocity_head = nn.Conv2d(out_ch, 1, kernel_size=1)
        self.sustain_head = nn.Sequential(
            ConvBNAct(out_ch, out_ch // 2, 3, 1, cfg.dropout),
            nn.Conv2d(out_ch // 2, 1, kernel_size=1),
        )

    def _pitch_time(self, head: nn.Module, z: torch.Tensor) -> torch.Tensor:
        return head(z).squeeze(1).transpose(1, 2).contiguous()  # [B, T, P]

    def forward(self, feat: torch.Tensor, target_frames: Optional[int] = None) -> Dict[str, torch.Tensor]:
        target_frames = target_frames or feat.shape[-1]
        features = self.encoder(feat)
        z = self.neck(features[-1])
        z = self.shared(z)
        z = F.interpolate(z, size=(self.cfg.n_pitches, target_frames), mode="bilinear", align_corners=False)
        sustain_logits = self.sustain_head(z).squeeze(1).mean(dim=1)  # [B, T]
        return {
            "onset": self._pitch_time(self.onset_head, z),
            "offset": self._pitch_time(self.offset_head, z),
            "frame": self._pitch_time(self.frame_head, z),
            "velocity": self._pitch_time(self.velocity_head, z),
            "sustain": sustain_logits,
        }


class PianoTranscriptionSystem(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.frontend = SpectrogramFrontend(cfg)  # waveform path for offline/live inference
        self.model = ResNetPianoTranscriber(cfg)

    def _cached_spec_to_model_features(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: [B, F, T] or [F, T], raw cached log spectrogram
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
        x = spec.float()
        d1 = torch.diff(x, dim=-1, prepend=x[..., :1])
        d2 = torch.diff(d1, dim=-1, prepend=d1[..., :1])
        feat = torch.stack([x, d1, d2], dim=1)  # [B, 3, F, T]
        mean = feat.mean(dim=(2, 3), keepdim=True)
        std = feat.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        feat = (feat - mean) / std
        feat = self.frontend.specaugment(feat)
        return feat

    def forward(self, x: torch.Tensor, target_frames: Optional[int] = None):
        if x.ndim in (2, 3) and x.shape[-2] in (self.cfg.n_mels, self.cfg.cqt_n_bins):
            feat = self._cached_spec_to_model_features(x)
        else:
            feat = self.frontend(x)
        return self.model(feat, target_frames=target_frames)


def count_parameters_millions(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
