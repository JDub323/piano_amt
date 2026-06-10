"""Multi-head piano transcription models."""
from __future__ import annotations

import importlib
from typing import Dict, Optional, Sequence

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


class FeaturePyramidDecoder(nn.Module):
    """Top-down skip decoder that preserves earlier time/frequency detail.

    The original prototype decoded only the deepest ResNet feature map. That is compact, but the
    deepest map is also the most downsampled in time. This decoder keeps the public model contract
    unchanged while fusing all encoder stages into a higher-resolution piano-roll representation.
    It is intentionally generic: any ``timm`` features-only encoder that returns a list of feature
    maps and channel counts can be used.
    """

    def __init__(self, encoder_channels: Sequence[int], out_ch: int, dropout: float = 0.0):
        super().__init__()
        if len(encoder_channels) == 0:
            raise ValueError("FeaturePyramidDecoder requires at least one encoder stage")
        self.laterals = nn.ModuleList([nn.Conv2d(ch, out_ch, kernel_size=1) for ch in encoder_channels])
        self.refines = nn.ModuleList(
            [ConvBNAct(out_ch, out_ch, 3, 1, dropout) for _ in encoder_channels]
        )
        self.fuse = nn.Sequential(
            ConvBNAct(out_ch * len(encoder_channels), out_ch, 3, 1, dropout),
            ConvBNAct(out_ch, out_ch, 3, 1, dropout),
        )

    def forward(self, features: Sequence[torch.Tensor], target_size: tuple[int, int]) -> torch.Tensor:
        if len(features) != len(self.laterals):
            raise ValueError(f"Expected {len(self.laterals)} feature maps, got {len(features)}")

        top_down: Optional[torch.Tensor] = None
        fused_at_target = []
        for feat, lateral, refine in zip(
            reversed(features), reversed(self.laterals), reversed(self.refines), strict=True
        ):
            current = lateral(feat)
            if top_down is not None:
                top_down = F.interpolate(
                    top_down, size=current.shape[-2:], mode="bilinear", align_corners=False
                )
                current = current + top_down
            top_down = refine(current)
            fused_at_target.append(
                F.interpolate(top_down, size=target_size, mode="bilinear", align_corners=False)
            )
        return self.fuse(torch.cat(list(reversed(fused_at_target)), dim=1))


class ResNetPianoTranscriber(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        try:
            timm = importlib.import_module("timm")
        except ImportError as exc:
            raise ImportError(
                "ResNetPianoTranscriber requires the optional dependency 'timm'. "
                "Install with `pip install -e .` or `pip install -e '.[train]'`."
            ) from exc

        self.encoder = timm.create_model(
            cfg.resnet_name,
            pretrained=cfg.pretrained,
            features_only=True,
            in_chans=3,
            out_indices=(1, 2, 3, 4),
        )
        encoder_channels = list(self.encoder.feature_info.channels())
        ch = cfg.decoder_channels
        decoder_type = cfg.decoder_type.lower()
        self.decoder_type = decoder_type

        if decoder_type == "legacy":
            enc_ch = encoder_channels[-1]
            self.decoder = nn.Sequential(
                ConvBNAct(enc_ch, ch, 3, 1, cfg.dropout),
                ConvBNAct(ch, ch, 3, 1, cfg.dropout),
            )
        elif decoder_type == "fpn":
            self.decoder = FeaturePyramidDecoder(encoder_channels, ch, cfg.dropout)
        else:
            raise ValueError("cfg.decoder_type must be either 'fpn' or 'legacy'")

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
            ConvBNAct(out_ch, max(1, out_ch // 2), 3, 1, cfg.dropout),
            nn.Conv2d(max(1, out_ch // 2), 1, kernel_size=1),
        )

    def _pitch_time(self, head: nn.Module, z: torch.Tensor) -> torch.Tensor:
        return head(z).squeeze(1).transpose(1, 2).contiguous()  # [B, T, P]

    def _decode_features(self, features: Sequence[torch.Tensor], target_frames: int) -> torch.Tensor:
        target_size = (self.cfg.n_pitches, target_frames)
        if self.decoder_type == "legacy":
            z = self.decoder(features[-1])
            return F.interpolate(z, size=target_size, mode="bilinear", align_corners=False)
        return self.decoder(features, target_size=target_size)

    def forward(self, feat: torch.Tensor, target_frames: Optional[int] = None) -> Dict[str, torch.Tensor]:
        target_frames = target_frames or feat.shape[-1]
        features = self.encoder(feat)
        z = self._decode_features(features, target_frames)
        z = self.shared(z)
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
