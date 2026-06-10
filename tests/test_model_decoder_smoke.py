import sys
import types

import torch
import torch.nn as nn

from piano_modeling.config import Config
from piano_modeling.models import ResNetPianoTranscriber


class _FeatureInfo:
    def channels(self):
        return [4, 8, 16, 32]


class _TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_info = _FeatureInfo()
        self.blocks = nn.ModuleList(
            [
                nn.Conv2d(3, 4, kernel_size=3, padding=1),
                nn.Conv2d(4, 8, kernel_size=3, padding=1),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
            ]
        )

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = torch.relu(block(x))
            features.append(x)
            if min(x.shape[-2:]) > 2:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return features


def test_fpn_decoder_outputs_piano_roll_shapes(monkeypatch):
    fake_timm = types.SimpleNamespace(create_model=lambda *args, **kwargs: _TinyEncoder())
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    cfg = Config(
        midi_min=60,
        midi_max=63,
        n_mels=32,
        pretrained=False,
        decoder_channels=8,
        decoder_type="fpn",
        enable_spec_augment=False,
    )
    model = ResNetPianoTranscriber(cfg)
    out = model(torch.randn(2, 3, 32, 12), target_frames=10)

    for key in ["onset", "offset", "frame", "velocity"]:
        assert out[key].shape == (2, 10, cfg.n_pitches)
    assert out["sustain"].shape == (2, 10)


def test_legacy_decoder_still_available(monkeypatch):
    fake_timm = types.SimpleNamespace(create_model=lambda *args, **kwargs: _TinyEncoder())
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    cfg = Config(midi_min=60, midi_max=63, pretrained=False, decoder_channels=8, decoder_type="legacy")
    model = ResNetPianoTranscriber(cfg)
    out = model(torch.randn(1, 3, 32, 12), target_frames=7)

    assert out["onset"].shape == (1, 7, cfg.n_pitches)
    assert out["sustain"].shape == (1, 7)
