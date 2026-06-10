import torch
import torch.nn as nn

from piano_modeling.config import Config
from piano_modeling.training import train_one_epoch, validate_quick


class TinySystem(nn.Module):
    def __init__(self, n_pitches: int):
        super().__init__()
        self.n_pitches = n_pitches
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x, target_frames=None):
        target_frames = target_frames or x.shape[-1]
        base = x.mean(dim=1)
        if base.shape[-1] != target_frames:
            base = torch.nn.functional.interpolate(
                base.unsqueeze(1), size=target_frames, mode="linear", align_corners=False
            ).squeeze(1)
        logits = self.scale * base.unsqueeze(-1).expand(-1, -1, self.n_pitches) + self.bias
        return {
            "onset": logits,
            "offset": logits,
            "frame": logits,
            "velocity": logits,
            "sustain": logits.mean(dim=-1),
        }


def _batch(batch_size=2, freq_bins=6, frames=8, pitches=4):
    torch.manual_seed(0)
    batch = {
        "spec": torch.randn(batch_size, freq_bins, frames),
        "onset": torch.zeros(batch_size, frames, pitches),
        "offset": torch.zeros(batch_size, frames, pitches),
        "frame": torch.zeros(batch_size, frames, pitches),
        "velocity": torch.zeros(batch_size, frames, pitches),
        "sustain": torch.zeros(batch_size, frames),
    }
    batch["onset"][:, 2, 1] = 1.0
    batch["offset"][:, 5, 1] = 1.0
    batch["frame"][:, 2:6, 1] = 1.0
    batch["velocity"][:, 2, 1] = 0.75
    batch["sustain"][:, 3:5] = 1.0
    return batch


def test_train_and_validate_smoke_on_synthetic_batch():
    cfg = Config(midi_min=60, midi_max=63, use_amp=False, grad_clip=0.0)
    system = TinySystem(cfg.n_pitches)
    optimizer = torch.optim.SGD(system.parameters(), lr=1e-2)
    loader = [_batch(pitches=cfg.n_pitches), _batch(pitches=cfg.n_pitches)]

    train_stats = train_one_epoch(system, loader, optimizer, None, cfg, epoch=1, device="cpu")
    val_stats = validate_quick(system, loader, cfg, device="cpu")

    assert train_stats["total"] > 0
    assert val_stats["total"] > 0
    assert 0.0 <= val_stats["onset_f1"] <= 1.0
    assert 0.0 <= val_stats["frame_f1"] <= 1.0
