import torch
import torch.nn as nn

from piano_modeling.config import Config
from piano_modeling.tensorboard_debug import log_model_debug_batch


class DummyWriter:
    def __init__(self):
        self.scalars = []
        self.histograms = []
        self.images = []

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.scalars.append(tag)

    def add_histogram(self, tag, values, global_step=None):
        self.histograms.append(tag)

    def add_image(self, tag, img_tensor, global_step=None):
        self.images.append(tag)

    def add_pr_curve(self, tag, labels, predictions, global_step=None):
        pass

    def flush(self):
        pass


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)


def test_log_model_debug_batch_accepts_generic_output_dict():
    cfg = Config(midi_min=60, midi_max=61)
    writer = DummyWriter()
    model = TinyModel()
    model_input = torch.randn(2, 4, 5)
    batch = {
        "onset": torch.zeros(2, 5, cfg.n_pitches),
        "offset": torch.zeros(2, 5, cfg.n_pitches),
        "frame": torch.zeros(2, 5, cfg.n_pitches),
        "velocity": torch.zeros(2, 5, cfg.n_pitches),
        "sustain": torch.zeros(2, 5),
    }
    pred = {k: torch.randn_like(v) for k, v in batch.items()}

    log_model_debug_batch(writer, model, model_input, batch, pred, {"total": 1.0}, cfg, step=0)

    assert "batch/loss/total" in writer.scalars
    assert "debug_probabilities/onset_mean" in writer.scalars
    assert any(tag.startswith("debug_logits/onset") for tag in writer.histograms)
    assert "debug/input_channel0" in writer.images
