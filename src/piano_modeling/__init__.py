"""Piano AMT modeling/training package.

The top-level package keeps imports light so utilities such as decoding and tests do not require the
full audio/MIDI/model dependency stack at import time. Heavier objects are loaded lazily on access.
"""
from __future__ import annotations

from .common import DEVICE, get_device, setup_runtime
from .config import CFG, Config, config_to_json, default_run_name
from .paths import ProjectPaths, make_colab_paths, print_paths

_LAZY_EXPORTS = {
    "PianoTranscriptionSystem": (".models", "PianoTranscriptionSystem"),
    "count_parameters_millions": (".models", "count_parameters_millions"),
    "load_checkpoint": (".training", "load_checkpoint"),
    "run_training": (".training", "run_training"),
    "save_checkpoint": (".training", "save_checkpoint"),
    "train_one_epoch": (".training", "train_one_epoch"),
    "validate_quick": (".training", "validate_quick"),
    "find_optimal_lr_fastai": (".lr_finder", "find_optimal_lr_fastai"),
    "find_learning_rate_range_test": (".lr_finder", "find_learning_rate_range_test"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    import importlib

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "CFG",
    "Config",
    "config_to_json",
    "default_run_name",
    "DEVICE",
    "get_device",
    "setup_runtime",
    "ProjectPaths",
    "make_colab_paths",
    "print_paths",
    *_LAZY_EXPORTS.keys(),
]
