"""Dataset and DataLoader helpers for pre-sliced and fallback MAESTRO training."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional
import random

import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from .audio import AudioAugmenter, load_audio_segment
from .common import DEVICE
from .config import CFG, Config
from .tokenization import PAD

SPEC_LRU_CACHE_SIZE = 64


class PreSlicedMaestroDataset(Dataset):
    """Fast dataset that loads one precomputed spectrogram/label chunk per item."""

    def __init__(self, manifest_df: pd.DataFrame, split: str, max_items: Optional[int] = None, training: bool = False):
        self.split = split
        self.training = training
        self.df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No sliced chunks found for split={split!r}")

        self.max_items = int(max_items) if max_items not in (None, 0) else None
        if self.max_items is not None and self.max_items <= 0:
            self.max_items = None

        # Validation/test caps use a random non-deterministic subset at construction.
        if (not training) and self.max_items is not None and self.max_items < len(self.df):
            self.df = self.df.sample(n=self.max_items, replace=False).reset_index(drop=True)
            self.max_items = None

    def __len__(self):
        if self.training and self.max_items is not None:
            return self.max_items
        return len(self.df)

    def __getitem__(self, idx: int):
        if self.training and self.max_items is not None:
            row_idx = random.randrange(len(self.df))
        else:
            row_idx = idx % len(self.df)
        row = self.df.iloc[row_idx]
        payload = torch.load(row["chunk_path"], map_location="cpu")
        item = {"spec": payload["spec"].float()}
        if "tokens" in payload:
            item["tokens"] = payload["tokens"].long()
            item["token_length"] = torch.tensor(int(item["tokens"].numel()), dtype=torch.long)
        else:
            for k in ["onset", "offset", "frame", "velocity", "sustain"]:
                item[k] = payload[k].float()
            if item["sustain"].ndim == 2 and item["sustain"].shape[-1] == 1:
                item["sustain"] = item["sustain"].squeeze(-1)
        item["start_sec"] = torch.tensor(float(row.get("start_sec", 0.0)), dtype=torch.float32)
        return item


def piano_collate(batch):
    if batch and "tokens" in batch[0]:
        out = {}
        out["spec"] = default_collate([b["spec"] for b in batch])
        out["tokens"] = pad_sequence([b["tokens"].long() for b in batch], batch_first=True, padding_value=PAD)
        out["token_length"] = torch.tensor([int(b["tokens"].numel()) for b in batch], dtype=torch.long)
        if "start_sec" in batch[0]:
            out["start_sec"] = default_collate([b["start_sec"] for b in batch])
        return out
    return default_collate(batch)


def _none_if_zero(value):
    if value is None:
        return None
    value = int(value)
    return None if value <= 0 else value


def make_sliced_loaders(
    sliced_df: pd.DataFrame,
    cfg: Config = CFG,
    train_samples_per_epoch: Optional[int] = None,
    val_samples: Optional[int] = None,
    device: str = DEVICE,
):
    train_cap = _none_if_zero(train_samples_per_epoch)
    val_cap = _none_if_zero(val_samples)
    train_ds = PreSlicedMaestroDataset(sliced_df, "train", max_items=train_cap, training=True)
    val_ds = PreSlicedMaestroDataset(sliced_df, "validation", max_items=val_cap, training=False)

    common_loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        collate_fn=piano_collate,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )
    print(f"Training chunks per epoch: {len(train_ds)}")
    print(f"Validation chunks: {len(val_ds)}")
    return train_loader, val_loader, train_ds, val_ds


@lru_cache(maxsize=SPEC_LRU_CACHE_SIZE)
def load_cached_spectrogram(spec_path: str) -> torch.Tensor:
    payload = torch.load(spec_path, map_location="cpu")
    spec = payload["spec"] if isinstance(payload, dict) and "spec" in payload else payload
    return spec.float().contiguous()


class MaestroCachedSpectrogramSegmentDataset(Dataset):
    """Fallback dataset: sample time windows from cached full-song spectrogram tensors."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        split: str,
        cfg: Config,
        training: bool,
        samples_per_epoch: Optional[int] = None,
        augment: bool = True,
    ):
        self.cfg = cfg
        self.training = training
        df = metadata[metadata["split"] == split].copy()
        df["spec_exists"] = df["spec_path"].apply(lambda p: Path(p).exists())
        df["midi_exists"] = df["midi_path"].apply(lambda p: Path(p).exists())
        self.df = df[df["spec_exists"] & df["midi_exists"]].reset_index(drop=True)
        self.samples_per_epoch = samples_per_epoch or len(self.df)
        assert len(self.df) > 0, f"No cached rows for split={split}. Run the MAESTRO preprocess/cache path first."

    def __len__(self):
        return self.samples_per_epoch

    def _duration_for_row(self, row, spec: torch.Tensor) -> float:
        if "duration" in row.index and not pd.isna(row["duration"]):
            return float(row["duration"])
        return max(0.0, (spec.shape[-1] - 1) / self.cfg.fps)

    def _choose_row_and_start(self, idx: int):
        row_idx = random.randrange(len(self.df)) if self.training else (idx % len(self.df))
        row = self.df.iloc[row_idx]
        spec = load_cached_spectrogram(str(row["spec_path"]))
        duration = self._duration_for_row(row, spec)
        max_start = max(0.0, duration - self.cfg.segment_seconds)
        if self.training:
            start = random.uniform(0.0, max_start) if max_start > 0 else 0.0
        else:
            start = ((idx // len(self.df)) * self.cfg.segment_seconds) % (max_start + 1e-8) if max_start > 0 else 0.0
        return row, spec, float(start)

    def _crop_spec(self, spec: torch.Tensor, start_sec: float) -> torch.Tensor:
        start_frame = int(round(start_sec * self.cfg.fps))
        n_frames = self.cfg.label_frames
        end_frame = start_frame + n_frames
        crop = spec[:, start_frame:end_frame]
        if crop.shape[-1] < n_frames:
            crop = F.pad(crop, (0, n_frames - crop.shape[-1]))
        elif crop.shape[-1] > n_frames:
            crop = crop[:, :n_frames]
        return crop.contiguous()

    def __getitem__(self, idx: int):
        row, full_spec, start_sec = self._choose_row_and_start(idx)
        spec = self._crop_spec(full_spec, start_sec)
        from .midi_labels import midi_to_targets

        targets = midi_to_targets(row["midi_path"], start_sec, self.cfg.segment_seconds, self.cfg, self.cfg.label_frames)
        item = {
            "spec": spec,
            "spec_path": row["spec_path"],
            "midi_path": row["midi_path"],
            "start_sec": torch.tensor(start_sec, dtype=torch.float32),
        }
        for k, v in targets.items():
            item[k] = torch.from_numpy(v)
        return item


def make_full_song_cached_loaders(
    meta: pd.DataFrame,
    cfg: Config,
    train_samples_per_epoch: int = 20000,
    val_samples: int = 2000,
):
    train_ds = MaestroCachedSpectrogramSegmentDataset(meta, "train", cfg, training=True, samples_per_epoch=train_samples_per_epoch, augment=False)
    val_ds = MaestroCachedSpectrogramSegmentDataset(meta, "validation", cfg, training=False, samples_per_epoch=val_samples, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader


class GenericAudioMidiSegmentDataset(Dataset):
    def __init__(self, manifest_csv: str, split: str, cfg: Config, training: bool, samples_per_epoch: Optional[int] = None, augment: bool = True):
        self.cfg = cfg
        self.training = training
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.samples_per_epoch = samples_per_epoch or len(self.df)
        self.augment = AudioAugmenter(cfg) if (training and augment and cfg.enable_audio_augment) else None
        assert len(self.df) > 0, f"No rows in {manifest_csv} for split={split}"

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        row_idx = random.randrange(len(self.df)) if self.training else idx % len(self.df)
        row = self.df.iloc[row_idx]
        duration = sf.info(row["audio_path"]).duration
        max_start = max(0.0, float(duration) - self.cfg.segment_seconds)
        start = random.uniform(0, max_start) if (self.training and max_start > 0) else 0.0
        wav = load_audio_segment(row["audio_path"], start, self.cfg.segment_seconds, self.cfg.sample_rate)
        if self.augment is not None:
            wav = self.augment(wav)
        from .midi_labels import midi_to_targets

        targets = midi_to_targets(row["midi_path"], start, self.cfg.segment_seconds, self.cfg, self.cfg.label_frames)
        item = {"audio": wav, "audio_path": row["audio_path"], "midi_path": row["midi_path"], "start_sec": torch.tensor(start, dtype=torch.float32)}
        for k, v in targets.items():
            item[k] = torch.from_numpy(v)
        return item


def scan_paired_audio_midi(root: str, out_csv: str, audio_exts=(".wav", ".flac", ".mp3"), midi_exts=(".mid", ".midi")):
    root = Path(root)
    rows = []
    audio_files = []
    for ext in audio_exts:
        audio_files.extend(root.rglob(f"*{ext}"))
    midi_by_stem = {}
    for ext in midi_exts:
        for mp in root.rglob(f"*{ext}"):
            midi_by_stem[mp.stem.lower()] = mp
    for ap in audio_files:
        mp = midi_by_stem.get(ap.stem.lower())
        if mp is not None:
            rows.append({"audio_path": str(ap), "midi_path": str(mp), "split": "train"})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"wrote {len(df)} pairs to {out_csv}")
    return df
