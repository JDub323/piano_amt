"""Restore, validate, and optionally rebuild the pre-sliced MAESTRO cache."""
from __future__ import annotations

import math
from pathlib import Path
import re
import shutil
import zipfile
from typing import Optional

import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import CFG, Config
from .midi_labels import midi_to_targets
from .paths import ProjectPaths

SPLITS = ("train", "validation", "test")


def has_sliced_files(root: Path) -> bool:
    root = Path(root)
    return any((root / split).exists() and any((root / split).rglob("*.pt")) for split in SPLITS)


def find_dataset_root_after_extract(tmp_root: Path) -> Path:
    candidates = [tmp_root] + [p for p in tmp_root.rglob("*") if p.is_dir()]
    for p in candidates:
        if (p / "sliced_manifest.csv").exists() or any((p / split).exists() for split in SPLITS):
            return p
    raise FileNotFoundError(f"Could not find a sliced dataset root inside the extracted archive at {tmp_root}.")


def restore_sliced_dataset_from_zip(
    paths: ProjectPaths,
    force_resplice_sliced_cache: bool = False,
    tmp_root: str | Path = "/content/_maestro_sliced_extract_tmp",
) -> Path:
    """Restore the Drive-backed sliced zip to local disk for fast training."""
    zip_path = Path(paths.sliced_zip_path)
    dst_root = Path(paths.sliced_root)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing sliced dataset zip: {zip_path}\n"
            "Safe default is to stop here so you do not accidentally redo expensive work. "
            "If you intentionally want to rebuild from the full-song cache, pass "
            "allow_rebuild_if_sliced_zip_missing=True to load_or_build_sliced_dataset()."
        )
    if dst_root.exists() and has_sliced_files(dst_root) and not force_resplice_sliced_cache:
        print(f"Found existing local sliced dataset at {dst_root}; not restoring again.")
        return dst_root

    print(f"Restoring pre-sliced dataset from Drive zip:\n  {zip_path}\n-> {dst_root}")
    tmp_root = Path(tmp_root)
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_root)
    extracted_root = find_dataset_root_after_extract(tmp_root)
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(extracted_root, dst_root)
    shutil.rmtree(tmp_root)
    print(f"Restored sliced dataset to {dst_root}")
    return dst_root


def start_sec_from_chunk_name(path: Path, cfg: Config) -> float:
    m = re.search(r"chunk(\d+)", Path(path).stem)
    if m:
        return int(m.group(1)) * float(cfg.segment_seconds)
    return 0.0


def rebuild_sliced_manifest_from_files(paths: ProjectPaths, cfg: Config = CFG) -> pd.DataFrame:
    rows = []
    root = Path(paths.sliced_root)
    for split in SPLITS:
        split_root = root / split
        if not split_root.exists():
            continue
        for chunk_path in sorted(split_root.rglob("*.pt")):
            rows.append({
                "chunk_path": str(chunk_path),
                "split": split,
                "start_sec": start_sec_from_chunk_name(chunk_path, cfg),
            })
    if not rows:
        raise FileNotFoundError(f"No sliced .pt files found under {root}")
    sliced_df = pd.DataFrame(rows)
    sliced_df.to_csv(paths.sliced_manifest, index=False)
    print(f"Rebuilt sliced manifest with {len(sliced_df)} chunks: {paths.sliced_manifest}")
    return sliced_df


def repair_manifest_paths(sliced_df: pd.DataFrame, paths: ProjectPaths, cfg: Config = CFG) -> pd.DataFrame:
    root = Path(paths.sliced_root)
    repaired = sliced_df.copy()
    repaired_paths = []
    missing = []
    basename_index = None
    for _, row in repaired.iterrows():
        original = Path(str(row["chunk_path"]))
        if original.exists():
            repaired_paths.append(str(original))
            continue
        split = str(row.get("split", ""))
        candidate = root / split / original.name
        if candidate.exists():
            repaired_paths.append(str(candidate))
            continue
        if basename_index is None:
            basename_index = {}
            for p in root.rglob("*.pt"):
                basename_index.setdefault(p.name, []).append(p)
        matches = basename_index.get(original.name, [])
        if len(matches) == 1:
            repaired_paths.append(str(matches[0]))
        else:
            repaired_paths.append(str(original))
            missing.append(str(original))

    repaired["chunk_path"] = repaired_paths
    if "start_sec" not in repaired.columns:
        repaired["start_sec"] = repaired["chunk_path"].apply(lambda p: start_sec_from_chunk_name(Path(p), cfg))

    still_missing = [p for p in repaired["chunk_path"] if not Path(p).exists()]
    if still_missing:
        raise FileNotFoundError(
            f"{len(still_missing)} chunk paths in the sliced manifest do not exist after repair. "
            f"First missing path: {still_missing[0]}"
        )
    repaired.to_csv(paths.sliced_manifest, index=False)
    if missing:
        print(f"Repaired manifest paths; {len(missing)} original paths needed remapping.")
    return repaired


def metadata_value_matches(saved_value, current_value) -> bool:
    if isinstance(current_value, float):
        try:
            return abs(float(saved_value) - float(current_value)) < 1e-6
        except Exception:
            return False
    return saved_value == current_value


def validate_sliced_cache_compatibility(sliced_df: pd.DataFrame, cfg: Config, samples_per_split: int = 3) -> None:
    required = {"spec", "onset", "offset", "frame", "velocity", "sustain"}
    expected_meta = {
        "feature_type": cfg.feature_type.lower(),
        "sample_rate": cfg.sample_rate,
        "hop_length": cfg.hop_length,
        "n_fft": cfg.n_fft,
        "n_mels": cfg.n_mels,
        "f_min": cfg.f_min,
        "f_max": cfg.f_max,
        "segment_seconds": cfg.segment_seconds,
        "midi_min": cfg.midi_min,
        "midi_max": cfg.midi_max,
    }

    sample_rows = []
    for split in SPLITS:
        split_df = sliced_df[sliced_df["split"] == split]
        if len(split_df):
            sample_rows.extend(split_df.head(samples_per_split).to_dict("records"))
    if not sample_rows:
        raise ValueError("Sliced manifest has no train/validation/test rows.")

    chunks_without_metadata = 0
    for row in sample_rows:
        path = Path(row["chunk_path"])
        if not path.exists():
            raise FileNotFoundError(f"Missing sliced chunk: {path}")
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload in {path}, got {type(payload)}")
        missing_keys = required - set(payload.keys())
        if missing_keys:
            raise KeyError(f"{path} is missing required keys: {sorted(missing_keys)}")

        spec = payload["spec"]
        if spec.ndim != 2:
            raise ValueError(f"{path}: expected spec shape [n_mels, frames], got {tuple(spec.shape)}")
        if int(spec.shape[0]) != int(cfg.n_mels):
            raise ValueError(f"{path}: n_mels mismatch. chunk={spec.shape[0]}, cfg.n_mels={cfg.n_mels}")
        if int(spec.shape[-1]) != int(cfg.label_frames):
            raise ValueError(f"{path}: frame count mismatch. chunk={spec.shape[-1]}, cfg.label_frames={cfg.label_frames}")

        expected_pitch_label_shape = (cfg.label_frames, cfg.n_pitches)
        for key in ["onset", "offset", "frame", "velocity"]:
            actual = tuple(payload[key].shape)
            if actual != expected_pitch_label_shape:
                raise ValueError(f"{path}: {key} shape mismatch. chunk={actual}, expected={expected_pitch_label_shape}")

        sustain_shape = tuple(payload["sustain"].shape)
        valid_sustain_shapes = {(cfg.label_frames,), (cfg.label_frames, 1)}
        if sustain_shape not in valid_sustain_shapes:
            raise ValueError(f"{path}: sustain shape mismatch. chunk={sustain_shape}, expected one of {sorted(valid_sustain_shapes)}")

        metadata = payload.get("metadata")
        if metadata is None:
            metadata = {k: payload[k] for k in expected_meta.keys() if k in payload}
        if metadata:
            mismatches = []
            for k, expected in expected_meta.items():
                if k in metadata and not metadata_value_matches(metadata[k], expected):
                    mismatches.append((k, metadata[k], expected))
            if mismatches:
                lines = "\n".join([f"  {k}: chunk={old!r}, cfg={new!r}" for k, old, new in mismatches])
                raise ValueError(f"Sliced cache metadata mismatch in {path}:\n{lines}")
        else:
            chunks_without_metadata += 1

    if chunks_without_metadata:
        print(
            f"Compatibility check passed by tensor shape for {len(sample_rows)} sampled chunks. "
            f"{chunks_without_metadata} sampled chunks were legacy files without metadata."
        )
    else:
        print(f"Compatibility check passed for {len(sample_rows)} sampled chunks with metadata.")


def full_song_cache_complete(meta_df: pd.DataFrame) -> bool:
    required_cols = {"spec_path", "midi_path"}
    if not required_cols.issubset(meta_df.columns):
        return False
    spec_ok = meta_df["spec_path"].apply(lambda p: Path(p).exists())
    midi_ok = meta_df["midi_path"].apply(lambda p: Path(p).exists())
    missing = int((~(spec_ok & midi_ok)).sum())
    if missing:
        print(f"Full-song cache is incomplete: {missing}/{len(meta_df)} rows are missing spec or MIDI files.")
    return missing == 0


def pre_slice_dataset(
    meta_df: pd.DataFrame,
    cfg: Config,
    paths: ProjectPaths,
    max_songs: Optional[int] = None,
    force_resplice_sliced_cache: bool = False,
) -> pd.DataFrame:
    rows = []
    df_to_process = meta_df if max_songs is None else meta_df.head(max_songs)
    print(f"Pre-slicing {len(df_to_process)} songs into {cfg.segment_seconds:g}s chunks...")

    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
        spec_path = Path(row["spec_path"])
        midi_path = Path(row["midi_path"])
        split = str(row["split"])
        source_id = str(row.get("audio_filename", spec_path.stem))
        safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id).replace(".wav", "")
        song_id = f"{idx:05d}_{Path(safe_source).stem}"
        try:
            payload = torch.load(spec_path, map_location="cpu")
            full_spec = payload["spec"] if isinstance(payload, dict) and "spec" in payload else payload
            full_spec = full_spec.float()
            duration = max(0.0, (full_spec.shape[-1] - 1) / cfg.fps)
            num_chunks = max(1, int(math.ceil(duration / cfg.segment_seconds)))
            for chunk_idx in range(num_chunks):
                start_sec = chunk_idx * cfg.segment_seconds
                chunk_name = f"{song_id}_chunk{chunk_idx:04d}.pt"
                chunk_path = paths.sliced_root / split / chunk_name
                rows.append({"chunk_path": str(chunk_path), "split": split, "start_sec": start_sec})
                if chunk_path.exists() and not force_resplice_sliced_cache:
                    continue
                start_frame = int(round(start_sec * cfg.fps))
                n_frames = cfg.label_frames
                end_frame = start_frame + n_frames
                crop = full_spec[:, start_frame:end_frame]
                if crop.shape[-1] < n_frames:
                    crop = torch.nn.functional.pad(crop, (0, n_frames - crop.shape[-1]))
                elif crop.shape[-1] > n_frames:
                    crop = crop[:, :n_frames]
                targets = midi_to_targets(midi_path, start_sec, cfg.segment_seconds, cfg, n_frames)
                save_dict = {
                    "spec": crop.half(),
                    "onset": torch.from_numpy(targets["onset"]).half(),
                    "offset": torch.from_numpy(targets["offset"]).half(),
                    "frame": torch.from_numpy(targets["frame"]).half(),
                    "velocity": torch.from_numpy(targets["velocity"]).half(),
                    "sustain": torch.from_numpy(targets["sustain"]).half(),
                    "metadata": {
                        "feature_type": cfg.feature_type.lower(),
                        "sample_rate": cfg.sample_rate,
                        "hop_length": cfg.hop_length,
                        "n_fft": cfg.n_fft,
                        "n_mels": cfg.n_mels,
                        "f_min": cfg.f_min,
                        "f_max": cfg.f_max,
                        "segment_seconds": cfg.segment_seconds,
                        "midi_min": cfg.midi_min,
                        "midi_max": cfg.midi_max,
                    },
                }
                chunk_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(save_dict, chunk_path)
        except Exception as e:
            print(f"Failed to slice {song_id}: {e}")

    sliced_df = pd.DataFrame(rows)
    paths.sliced_root.mkdir(parents=True, exist_ok=True)
    sliced_df.to_csv(paths.sliced_manifest, index=False)
    print(f"Saved {len(sliced_df)} chunk references to {paths.sliced_manifest}")
    return sliced_df


def backup_sliced_dataset_to_drive(paths: ProjectPaths) -> None:
    if not has_sliced_files(paths.sliced_root):
        raise FileNotFoundError(f"No sliced files to back up under {paths.sliced_root}")
    local_zip_base = Path("/content/local_maestro_sliced")
    local_zip = local_zip_base.with_suffix(".zip")
    if local_zip.exists():
        local_zip.unlink()
    print("Compressing local sliced dataset. This can take a few minutes...")
    shutil.make_archive(str(local_zip_base), "zip", root_dir=str(paths.sliced_root.parent), base_dir=paths.sliced_root.name)
    paths.sliced_zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying sliced backup to Drive: {paths.sliced_zip_path}")
    shutil.copy(local_zip, paths.sliced_zip_path)
    print("Backup complete.")


def load_or_build_sliced_dataset(
    paths: ProjectPaths,
    cfg: Config = CFG,
    meta_df: Optional[pd.DataFrame] = None,
    use_pre_sliced_dataset: bool = True,
    allow_rebuild_if_sliced_zip_missing: bool = False,
    force_resplice_sliced_cache: bool = False,
) -> pd.DataFrame:
    if not use_pre_sliced_dataset:
        raise RuntimeError("use_pre_sliced_dataset=False, but the cleaned training path expects pre-sliced data.")

    if force_resplice_sliced_cache:
        print("force_resplice_sliced_cache=True. Rebuilding sliced cache from full-song spec/MIDI cache.")
        if paths.sliced_root.exists():
            shutil.rmtree(paths.sliced_root)
        paths.sliced_root.mkdir(parents=True, exist_ok=True)
        if meta_df is None or not full_song_cache_complete(meta_df):
            raise FileNotFoundError(
                "Cannot reslice because the full-song spectrogram/MIDI cache is incomplete. "
                "Restore the sliced zip, or rebuild the full-song cache first."
            )
        sliced_df = pre_slice_dataset(meta_df, cfg, paths, max_songs=None, force_resplice_sliced_cache=True)
        backup_sliced_dataset_to_drive(paths)
        return sliced_df

    if not has_sliced_files(paths.sliced_root):
        if paths.sliced_zip_path.exists():
            restore_sliced_dataset_from_zip(paths, force_resplice_sliced_cache=False)
        elif allow_rebuild_if_sliced_zip_missing:
            print("Sliced zip is missing, but rebuild is explicitly allowed.")
            if meta_df is None or not full_song_cache_complete(meta_df):
                raise FileNotFoundError(
                    "Cannot rebuild sliced cache because full-song spec/MIDI cache is incomplete. "
                    "Rebuild it first, or restore the sliced zip."
                )
            paths.sliced_root.mkdir(parents=True, exist_ok=True)
            sliced_df = pre_slice_dataset(meta_df, cfg, paths, max_songs=None)
            backup_sliced_dataset_to_drive(paths)
            return sliced_df
        else:
            raise FileNotFoundError(
                f"Missing local sliced dataset and Drive zip:\n  local: {paths.sliced_root}\n  zip:   {paths.sliced_zip_path}\n"
                "Safe default is to stop rather than redownload/reslice. "
                "Restore the zip to Drive or set allow_rebuild_if_sliced_zip_missing=True."
            )

    if paths.sliced_manifest.exists():
        print(f"Loading sliced manifest: {paths.sliced_manifest}")
        sliced_df = pd.read_csv(paths.sliced_manifest)
    else:
        print("sliced_manifest.csv not found; rebuilding it by scanning existing chunks.")
        sliced_df = rebuild_sliced_manifest_from_files(paths, cfg)

    sliced_df = repair_manifest_paths(sliced_df, paths, cfg)
    validate_sliced_cache_compatibility(sliced_df, cfg)
    print("Sliced chunk counts:")
    print(sliced_df["split"].value_counts())
    return sliced_df
