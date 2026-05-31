"""MAESTRO metadata and optional full-song spectrogram cache utilities.

The normal experiment path uses the pre-sliced dataset zip. Full-song MAESTRO
preprocessing is kept here as an explicit fallback/rebuild path and is disabled
by default in the notebook.
"""
from __future__ import annotations

import gc
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests
import soundfile as sf
import torch
import torchaudio
from tqdm.auto import tqdm

from .config import CFG, Config
from .paths import ProjectPaths

try:
    from remotezip import RemoteZip
except ImportError:  # keeps the Colab notebook self-healing
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "remotezip"])
    from remotezip import RemoteZip


MAESTRO_BASE_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
MAESTRO_ZIP_URL = MAESTRO_BASE_URL + "maestro-v3.0.0.zip"
MAESTRO_CSV_URL = MAESTRO_BASE_URL + "maestro-v3.0.0.csv"
MAESTRO_JSON_URL = MAESTRO_BASE_URL + "maestro-v3.0.0.json"


def _metadata_csv_path(paths: ProjectPaths) -> Path:
    return paths.maestro_meta_root / "maestro-v3.0.0.csv"


def _metadata_json_path(paths: ProjectPaths) -> Path:
    return paths.maestro_meta_root / "maestro-v3.0.0.json"


def download_small_file(url: str, dest: Path, retries: int = 3, chunk_size: int = 1024 * 1024) -> None:
    """Download small metadata files using requests instead of wget."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            with requests.get(url, stream=True, timeout=(10, 120)) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            if tmp.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty file: {url}")
            tmp.replace(dest)
            return
        except Exception as e:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise
            print(f"metadata download failed attempt {attempt}/{retries}: {url} -> {repr(e)}")
            time.sleep(2 * attempt)


def zip_member_candidates(relative_path: str):
    rel = str(relative_path).replace("\\", "/").lstrip("/")
    prefix = "maestro-v3.0.0/"
    candidates = [rel, prefix + rel]
    if rel.startswith(prefix):
        candidates.append(rel[len(prefix):])
    seen, out = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def find_zip_member(zip_names, relative_path: str) -> str:
    for candidate in zip_member_candidates(relative_path):
        if candidate in zip_names:
            return candidate
    rel = str(relative_path).replace("\\", "/").lstrip("/")
    suffix_matches = [name for name in zip_names if name.endswith("/" + rel) or name == rel]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        return sorted(suffix_matches, key=len)[0]
    raise FileNotFoundError(f"Could not find {relative_path} inside {MAESTRO_ZIP_URL}")


def extract_zip_member(rz, zip_names, relative_path: str, dest: Path, retries: int = 3) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    member = find_zip_member(zip_names, relative_path)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            with rz.open(member) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            if tmp.stat().st_size == 0:
                raise RuntimeError(f"Extracted empty file from zip: {member}")
            tmp.replace(dest)
            return
        except Exception as e:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise
            print(f"zip extraction failed attempt {attempt}/{retries}: {member} -> {repr(e)}")
            time.sleep(2 * attempt)


def download_metadata(paths: ProjectPaths) -> Path:
    """Download MAESTRO CSV/JSON metadata; falls back to the remote zip."""
    paths.maestro_meta_root.mkdir(parents=True, exist_ok=True)
    metadata_csv = _metadata_csv_path(paths)
    metadata_json = _metadata_json_path(paths)
    missing = []

    if not metadata_csv.exists():
        try:
            download_small_file(MAESTRO_CSV_URL, metadata_csv)
        except Exception as e:
            print("CSV metadata direct download failed; will try zip fallback:", repr(e))
            missing.append(("maestro-v3.0.0.csv", metadata_csv))

    if not metadata_json.exists():
        try:
            download_small_file(MAESTRO_JSON_URL, metadata_json)
        except Exception as e:
            print("JSON metadata direct download failed; will try zip fallback:", repr(e))
            missing.append(("maestro-v3.0.0.json", metadata_json))

    if missing:
        print("opening remote MAESTRO zip for metadata fallback...")
        with RemoteZip(MAESTRO_ZIP_URL) as rz:
            zip_names = set(rz.namelist())
            for member_name, dest in missing:
                extract_zip_member(rz, zip_names, member_name, dest)

    print("metadata_csv:", metadata_csv, metadata_csv.exists())
    return metadata_csv


def cache_spec_path(paths: ProjectPaths, audio_filename: str, cfg: Config = CFG) -> Path:
    rel = Path(str(audio_filename))
    return (paths.maestro_spec_root / rel).with_suffix(f".{cfg.feature_type.lower()}.pt")


def cache_midi_path(paths: ProjectPaths, midi_filename: str) -> Path:
    return paths.maestro_midi_root / str(midi_filename)


def work_audio_path(paths: ProjectPaths, audio_filename: str) -> Path:
    return paths.maestro_work_root / str(audio_filename)


def load_full_audio_mono(path: Path, target_sr: int) -> torch.Tensor:
    audio, native_sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    wav = torch.from_numpy(audio)
    if int(native_sr) != int(target_sr) and wav.numel() > 0:
        wav = torchaudio.functional.resample(wav, int(native_sr), int(target_sr))
    return wav.float().clamp(-1.0, 1.0)


def make_spectrogram_transform(cfg: Config, device: str):
    if cfg.feature_type.lower() == "mel":
        return torchaudio.transforms.MelSpectrogram(
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
        ).to(device)
    if cfg.feature_type.lower() == "cqt":
        from nnAudio.features import CQT1992v2
        return CQT1992v2(
            sr=cfg.sample_rate,
            hop_length=cfg.hop_length,
            fmin=cfg.f_min,
            n_bins=cfg.cqt_n_bins,
            bins_per_octave=cfg.cqt_bins_per_octave,
            output_format="Magnitude",
            verbose=False,
        ).to(device)
    raise ValueError(f"Unsupported cfg.feature_type={cfg.feature_type}")


@torch.no_grad()
def audio_to_cached_spectrogram(audio_path: Path, transform, cfg: Config, device: str, save_dtype: str = "float16") -> torch.Tensor:
    wav = load_full_audio_mono(audio_path, cfg.sample_rate).to(device)
    spec = transform(wav.unsqueeze(0))
    if isinstance(spec, tuple):
        spec = spec[0]
    if spec.ndim == 4:
        spec = spec.squeeze(1)
    spec = torch.log1p(spec.squeeze(0).clamp_min(0) * 1000.0).cpu()
    spec = spec.half() if save_dtype == "float16" else spec.float()
    return spec.contiguous()


def write_cache_manifest(paths: ProjectPaths, meta: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    manifest = meta.copy()
    manifest["spec_path"] = manifest["audio_filename"].apply(lambda x: str(cache_spec_path(paths, x, cfg)))
    manifest["midi_path"] = manifest["midi_filename"].apply(lambda x: str(cache_midi_path(paths, x)))
    manifest["feature_type"] = cfg.feature_type.lower()
    manifest["sample_rate"] = cfg.sample_rate
    manifest["hop_length"] = cfg.hop_length
    manifest["spec_exists"] = manifest["spec_path"].apply(lambda p: Path(p).exists())
    manifest["midi_exists"] = manifest["midi_path"].apply(lambda p: Path(p).exists())
    paths.maestro_spec_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(paths.maestro_cache_manifest, index=False)
    print(
        f"cached specs: {int(manifest.spec_exists.sum())}/{len(manifest)} | "
        f"cached MIDI: {int(manifest.midi_exists.sum())}/{len(manifest)}"
    )
    print("manifest:", paths.maestro_cache_manifest)
    return manifest


def process_one_maestro_row(
    row,
    transform,
    rz,
    zip_names,
    paths: ProjectPaths,
    cfg: Config,
    device: str,
    overwrite_existing_specs: bool = False,
    delete_audio_file_immediately: bool = True,
    save_spectrogram_dtype: str = "float16",
) -> str:
    audio_filename = str(row["audio_filename"])
    midi_filename = str(row["midi_filename"])
    spec_path = cache_spec_path(paths, audio_filename, cfg)
    midi_cache_path = cache_midi_path(paths, midi_filename)
    work_audio = work_audio_path(paths, audio_filename)

    if not midi_cache_path.exists():
        extract_zip_member(rz, zip_names, midi_filename, midi_cache_path)

    if spec_path.exists() and not overwrite_existing_specs:
        return "skipped"

    extract_zip_member(rz, zip_names, audio_filename, work_audio)
    spec = audio_to_cached_spectrogram(work_audio, transform, cfg, device, save_spectrogram_dtype)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec": spec,
        "audio_filename": audio_filename,
        "midi_filename": midi_filename,
        "feature_type": cfg.feature_type.lower(),
        "sample_rate": cfg.sample_rate,
        "hop_length": cfg.hop_length,
        "n_fft": cfg.n_fft,
        "n_mels": cfg.n_mels,
        "f_min": cfg.f_min,
        "f_max": cfg.f_max,
        "duration": float(row["duration"]) if "duration" in row.index else None,
    }
    torch.save(payload, spec_path)

    if delete_audio_file_immediately and work_audio.exists():
        work_audio.unlink()

    return "cached"


def preprocess_maestro_in_download_batches(
    paths: ProjectPaths,
    cfg: Config = CFG,
    num_batches: int = 2,
    start_batch: int = 0,
    end_batch: Optional[int] = None,
    overwrite_existing_specs: bool = False,
    delete_audio_file_immediately: bool = True,
    delete_working_dir_after_each_batch: bool = True,
    save_spectrogram_dtype: str = "float16",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> pd.DataFrame:
    """Fallback-only full-song MAESTRO preprocessing path."""
    metadata_csv = download_metadata(paths)
    meta = pd.read_csv(metadata_csv)
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1")
    end_batch = num_batches if end_batch is None or end_batch < 0 else min(end_batch, num_batches)
    batch_indices = np.array_split(np.arange(len(meta)), num_batches)
    transform = make_spectrogram_transform(cfg, device).eval()

    print(f"Preprocessing on {device}; batches {start_batch} through {end_batch - 1}")
    print("Using remote zip:", MAESTRO_ZIP_URL)
    print("reading remote zip directory...")
    with RemoteZip(MAESTRO_ZIP_URL) as rz_for_names:
        zip_names = set(rz_for_names.namelist())

    for batch_idx in range(start_batch, end_batch):
        rows = meta.iloc[batch_indices[batch_idx]].reset_index(drop=True)
        print(f"\n=== MAESTRO batch {batch_idx + 1}/{num_batches}: {len(rows)} songs ===")
        counts = {"cached": 0, "skipped": 0, "failed": 0}
        with RemoteZip(MAESTRO_ZIP_URL) as rz:
            for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"cache batch {batch_idx + 1}/{num_batches}"):
                try:
                    status = process_one_maestro_row(
                        row, transform, rz, zip_names, paths, cfg, device,
                        overwrite_existing_specs=overwrite_existing_specs,
                        delete_audio_file_immediately=delete_audio_file_immediately,
                        save_spectrogram_dtype=save_spectrogram_dtype,
                    )
                    counts[status] = counts.get(status, 0) + 1
                except Exception as e:
                    counts["failed"] += 1
                    print("FAILED:", row.get("audio_filename", "<unknown>"), repr(e))
        write_cache_manifest(paths, meta, cfg)
        if delete_working_dir_after_each_batch and paths.maestro_work_root.exists():
            shutil.rmtree(paths.maestro_work_root)
            paths.maestro_work_root.mkdir(parents=True, exist_ok=True)
            print("deleted temporary working audio:", paths.maestro_work_root)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("batch counts:", counts)
    return write_cache_manifest(paths, meta, cfg)


def load_cached_maestro_manifest(
    paths: ProjectPaths,
    cfg: Config = CFG,
    download_if_missing: bool = True,
    disable_audio_augment: bool = True,
) -> pd.DataFrame:
    """Load the full-song manifest without doing long downloads/rebuilds."""
    metadata_csv = _metadata_csv_path(paths)
    if download_if_missing or not metadata_csv.exists():
        metadata_csv = download_metadata(paths)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing {metadata_csv}. Run download_metadata(paths) first.")

    raw_meta = pd.read_csv(metadata_csv)
    if paths.maestro_cache_manifest.exists():
        meta = pd.read_csv(paths.maestro_cache_manifest)
    else:
        print("Cache manifest not found yet; building a manifest from expected cache paths.")
        meta = raw_meta.copy()
        meta["spec_path"] = meta["audio_filename"].apply(lambda x: str(cache_spec_path(paths, x, cfg)))
        meta["midi_path"] = meta["midi_filename"].apply(lambda x: str(cache_midi_path(paths, x)))
        meta["feature_type"] = cfg.feature_type.lower()
        meta["sample_rate"] = cfg.sample_rate
        meta["hop_length"] = cfg.hop_length

    meta["spec_exists"] = meta["spec_path"].apply(lambda p: Path(p).exists())
    meta["midi_exists"] = meta["midi_path"].apply(lambda p: Path(p).exists())

    print(meta.head())
    print(meta["split"].value_counts())
    print(f"Cached spectrogram files: {int(meta['spec_exists'].sum())} / {len(meta)}")
    print(f"Cached MIDI files: {int(meta['midi_exists'].sum())} / {len(meta)}")
    missing = meta[~(meta["spec_exists"] & meta["midi_exists"])]
    if len(missing):
        print(f"WARNING: {len(missing)} rows are not cached yet. This is okay if you train from the pre-sliced zip.")
        print(missing[["split", "audio_filename", "midi_filename", "spec_exists", "midi_exists"]].head(10))
    else:
        print("All MAESTRO rows are cached as spectrogram tensors + MIDI.")

    if disable_audio_augment and cfg.enable_audio_augment:
        cfg.enable_audio_augment = False
        print("Set cfg.enable_audio_augment = False because training reads cached/pre-sliced spectrograms, not waveforms.")
    return meta


def copy_full_song_cache_to_local(
    paths: ProjectPaths,
    meta: Optional[pd.DataFrame] = None,
    local_dir: str | Path = "/content/local_maestro",
) -> Optional[pd.DataFrame]:
    """Optional fallback utility for the old full-song cached-spectrogram dataset."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_spec_dir = local_dir / paths.maestro_spec_root.name
    local_midi_dir = local_dir / paths.maestro_midi_root.name
    t0 = time.time()
    if paths.maestro_spec_root.exists():
        print(f"Copying spectrograms to {local_spec_dir}...")
        shutil.copytree(paths.maestro_spec_root, local_spec_dir, dirs_exist_ok=True)
    else:
        print(f"WARNING: full-song spec cache does not exist: {paths.maestro_spec_root}")
    if paths.maestro_midi_root.exists():
        print(f"Copying MIDI to {local_midi_dir}...")
        shutil.copytree(paths.maestro_midi_root, local_midi_dir, dirs_exist_ok=True)
    else:
        print(f"WARNING: MIDI cache does not exist: {paths.maestro_midi_root}")
    print(f"\nCopy finished in {time.time() - t0:.1f} seconds.")
    if meta is None:
        return None
    meta = meta.copy()
    meta["spec_path"] = meta["spec_path"].str.replace(str(paths.maestro_spec_root), str(local_spec_dir), regex=False)
    meta["midi_path"] = meta["midi_path"].str.replace(str(paths.maestro_midi_root), str(local_midi_dir), regex=False)
    print("Updated manifest dataframe to point to local full-song cache paths.")
    return meta
