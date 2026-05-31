"""Formal evaluation helpers for note, velocity, and pedal metrics."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import mir_eval
import numpy as np
import pandas as pd
import pretty_midi
from tqdm.auto import tqdm

from .config import CFG, Config
from .decoding import decode_notes_from_probs
from .inference import infer_audio_file
from .midi_labels import midi_note_events, midi_to_targets
from .models import PianoTranscriptionSystem
from .training import load_checkpoint


def events_df_to_mir_eval(events: pd.DataFrame):
    if events is None or len(events) == 0:
        return np.empty((0, 2)), np.empty((0,)), np.empty((0,))
    intervals = events[["onset_sec", "offset_sec"]].to_numpy(dtype=np.float64)
    pitches = np.array([pretty_midi.note_number_to_hz(int(p)) for p in events["midi_pitch"]], dtype=np.float64)
    velocities = events["velocity"].to_numpy(dtype=np.float64) / 127.0
    return intervals, pitches, velocities


def pedal_reference_from_midi(midi_path: str, duration: float, cfg: Config = CFG) -> np.ndarray:
    T = int(math.ceil(duration * cfg.fps)) + 1
    return midi_to_targets(midi_path, 0.0, duration, cfg, n_frames=T)["sustain"]


def frame_prf(est_binary: np.ndarray, ref_binary: np.ndarray):
    est = est_binary.astype(bool)
    ref = ref_binary.astype(bool)
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    tp = np.logical_and(est, ref).sum()
    fp = np.logical_and(est, ~ref).sum()
    fn = np.logical_and(~est, ref).sum()
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return {"precision": p, "recall": r, "f1": f1}


def evaluate_one_track(row: pd.Series, system: PianoTranscriptionSystem, cfg: Config = CFG):
    probs = infer_audio_file(row["audio_path"], system, cfg)
    pred_notes = decode_notes_from_probs(probs["onset"], probs["frame"], probs["offset"], probs["velocity"], cfg)
    est_i, est_p, est_v = events_df_to_mir_eval(pred_notes)
    ref_i, ref_p, ref_v = midi_note_events(row["midi_path"], cfg)

    onset_p, onset_r, onset_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p, onset_tolerance=0.05, offset_ratio=None
    )
    off_p, off_r, off_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p, onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05
    )
    try:
        vel_p, vel_r, vel_f1, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
            ref_i, ref_p, ref_v, est_i, est_p, est_v,
            onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05, velocity_tolerance=0.1,
        )
    except Exception:
        vel_p = vel_r = vel_f1 = np.nan

    ref_pedal = pedal_reference_from_midi(row["midi_path"], probs["duration"], cfg)
    pedal_metrics = frame_prf(probs["sustain"] >= 0.5, ref_pedal >= 0.5)
    return {
        "audio_filename": row.get("audio_filename", Path(row["audio_path"]).name),
        "note_onset_precision": onset_p,
        "note_onset_recall": onset_r,
        "note_onset_f1": onset_f1,
        "note_onset_offset_precision": off_p,
        "note_onset_offset_recall": off_r,
        "note_onset_offset_f1": off_f1,
        "velocity_precision": vel_p,
        "velocity_recall": vel_r,
        "velocity_f1": vel_f1,
        "pedal_precision": pedal_metrics["precision"],
        "pedal_recall": pedal_metrics["recall"],
        "pedal_f1": pedal_metrics["f1"],
        "num_pred_notes": len(pred_notes),
        "num_ref_notes": len(ref_i),
    }


def evaluate_split_sample(
    meta: pd.DataFrame,
    system: PianoTranscriptionSystem,
    export_dir: str | Path,
    run_name: str,
    split: str = "test",
    n_tracks: int = 20,
    checkpoint_path: Optional[Path] = None,
    cfg: Config = CFG,
    device: str = "cuda",
):
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, system, map_location=device, cfg=cfg)
    system.eval()
    df = meta[(meta["split"] == split) & meta["audio_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    df = df.head(n_tracks)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"eval {split}"):
        rows.append(evaluate_one_track(row, system, cfg))
    report = pd.DataFrame(rows)
    print(report)
    print(report.mean(numeric_only=True))
    out_path = Path(export_dir) / f"eval_{split}_{run_name}.csv"
    report.to_csv(out_path, index=False)
    print("saved:", out_path)
    return report
