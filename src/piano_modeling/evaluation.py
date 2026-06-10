"""Formal evaluation helpers and CLI for note, velocity, and pedal metrics."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, fields
from pathlib import Path
from typing import Mapping, Optional

import mir_eval
import numpy as np
import pandas as pd
import pretty_midi
import torch
from tqdm.auto import tqdm

from .config import CFG, Config
from .decoding import decode_notes_from_probs
from .inference import infer_audio_file
from .midi_labels import midi_note_events, midi_to_targets
from .models import PianoTranscriptionSystem
from .training import load_checkpoint


SUMMARY_METRICS = (
    "note_onset_precision",
    "note_onset_recall",
    "note_onset_f1",
    "note_onset_offset_precision",
    "note_onset_offset_recall",
    "note_onset_offset_f1",
    "velocity_precision",
    "velocity_recall",
    "velocity_f1",
    "pedal_precision",
    "pedal_recall",
    "pedal_f1",
    "num_pred_notes",
    "num_ref_notes",
)


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
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def summarize_evaluation_report(report: pd.DataFrame) -> dict[str, float]:
    """Return stable numeric summary metrics from a per-track evaluation report."""
    summary = report.mean(numeric_only=True).to_dict() if len(report) else {}
    return {k: float(summary[k]) for k in SUMMARY_METRICS if k in summary and pd.notna(summary[k])}


def save_evaluation_outputs(report: pd.DataFrame, output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    """Save per-track CSV and aggregate JSON summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}_summary.json"
    report.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summarize_evaluation_report(report), indent=2))
    return csv_path, json_path


def config_from_mapping(values: Mapping | None, fallback: Config = CFG) -> Config:
    """Build Config from a partial mapping while ignoring unknown/stale checkpoint keys."""
    allowed = {f.name for f in fields(Config)}
    base = asdict(fallback)
    if values:
        base.update({k: v for k, v in values.items() if k in allowed})
    return Config(**base)


def config_from_checkpoint(checkpoint_path: str | Path, fallback: Config = CFG) -> Config:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return config_from_mapping(ckpt.get("cfg") or {}, fallback=fallback)


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


def _filter_existing_split(meta: pd.DataFrame, split: str) -> pd.DataFrame:
    required = {"audio_path", "midi_path", "split"}
    missing = sorted(required - set(meta.columns))
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")
    df = meta[(meta["split"] == split) & meta["audio_path"].apply(lambda p: Path(p).exists())]
    df = df[df["midi_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    return df


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
    system.to(device)
    system.eval()
    df = _filter_existing_split(meta, split).head(n_tracks)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"eval {split}"):
        rows.append(evaluate_one_track(row, system, cfg))
    report = pd.DataFrame(rows)
    summary = summarize_evaluation_report(report)
    print(report)
    print(pd.Series(summary, name="mean"))
    stem = f"eval_{split}_{run_name}"
    csv_path, json_path = save_evaluation_outputs(report, export_dir, stem)
    print("saved:", csv_path)
    print("saved:", json_path)
    return report


def evaluate_checkpoint(
    metadata_csv: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    n_tracks: int = 20,
    run_name: Optional[str] = None,
    cfg: Optional[Config] = None,
    device: str = "cuda",
) -> pd.DataFrame:
    """Load a checkpoint and metadata CSV, run formal evaluation, and save reports."""
    checkpoint_path = Path(checkpoint_path)
    cfg = cfg or config_from_checkpoint(checkpoint_path)
    run_name = run_name or checkpoint_path.parent.name or checkpoint_path.stem
    meta = pd.read_csv(metadata_csv)
    system = PianoTranscriptionSystem(cfg).to(device)
    return evaluate_split_sample(
        meta,
        system,
        output_dir,
        run_name,
        split=split,
        n_tracks=n_tracks,
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        device=device,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a piano AMT checkpoint on paired audio/MIDI metadata.")
    parser.add_argument("--metadata-csv", required=True, help="CSV with audio_path, midi_path, and split columns")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    parser.add_argument("--output-dir", default="evaluation_reports", help="Directory for CSV/JSON outputs")
    parser.add_argument("--split", default="test", help="Metadata split to evaluate")
    parser.add_argument("--n-tracks", type=int, default=20, help="Number of tracks from the split to evaluate")
    parser.add_argument("--run-name", default=None, help="Optional report name suffix; defaults to checkpoint parent")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional JSON file with Config overrides. Checkpoint config is used as the base.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = config_from_checkpoint(args.checkpoint)
    if args.config_json:
        overrides = json.loads(Path(args.config_json).read_text())
        cfg = config_from_mapping(overrides, fallback=cfg)
    report = evaluate_checkpoint(
        args.metadata_csv,
        args.checkpoint,
        args.output_dir,
        split=args.split,
        n_tracks=args.n_tracks,
        run_name=args.run_name,
        cfg=cfg,
        device=args.device,
    )
    print(json.dumps(summarize_evaluation_report(report), indent=2))


if __name__ == "__main__":
    main()
