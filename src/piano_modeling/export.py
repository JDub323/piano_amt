"""Export predictions as MIDI, piano-roll NPY, CSV, and probability arrays."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pretty_midi

from .config import CFG, Config
from .decoding import decode_notes_from_probs, decode_pedal_from_probs
from .inference import infer_audio_file
from .models import PianoTranscriptionSystem


def events_to_pianoroll(events: pd.DataFrame, duration_sec: float, cfg: Config = CFG) -> np.ndarray:
    T = int(math.ceil(duration_sec * cfg.fps)) + 1
    roll = np.zeros((T, cfg.n_pitches), dtype=np.float32)
    if events is None or len(events) == 0:
        return roll
    for _, row in events.iterrows():
        p = int(row["midi_pitch"]) - cfg.midi_min
        if not 0 <= p < cfg.n_pitches:
            continue
        lo = max(0, int(round(float(row["onset_sec"]) * cfg.fps)))
        hi = min(T, int(round(float(row["offset_sec"]) * cfg.fps)))
        if hi > lo:
            roll[lo:hi, p] = max(roll[lo:hi, p].max(initial=0), float(row.get("velocity", 100)) / 127.0)
    return roll


def save_events_to_midi(events: pd.DataFrame, midi_path: str | Path, pedal_events: Optional[pd.DataFrame] = None, program: int = 0) -> None:
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=program, name="Transcribed Piano")
    if events is not None and len(events):
        for _, row in events.iterrows():
            start = float(row["onset_sec"])
            end = max(start + 0.01, float(row["offset_sec"]))
            pitch = int(row["midi_pitch"])
            velocity = int(np.clip(row.get("velocity", 100), 1, 127))
            piano.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    if pedal_events is not None and len(pedal_events):
        for _, row in pedal_events.iterrows():
            piano.control_changes.append(pretty_midi.ControlChange(number=64, value=127, time=float(row["onset_sec"])))
            piano.control_changes.append(pretty_midi.ControlChange(number=64, value=0, time=float(row["offset_sec"])))
    pm.instruments.append(piano)
    pm.write(str(midi_path))


def export_prediction_bundle(
    audio_path: str,
    system: PianoTranscriptionSystem,
    out_stem: str,
    export_dir: str | Path,
    cfg: Config = CFG,
):
    probs = infer_audio_file(audio_path, system, cfg)
    notes = decode_notes_from_probs(probs["onset"], probs["frame"], probs["offset"], probs["velocity"], cfg)
    pedal = decode_pedal_from_probs(probs["sustain"], cfg)
    out_dir = Path(export_dir) / out_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_path = out_dir / f"{out_stem}.mid"
    csv_path = out_dir / f"{out_stem}_events.csv"
    pedal_csv_path = out_dir / f"{out_stem}_pedal.csv"
    roll_path = out_dir / f"{out_stem}_pianoroll.npy"
    prob_path = out_dir / f"{out_stem}_probabilities.npz"

    save_events_to_midi(notes, midi_path, pedal)
    notes.to_csv(csv_path, index=False)
    pedal.to_csv(pedal_csv_path, index=False)
    np.save(roll_path, events_to_pianoroll(notes, probs["duration"], cfg))
    np.savez_compressed(
        prob_path,
        onset=probs["onset"],
        offset=probs["offset"],
        frame=probs["frame"],
        velocity=probs["velocity"],
        sustain=probs["sustain"],
    )
    return {"midi": midi_path, "csv": csv_path, "pedal_csv": pedal_csv_path, "pianoroll": roll_path, "probabilities": prob_path, "notes": notes, "pedal": pedal}
