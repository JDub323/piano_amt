"""MIDI parsing and frame-aligned target construction."""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pretty_midi

from .config import CFG, Config


@lru_cache(maxsize=4096)
def parse_midi_cached(midi_path: str):
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    sustain_cc = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append((float(n.start), float(n.end), int(n.pitch), int(n.velocity)))
        for cc in inst.control_changes:
            if int(cc.number) == 64:
                sustain_cc.append((float(cc.time), int(cc.value)))
    notes.sort(key=lambda x: (x[0], x[2], x[1]))
    sustain_cc.sort(key=lambda x: x[0])
    return tuple(notes), tuple(sustain_cc), float(pm.get_end_time())


def _paint_event(target: np.ndarray, frame_idx: int, pitch_idx: int, width: int, value: float = 1.0) -> None:
    T, P = target.shape
    lo = max(0, frame_idx - width // 2)
    hi = min(T, frame_idx + width // 2 + 1)
    if 0 <= pitch_idx < P and lo < hi:
        target[lo:hi, pitch_idx] = value


def midi_to_targets(
    midi_path: str | Path,
    start_sec: float,
    duration_sec: float,
    cfg: Config = CFG,
    n_frames: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Create segment-level targets from a MIDI file."""
    if n_frames is None:
        n_frames = cfg.label_frames
    T, P = n_frames, cfg.n_pitches
    fps = cfg.fps
    seg_end = start_sec + duration_sec

    onset = np.zeros((T, P), dtype=np.float32)
    offset = np.zeros((T, P), dtype=np.float32)
    frame = np.zeros((T, P), dtype=np.float32)
    velocity = np.zeros((T, P), dtype=np.float32)
    sustain = np.zeros((T,), dtype=np.float32)

    notes, sustain_cc, _ = parse_midi_cached(str(midi_path))

    for note_start, note_end, pitch, vel in notes:
        if note_end <= start_sec or note_start >= seg_end:
            continue
        if pitch < cfg.midi_min or pitch > cfg.midi_max:
            continue
        p = pitch - cfg.midi_min
        start_f = int(round((note_start - start_sec) * fps))
        end_f = int(round((note_end - start_sec) * fps))
        active_lo = max(0, int(math.floor((note_start - start_sec) * fps)))
        active_hi = min(T, int(math.ceil((note_end - start_sec) * fps)))
        if active_lo < active_hi:
            frame[active_lo:active_hi, p] = 1.0
        if 0 <= start_f < T:
            _paint_event(onset, start_f, p, cfg.onset_width_frames, 1.0)
            _paint_event(velocity, start_f, p, cfg.onset_width_frames, vel / 127.0)
        if 0 <= end_f < T:
            _paint_event(offset, end_f, p, cfg.offset_width_frames, 1.0)

    # Sustain pedal label from CC64 events.
    state = 0
    for t, val in sustain_cc:
        if t <= start_sec:
            state = 1 if val >= cfg.sustain_threshold else 0
        else:
            break

    last_t = start_sec
    for t, val in sustain_cc:
        if t < start_sec:
            continue
        if t > seg_end:
            break
        lo = max(0, int(math.floor((last_t - start_sec) * fps)))
        hi = min(T, int(math.ceil((t - start_sec) * fps)))
        if lo < hi:
            sustain[lo:hi] = state
        state = 1 if val >= cfg.sustain_threshold else 0
        last_t = t
    lo = max(0, int(math.floor((last_t - start_sec) * fps)))
    if lo < T:
        sustain[lo:T] = state

    return {
        "onset": onset,
        "offset": offset,
        "frame": frame,
        "velocity": velocity,
        "sustain": sustain,
    }


def midi_note_events(midi_path: str | Path, cfg: Config = CFG) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return intervals, pitches in Hz, and normalized velocities for mir_eval."""
    notes, _, _ = parse_midi_cached(str(midi_path))
    intervals, pitches, velocities = [], [], []
    for start, end, pitch, vel in notes:
        if cfg.midi_min <= pitch <= cfg.midi_max and end > start:
            intervals.append([start, end])
            pitches.append(pretty_midi.note_number_to_hz(pitch))
            velocities.append(vel / 127.0)
    return (
        np.asarray(intervals, dtype=np.float64).reshape(-1, 2),
        np.asarray(pitches, dtype=np.float64),
        np.asarray(velocities, dtype=np.float64),
    )
