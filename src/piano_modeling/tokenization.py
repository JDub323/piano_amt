"""Event-token representation for experimental seq2seq piano transcription."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .config import Config

PAD = 0
BOS = 1
EOS = 2
FRAME_FORWARD = 3
PEDAL_ON = 4
PEDAL_OFF = 5
NOTE_ON_BASE = 6


@dataclass(frozen=True)
class TokenSpec:
    midi_min: int
    midi_max: int
    velocity_bins: int

    @property
    def n_pitches(self) -> int:
        return self.midi_max - self.midi_min + 1

    @property
    def note_off_base(self) -> int:
        return NOTE_ON_BASE + self.n_pitches * self.velocity_bins

    @property
    def vocab_size(self) -> int:
        return self.note_off_base + self.n_pitches


def token_spec_from_config(cfg: Config) -> TokenSpec:
    return TokenSpec(cfg.midi_min, cfg.midi_max, int(cfg.token_velocity_bins))


def token_vocab_size(cfg: Config) -> int:
    return token_spec_from_config(cfg).vocab_size


def velocity_to_bin(velocity: int, velocity_bins: int) -> int:
    velocity = int(max(1, min(127, velocity)))
    return min(velocity_bins - 1, int(round((velocity - 1) * (velocity_bins - 1) / 126)))


def bin_to_velocity(bin_idx: int, velocity_bins: int) -> int:
    bin_idx = int(max(0, min(velocity_bins - 1, bin_idx)))
    if velocity_bins <= 1:
        return 64
    return int(round(1 + bin_idx * 126 / (velocity_bins - 1)))


def note_on_token(pitch_idx: int, velocity_bin: int, spec: TokenSpec) -> int:
    return NOTE_ON_BASE + int(pitch_idx) * spec.velocity_bins + int(velocity_bin)


def note_off_token(pitch_idx: int, spec: TokenSpec) -> int:
    return spec.note_off_base + int(pitch_idx)


def token_kind(token: int, spec: TokenSpec) -> str:
    token = int(token)
    if token == PAD:
        return "pad"
    if token == BOS:
        return "bos"
    if token == EOS:
        return "eos"
    if token == FRAME_FORWARD:
        return "frame_forward"
    if token == PEDAL_ON:
        return "pedal_on"
    if token == PEDAL_OFF:
        return "pedal_off"
    if NOTE_ON_BASE <= token < spec.note_off_base:
        return "note_on"
    if spec.note_off_base <= token < spec.vocab_size:
        return "note_off"
    return "unknown"


def decode_note_on(token: int, spec: TokenSpec) -> Tuple[int, int]:
    rel = int(token) - NOTE_ON_BASE
    pitch_idx = rel // spec.velocity_bins
    velocity_bin = rel % spec.velocity_bins
    return spec.midi_min + pitch_idx, bin_to_velocity(velocity_bin, spec.velocity_bins)


def decode_note_off(token: int, spec: TokenSpec) -> int:
    pitch_idx = int(token) - spec.note_off_base
    return spec.midi_min + pitch_idx


def _paint_repeated_forward(tokens: List[int], n: int) -> None:
    if n > 0:
        tokens.extend([FRAME_FORWARD] * int(n))


def midi_to_event_tokens(
    midi_path: str,
    start_sec: float,
    segment_seconds: float,
    cfg: Config,
    *,
    add_bos: bool = True,
    add_eos: bool = True,
    max_seq_len: int | None = None,
) -> np.ndarray:
    """Convert a MIDI segment to compact autoregressive event tokens.

    The representation is a sparse MIDI-event stream over frame time:

    BOS, events at current frame, FRAME_FORWARD, ..., EOS

    Each FRAME_FORWARD advances exactly one label frame. Note activity is encoded as NOTE_ON
    with quantized velocity and NOTE_OFF. Sustain pedal uses binary PEDAL_ON/PEDAL_OFF events.
    This avoids storing dense onset/offset/frame/velocity target matrices.
    """
    spec = token_spec_from_config(cfg)
    max_seq_len = int(max_seq_len or cfg.token_max_seq_len)
    end_sec = float(start_sec) + float(segment_seconds)
    n_frames = cfg.label_frames
    from .midi_labels import parse_midi_cached

    notes, sustain_cc, _ = parse_midi_cached(str(midi_path))

    events_by_frame: dict[int, list[tuple[int, int]]] = {}

    def add_event(frame: int, priority: int, token: int) -> None:
        frame = max(0, min(n_frames - 1, int(frame)))
        events_by_frame.setdefault(frame, []).append((priority, int(token)))

    for note_start, note_end, pitch, velocity in notes:
        if note_end < start_sec or note_start >= end_sec:
            continue
        if not (cfg.midi_min <= pitch <= cfg.midi_max):
            continue
        pitch_idx = pitch - cfg.midi_min
        on_frame = int(round((max(float(note_start), float(start_sec)) - start_sec) * cfg.fps))
        off_frame = int(round((min(float(note_end), float(end_sec)) - start_sec) * cfg.fps))
        vel_bin = velocity_to_bin(int(velocity), spec.velocity_bins)
        add_event(on_frame, 1, note_on_token(pitch_idx, vel_bin, spec))
        add_event(off_frame, 0, note_off_token(pitch_idx, spec))

    for cc_time, cc_value in sustain_cc:
        if cc_time < start_sec or cc_time >= end_sec:
            continue
        frame = int(round((float(cc_time) - start_sec) * cfg.fps))
        add_event(frame, 2, PEDAL_ON if int(cc_value) >= cfg.sustain_threshold else PEDAL_OFF)

    tokens: List[int] = [BOS] if add_bos else []
    current_frame = 0
    for frame in sorted(events_by_frame):
        _paint_repeated_forward(tokens, frame - current_frame)
        current_frame = frame
        for _, token in sorted(events_by_frame[frame], key=lambda x: (x[0], x[1])):
            tokens.append(token)
            if len(tokens) >= max_seq_len - int(add_eos):
                break
        if len(tokens) >= max_seq_len - int(add_eos):
            break
    if add_eos:
        tokens.append(EOS)
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
        tokens[-1] = EOS
    return np.asarray(tokens, dtype=np.int32)


def token_sequence_to_events(tokens: Sequence[int], cfg: Config) -> list[dict]:
    """Best-effort conversion from predicted tokens back to note/pedal events."""
    spec = token_spec_from_config(cfg)
    frame = 0
    active: dict[int, tuple[float, int]] = {}
    events: list[dict] = []
    for token in tokens:
        token = int(token)
        if token in (PAD, BOS):
            continue
        if token == EOS:
            break
        if token == FRAME_FORWARD:
            frame += 1
            continue
        time_sec = frame / cfg.fps
        if token == PEDAL_ON:
            events.append({"type": "pedal_on", "time": time_sec})
        elif token == PEDAL_OFF:
            events.append({"type": "pedal_off", "time": time_sec})
        elif NOTE_ON_BASE <= token < spec.note_off_base:
            pitch, velocity = decode_note_on(token, spec)
            if pitch in active:
                onset, old_velocity = active[pitch]
                events.append({"type": "note", "onset": onset, "offset": time_sec, "pitch": pitch, "velocity": old_velocity})
            active[pitch] = (time_sec, velocity)
        elif spec.note_off_base <= token < spec.vocab_size:
            pitch = decode_note_off(token, spec)
            if pitch in active:
                onset, velocity = active.pop(pitch)
                if time_sec > onset:
                    events.append({"type": "note", "onset": onset, "offset": time_sec, "pitch": pitch, "velocity": velocity})
    final_time = frame / cfg.fps
    for pitch, (onset, velocity) in active.items():
        if final_time > onset:
            events.append({"type": "note", "onset": onset, "offset": final_time, "pitch": pitch, "velocity": velocity})
    return events
