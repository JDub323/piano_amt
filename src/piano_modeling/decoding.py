"""Decode dense model probabilities into note and pedal events."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG, Config


def local_maxima_1d(x: np.ndarray, threshold: float) -> np.ndarray:
    if len(x) == 0:
        return np.zeros_like(x, dtype=bool)
    left = np.r_[x[0], x[:-1]]
    right = np.r_[x[1:], x[-1]]
    return (x >= threshold) & (x >= left) & (x >= right)


def decode_notes_from_probs(
    onset_prob: np.ndarray,
    frame_prob: np.ndarray,
    offset_prob: np.ndarray,
    velocity_prob: np.ndarray,
    cfg: Config = CFG,
) -> pd.DataFrame:
    """Decode note events from [T, 88] probability arrays."""
    T, P = onset_prob.shape
    fps = cfg.fps
    min_len = int(round(cfg.min_note_seconds * fps))
    merge_frames = int(round(cfg.merge_onset_seconds * fps))
    events = []
    for p in range(P):
        midi_pitch = p + cfg.midi_min
        onset_peaks = local_maxima_1d(onset_prob[:, p], cfg.onset_threshold)
        onset_idxs = np.flatnonzero(onset_peaks)
        last_onset = -10 ** 9
        for t0 in onset_idxs:
            if t0 - last_onset < merge_frames:
                continue
            last_onset = t0
            t = t0 + 1
            while t < T:
                off = offset_prob[t, p] >= cfg.offset_threshold
                inactive = frame_prob[t, p] < cfg.frame_threshold and t - t0 >= min_len
                if off or inactive:
                    break
                t += 1
            t1 = max(t, t0 + max(1, min_len))
            if t1 > T:
                t1 = T
            velocity = int(np.clip(round(velocity_prob[t0, p] * 127), 1, 127))
            events.append({
                "onset_sec": t0 / fps,
                "offset_sec": t1 / fps,
                "midi_pitch": midi_pitch,
                "velocity": velocity,
                "onset_prob": float(onset_prob[t0, p]),
            })
    events = pd.DataFrame(events)
    if len(events):
        events = events.sort_values(["onset_sec", "midi_pitch"]).reset_index(drop=True)
    return events


def decode_pedal_from_probs(sustain_prob: np.ndarray, cfg: Config = CFG) -> pd.DataFrame:
    active = sustain_prob >= 0.5
    events = []
    in_pedal = False
    start = 0
    for i, a in enumerate(active):
        if a and not in_pedal:
            start = i
            in_pedal = True
        elif not a and in_pedal:
            events.append({"onset_sec": start / cfg.fps, "offset_sec": i / cfg.fps, "cc": 64, "value": 127})
            in_pedal = False
    if in_pedal:
        events.append({"onset_sec": start / cfg.fps, "offset_sec": len(active) / cfg.fps, "cc": 64, "value": 127})
    return pd.DataFrame(events)
