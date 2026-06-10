import numpy as np

from piano_modeling.config import Config
from piano_modeling.decoding import decode_notes_from_probs, decode_pedal_from_probs


def test_decode_single_note_from_synthetic_probabilities():
    cfg = Config(
        midi_min=60,
        midi_max=63,
        sample_rate=100,
        hop_length=10,
        onset_threshold=0.5,
        frame_threshold=0.35,
        offset_threshold=0.35,
        min_note_seconds=0.1,
        merge_onset_seconds=0.0,
    )
    T, P = 12, cfg.n_pitches
    onset = np.zeros((T, P), dtype=np.float32)
    frame = np.zeros((T, P), dtype=np.float32)
    offset = np.zeros((T, P), dtype=np.float32)
    velocity = np.zeros((T, P), dtype=np.float32)

    onset[2, 1] = 0.95
    frame[2:7, 1] = 0.9
    offset[7, 1] = 0.8
    velocity[2, 1] = 0.5

    events = decode_notes_from_probs(onset, frame, offset, velocity, cfg)

    assert len(events) == 1
    event = events.iloc[0]
    assert event["midi_pitch"] == 61
    assert event["onset_sec"] == 0.2
    assert event["offset_sec"] == 0.7
    assert 60 <= event["velocity"] <= 65


def test_decode_pedal_segments():
    cfg = Config(sample_rate=100, hop_length=10)
    sustain = np.array([0.0, 0.7, 0.8, 0.2, 0.1, 0.9], dtype=np.float32)
    events = decode_pedal_from_probs(sustain, cfg)

    assert len(events) == 2
    assert events.iloc[0]["onset_sec"] == 0.1
    assert events.iloc[0]["offset_sec"] == 0.3
    assert events.iloc[1]["onset_sec"] == 0.5
