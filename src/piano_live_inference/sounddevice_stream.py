"""Local microphone streaming using sounddevice.

This is intended for a local Python runtime. Colab typically cannot access low-level host audio devices this way.
"""
from __future__ import annotations

import time
from typing import Optional

from piano_modeling.config import CFG, Config
from piano_modeling.models import PianoTranscriptionSystem

from .realtime import RealTimePianoTranscriber


def run_sounddevice_realtime(system: PianoTranscriptionSystem, cfg: Config = CFG, seconds: Optional[float] = None):
    import sounddevice as sd

    rt = RealTimePianoTranscriber(system, cfg)
    start_wall = time.time()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        events = rt.push(indata.copy(), cfg.sample_rate)
        if len(events):
            print(events[["onset_sec", "offset_sec", "midi_pitch", "velocity"]].tail(10).to_string(index=False))

    print("Starting stream. Press Ctrl+C to stop.")
    with sd.InputStream(channels=1, samplerate=cfg.sample_rate, blocksize=int(cfg.sample_rate * 0.05), callback=callback):
        try:
            while True:
                time.sleep(0.1)
                if seconds is not None and time.time() - start_wall >= seconds:
                    break
        except KeyboardInterrupt:
            pass
