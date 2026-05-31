"""Colab-friendly Gradio microphone streaming demo."""
from __future__ import annotations

import numpy as np
import pandas as pd

from piano_modeling.config import CFG, Config
from piano_modeling.models import PianoTranscriptionSystem

from .realtime import RealTimePianoTranscriber


class GradioRealtimeAdapter:
    def __init__(self, rt: RealTimePianoTranscriber):
        self.rt = rt
        self.last_len = 0
        self.last_sr = None
        self.events = pd.DataFrame(columns=["onset_sec", "offset_sec", "midi_pitch", "velocity"])

    def reset(self):
        self.rt.reset()
        self.last_len = 0
        self.last_sr = None
        self.events = pd.DataFrame(columns=["onset_sec", "offset_sec", "midi_pitch", "velocity"])
        return self.events

    def __call__(self, audio):
        if audio is None:
            return self.events.tail(50)
        sr, data = audio
        data = np.asarray(data)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 1.5:
            data = data / np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else data / np.max(np.abs(data))

        if self.last_sr == sr and len(data) >= self.last_len:
            chunk = data[self.last_len:]
            self.last_len = len(data)
        else:
            self.rt.reset()
            chunk = data
            self.last_len = len(data)
            self.last_sr = sr
        new_events = self.rt.push(chunk, sr)
        if len(new_events):
            self.events = pd.concat([self.events, new_events[["onset_sec", "offset_sec", "midi_pitch", "velocity"]]], ignore_index=True)
        return self.events.tail(50)


def launch_gradio_realtime(system: PianoTranscriptionSystem, cfg: Config = CFG):
    import gradio as gr

    adapter = GradioRealtimeAdapter(RealTimePianoTranscriber(system, cfg))
    with gr.Blocks() as demo:
        gr.Markdown("# Real-time Piano Transcription Demo")
        gr.Markdown("Speak/play piano into the microphone. The table shows recently emitted note events. For true low latency, use the local sounddevice function.")
        mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Microphone")
        out = gr.Dataframe(headers=["onset_sec", "offset_sec", "midi_pitch", "velocity"], label="Recent note events")
        reset_btn = gr.Button("Reset stream state")
        mic.stream(adapter, inputs=mic, outputs=out)
        reset_btn.click(adapter.reset, outputs=out)
    demo.launch(share=True, debug=False)
