# piano_amt

Retry of my automatic music transcription project, but I will fully utilize AI to speed up banal functions, so I can focus on the cool stuff. This is set up for Colab since I fried my laptop when I tried to run last project locally.

## What it does

This repo trains and evaluates a piano automatic music transcription model. Audio is converted to spectrogram-like features, the model predicts dense piano-roll heads for onset, offset, frame, velocity, and sustain pedal, and decoding/export utilities turn those probabilities into note events or MIDI.

## Install

```bash
pip install -e ".[train,dev]"
```

For live demos, also install:

```bash
pip install -e ".[live]"
```

## Smoke tests

The test suite uses synthetic data and a fake tiny encoder where possible, so it can catch shape, decoding, training-loop, and TensorBoard logging regressions without downloading MAESTRO or a large checkpoint.

```bash
pytest -q
```

## Training diagnostics

`run_training` now writes TensorBoard logs by default under:

```text
<checkpoint_dir>/<run_name>/tensorboard
```

Useful TensorBoard panels to watch first:

- `epoch/train/*` and `epoch/validation/*`: loss components and quick dense-label F1.
- `debug_probabilities/*`: whether heads are saturated near 0 or 1.
- `debug_activation_rate/*` vs `debug_target_density/*`: whether sparse heads such as onset/offset are firing at a plausible rate.
- `debug_pr_curve/*`: precision/recall behavior by head.
- `debug_global_norm/*`, `debug_param_norm/*`, and `debug_grad_norm/*`: dead/exploding parameter or gradient signals.
- `debug_predictions/*` and `debug_targets/*`: visual piano-roll sanity checks.

You can change the cadence or turn logging off:

```python
run_training(
    system,
    cfg,
    paths,
    sliced_meta,
    enable_tensorboard=True,
    tensorboard_log_every=100,
    tensorboard_log_graph=True,
)
```

## First-class evaluation

Use the CLI to run formal note-level evaluation from a checkpoint and a metadata CSV with `audio_path`, `midi_path`, and `split` columns:

```bash
piano-evaluate \
  --metadata-csv /path/to/metadata.csv \
  --checkpoint /path/to/best.pt \
  --output-dir /path/to/eval_reports \
  --split test \
  --n-tracks 20 \
  --device cuda
```

This writes a per-track CSV and a JSON summary containing note onset F1, note onset+offset F1, velocity F1, pedal F1, and note counts.

You can also opt into formal evaluation during training whenever a new best quick-validation checkpoint is found:

```python
run_training(
    system,
    cfg,
    paths,
    sliced_meta,
    formal_eval_meta=maestro_meta,
    formal_eval_split="validation",
    formal_eval_n_tracks=5,
)
```

Keep this small during training; full-track inference is much slower than the dense-label validation pass.

## Model decoder

The default model decoder is now `decoder_type="fpn"`, a feature-pyramid skip decoder that fuses all `timm` encoder stages before the output heads. This should preserve more time/frequency detail than the original deepest-feature-only decoder, which is especially important for precise onset timing.

For old experiments or checkpoint comparisons, use:

```python
cfg.decoder_type = "legacy"
```
