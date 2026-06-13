"""Experiment configuration for piano AMT training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass
class Config:
    # Dataset
    maestro_version: str = "v3.0.0"
    midi_min: int = 21                  # A0
    midi_max: int = 108                 # C8
    sample_rate: int = 16000
    segment_seconds: float = 12.0       # Long enough for note context, still batchable on A100
    hop_length: int = 160               # 10 ms at 16 kHz -> 100 fps
    n_fft: int = 2048
    n_mels: int = 229
    f_min: float = 27.5                 # A0
    f_max: float = 8000.0
    feature_type: str = "mel"           # Existing pre-sliced cache is mel.
    cqt_bins_per_octave: int = 48
    cqt_n_bins: int = 48 * 8            # Kept for old/fallback experiments only.

    # Model
    resnet_name: str = "resnet34.a1_in1k"
    pretrained: bool = True
    decoder_channels: int = 256
    decoder_type: str = "fpn"             # "fpn" uses encoder skip features; "legacy" uses deepest map only.
    model_type: str = "framewise"         # "framewise" or experimental "token_seq2seq".
    training_data_format: str = "framewise"  # "framewise" dense labels or "tokenwise" event-token labels.
    dropout: float = 0.10
    compile_model: bool = True

    # Experimental token seq2seq model/cache
    token_velocity_bins: int = 32
    token_max_seq_len: int = 4096
    token_encoder_dim: int = 256
    token_encoder_layers: int = 4
    token_decoder_layers: int = 4
    token_num_heads: int = 8
    token_ff_dim: int = 1024
    token_dropout: float = 0.10
    token_label_smoothing: float = 0.02
    token_frame_forward_weight: float = 0.25
    token_pedal_weight: float = 0.75
    token_eos_weight: float = 1.0
    delete_other_training_data_on_rebuild: bool = True

    # Sliced-cache backup/restore
    # New default: pre-slice songs in small durable shards and copy each shard to Drive
    # as soon as it finishes, so a dying Colab runtime does not lose all progress.
    use_sharded_preslicing: bool = True
    sliced_preslice_songs_per_shard: int = 80
    resume_sharded_preslicing: bool = True
    backup_preslice_shard_immediately: bool = True
    skip_final_backup_when_preslice_shards_exist: bool = True

    # Optional final backup sharding. This is mostly useful if you disable the
    # pre-slice build shards above and still want a resumable post-build backup.
    use_sharded_sliced_backups: bool = True
    sliced_backup_num_shards: int = 4
    sliced_backup_compression: str = "stored"    # "stored" is much faster on CPU; use "deflated" for smaller zips.
    sliced_backup_compresslevel: int = 0           # Only used for deflated backups.
    resume_sharded_sliced_backup: bool = True      # Reuse completed shard zips after Colab interruption.
    verify_sliced_zip_after_write: bool = False    # testzip() rereads every shard and is slow; atomic copies protect final files.
    verify_existing_preslice_shards: bool = False  # Avoid testzip() over Drive before restore; restore itself validates the zip.
    compact_sliced_tensors_before_save: bool = True  # Clone crops so torch.save does not serialize full-song tensor storage.
    delete_local_preslice_shard_after_backup: bool = True   # Free Colab disk after each build shard is safely on Drive.
    delete_local_preslice_zip_after_copy: bool = True        # Do not keep a duplicate local shard zip after Drive copy.
    use_zip_backed_sliced_dataset: bool = True               # Load chunks directly from Drive shard zips after local chunks are pruned.

    # Training
    FIND_OPTIMAL_SETTINGS: bool = False  # Legacy notebook flag; prefer the explicit flags below.
    use_found_training_settings: bool = False
    use_found_learning_rate: bool = False
    found_training_settings_path: str = ""
    found_learning_rate_path: str = ""
    lr_finder_start_lr: float = 1e-7
    lr_finder_end_lr: float = 1e-1
    lr_finder_num_iters: int = 100
    lr_finder_train_samples: int = 2048
    lr_finder_smoothing: float = 0.05
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 40
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True

    # Profiler / flame-chart debugging
    enable_flame_charts: bool = False
    flame_chart_batches: int = 8
    flame_chart_wait_batches: int = 1
    flame_chart_warmup_batches: int = 1
    flame_chart_output_dir: str = ""
    flame_chart_record_shapes: bool = True
    flame_chart_profile_memory: bool = True
    flame_chart_with_stack: bool = True

    # Loss weights
    onset_loss_weight: float = 2.0
    offset_loss_weight: float = 1.0
    frame_loss_weight: float = 1.0
    velocity_loss_weight: float = 0.5
    sustain_loss_weight: float = 0.5
    onset_pos_weight: float = 16.0
    offset_pos_weight: float = 12.0
    frame_pos_weight: float = 3.0
    sustain_pos_weight: float = 2.0

    # Label construction
    onset_width_frames: int = 2
    offset_width_frames: int = 2
    sustain_threshold: int = 64

    # Augmentation
    enable_audio_augment: bool = False
    enable_spec_augment: bool = True
    noise_probability: float = 0.35
    reverb_probability: float = 0.20
    gain_probability: float = 0.90
    eq_probability: float = 0.30
    spec_mask_probability: float = 0.50

    # Decoding
    onset_threshold: float = 0.50
    frame_threshold: float = 0.35
    offset_threshold: float = 0.35
    min_note_seconds: float = 0.03
    merge_onset_seconds: float = 0.03

    @property
    def n_pitches(self) -> int:
        return self.midi_max - self.midi_min + 1

    @property
    def fps(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def segment_samples(self) -> int:
        return int(round(self.segment_seconds * self.sample_rate))

    @property
    def label_frames(self) -> int:
        # Matches torch STFT center=True behavior reasonably well: 1 + n_samples // hop.
        return 1 + self.segment_samples // self.hop_length


def config_to_json(cfg: Config) -> str:
    return json.dumps(asdict(cfg), indent=2)


def default_run_name(cfg: Config) -> str:
    return (
        f"{cfg.resnet_name}_maestro_{cfg.maestro_version}_{cfg.feature_type}"
        f"_sr{cfg.sample_rate}_hop{cfg.hop_length}_seg{int(cfg.segment_seconds)}"
    )


CFG = Config()
