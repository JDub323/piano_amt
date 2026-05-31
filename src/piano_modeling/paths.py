"""Path construction for Colab/Drive-backed experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ProjectPaths:
    project_root: Path
    datasets_root: Path
    maestro_meta_root: Path
    maestro_work_root: Path
    maestro_spec_root: Path
    maestro_midi_root: Path
    maestro_cache_manifest: Path
    sliced_zip_path: Path
    sliced_root: Path
    sliced_manifest: Path
    checkpoint_dir: Path
    export_dir: Path
    cache_dir: Path

    def mkdirs(self, include_local: bool = True) -> None:
        roots: Iterable[Path] = [
            self.project_root,
            self.datasets_root,
            self.maestro_meta_root,
            self.maestro_work_root,
            self.maestro_spec_root,
            self.maestro_midi_root,
            self.checkpoint_dir,
            self.export_dir,
            self.cache_dir,
        ]
        if include_local:
            roots = list(roots) + [self.sliced_root]
        for p in roots:
            p.mkdir(parents=True, exist_ok=True)

    @property
    def maestro_root(self) -> Path:
        # Backwards-compatible alias used by the original notebook.
        return self.maestro_meta_root


def make_colab_paths(
    project_root: str | Path = "/content/drive/MyDrive/piano_transcription_resnet",
    local_sliced_root: str | Path = "/content/local_maestro_sliced",
    local_work_root: str | Path = "/content/maestro_working_audio",
) -> ProjectPaths:
    project_root = Path(project_root)
    datasets_root = project_root / "datasets"
    maestro_meta_root = datasets_root / "maestro-v3.0.0_metadata"
    maestro_spec_root = datasets_root / "maestro-v3.0.0_spectrogram_tensors"
    maestro_midi_root = datasets_root / "maestro-v3.0.0_midi"
    sliced_root = Path(local_sliced_root)
    paths = ProjectPaths(
        project_root=project_root,
        datasets_root=datasets_root,
        maestro_meta_root=maestro_meta_root,
        maestro_work_root=Path(local_work_root),
        maestro_spec_root=maestro_spec_root,
        maestro_midi_root=maestro_midi_root,
        maestro_cache_manifest=maestro_spec_root / "maestro-v3.0.0_cached_manifest.csv",
        sliced_zip_path=datasets_root / "maestro-v3.0.0_sliced.zip",
        sliced_root=sliced_root,
        sliced_manifest=sliced_root / "sliced_manifest.csv",
        checkpoint_dir=project_root / "checkpoints",
        export_dir=project_root / "exports",
        cache_dir=project_root / "cache",
    )
    paths.mkdirs(include_local=False)
    return paths


def print_paths(paths: ProjectPaths) -> None:
    for name, value in paths.__dict__.items():
        print(f"{name}: {value}")
