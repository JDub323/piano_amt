"""Small diagnostics for checking expected Colab/Drive artifacts."""
from __future__ import annotations

from pathlib import Path

from .paths import ProjectPaths


def check_dataset_artifacts(paths: ProjectPaths) -> dict:
    checks = {
        "project_root": paths.project_root,
        "datasets_root": paths.datasets_root,
        "maestro_meta_root": paths.maestro_meta_root,
        "maestro_spec_root": paths.maestro_spec_root,
        "maestro_midi_root": paths.maestro_midi_root,
        "maestro_cache_manifest": paths.maestro_cache_manifest,
        "sliced_zip_path": paths.sliced_zip_path,
        "sliced_root": paths.sliced_root,
        "sliced_manifest": paths.sliced_manifest,
        "checkpoint_dir": paths.checkpoint_dir,
        "export_dir": paths.export_dir,
    }
    out = {}
    for name, p in checks.items():
        p = Path(p)
        exists = p.exists()
        out[name] = exists
        if p.is_dir():
            count = sum(1 for _ in p.rglob("*"))
            print(f"{name}: {p} exists=True files/dirs={count}")
        else:
            size = p.stat().st_size if exists else 0
            print(f"{name}: {p} exists={exists} size={size}")
    return out
