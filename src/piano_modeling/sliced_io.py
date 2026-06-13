"""I/O helpers for local or zip-backed sliced-cache chunks."""
from __future__ import annotations

import io
from pathlib import Path
import zipfile

import pandas as pd
import torch


def row_value(row, key: str):
    try:
        value = row[key]
    except Exception:
        if isinstance(row, dict):
            value = row.get(key)
        else:
            value = getattr(row, key, None)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def row_has_zip_payload(row) -> bool:
    zip_path = row_value(row, "zip_path")
    zip_member = row_value(row, "zip_member")
    return bool(zip_path) and bool(zip_member) and Path(str(zip_path)).exists()


def load_sliced_payload_from_row(row):
    """Load a sliced chunk from either a local .pt file or a zip-backed manifest row."""
    chunk_path = row_value(row, "chunk_path")
    if chunk_path and Path(str(chunk_path)).exists():
        return torch.load(str(chunk_path), map_location="cpu")

    zip_path = row_value(row, "zip_path")
    zip_member = row_value(row, "zip_member")
    if zip_path and zip_member and Path(str(zip_path)).exists():
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            with zf.open(str(zip_member), "r") as fh:
                return torch.load(io.BytesIO(fh.read()), map_location="cpu")

    raise FileNotFoundError(
        f"Missing sliced chunk. chunk_path={chunk_path!r}, zip_path={zip_path!r}, zip_member={zip_member!r}"
    )
