"""UMI / diffusion-policy Zarr replay-buffer exporter.

The native UMI replay buffer is a downstream training format: samples are
stored by row index at fixed ``dt`` with no per-sensor timestamp arrays. This
module therefore exports aligned episodes into that shape rather than treating
Zarr buffers as sync-validation inputs (D-0036).
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from embodied_sync.core.episode import AlignedRun

__all__ = ["export_umi_zarr"]


def _numeric_vector(payload: Any) -> list[float] | None:
    """Payload as a float vector if it is one; scalars become length-1."""
    if isinstance(payload, bool):
        return None
    if isinstance(payload, (int, float)):
        return [float(payload)]
    if isinstance(payload, (list, tuple)):
        out: list[float] = []
        for v in payload:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            out.append(float(v))
        return out
    return None


def _stream_dims(aligned: AlignedRun) -> dict[str, int]:
    dims: dict[str, int] = {}
    if not aligned.frames:
        return dims
    for name in aligned.frames[0].samples:
        dim: int | None = None
        for frame in aligned.frames:
            sample = frame.samples.get(name)
            if sample is None:
                continue
            vec = _numeric_vector(sample.payload)
            if vec is None:
                dim = None
                break
            if dim is None:
                dim = len(vec)
            elif dim != len(vec):
                dim = None
                break
        if dim is None or dim == 0:
            warnings.warn(
                f"export_umi_zarr: skipping non-numeric stream {name!r} "
                "(video/image payload refs are not transcoded)",
                stacklevel=2,
            )
        else:
            dims[name] = dim
    return dims


def _safe_zarr_key(name: str, used: set[str]) -> str:
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip("/")) or "stream"
    candidate = key
    suffix = 1
    while candidate in used:
        suffix += 1
        candidate = f"{key}_{suffix}"
    used.add(candidate)
    return candidate


def _write_group(path: Path, attrs: dict[str, Any] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text('{"zarr_format":2}\n', encoding="utf-8")
    (path / ".zattrs").write_text(
        json.dumps(attrs or {}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_array(path: Path, array: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(array)
    chunks = list(contiguous.shape)
    metadata = {
        "zarr_format": 2,
        "shape": list(contiguous.shape),
        "chunks": chunks,
        "dtype": contiguous.dtype.str,
        "compressor": None,
        "fill_value": None,
        "order": "C",
        "filters": None,
    }
    (path / ".zarray").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (path / ".zattrs").write_text("{}\n", encoding="utf-8")
    chunk_name = ".".join("0" for _ in contiguous.shape)
    (path / chunk_name).write_bytes(contiguous.tobytes(order="C"))


def export_umi_zarr(
    aligned: AlignedRun,
    out_dir: str | Path,
    *,
    target_rate_hz: float,
) -> Path:
    """Write ``aligned`` as a UMI-style Zarr replay buffer.

    The layout mirrors diffusion-policy replay buffers:
    ``data/<key>`` arrays hold numeric streams as ``N x C`` float32 and
    ``meta/episode_ends`` stores int64 cumulative episode lengths. Missing
    aligned frames export as NaN. Non-numeric streams are skipped with a
    warning.
    """
    import zarr  # noqa: F401, PLC0415

    if target_rate_hz <= 0:
        raise ValueError(f"target_rate_hz must be > 0, got {target_rate_hz!r}")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"refusing to export UMI Zarr buffer into non-empty directory: {out}"
        )
    if not aligned.frames:
        raise ValueError("cannot export an aligned episode with zero frames")
    dims = _stream_dims(aligned)
    if not dims:
        raise ValueError("no numeric streams to export")

    key_map: dict[str, str] = {}
    used_keys: set[str] = set()
    arrays: dict[str, np.ndarray] = {}
    n = len(aligned.frames)
    for name, dim in dims.items():
        key = _safe_zarr_key(name, used_keys)
        key_map[key] = name
        values = np.empty((n, dim), dtype=np.float32)
        for i, frame in enumerate(aligned.frames):
            sample = frame.samples.get(name)
            vec = _numeric_vector(sample.payload) if sample is not None else None
            values[i, :] = vec if vec is not None else [math.nan] * dim
        arrays[key] = values

    _write_group(
        out,
        {
            "format": "embodied_sync.umi_zarr.v0",
            "target_rate_hz": float(target_rate_hz),
            "stream_names": key_map,
        },
    )
    _write_group(out / "data")
    _write_group(out / "meta")
    for key, array in arrays.items():
        _write_array(out / "data" / key, array)
    _write_array(out / "meta" / "episode_ends", np.array([n], dtype=np.int64))
    return out
