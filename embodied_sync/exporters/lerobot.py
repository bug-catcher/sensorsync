"""LeRobot exporters: minimal v3.0 dataset writer + contract path.

:func:`export_lerobot_dataset` (D-0033) writes an aligned episode as a
minimal LeRobot **v3.0** dataset directory — ``meta/info.json``,
``meta/tasks.parquet``, ``meta/episodes/chunk-000/file-000.parquet``,
and one data parquet file. Numeric streams only: video/image payload-ref
streams are skipped with a warning (no transcoding). Frames where a
stream is missing export as NaN — LeRobot's row model has no missing
concept, and inventing values would defeat the sync-quality story.

The older ``save_lerobot_run`` / ``save_lerobot_episode`` functions are
the deterministic Milestone 5 CI contract path and stay unchanged.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

from embodied_sync.core.episode import AlignedRun
from embodied_sync.core.sample import Sample
from embodied_sync.datasets.io import _frame_to_record, _sample_to_record

__all__ = ["export_lerobot_dataset", "save_lerobot_episode", "save_lerobot_run"]

_FORMAT = "embodied_sync.lerobot.contract.v0"
_DATASET_NAME = "dataset.json"


def _numeric_vector(payload: Any) -> list[float] | None:
    """Payload as a float vector if it is one (scalars become length-1)."""
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


def export_lerobot_dataset(
    aligned: AlignedRun,
    out_dir: str | Path,
    *,
    target_rate_hz: float,
    task: str = "unknown",
    robot_type: str = "unknown",
) -> Path:
    """Write ``aligned`` as a minimal single-episode LeRobot v3.0 dataset.

    Requires ``pyarrow`` (installed with the ``lerobot`` extra). Fails if
    ``out_dir`` exists and is non-empty. Returns ``out_dir``.

    Exported timestamps are episode-relative float seconds on the target
    grid (``(target_time_ns - first_target_ns) / 1e9``) with ``fps`` set
    to ``target_rate_hz`` — the shape LeRobot's tolerance-based lookup
    expects. Non-numeric streams (video/image refs) are skipped with a
    warning; missing frames export as NaN.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if target_rate_hz <= 0:
        raise ValueError(f"target_rate_hz must be > 0, got {target_rate_hz!r}")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"refusing to export LeRobot dataset into non-empty directory: {out_dir}"
        )
    frames = aligned.frames
    if not frames:
        raise ValueError("cannot export an aligned episode with zero frames")

    stream_names = list(frames[0].samples.keys())
    dims: dict[str, int] = {}
    for name in stream_names:
        dim: int | None = None
        for frame in frames:
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
                f"export_lerobot_dataset: skipping non-numeric stream {name!r} "
                "(video/image payload refs are not transcoded)",
                stacklevel=2,
            )
        else:
            dims[name] = dim
    if not dims:
        raise ValueError("no numeric streams to export")

    n = len(frames)
    first_target = frames[0].target_time_ns
    columns: dict[str, Any] = {}
    for name, dim in dims.items():
        flat: list[float] = []
        for frame in frames:
            sample = frame.samples.get(name)
            vec = _numeric_vector(sample.payload) if sample is not None else None
            flat.extend(vec if vec is not None else [math.nan] * dim)
        values = pa.array(flat, type=pa.float32())
        columns[name] = pa.FixedSizeListArray.from_arrays(values, dim)
    columns["timestamp"] = pa.array(
        [(f.target_time_ns - first_target) / 1e9 for f in frames], type=pa.float32()
    )
    columns["frame_index"] = pa.array(range(n), type=pa.int64())
    columns["episode_index"] = pa.array([0] * n, type=pa.int64())
    columns["index"] = pa.array(range(n), type=pa.int64())
    columns["task_index"] = pa.array([0] * n, type=pa.int64())

    data_dir = out_dir / "data" / "chunk-000"
    episodes_dir = out_dir / "meta" / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), data_dir / "file-000.parquet")

    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                # Column name mirrors real LeRobot exports (a pandas index).
                "__index_level_0__": pa.array([task], type=pa.string()),
            }
        ),
        out_dir / "meta" / "tasks.parquet",
    )

    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0], type=pa.int64()),
                "data/chunk_index": pa.array([0], type=pa.int64()),
                "data/file_index": pa.array([0], type=pa.int64()),
                "dataset_from_index": pa.array([0], type=pa.int64()),
                "dataset_to_index": pa.array([n], type=pa.int64()),
                "tasks": pa.array([[task]], type=pa.list_(pa.string())),
                "length": pa.array([n], type=pa.int64()),
                "meta/episodes/chunk_index": pa.array([0], type=pa.int64()),
                "meta/episodes/file_index": pa.array([0], type=pa.int64()),
            }
        ),
        episodes_dir / "file-000.parquet",
    )

    fps: float | int = (
        int(target_rate_hz) if float(target_rate_hz).is_integer() else target_rate_hz
    )
    features: dict[str, Any] = {}
    for name, dim in dims.items():
        features[name] = {"dtype": "float32", "shape": [dim], "names": None, "fps": fps}
    for name, dtype in (
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ):
        features[name] = {"dtype": dtype, "shape": [1], "names": None, "fps": fps}
    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": 1,
        "total_frames": n,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (out_dir / "meta" / "info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    return out_dir


def save_lerobot_run(run: dict[str, list[Sample]], path: str | Path) -> Path:
    """Write a deterministic local LeRobot-style dataset directory.

    The contract path preserves stream names, episode boundaries (one run is
    one episode in v0), frame timing, state/action timing, payload refs, and
    quality flags without importing the optional ``lerobot`` package.
    """
    document: dict[str, Any] = {
        "format": _FORMAT,
        "type": "run",
        "episodes": [
            {
                "episode_index": 0,
                "streams": {
                    name: [_sample_to_record(sample) for sample in samples]
                    for name, samples in run.items()
                },
            }
        ],
    }
    return _write_dataset(document, path)


def save_lerobot_episode(episode: AlignedRun, path: str | Path) -> Path:
    """Write aligned frames to the deterministic LeRobot contract directory."""
    document: dict[str, Any] = {
        "format": _FORMAT,
        "type": "aligned_episode",
        "episodes": [
            {
                "episode_index": 0,
                "frames": [_frame_to_record(frame) for frame in episode.frames],
                "report": {
                    "missing_count": episode.report.missing_count,
                    "ground_truth_missing_count": (
                        episode.report.ground_truth_missing_count
                    ),
                    "median_skew_ns": episode.report.median_skew_ns,
                },
            }
        ],
    }
    return _write_dataset(document, path)


def _write_dataset(document: dict[str, Any], path: str | Path) -> Path:
    dataset_dir = Path(path)
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        raise FileExistsError(
            f"refusing to write LeRobot dataset into non-empty directory: {dataset_dir}"
        )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / _DATASET_NAME
    dataset_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, separators=(",", ": ")) + "\n",
        encoding="utf-8",
    )
    return dataset_dir
