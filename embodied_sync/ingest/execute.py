"""Deterministic execution of reviewed import plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embodied_sync.core.sample import Modality, Sample
from embodied_sync.ingest.model import ImportPlan

__all__ = ["execute_import_plan", "plan_source_rate_hz"]

_UNIT_NS = {
    "seconds": 1_000_000_000.0,
    "milliseconds": 1_000_000.0,
    "microseconds": 1_000.0,
    "nanoseconds": 1.0,
}


def _dict(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"import-plan parameter {name!r} must be an object")
    return {str(key): item for key, item in value.items()}


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"import-plan parameter {name!r} must be > 0")
    return float(value)


def _relative_path(root: Path, value: object, *, name: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"import-plan parameter {name!r} must stay within the dataset root")
    return root / relative


def _relative_glob(value: object, *, name: str) -> str:
    pattern = str(value)
    path = Path(pattern)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"import-plan parameter {name!r} must stay within the dataset root")
    return pattern


def _numeric_scalar(value: object) -> float | None:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _to_builtin(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def plan_source_rate_hz(plan: ImportPlan) -> float | None:
    """Return a plan's native fixed rate when it declares one."""

    if plan.executor == "indexed_episode":
        clock = plan.parameters.get("clock")
        if isinstance(clock, dict):
            value = clock.get("rate_hz")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    value = plan.parameters.get("source_rate_hz")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _clock_times(
    rows: list[dict[str, Any]], clock: dict[str, Any], run_offset_ns: int
) -> tuple[list[int], list[float | None], int]:
    rate_hz = _positive_float(clock.get("rate_hz"), name="clock.rate_hz")
    period_ns = round(1_000_000_000 / rate_hz)
    strategy = str(clock.get("strategy"))
    if strategy == "row_index":
        return (
            [run_offset_ns + index * period_ns for index in range(len(rows))],
            [None] * len(rows),
            period_ns,
        )
    if strategy != "timestamp_field":
        raise ValueError(f"unknown indexed-episode clock strategy {strategy!r}")

    field = str(clock.get("field"))
    unit = str(clock.get("unit"))
    if unit not in _UNIT_NS:
        raise ValueError(f"unknown timestamp unit {unit!r}; known: {sorted(_UNIT_NS)}")
    source_times = [_numeric_scalar(row.get(field)) for row in rows]
    if any(value is None for value in source_times):
        raise ValueError(f"timestamp field {field!r} is absent or non-numeric in a row")
    numeric_times = [float(value) for value in source_times if value is not None]
    start = numeric_times[0]
    acquisition = [
        run_offset_ns + round((value - start) * _UNIT_NS[unit]) for value in numeric_times
    ]
    if any(right < left for left, right in zip(acquisition, acquisition[1:])):
        raise ValueError(f"timestamp field {field!r} is non-monotonic")
    return acquisition, source_times, period_ns


def _indexed_episode_import(
    root: Path, plan: ImportPlan, *, max_episodes: int | None
) -> tuple[dict[str, list[Sample]], dict[str, Any]]:
    parameters = plan.parameters
    episode_glob = _relative_glob(parameters.get("episode_glob"), name="episode_glob")
    row_file = str(parameters.get("row_file"))
    if Path(row_file).name != row_file:
        raise ValueError("import-plan row_file must be a filename")
    episode_paths = sorted(
        path for path in root.glob(episode_glob) if (path / row_file).is_file()
    )
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be > 0")
        episode_paths = episode_paths[:max_episodes]
    if not episode_paths:
        raise FileNotFoundError(
            f"no episodes matching {episode_glob!r} with {row_file!r} under {root}"
        )

    state_streams_raw = _dict(parameters.get("state_streams"), name="state_streams")
    state_streams = {
        name: Modality(str(modality)) for name, modality in state_streams_raw.items()
    }
    clock = _dict(parameters.get("clock"), name="clock")
    camera = _dict(parameters.get("camera", {"source": "none"}), name="camera")
    camera_source = str(camera.get("source", "none"))
    if camera_source not in {"hdf5", "video", "none"}:
        raise ValueError(f"unknown camera source {camera_source!r}")
    clock_domain = str(parameters.get("source_clock_domain", "inferred.indexed_episode"))
    source_time_field_raw = parameters.get("source_time_field")
    source_time_field = (
        str(source_time_field_raw) if source_time_field_raw is not None else None
    )

    hdf5_path: Path | None = None
    hdf5_camera_names: list[str] = []
    if camera_source == "hdf5":
        hdf5_path = _relative_path(root, camera.get("path"), name="camera.path")
        if not hdf5_path.is_file():
            raise FileNotFoundError(f"planned HDF5 camera source is missing: {hdf5_path}")
        raw_names = camera.get("camera_names")
        if not isinstance(raw_names, list) or not all(isinstance(name, str) for name in raw_names):
            raise ValueError("import-plan camera.camera_names must be an array of strings")
        hdf5_camera_names = list(raw_names)

    video_glob = (
        _relative_glob(camera.get("glob"), name="camera.glob")
        if camera_source == "video"
        else ""
    )
    run: dict[str, list[Sample]] = {}
    sequence_ids: dict[str, int] = {}
    boundaries: list[dict[str, object]] = []
    run_offset_ns = 0

    for episode_path in episode_paths:
        document = json.loads((episode_path / row_file).read_text(encoding="utf-8"))
        if not isinstance(document, list) or not all(isinstance(row, dict) for row in document):
            raise ValueError(f"{episode_path / row_file} must contain an array of objects")
        rows: list[dict[str, Any]] = document
        if not rows:
            continue
        acquisition_times, clock_source_times, period_ns = _clock_times(
            rows, clock, run_offset_ns
        )
        source_times = (
            [_numeric_scalar(row.get(source_time_field)) for row in rows]
            if source_time_field is not None
            else clock_source_times
        )

        for stream_name, modality in state_streams.items():
            samples = run.setdefault(stream_name, [])
            sequence_ids.setdefault(stream_name, len(samples))
            for frame_index, row in enumerate(rows):
                if stream_name not in row:
                    continue
                samples.append(
                    Sample(
                        stream_name=stream_name,
                        modality=modality,
                        sequence_id=sequence_ids[stream_name],
                        acquisition_time_ns=acquisition_times[frame_index],
                        receive_time_ns=acquisition_times[frame_index],
                        source_clock_domain=clock_domain,
                        payload=_to_builtin(row[stream_name]),
                        payload_ref=(
                            f"{(episode_path / row_file).relative_to(root)}#row={frame_index}"
                        ),
                    )
                )
                sequence_ids[stream_name] += 1

        if camera_source == "hdf5":
            assert hdf5_path is not None
            camera_sources = [
                (
                    camera_name,
                    f"{hdf5_path.relative_to(root)}:/"
                    f"{episode_path.name}/{camera_name}[{{frame_index}}]",
                )
                for camera_name in hdf5_camera_names
            ]
        elif camera_source == "video":
            camera_sources = [
                (
                    video_path.stem,
                    f"{video_path.relative_to(root)}#frame={{frame_index}}",
                )
                for video_path in sorted(episode_path.glob(video_glob))
            ]
        else:
            camera_sources = []

        for camera_name, reference in camera_sources:
            stream_name = f"camera.{camera_name}"
            samples = run.setdefault(stream_name, [])
            sequence_ids.setdefault(stream_name, len(samples))
            for frame_index, acquisition_time_ns in enumerate(acquisition_times):
                payload: dict[str, object] = {
                    "episode_id": episode_path.name,
                    "frame_index": frame_index,
                }
                if source_times[frame_index] is not None:
                    payload["source_time"] = source_times[frame_index]
                samples.append(
                    Sample(
                        stream_name=stream_name,
                        modality=Modality.CAMERA,
                        sequence_id=sequence_ids[stream_name],
                        acquisition_time_ns=acquisition_time_ns,
                        receive_time_ns=acquisition_time_ns,
                        source_clock_domain=clock_domain,
                        payload=payload,
                        payload_ref=reference.format(frame_index=frame_index),
                    )
                )
                sequence_ids[stream_name] += 1

        boundaries.append(
            {
                "episode_id": episode_path.name,
                "start_time_ns": run_offset_ns,
                "length": len(rows),
                "source_start_time": source_times[0],
                "source_end_time": source_times[-1],
                "camera_count": len(camera_sources),
            }
        )
        run_offset_ns = acquisition_times[-1] + period_ns

    if not run:
        raise ValueError("import plan produced no streams")
    info: dict[str, Any] = {
        "source_path": str(root),
        "format": "inferred.indexed_episode.v0",
        "executor": plan.executor,
        "confidence": plan.confidence,
        "source_rate_hz": plan_source_rate_hz(plan),
        "imported_episodes": len(boundaries),
        "episodes": boundaries,
        "clock": clock,
        "camera": camera,
        "state_streams": sorted(state_streams),
    }
    return run, info


def _specialized_import(
    root: Path, plan: ImportPlan, *, max_episodes: int | None
) -> tuple[dict[str, list[Sample]], dict[str, Any]]:
    if plan.executor == "canonical_run":
        from embodied_sync.datasets.io import load_run

        return load_run(root), {"source_path": str(root), "format": "canonical_run.v0"}
    if plan.executor == "lerobot_v3":
        from embodied_sync.adapters.lerobot import load_lerobot_dataset

        return load_lerobot_dataset(root, max_episodes=max_episodes)
    if plan.executor == "surg_sync_v1":
        from embodied_sync.adapters.surg_sync import load_surg_sync_dataset

        return load_surg_sync_dataset(root, max_episodes=max_episodes)
    if plan.executor == "mcap":
        from embodied_sync.adapters.mcap import load_mcap_run

        return load_mcap_run(root), {"source_path": str(root), "format": "mcap"}
    if plan.executor == "xdf":
        from embodied_sync.adapters.lsl import load_xdf_file

        return load_xdf_file(root)
    if plan.executor == "umi_contract":
        from embodied_sync.adapters.umi import load_umi_replay_buffer

        return load_umi_replay_buffer(root), {
            "source_path": str(root),
            "format": "umi_contract",
        }
    raise ValueError(f"unknown import-plan executor {plan.executor!r}")


def execute_import_plan(
    path: str | Path,
    plan: ImportPlan,
    *,
    max_episodes: int | None = None,
) -> tuple[dict[str, list[Sample]], dict[str, Any]]:
    """Execute a plan through a registered importer; never evaluate generated code."""

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"dataset path does not exist: {root}")
    if plan.executor == "indexed_episode":
        if not root.is_dir():
            raise ValueError("indexed_episode executor requires a directory")
        return _indexed_episode_import(root, plan, max_episodes=max_episodes)
    return _specialized_import(root, plan, max_episodes=max_episodes)
