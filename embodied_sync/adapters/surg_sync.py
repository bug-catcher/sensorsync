"""SurgSync adapter: native v1.0 reader + synthetic contract path.

SurgSync-shaped datasets bundle multi-modal surgical-robot signals —
endoscope video, external camera, kinematic state, force readings, and
workflow phase markers — with measured timestamp deltas to a stereo-left
master clock. The native :func:`load_surg_sync_dataset` reader loads the
open CC-BY-NC-4.0 SurgSync v1.0 Hugging Face release from a local path
and maps each topic timestamp as ``master_timestamp_ns + delta`` for
``acquisition_time_ns`` while keeping the snapped master grid in
``receive_time_ns`` (D-0035). The deterministic
:func:`load_surg_sync_run` JSON contract path remains dependency-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embodied_sync.core.sample import QUALITY_GAP_BEFORE, Modality, Sample

__all__ = ["load_surg_sync_dataset", "load_surg_sync_run"]

_FORMAT = "embodied_sync.surg_sync.contract.v0"
_PARTITIONS = ("online_data", "offline_data")
_VIDEO_DELTAS = {
    "video_raw.stereo_left": None,
    "video_raw.stereo_right": "delta_to_master.image_right_ns",
    "video_raw.side": "delta_to_master.image_side_ns",
}
_JAW_TOPICS = {
    "jaw_measured": "jaw.measured_position",
    "jaw_setpoint": "jaw.setpoint_position",
}


def _discover_episodes(root: Path, partition: str | None) -> list[Path]:
    if (root / "episode_meta.json").is_file():
        if partition is not None:
            meta = json.loads((root / "episode_meta.json").read_text(encoding="utf-8"))
            variant = str(meta.get("recorder_variant", ""))
            if variant != partition.removesuffix("_data"):
                raise ValueError(
                    f"episode {root} has partition {variant!r}, not {partition!r}"
                )
        return [root]
    partitions = _PARTITIONS if partition is None else (partition,)
    episodes: list[Path] = []
    for part in partitions:
        if part not in _PARTITIONS:
            raise ValueError(
                f"unknown SurgSync partition {part!r}; expected one of {_PARTITIONS}"
            )
        base = root / part / "episodes"
        if not base.is_dir():
            continue
        episodes.extend(
            sorted(p for p in base.glob("*/*") if (p / "episode_meta.json").is_file())
        )
    if not episodes:
        raise FileNotFoundError(f"no SurgSync episodes found under {root}")
    return episodes


def _to_builtin(value: object) -> object:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


def _payload_from_row(row: dict[str, Any], columns: list[str]) -> dict[str, object]:
    return {col: _to_builtin(row.get(col)) for col in columns}


def _topic_columns(rows_schema_names: list[str], topic: str) -> list[str]:
    if topic in _JAW_TOPICS:
        col = _JAW_TOPICS[topic]
        return [col] if col in rows_schema_names else []
    prefix = f"{topic}."
    return [
        name
        for name in rows_schema_names
        if name.startswith(prefix) and name not in {"frame_index", "master_timestamp_ns"}
    ]


def _delta_topic(delta_col: str, arm: str) -> str:
    prefix = f"delta_to_master.{arm}."
    topic = delta_col.removeprefix(prefix).removesuffix("_ns")
    return topic


def load_surg_sync_dataset(
    path: str | Path,
    *,
    partition: str | None = None,
    max_episodes: int | None = None,
) -> tuple[dict[str, list[Sample]], dict[str, Any]]:
    """Load a real SurgSync v1.0 dataset directory as a canonical run.

    ``pyarrow`` is imported lazily. Per-clip master timestamps are composed
    onto one monotonic run timeline by offsetting each imported episode to
    the previous episode's end. Within an episode, source acquisition time
    is exactly ``master_timestamp_ns + delta_to_master.<topic>_ns`` and
    receive time is the snapped master timestamp.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    root = Path(path)
    if not (root / "meta").is_dir() and not (root / "episode_meta.json").is_file():
        raise FileNotFoundError(f"not a SurgSync dataset (no meta/): {root}")

    episode_paths = _discover_episodes(root, partition)
    if max_episodes is not None:
        episode_paths = episode_paths[:max_episodes]
    if not episode_paths:
        raise ValueError(f"SurgSync dataset has no episodes selected: {root}")

    run: dict[str, list[Sample]] = {}
    seq: dict[str, int] = {}
    boundaries: list[dict[str, Any]] = []
    drop_ground_truth: list[dict[str, int | bool | str]] = []
    run_offset_ns = 0

    for episode_path in episode_paths:
        meta = json.loads((episode_path / "episode_meta.json").read_text(encoding="utf-8"))
        modalities_path = episode_path / "modalities.json"
        modalities = (
            json.loads(modalities_path.read_text(encoding="utf-8"))
            if modalities_path.is_file()
            else {}
        )
        timestamp_rows = pq.read_table(episode_path / "timestamp.parquet").to_pylist()
        if not timestamp_rows:
            continue
        task = str(meta.get("task", episode_path.parent.name))
        variant = str(meta.get("recorder_variant", episode_path.parents[2].name))
        master_t0_ns = int(meta.get("master_t0_ns", 0))
        episode_id = str(meta.get("episode_id", episode_path.name))
        start_time_ns = run_offset_ns
        last_master_ns = 0

        for row in timestamp_rows:
            frame_index = int(row["frame_index"])
            master_ns = run_offset_ns + int(row["master_timestamp_ns"])
            last_master_ns = max(last_master_ns, master_ns - run_offset_ns)
            drop_count = int(row.get("drop_count_since_prev", 0) or 0)
            contiguous = bool(row.get("is_contiguous_to_prev", True))
            if drop_count or not contiguous:
                drop_ground_truth.append(
                    {
                        "episode_id": episode_id,
                        "frame_index": frame_index,
                        "master_time_ns": master_ns,
                        "drop_count_since_prev": drop_count,
                        "is_contiguous_to_prev": contiguous,
                    }
                )

        video_raw = modalities.get("video_raw", {}) if isinstance(modalities, dict) else {}
        for stream_name, delta_col in _VIDEO_DELTAS.items():
            short_name = stream_name.split(".", 1)[1]
            spec = video_raw.get(short_name, {}) if isinstance(video_raw, dict) else {}
            if isinstance(spec, dict) and spec.get("present") is False:
                continue
            samples = run.setdefault(stream_name, [])
            seq.setdefault(stream_name, len(samples))
            rel_video = Path("video_raw") / f"{short_name}.mkv"
            for row in timestamp_rows:
                delta = 0 if delta_col is None else row.get(delta_col)
                if delta is None:
                    continue
                frame_index = int(row["frame_index"])
                master_ns = run_offset_ns + int(row["master_timestamp_ns"])
                delta_ns = int(delta)
                quality = (
                    frozenset({QUALITY_GAP_BEFORE})
                    if int(row.get("drop_count_since_prev", 0) or 0) > 0
                    else frozenset()
                )
                samples.append(
                    Sample(
                        stream_name=stream_name,
                        modality=Modality.CAMERA,
                        sequence_id=seq[stream_name],
                        acquisition_time_ns=master_ns + delta_ns,
                        receive_time_ns=master_ns,
                        source_clock_domain=f"surgsync.{variant}.{stream_name}",
                        payload={
                            "episode_id": episode_id,
                            "frame_index": frame_index,
                            "master_t0_ns": master_t0_ns,
                            "master_timestamp_ns": master_ns,
                            "delta_to_master_ns": delta_ns,
                        },
                        payload_ref=f"{episode_path.relative_to(root)}/{rel_video}#frame={frame_index}",
                        quality_flags=quality,
                    )
                )
                seq[stream_name] += 1

        for arm in ("ECM", "PSM1", "PSM2"):
            parquet_path = episode_path / f"{arm}.parquet"
            if not parquet_path.is_file():
                continue
            table = pq.read_table(parquet_path)
            schema_names = table.schema.names
            kinematic_rows = table.to_pylist()
            delta_cols = [
                name
                for name in timestamp_rows[0]
                if name.startswith(f"delta_to_master.{arm}.") and name.endswith("_ns")
            ]
            for delta_col in delta_cols:
                topic = _delta_topic(delta_col, arm)
                columns = _topic_columns(schema_names, topic)
                if not columns:
                    continue
                stream_name = f"{arm}.{topic}"
                samples = run.setdefault(stream_name, [])
                seq.setdefault(stream_name, len(samples))
                for ts_row, kin_row in zip(timestamp_rows, kinematic_rows):
                    delta = ts_row.get(delta_col)
                    if delta is None:
                        continue
                    frame_index = int(ts_row["frame_index"])
                    master_ns = run_offset_ns + int(ts_row["master_timestamp_ns"])
                    delta_ns = int(delta)
                    quality = (
                        frozenset({QUALITY_GAP_BEFORE})
                        if int(ts_row.get("drop_count_since_prev", 0) or 0) > 0
                        else frozenset()
                    )
                    payload = _payload_from_row(kin_row, columns)
                    payload.update(
                        {
                            "episode_id": episode_id,
                            "frame_index": frame_index,
                            "master_t0_ns": master_t0_ns,
                            "master_timestamp_ns": master_ns,
                            "delta_to_master_ns": delta_ns,
                        }
                    )
                    samples.append(
                        Sample(
                            stream_name=stream_name,
                            modality=Modality.ROBOT_STATE,
                            sequence_id=seq[stream_name],
                            acquisition_time_ns=master_ns + delta_ns,
                            receive_time_ns=master_ns,
                            source_clock_domain=f"surgsync.{variant}.{arm}.{topic}",
                            payload=payload,
                            quality_flags=quality,
                        )
                    )
                    seq[stream_name] += 1

        annotation_path = episode_path / "annotation.parquet"
        if annotation_path.is_file():
            annotation_rows = pq.read_table(annotation_path).to_pylist()
            stream_name = "annotation"
            samples = run.setdefault(stream_name, [])
            seq.setdefault(stream_name, len(samples))
            for row in annotation_rows:
                frame_index = int(row["frame_index"])
                master_ns = run_offset_ns + int(row["master_timestamp_ns"])
                payload = {
                    name: _to_builtin(value)
                    for name, value in row.items()
                    if name not in {"frame_index", "master_timestamp_ns"}
                }
                payload.update(
                    {
                        "episode_id": episode_id,
                        "frame_index": frame_index,
                        "master_t0_ns": master_t0_ns,
                    }
                )
                samples.append(
                    Sample(
                        stream_name=stream_name,
                        modality=Modality.EVENT,
                        sequence_id=seq[stream_name],
                        acquisition_time_ns=master_ns,
                        receive_time_ns=master_ns,
                        source_clock_domain=f"surgsync.{variant}.master",
                        payload=payload,
                    )
                )
                seq[stream_name] += 1

        boundaries.append(
            {
                "episode_id": episode_id,
                "partition": variant,
                "task": task,
                "clip": episode_path.name,
                "start_time_ns": start_time_ns,
                "length": int(meta.get("length_frames", len(timestamp_rows))),
                "master_t0_ns": master_t0_ns,
                "drop_count_total": sum(
                    int(row.get("drop_count_since_prev", 0) or 0)
                    for row in timestamp_rows
                ),
                "non_contiguous_count": sum(
                    1 for row in timestamp_rows if not bool(row.get("is_contiguous_to_prev", True))
                ),
            }
        )
        run_offset_ns += last_master_ns + 100_000_000

    dataset_info: dict[str, Any] = {
        "source_path": str(root),
        "format": "surgsync.v1.0",
        "license": "CC-BY-NC-4.0",
        "imported_episodes": len(boundaries),
        "episodes": boundaries,
        "drop_ground_truth": drop_ground_truth,
        "timestamp_mapping": {
            "acquisition_time_ns": "master_timestamp_ns + delta_to_master.<topic>_ns",
            "receive_time_ns": "master_timestamp_ns",
        },
    }
    return run, dataset_info


def load_surg_sync_run(path: str | Path) -> dict[str, list[Sample]]:
    """Load a SurgSync-style contract JSON file into a run dict.

    The contract document shape (v0)::

        {
          "format": "embodied_sync.surg_sync.contract.v0",
          "streams": {
            "endoscope": {
              "modality": "camera",
              "source_clock_domain": "endoscope_hw",
              "samples": [
                 {"sequence_id": 0, "acquisition_time_ns": ...,
                  "receive_time_ns": ..., "payload": {...},
                  "quality_flags": []}
              ]
            },
            ...
          }
        }

    Timestamps are preserved verbatim (integer ns, D-0002). Modality
    strings are validated against :class:`Modality`. Unknown clock
    domains stay as free strings on ``Sample.source_clock_domain`` —
    the caller can lift them into typed
    :class:`~embodied_sync.time.ClockDomain` values via
    :func:`~embodied_sync.time.resolve_clock_domain`.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != _FORMAT:
        raise ValueError(
            f"unsupported SurgSync contract format {document.get('format')!r} "
            f"in {path}"
        )
    streams = document.get("streams")
    if not isinstance(streams, dict):
        raise ValueError(f"invalid SurgSync run in {path}: missing streams")
    run: dict[str, list[Sample]] = {}
    for name, stream in streams.items():
        modality = Modality(stream.get("modality", Modality.OTHER.value))
        clock_domain = str(stream.get("source_clock_domain", "unknown"))
        records = stream.get("samples")
        if not isinstance(records, list):
            raise ValueError(f"invalid SurgSync stream {name!r}: missing samples")
        samples: list[Sample] = []
        for index, record in enumerate(records):
            samples.append(
                Sample(
                    stream_name=name,
                    modality=modality,
                    sequence_id=int(record.get("sequence_id", index)),
                    acquisition_time_ns=int(record["acquisition_time_ns"]),
                    receive_time_ns=int(
                        record.get("receive_time_ns", record["acquisition_time_ns"])
                    ),
                    source_clock_domain=clock_domain,
                    payload=record.get("payload"),
                    payload_ref=record.get("payload_ref"),
                    quality_flags=frozenset(record.get("quality_flags", ())),
                )
            )
        run[name] = samples
    return run
