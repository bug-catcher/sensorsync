"""Read-only dataset inspection for automatic import planning."""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_sync.ingest.model import DatasetProfile

__all__ = ["inspect_dataset"]

_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
_MAX_JSON_FILES = 512
_MAX_JSON_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _JsonRows:
    path: Path
    row_count: int
    common_fields: frozenset[str]
    timestamp_values: dict[str, list[float]]


def _numeric_scalar(value: object) -> float | None:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _known_signatures(path: Path) -> tuple[str, ...]:
    signatures: list[str] = []
    if path.is_file():
        if path.suffix.lower() == ".mcap":
            signatures.append("mcap")
        if path.suffix.lower() == ".xdf":
            signatures.append("xdf")
        if path.suffix.lower() == ".json":
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                document = None
            if isinstance(document, dict) and str(document.get("format", "")).startswith(
                "embodied_sync.umi"
            ):
                signatures.append("umi_contract")
        return tuple(signatures)

    manifest = path / "manifest.json"
    if manifest.is_file() and (path / "streams").is_dir():
        try:
            document = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            document = None
        if isinstance(document, dict) and isinstance(document.get("streams"), dict):
            signatures.append("canonical_run")

    lerobot_info = path / "meta" / "info.json"
    if lerobot_info.is_file():
        try:
            info = json.loads(lerobot_info.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            info = None
        if isinstance(info, dict) and str(info.get("codebase_version", "")).startswith("v3"):
            signatures.append("lerobot_v3")

    surg_episode = (path / "episode_meta.json").is_file()
    surg_root = any(path.glob("*_data/episodes/*/*/episode_meta.json"))
    if surg_episode or surg_root:
        signatures.append("surg_sync_v1")

    mcap_files = list(path.glob("*.mcap"))
    if len(mcap_files) == 1:
        signatures.append("mcap")
    return tuple(signatures)


def _json_row_groups(root: Path) -> dict[str, list[_JsonRows]]:
    groups: dict[str, list[_JsonRows]] = defaultdict(list)
    paths = sorted(root.rglob("*.json"))[:_MAX_JSON_FILES]
    for path in paths:
        try:
            if path.stat().st_size > _MAX_JSON_BYTES:
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(document, list) or not document:
            continue
        if not all(isinstance(row, dict) for row in document):
            continue
        common_fields = {str(name) for name in document[0]}
        for row in document[1:]:
            common_fields.intersection_update(str(name) for name in row)
        timestamp_values: dict[str, list[float]] = {}
        for field in common_fields:
            if "time" not in field.lower() and "stamp" not in field.lower():
                continue
            values = [_numeric_scalar(row.get(field)) for row in document]
            if not any(value is None for value in values):
                timestamp_values[field] = [
                    float(value) for value in values if value is not None
                ]
        groups[path.name].append(
            _JsonRows(
                path=path,
                row_count=len(document),
                common_fields=frozenset(common_fields),
                timestamp_values=timestamp_values,
            )
        )
    return groups


def _timestamp_profiles(
    episodes: list[_JsonRows], common_fields: set[str]
) -> list[dict[str, object]]:
    profiles: list[dict[str, object]] = []
    candidate_fields = sorted(
        name for name in common_fields if "time" in name.lower() or "stamp" in name.lower()
    )
    for field in candidate_fields:
        values_by_episode: list[list[float]] = []
        for episode in episodes:
            values = episode.timestamp_values.get(field)
            if values is None:
                break
            values_by_episode.append(values)
        if len(values_by_episode) != len(episodes):
            continue
        diffs = [
            values[index + 1] - values[index]
            for values in values_by_episode
            for index in range(len(values) - 1)
        ]
        if not diffs:
            continue
        positive = sum(delta > 0 for delta in diffs)
        profiles.append(
            {
                "field": field,
                "episode_spans": [values[-1] - values[0] for values in values_by_episode],
                "median_delta": statistics.median(diffs),
                "mean_delta": statistics.mean(diffs),
                "min_delta": min(diffs),
                "max_delta": max(diffs),
                "monotonic_fraction": positive / len(diffs),
            }
        )
    return profiles


def _probe_hdf5(
    root: Path, episode_ids: list[str], row_counts: dict[str, int]
) -> tuple[dict[str, object] | None, str | None]:
    files = sorted((*root.glob("*.h5"), *root.glob("*.hdf5")))
    if not files:
        return None, None
    try:
        import h5py  # noqa: PLC0415
    except ModuleNotFoundError:
        return None, "HDF5 files found but h5py is unavailable; array shapes were not inspected"

    best: dict[str, object] | None = None
    for path in files:
        try:
            with h5py.File(path, "r") as h5:
                matching_ids = [episode_id for episode_id in episode_ids if episode_id in h5]
                if not matching_ids:
                    continue
                name_sets = [
                    {
                        str(name)
                        for name, value in h5[episode_id].items()
                        if hasattr(value, "shape") and len(value.shape) >= 1
                    }
                    for episode_id in matching_ids
                ]
                camera_names = sorted(set.intersection(*name_sets)) if name_sets else []
                matches = 0
                checks = 0
                shapes: dict[str, list[int]] = {}
                for episode_id in matching_ids:
                    for camera_name in camera_names:
                        dataset = h5[episode_id][camera_name]
                        checks += 1
                        if int(dataset.shape[0]) == row_counts[episode_id]:
                            matches += 1
                        shapes.setdefault(camera_name, [int(value) for value in dataset.shape[1:]])
                candidate: dict[str, object] = {
                    "path": str(path.relative_to(root)),
                    "matching_episode_groups": len(matching_ids),
                    "episode_groups": len(episode_ids),
                    "camera_names": camera_names,
                    "count_matches": matches,
                    "count_checks": checks,
                    "count_match_ratio": matches / checks if checks else 0.0,
                    "sample_shapes": shapes,
                }
                previous_matches = best.get("count_matches") if best is not None else None
                if best is None or not isinstance(previous_matches, int) or matches > previous_matches:
                    best = candidate
        except OSError:
            continue
    return best, None


def _parse_rate(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            den = float(denominator)
            return float(numerator) / den if den else None
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def _ffprobe_video(path: Path) -> dict[str, object] | None:
    executable = shutil.which("ffprobe")
    if executable is None or path.stat().st_size == 0:
        return None
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        document = json.loads(result.stdout) if result.returncode == 0 else {}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    streams = document.get("streams") if isinstance(document, dict) else None
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        return None
    stream = streams[0]
    rate = _parse_rate(stream.get("avg_frame_rate"))
    raw_frames = stream.get("nb_frames")
    try:
        frame_count = (
            int(raw_frames)
            if isinstance(raw_frames, (str, int, float))
            and not isinstance(raw_frames, bool)
            and raw_frames != "N/A"
            else None
        )
    except (TypeError, ValueError):
        frame_count = None
    try:
        duration_s = float(stream["duration"]) if stream.get("duration") is not None else None
    except (TypeError, ValueError):
        duration_s = None
    return {
        "path": str(path),
        "rate_hz": rate,
        "frame_count": frame_count,
        "duration_s": duration_s,
    }


def _probe_videos(
    root: Path, episode_paths: list[Path], row_counts: dict[str, int]
) -> dict[str, object] | None:
    by_episode: dict[str, list[Path]] = {}
    for episode_path in episode_paths:
        videos = sorted(
            path
            for path in episode_path.rglob("*")
            if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES
        )
        if videos:
            by_episode[episode_path.name] = videos
    all_videos = [path for paths in by_episode.values() for path in paths]
    if not all_videos:
        return None

    sample_indices = sorted({0, len(all_videos) // 2, len(all_videos) - 1})
    observations = [
        observation
        for index in sample_indices
        if (observation := _ffprobe_video(all_videos[index])) is not None
    ]
    rates = [
        float(value)
        for item in observations
        if isinstance((value := item["rate_hz"]), (int, float))
        and not isinstance(value, bool)
    ]
    stable_rate = (
        statistics.median(rates)
        if rates and max(rates) - min(rates) <= max(0.01, statistics.median(rates) * 0.001)
        else None
    )
    frame_matches = 0
    frame_checks = 0
    for item in observations:
        frame_count = item["frame_count"]
        episode_id = Path(str(item["path"])).parents[1].name
        if isinstance(frame_count, int) and episode_id in row_counts:
            frame_checks += 1
            if frame_count == row_counts[episode_id]:
                frame_matches += 1

    first_episode = next(iter(by_episode.values()))
    first_relative = first_episode[0].relative_to(first_episode[0].parents[1])
    video_glob = str(first_relative.parent / f"*{first_relative.suffix.lower()}")
    return {
        "file_count": len(all_videos),
        "episodes_with_video": len(by_episode),
        "files_per_episode_min": min(len(paths) for paths in by_episode.values()),
        "files_per_episode_max": max(len(paths) for paths in by_episode.values()),
        "video_glob": video_glob,
        "sampled_observations": observations,
        "rate_hz": stable_rate,
        "frame_count_matches": frame_matches,
        "frame_count_checks": frame_checks,
        "frame_count_match_ratio": frame_matches / frame_checks if frame_checks else None,
    }


def _indexed_episode_facts(root: Path) -> tuple[dict[str, object] | None, list[str]]:
    warnings: list[str] = []
    groups = _json_row_groups(root)
    if not groups:
        return None, warnings
    _, episodes = max(groups.items(), key=lambda item: (len(item[1]), item[0]))
    episode_paths = [episode.path.parent for episode in episodes]
    common_parent = Path(os.path.commonpath([str(path) for path in episode_paths]))
    if not all(path.parent == common_parent for path in episode_paths):
        warnings.append("row JSON files are not direct children of one episode directory")
        return None, warnings

    row_file = episodes[0].path.name
    episode_glob = str(common_parent.relative_to(root) / "*")
    episode_ids = [episode.path.parent.name for episode in episodes]
    row_counts = {
        episode.path.parent.name: episode.row_count for episode in episodes
    }
    common_fields = set(episodes[0].common_fields)
    for episode in episodes[1:]:
        common_fields.intersection_update(episode.common_fields)

    timestamps = _timestamp_profiles(episodes, common_fields)
    hdf5, hdf5_warning = _probe_hdf5(root, episode_ids, row_counts)
    if hdf5_warning:
        warnings.append(hdf5_warning)
    videos = _probe_videos(root, episode_paths, row_counts)
    return {
        "episode_glob": episode_glob,
        "row_file": row_file,
        "episode_ids": episode_ids,
        "episode_count": len(episodes),
        "row_counts": row_counts,
        "total_rows": sum(row_counts.values()),
        "common_fields": sorted(common_fields),
        "timestamp_fields": timestamps,
        "hdf5": hdf5,
        "videos": videos,
    }, warnings


def inspect_dataset(path: str | Path) -> DatasetProfile:
    """Inspect a dataset without importing payload arrays or changing files."""

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"dataset path does not exist: {root}")
    facts: dict[str, Any] = {}
    warnings: list[str] = []
    if root.is_dir():
        indexed, indexed_warnings = _indexed_episode_facts(root)
        warnings.extend(indexed_warnings)
        if indexed is not None:
            facts["indexed_episode"] = indexed
    return DatasetProfile(
        root=str(root),
        path_kind="directory" if root.is_dir() else "file",
        signatures=_known_signatures(root),
        facts=facts,
        warnings=tuple(warnings),
    )
