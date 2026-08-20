"""Generate and confidence-gate deterministic dataset import plans."""

from __future__ import annotations

import math
import statistics
from typing import Any

from embodied_sync.core.sample import Modality
from embodied_sync.ingest.model import DatasetProfile, Evidence, ImportPlan, InferenceResult

__all__ = ["infer_import"]

_UNIT_NS = {
    "seconds": 1_000_000_000.0,
    "milliseconds": 1_000_000.0,
    "microseconds": 1_000.0,
    "nanoseconds": 1.0,
}
_COMMON_RATES_HZ = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 24.0, 25.0, 30.0, 50.0, 60.0, 100.0, 200.0, 250.0)


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _infer_modality(field: str) -> Modality:
    name = field.lower()
    if "success_rate" in name:
        return Modality.OTHER
    if any(token in name for token in ("action", "command", "setpoint")):
        return Modality.ACTION
    if any(token in name for token in ("force", "wrench", "torque", "tau", "tactile")):
        return Modality.TACTILE
    if name in {"q", "dq"} or any(
        token in name
        for token in ("robot", "joint", "gripper", "pose", "transform", "elbow")
    ):
        return Modality.ROBOT_STATE
    return Modality.OTHER


def _specialized_candidates(profile: DatasetProfile) -> list[ImportPlan]:
    candidates: list[ImportPlan] = []
    mapping = {
        "canonical_run": "canonical_run",
        "lerobot_v3": "lerobot_v3",
        "surg_sync_v1": "surg_sync_v1",
        "mcap": "mcap",
        "xdf": "xdf",
        "umi_contract": "umi_contract",
    }
    for signature in profile.signatures:
        executor = mapping.get(signature)
        if executor is None:
            continue
        parameters: dict[str, Any] = {}
        if executor == "lerobot_v3":
            parameters["source_rate_hz"] = _lerobot_rate(profile.root)
        candidates.append(
            ImportPlan(
                executor=executor,
                confidence=0.99,
                parameters=parameters,
                evidence=(
                    Evidence(
                        code="known_format_signature",
                        message=f"found the canonical {signature} on-disk signature",
                        score=0.99,
                    ),
                ),
            )
        )
    return candidates


def _lerobot_rate(root: str) -> float | None:
    import json
    from pathlib import Path

    info_path = Path(root) / "meta" / "info.json"
    try:
        document = json.loads(info_path.read_text(encoding="utf-8"))
        value = document.get("fps")
    except (OSError, json.JSONDecodeError):
        return None
    return _float(value)


def _timestamp_profile(indexed: dict[str, Any]) -> dict[str, Any] | None:
    raw = indexed.get("timestamp_fields")
    if not isinstance(raw, list) or not raw:
        return None
    profiles = [item for item in raw if isinstance(item, dict)]
    if not profiles:
        return None
    return max(profiles, key=lambda item: float(item.get("monotonic_fraction", 0.0)))


def _fallback_rate(timestamp: dict[str, Any] | None) -> tuple[float | None, str]:
    if timestamp is None:
        return None, "none"
    delta = _float(timestamp.get("median_delta"))
    if delta is None or delta <= 0:
        return None, "none"

    if delta >= 1_000_000:
        raw_rate = 1_000_000_000 / delta
    elif delta >= 1.0:
        raw_rate = 1_000 / delta
    else:
        raw_rate = 1 / delta
    nearest = min(_COMMON_RATES_HZ, key=lambda rate: abs(math.log(rate / raw_rate)))
    relative_error = abs(nearest - raw_rate) / nearest
    if relative_error <= 0.10:
        return nearest, "timestamp_median_quantized"
    return raw_rate, "timestamp_median"


def _rate(indexed: dict[str, Any], override: float | None) -> tuple[float | None, str]:
    if override is not None:
        if override <= 0:
            raise ValueError(f"rate_hz must be > 0, got {override!r}")
        return float(override), "user_override"
    videos = _dict(indexed.get("videos"))
    video_rate = _float(videos.get("rate_hz"))
    if video_rate is not None and video_rate > 0:
        return video_rate, "video_metadata"
    return _fallback_rate(_timestamp_profile(indexed))


def _time_unit(timestamp: dict[str, Any], rate_hz: float) -> str:
    delta = _float(timestamp.get("median_delta"))
    if delta is None or delta <= 0:
        return "nanoseconds"
    period_ns = 1_000_000_000 / rate_hz
    return min(
        _UNIT_NS,
        key=lambda unit: abs(math.log(max(delta * _UNIT_NS[unit], 1e-12) / period_ns)),
    )


def _duration_error(
    indexed: dict[str, Any], timestamp: dict[str, Any], rate_hz: float, unit: str
) -> float | None:
    spans = timestamp.get("episode_spans")
    row_counts = indexed.get("row_counts")
    episode_ids = indexed.get("episode_ids")
    if not isinstance(spans, list) or not isinstance(row_counts, dict):
        return None
    if not isinstance(episode_ids, list) or len(spans) != len(episode_ids):
        return None
    errors: list[float] = []
    for episode_id, raw_span in zip(episode_ids, spans):
        count = row_counts.get(str(episode_id))
        span = _float(raw_span)
        if not isinstance(count, int) or count < 2 or span is None:
            continue
        expected_ns = (count - 1) * 1_000_000_000 / rate_hz
        actual_ns = span * _UNIT_NS[unit]
        errors.append(abs(actual_ns - expected_ns) / expected_ns)
    return statistics.median(errors) if errors else None


def _camera_parameters(indexed: dict[str, Any]) -> tuple[dict[str, Any], float, list[Evidence]]:
    hdf5 = _dict(indexed.get("hdf5"))
    h5_ratio = _float(hdf5.get("count_match_ratio")) or 0.0
    camera_names = hdf5.get("camera_names")
    if h5_ratio >= 0.99 and isinstance(camera_names, list) and camera_names:
        score = 0.35 * h5_ratio
        return (
            {
                "source": "hdf5",
                "path": str(hdf5["path"]),
                "camera_names": [str(name) for name in camera_names],
            },
            score,
            [
                Evidence(
                    code="hdf5_row_count_match",
                    message=(
                        f"{hdf5.get('count_matches', 0)}/{hdf5.get('count_checks', 0)} "
                        "HDF5 camera arrays have one entry per state row"
                    ),
                    score=score,
                )
            ],
        )

    videos = _dict(indexed.get("videos"))
    if videos:
        ratio = _float(videos.get("frame_count_match_ratio"))
        score = 0.25 * ratio if ratio is not None else 0.10
        return (
            {"source": "video", "glob": str(videos.get("video_glob", "video/*.mp4"))},
            score,
            [
                Evidence(
                    code="video_row_count_match",
                    message=(
                        "sampled video frame counts match state rows"
                        if ratio is not None and ratio >= 0.99
                        else "episode-local video files are present"
                    ),
                    score=score,
                )
            ],
        )
    return {"source": "none"}, 0.0, []


def _indexed_candidates(
    profile: DatasetProfile, *, rate_hz: float | None
) -> list[ImportPlan]:
    indexed = _dict(profile.facts.get("indexed_episode"))
    if not indexed:
        return []
    resolved_rate, rate_source = _rate(indexed, rate_hz)
    if resolved_rate is None:
        return []

    common_fields = indexed.get("common_fields")
    fields = [str(field) for field in common_fields] if isinstance(common_fields, list) else []
    timestamp = _timestamp_profile(indexed)
    timestamp_names = {
        str(item.get("field"))
        for item in indexed.get("timestamp_fields", [])
        if isinstance(item, dict)
    }
    state_fields = [field for field in fields if field not in timestamp_names]
    state_streams = {field: _infer_modality(field).value for field in state_fields}
    camera, camera_score, camera_evidence = _camera_parameters(indexed)

    layout_score = 0.20 if int(indexed.get("episode_count", 0)) > 1 else 0.12
    schema_score = 0.10 if state_streams else 0.0
    rate_score = 0.20 if rate_source in {"video_metadata", "user_override"} else 0.08
    row_evidence = [
        Evidence(
            code="repeated_episode_rows",
            message=(
                f"found {indexed.get('episode_count', 0)} episodes and "
                f"{indexed.get('total_rows', 0)} rows with one common row schema"
            ),
            score=layout_score + schema_score,
        ),
        Evidence(
            code="fixed_rate",
            message=f"native rate {resolved_rate:g} Hz from {rate_source}",
            score=rate_score,
        ),
        *camera_evidence,
    ]
    videos = _dict(indexed.get("videos"))
    frame_ratio = _float(videos.get("frame_count_match_ratio"))
    frame_score = 0.10 * frame_ratio if frame_ratio is not None else 0.0
    if frame_score:
        row_evidence.append(
            Evidence(
                code="video_frame_count_match",
                message="sampled MP4 frame counts equal their episode row counts",
                score=frame_score,
            )
        )
    row_confidence = min(0.99, layout_score + schema_score + rate_score + camera_score + frame_score)

    base_parameters: dict[str, Any] = {
        "episode_glob": str(indexed["episode_glob"]),
        "row_file": str(indexed["row_file"]),
        "state_streams": state_streams,
        "camera": camera,
        "source_clock_domain": "inferred.indexed_episode",
    }
    if timestamp is not None:
        base_parameters["source_time_field"] = str(timestamp["field"])
    row_warnings: list[str] = []
    if timestamp is not None:
        unit = _time_unit(timestamp, resolved_rate)
        error = _duration_error(indexed, timestamp, resolved_rate, unit)
        if error is not None and error > 0.05:
            row_warnings.append(
                f"{timestamp.get('field')} duration differs from the {resolved_rate:g} Hz "
                f"row clock by a median {error * 100:.1f}%"
            )
    row_plan = ImportPlan(
        executor="indexed_episode",
        confidence=row_confidence,
        parameters={
            **base_parameters,
            "clock": {"strategy": "row_index", "rate_hz": resolved_rate},
        },
        evidence=tuple(row_evidence),
        warnings=tuple(row_warnings),
    )

    candidates = [row_plan]
    if timestamp is not None:
        unit = _time_unit(timestamp, resolved_rate)
        monotonic = _float(timestamp.get("monotonic_fraction")) or 0.0
        error = _duration_error(indexed, timestamp, resolved_rate, unit)
        agreement = max(0.0, 1.0 - (error or 0.0) / 0.10)
        timestamp_score = (
            layout_score
            + schema_score
            + 0.15 * monotonic
            + 0.25 * agreement
            + min(camera_score, 0.10)
            + min(rate_score, 0.10)
        )
        timestamp_evidence = [
            Evidence(
                code="monotonic_timestamp_field",
                message=(
                    f"{timestamp.get('field')} is monotonic for "
                    f"{monotonic * 100:.1f}% of adjacent rows"
                ),
                score=0.15 * monotonic,
            ),
            Evidence(
                code="timestamp_duration_agreement",
                message=(
                    f"timestamp-derived episode duration median error is "
                    f"{(error or 0.0) * 100:.1f}% against the {resolved_rate:g} Hz media clock"
                ),
                score=0.25 * agreement,
            ),
        ]
        timestamp_plan = ImportPlan(
            executor="indexed_episode",
            confidence=min(0.99, timestamp_score),
            parameters={
                **base_parameters,
                "clock": {
                    "strategy": "timestamp_field",
                    "rate_hz": resolved_rate,
                    "field": str(timestamp["field"]),
                    "unit": unit,
                },
            },
            evidence=tuple(timestamp_evidence),
            warnings=(
                (f"timestamp duration disagrees with media by {(error or 0.0) * 100:.1f}%",)
                if error is not None and error > 0.05
                else ()
            ),
        )
        candidates.append(timestamp_plan)
    return candidates


def infer_import(
    profile: DatasetProfile,
    *,
    rate_hz: float | None = None,
    min_confidence: float = 0.75,
    min_margin: float = 0.12,
) -> InferenceResult:
    """Rank adapter/clock interpretations and select only decisive winners."""

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    if not 0.0 <= min_margin <= 1.0:
        raise ValueError("min_margin must be in [0, 1]")
    candidates = [*_specialized_candidates(profile), *_indexed_candidates(profile, rate_hz=rate_hz)]
    candidates.sort(key=lambda plan: (-plan.confidence, plan.executor, str(plan.parameters)))
    if not candidates:
        return InferenceResult(
            profile=profile,
            candidates=(),
            selected=None,
            decision="no registered executor can interpret the observed layout",
        )

    top = candidates[0]
    if top.confidence < min_confidence:
        return InferenceResult(
            profile=profile,
            candidates=tuple(candidates),
            selected=None,
            decision=(
                f"top confidence {top.confidence:.3f} is below the required "
                f"{min_confidence:.3f}"
            ),
        )
    if len(candidates) > 1:
        margin = top.confidence - candidates[1].confidence
        if margin < min_margin:
            return InferenceResult(
                profile=profile,
                candidates=tuple(candidates),
                selected=None,
                decision=(
                    f"top-two confidence margin {margin:.3f} is below the required "
                    f"{min_margin:.3f}"
                ),
            )
    return InferenceResult(
        profile=profile,
        candidates=tuple(candidates),
        selected=top,
        decision=f"selected {top.executor} with confidence {top.confidence:.3f}",
    )
