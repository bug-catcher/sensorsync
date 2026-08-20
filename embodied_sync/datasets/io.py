"""Run-on-disk format v0: ``manifest.json`` + ``streams/<name>.jsonl``.

Design (DECISIONS.md D-0005). Currently the *designed API* under test; bodies
raise :class:`NotImplementedError` until implemented (TDD red).

Layout::

    run_dir/
      manifest.json           # {"format_version": 0, "streams": {...}, ...}
      streams/<name>.jsonl    # one JSON object per Sample

Contracts
---------
- ``save_run`` then ``load_run`` round-trips every Sample field exactly.
  Timestamps in particular are integers end-to-end: JSON must never pass
  through a float conversion (this is the timestamp-preservation contract).
- Samples are written in-order; loading preserves order.
- ``payload`` must be JSON-serializable in v0 (numbers, strings, lists,
  dicts, None). Numpy arrays are converted to lists on save and come back as
  lists — adapters that need arrays convert at the edge. Bulk binary payloads
  use ``payload_ref`` instead.
- ``manifest.json`` records ``format_version``, stream names, modalities,
  clock domains, sample counts, and (for synthetic runs) the generator seed
  and duration, so a run is reproducible from its manifest alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from embodied_sync.align.engine import (
    AlignedFrame,
    AlignedRun,
    AlignedSampleMetadata,
    AlignmentReport,
    MethodArg,
)
from embodied_sync.core.sample import Modality, Sample

FORMAT_VERSION = 0

_MANIFEST_NAME = "manifest.json"
_STREAMS_DIR = "streams"
_FRAMES_NAME = "frames.jsonl"
CORRUPTION_GROUND_TRUTH_NAME = "corruption_ground_truth.json"
_EPISODE_TYPE = "aligned_episode"


def _jsonify_payload(obj: Any) -> Any:
    """Convert numpy containers/scalars to plain Python; leave the rest as-is.

    Tuples become lists (JSON has no tuple); they come back as lists.
    Timestamps never pass through here — only payloads do.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {key: _jsonify_payload(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify_payload(value) for value in obj]
    return obj


def _sample_to_record(sample: Sample) -> dict[str, Any]:
    return {
        "stream_name": sample.stream_name,
        "modality": sample.modality.value,
        "sequence_id": sample.sequence_id,
        "acquisition_time_ns": sample.acquisition_time_ns,
        "receive_time_ns": sample.receive_time_ns,
        "source_clock_domain": sample.source_clock_domain,
        "payload": _jsonify_payload(sample.payload),
        "payload_ref": sample.payload_ref,
        # Sorted so files are byte-stable for identical runs (diffable fixtures).
        "quality_flags": sorted(sample.quality_flags),
    }


def _record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        stream_name=record["stream_name"],
        modality=Modality(record["modality"]),
        sequence_id=record["sequence_id"],
        acquisition_time_ns=record["acquisition_time_ns"],
        receive_time_ns=record["receive_time_ns"],
        source_clock_domain=record["source_clock_domain"],
        payload=record["payload"],
        payload_ref=record["payload_ref"],
        quality_flags=frozenset(record["quality_flags"]),
    )


#: Public aliases for the run-format-v0 record codec. :func:`save_run` writes a
#: whole run at once; a live recorder (:mod:`embodied_sync.session`, D-0037)
#: appends one line at a time as samples arrive and must produce *byte-identical*
#: records, so it shares this codec instead of re-deriving the schema.
sample_to_record = _sample_to_record
record_to_sample = _record_to_sample
jsonify_payload = _jsonify_payload


def save_run(
    run: dict[str, list[Sample]],
    run_dir: str | Path,
    *,
    extra_manifest: dict[str, object] | None = None,
) -> Path:
    """Write a run directory. Fails if ``run_dir`` exists and is non-empty.

    Returns the run directory path. Timestamp integers are serialized by
    Python's ``json`` directly (arbitrary-precision int → decimal literal),
    so there is no float path anywhere.
    """
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to write run into non-empty directory: {run_dir}")
    streams_dir = run_dir / _STREAMS_DIR
    streams_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = dict(extra_manifest or {})
    # Reserved keys always win over extra_manifest.
    manifest["format_version"] = FORMAT_VERSION
    manifest["streams"] = {}
    for name, samples in run.items():
        with (streams_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for sample in samples:
                json.dump(_sample_to_record(sample), f, separators=(",", ":"))
                f.write("\n")
        manifest["streams"][name] = {
            "modality": samples[0].modality.value if samples else None,
            "clock_domains": sorted({s.source_clock_domain for s in samples}),
            "sample_count": len(samples),
        }
    with (run_dir / _MANIFEST_NAME).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return run_dir


def load_run(run_dir: str | Path) -> dict[str, list[Sample]]:
    """Load a run directory written by :func:`save_run`.

    Round-trip guarantee: ``load_run(save_run(run, d)) == run`` for JSON-
    serializable payloads, with all timestamp fields preserved exactly.
    Stream order follows the manifest (which preserves save order).
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"not a run directory (no {_MANIFEST_NAME}): {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported run format_version {version!r} in {manifest_path} "
            f"(this build reads version {FORMAT_VERSION})"
        )

    run: dict[str, list[Sample]] = {}
    for name, info in manifest["streams"].items():
        stream_path = run_dir / _STREAMS_DIR / f"{name}.jsonl"
        with stream_path.open("r", encoding="utf-8") as f:
            samples = [_record_to_sample(json.loads(line)) for line in f if line.strip()]
        if len(samples) != info["sample_count"]:
            raise ValueError(
                f"stream {name!r}: manifest says {info['sample_count']} samples, "
                f"found {len(samples)} in {stream_path}"
            )
        run[name] = samples
    return run


def save_corruption_ground_truth(
    dropped: Mapping[str, Sequence[Sample]],
    run_dir: str | Path,
    *,
    extra_metadata: dict[str, object] | None = None,
) -> Path:
    """Write dropped-sample ground truth next to a corrupted run.

    The sidecar is intentionally separate from ``manifest.json``: it is
    validation truth, not recorder-observable run metadata. Samples use the
    same JSON record shape as stream JSONL files, preserving every timestamp
    as an integer.
    """
    run_dir = Path(run_dir)
    metadata: dict[str, Any] = dict(extra_metadata or {})
    metadata["format_version"] = FORMAT_VERSION
    metadata["type"] = "corruption_ground_truth"
    metadata["dropped"] = {
        stream: [_sample_to_record(sample) for sample in samples]
        for stream, samples in dropped.items()
    }
    path = run_dir / CORRUPTION_GROUND_TRUTH_NAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def load_corruption_ground_truth(run_dir: str | Path) -> dict[str, tuple[Sample, ...]]:
    """Load dropped-sample ground truth written by ``save_corruption_ground_truth``."""
    path = Path(run_dir) / CORRUPTION_GROUND_TRUTH_NAME
    metadata = json.loads(path.read_text(encoding="utf-8"))
    version = metadata.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported corruption ground truth format_version {version!r} in {path} "
            f"(this build reads version {FORMAT_VERSION})"
        )
    dropped = metadata.get("dropped")
    if not isinstance(dropped, dict):
        raise ValueError(f"invalid corruption ground truth in {path}: missing dropped mapping")
    return {
        stream: tuple(_record_to_sample(record) for record in records)
        for stream, records in dropped.items()
    }


def _metadata_to_record(metadata: AlignedSampleMetadata) -> dict[str, Any]:
    return {
        "source_time_ns": metadata.source_time_ns,
        "skew_ns": metadata.skew_ns,
        "method": metadata.method,
        "missing": metadata.missing,
        "confidence": metadata.confidence,
    }


def _record_to_metadata(record: dict[str, Any]) -> AlignedSampleMetadata:
    return AlignedSampleMetadata(
        source_time_ns=record["source_time_ns"],
        skew_ns=record["skew_ns"],
        method=record["method"],
        missing=record["missing"],
        confidence=record["confidence"],
    )


def _frame_to_record(frame: AlignedFrame) -> dict[str, Any]:
    return {
        "target_time_ns": frame.target_time_ns,
        "samples": {
            name: (_sample_to_record(sample) if sample is not None else None)
            for name, sample in frame.samples.items()
        },
        "metadata": {
            name: _metadata_to_record(md) for name, md in frame.metadata.items()
        },
    }


def _record_to_frame(record: dict[str, Any]) -> AlignedFrame:
    samples = {
        name: (_record_to_sample(payload) if payload is not None else None)
        for name, payload in record["samples"].items()
    }
    metadata = {
        name: _record_to_metadata(md) for name, md in record["metadata"].items()
    }
    return AlignedFrame(
        target_time_ns=record["target_time_ns"],
        samples=samples,
        metadata=metadata,
    )


def _alignment_policy_to_manifest(policy: object) -> Any:
    if policy is None:
        return None
    if isinstance(policy, str):
        return policy
    if not isinstance(policy, Mapping):
        return policy
    return {
        stream: (
            entry
            if isinstance(entry, (str, Mapping))
            else {
                "method": entry.method,
                "tolerance_ns": entry.tolerance_ns,
            }
        )
        for stream, entry in policy.items()
    }


def _alignment_policy_from_manifest(raw: object) -> object | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        return raw
    policy: dict[str, object] = {}
    for stream, entry in raw.items():
        if isinstance(entry, str):
            policy[str(stream)] = entry
        elif isinstance(entry, dict):
            policy[str(stream)] = dict(entry)
        else:
            policy[str(stream)] = entry
    return policy


def save_episode(
    aligned: AlignedRun,
    episode_dir: str | Path,
    *,
    target_rate_hz: float,
    alignment_policy: MethodArg | None = None,
    extra_manifest: dict[str, object] | None = None,
) -> Path:
    """Write an aligned episode (D-0021) with ``manifest.json`` + ``frames.jsonl``.

    Fails if ``episode_dir`` exists and is non-empty. Timestamps stay as
    integer nanoseconds throughout — the JSON path never introduces a
    float. ``target_rate_hz`` is echoed in the manifest so consumers can
    verify the requested rate matches the frame period without inferring
    from spacing.
    """
    episode_dir = Path(episode_dir)
    if episode_dir.exists() and any(episode_dir.iterdir()):
        raise FileExistsError(
            f"refusing to write episode into non-empty directory: {episode_dir}"
        )
    episode_dir.mkdir(parents=True, exist_ok=True)

    stream_names = (
        list(aligned.frames[0].samples.keys())
        if aligned.frames
        else sorted(aligned.report.missing_count.keys())
    )
    target_period_ns = round(1e9 / target_rate_hz)

    manifest: dict[str, Any] = dict(extra_manifest or {})
    policy_for_manifest = (
        alignment_policy
        if alignment_policy is not None
        else aligned.report.alignment_policy
    )
    manifest["format_version"] = FORMAT_VERSION
    manifest["type"] = _EPISODE_TYPE
    manifest["target_rate_hz"] = target_rate_hz
    manifest["target_period_ns"] = target_period_ns
    manifest["streams"] = stream_names
    manifest["frame_count"] = len(aligned.frames)
    manifest["missing_count"] = dict(aligned.report.missing_count)
    manifest["ground_truth_missing_count"] = dict(
        aligned.report.ground_truth_missing_count
    )
    manifest["median_skew_ns"] = dict(aligned.report.median_skew_ns)
    if policy_for_manifest is not None:
        manifest["alignment_policy"] = _alignment_policy_to_manifest(
            policy_for_manifest
        )

    with (episode_dir / _FRAMES_NAME).open("w", encoding="utf-8") as f:
        for frame in aligned.frames:
            json.dump(_frame_to_record(frame), f, separators=(",", ":"))
            f.write("\n")
    with (episode_dir / _MANIFEST_NAME).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return episode_dir


def load_episode(episode_dir: str | Path) -> AlignedRun:
    """Load an aligned episode written by :func:`save_episode`.

    Round-trip guarantee: ``load_episode(save_episode(a, d, ...))``
    reconstructs an :class:`AlignedRun` equal to ``a``.
    """
    episode_dir = Path(episode_dir)
    manifest_path = episode_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"not an episode directory (no {_MANIFEST_NAME}): {episode_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported episode format_version {version!r} in {manifest_path} "
            f"(this build reads version {FORMAT_VERSION})"
        )
    if manifest.get("type") != _EPISODE_TYPE:
        raise ValueError(
            f"expected episode type {_EPISODE_TYPE!r}, got {manifest.get('type')!r}"
        )

    frames_path = episode_dir / _FRAMES_NAME
    with frames_path.open("r", encoding="utf-8") as f:
        frames = [_record_to_frame(json.loads(line)) for line in f if line.strip()]
    if len(frames) != manifest["frame_count"]:
        raise ValueError(
            f"manifest says {manifest['frame_count']} frames, found {len(frames)} "
            f"in {frames_path}"
        )
    median_skew_raw = manifest.get("median_skew_ns", {})
    median_skew_ns: dict[str, int | None] = {
        name: (int(value) if value is not None else None)
        for name, value in median_skew_raw.items()
    }
    report = AlignmentReport(
        missing_count=dict(manifest["missing_count"]),
        ground_truth_missing_count=dict(manifest.get("ground_truth_missing_count", {})),
        median_skew_ns=median_skew_ns,
        alignment_policy=_alignment_policy_from_manifest(
            manifest.get("alignment_policy")
        ),
    )
    return AlignedRun(frames=frames, report=report)
