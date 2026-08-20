"""MCAP adapter (Milestone 4 synthetic contract path).

Optional-dependency discipline: this module imports at base-install
time (no top-level ``import mcap``), so ``import
embodied_sync.adapters.mcap`` succeeds without ``pip install
embodied-sync[mcap]``. The concrete functions do their heavy imports
inside external-format function bodies.

See ``docs/developer/adapter_authoring_guide.md`` for the contract every
adapter (including this one) must satisfy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embodied_sync.core.sample import (
    QUALITY_DUPLICATE,
    QUALITY_GAP_BEFORE,
    QUALITY_NON_MONOTONIC,
    Modality,
    Sample,
)
from embodied_sync.datasets.io import _record_to_sample
from embodied_sync.exporters.mcap import _FORMAT

__all__ = ["load_mcap_run"]

_INLINE_PAYLOAD_BYTES = 4096


def load_mcap_run(
    path: str | Path,
    *,
    topics: list[str] | None = None,
    start_time_ns: int | None = None,
    end_time_ns: int | None = None,
) -> dict[str, list[Sample]]:
    """Load an MCAP file into a run of ``Sample``s.

    Deterministic contract documents written by
    :func:`embodied_sync.exporters.mcap.save_mcap_run` are read without the
    optional ``mcap`` dependency. True MCAP files use ``mcap.reader`` lazily
    and preserve integer log/publish timestamps without ROS message decoding.

    ``topics`` filters to a subset of channel topics (``None`` = all).
    ``start_time_ns`` / ``end_time_ns`` bound the loaded window against
    the reader's ``log_time`` axis; both are inclusive on the low side
    and exclusive on the high side per ``mcap.reader.iter_messages``.
    Both filters are used at read time so no data outside the window
    hits Python objects — necessary for surveying large SLAM-style
    recordings without materialising the whole file.

    The contract-document path ignores the two time filters (the
    contract document has no cheap index; downstream code slices in
    Python).
    """
    mcap_path = _resolve_mcap_path(Path(path))
    with mcap_path.open("rb") as f:
        prefix = f.read(1)
    if prefix == b"{":
        return _load_contract_mcap_run(mcap_path)
    return _load_true_mcap_run(
        mcap_path,
        topics=topics,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
    )


def _resolve_mcap_path(path: Path) -> Path:
    """Resolve direct MCAP files and rosbag2 directories to one MCAP file."""
    if path.is_dir():
        candidates = sorted(path.rglob("*.mcap"))
        if not candidates:
            raise FileNotFoundError(f"no .mcap files found under rosbag directory: {path}")
        if len(candidates) > 1:
            raise ValueError(
                f"rosbag directory {path} contains multiple .mcap files; "
                "pass one file explicitly"
            )
        return candidates[0]
    if path.suffix == ".zip":
        if path.stat().st_size == 0:
            raise ValueError(f"empty MCAP/rosbag zip placeholder: {path}")
        raise ValueError(
            f"cannot load compressed MCAP/rosbag zip directly: {path}; "
            "extract it and pass the rosbag directory or .mcap file"
        )
    return path


def _load_contract_mcap_run(path: Path) -> dict[str, list[Sample]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != _FORMAT:
        raise ValueError(
            f"unsupported MCAP contract format {document.get('format')!r} in {path}"
        )
    if document.get("type") != "run":
        raise ValueError(f"expected MCAP run document, got {document.get('type')!r}")
    streams = document.get("streams")
    if not isinstance(streams, dict):
        raise ValueError(f"invalid MCAP run document in {path}: missing streams")
    return {
        name: [_record_to_sample(record) for record in records]
        for name, records in streams.items()
    }


def _load_true_mcap_run(
    path: Path,
    *,
    topics: list[str] | None,
    start_time_ns: int | None = None,
    end_time_ns: int | None = None,
) -> dict[str, list[Sample]]:
    from mcap.reader import make_reader  # noqa: PLC0415

    run: dict[str, list[Sample]] = {}
    per_stream_counts: dict[str, int] = {}
    last_sequence: dict[str, int] = {}
    last_receive_time: dict[str, int] = {}

    with path.open("rb") as f:
        reader = make_reader(f)
        iter_kwargs: dict[str, Any] = {"topics": topics}
        if start_time_ns is not None:
            iter_kwargs["start_time"] = int(start_time_ns)
        if end_time_ns is not None:
            iter_kwargs["end_time"] = int(end_time_ns)
        for schema, channel, message in reader.iter_messages(**iter_kwargs):
            stream_name = channel.topic
            index = per_stream_counts.get(stream_name, 0)
            sequence_id = (
                int(message.sequence)
                if message.sequence is not None and message.sequence != 0
                else index
            )
            receive_time_ns = int(message.log_time)
            acquisition_time_ns = int(message.publish_time or message.log_time)
            quality_flags = _quality_flags_for_message(
                stream_name=stream_name,
                sequence_id=sequence_id,
                receive_time_ns=receive_time_ns,
                last_sequence=last_sequence,
                last_receive_time=last_receive_time,
            )
            sample = Sample(
                stream_name=stream_name,
                modality=_infer_modality(channel.topic, schema.name if schema else None),
                sequence_id=sequence_id,
                acquisition_time_ns=acquisition_time_ns,
                receive_time_ns=receive_time_ns,
                source_clock_domain=(
                    "mcap_publish_time"
                    if message.publish_time is not None
                    else "mcap_log_time"
                ),
                payload=_payload_for_message(
                    schema_name=schema.name if schema else None,
                    schema_encoding=schema.encoding if schema else None,
                    channel_topic=channel.topic,
                    message_encoding=channel.message_encoding,
                    data=message.data,
                ),
                payload_ref=_payload_ref_for_message(path, channel.topic, sequence_id, message.data),
                quality_flags=quality_flags,
            )
            run.setdefault(stream_name, []).append(sample)
            per_stream_counts[stream_name] = index + 1
            last_sequence[stream_name] = sequence_id
            last_receive_time[stream_name] = receive_time_ns
    return run


def _infer_modality(topic: str, schema_name: str | None) -> Modality:
    haystack = f"{topic} {schema_name or ''}".lower()
    if "image" in haystack or "camera" in haystack:
        return Modality.CAMERA
    if "imu" in haystack:
        return Modality.TACTILE
    if "tf" in haystack or "log" in haystack or "event" in haystack:
        return Modality.EVENT
    return Modality.OTHER


def _quality_flags_for_message(
    *,
    stream_name: str,
    sequence_id: int,
    receive_time_ns: int,
    last_sequence: dict[str, int],
    last_receive_time: dict[str, int],
) -> frozenset[str]:
    flags: set[str] = set()
    previous_sequence = last_sequence.get(stream_name)
    if previous_sequence is not None:
        if sequence_id == previous_sequence:
            flags.add(QUALITY_DUPLICATE)
        elif sequence_id > previous_sequence + 1:
            flags.add(QUALITY_GAP_BEFORE)
    previous_receive_time = last_receive_time.get(stream_name)
    if previous_receive_time is not None and receive_time_ns < previous_receive_time:
        flags.add(QUALITY_NON_MONOTONIC)
    return frozenset(flags)


def _payload_for_message(
    *,
    schema_name: str | None,
    schema_encoding: str | None,
    channel_topic: str,
    message_encoding: str,
    data: bytes,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_name": schema_name,
        "schema_encoding": schema_encoding,
        "topic": channel_topic,
        "message_encoding": message_encoding,
        "byte_length": len(data),
    }
    if len(data) <= _INLINE_PAYLOAD_BYTES:
        payload["data_hex"] = data.hex()
    return payload


def _payload_ref_for_message(
    path: Path,
    topic: str,
    sequence_id: int,
    data: bytes,
) -> str | None:
    if len(data) <= _INLINE_PAYLOAD_BYTES:
        return None
    return f"{path}#topic={topic};sequence_id={sequence_id};byte_length={len(data)}"
