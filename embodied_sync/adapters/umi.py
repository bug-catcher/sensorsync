"""UMI replay-buffer adapter (Milestone 6 synthetic contract path)."""

from __future__ import annotations

import json
from pathlib import Path

from embodied_sync.core.sample import Modality, Sample

__all__ = ["load_umi_replay_buffer"]

_FORMAT = "embodied_sync.umi.replay_buffer.v0"


def load_umi_replay_buffer(path: str | Path) -> dict[str, list[Sample]]:
    """Load a small UMI-like replay-buffer JSON fixture.

    The fixture records explicit per-stream latency offsets in nanoseconds.
    ``acquisition_time_ns`` is preserved from the file and
    ``receive_time_ns`` is computed as ``acquisition + latency_offset_ns`` so
    latency handling is visible and testable.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != _FORMAT:
        raise ValueError(
            f"unsupported UMI replay-buffer format {document.get('format')!r} in {path}"
        )
    streams = document.get("streams")
    if not isinstance(streams, dict):
        raise ValueError(f"invalid UMI replay buffer in {path}: missing streams")

    run: dict[str, list[Sample]] = {}
    for name, stream in streams.items():
        latency_offset_ns = int(stream.get("latency_offset_ns", 0))
        modality = Modality(stream.get("modality", Modality.OTHER.value))
        clock_domain = str(stream.get("source_clock_domain", "umi"))
        records = stream.get("samples")
        if not isinstance(records, list):
            raise ValueError(f"invalid UMI stream {name!r}: missing samples")
        run[name] = [
            Sample(
                stream_name=name,
                modality=modality,
                sequence_id=int(record.get("sequence_id", index)),
                acquisition_time_ns=int(record["acquisition_time_ns"]),
                receive_time_ns=int(record["acquisition_time_ns"]) + latency_offset_ns,
                source_clock_domain=clock_domain,
                payload=record.get("payload"),
                payload_ref=record.get("payload_ref"),
                quality_flags=frozenset(record.get("quality_flags", ())),
            )
            for index, record in enumerate(records)
        ]
    return run
