"""Incremental run-format-v0 writer for a live session.

The point of this module is a contract, not a file format: **a recorded
session is a valid ``embsync align`` / ``embsync report`` input.** Live
capture and offline replay share one on-disk shape, so the run a
researcher records at the bench loads with
:func:`~embodied_sync.datasets.io.load_run` and aligns with
:func:`~embodied_sync.align.align_run` without a conversion step, and
the sync report they get from a recording is computed by the same code
that produced the report they trust from a fixture.

:func:`~embodied_sync.datasets.io.save_run` writes a whole run at once
from a materialised ``dict[str, list[Sample]]``; a live session has no
such dict — that is the memory growth the ring buffers exist to avoid.
So :class:`RunRecorder` holds one append-mode file handle per stream and
writes each sample as it arrives, using the *same* record codec
(:func:`~embodied_sync.datasets.io.sample_to_record`) so the bytes are
identical to a saved run.

``manifest.json`` is written by :meth:`RunRecorder.flush` and again by
:meth:`RunRecorder.close`, because ``load_run`` cross-checks each stream's
``sample_count`` against the lines on disk. A session killed with
``SIGKILL`` between flushes therefore leaves a manifest that undercounts
— the JSONL is still intact, and re-running ``flush``-less recovery is
out of scope for v1; the context-manager form (``with embsync.init(...)
as sync:``) closes on the way out of the block including on exceptions.

Persist modes
-------------
``"metadata"`` (default) writes timing, sequence id, clock domain and
quality flags with ``payload=None``. A camera session writing 30 frames
per second of JPEG bytes into a JSONL file is not a feature, and the
timing record is what this library is about.
``"full"`` also writes the payload, and fails loudly with a pointer at
the ``serialize=`` hook when the payload is not JSON-able.
``"off"`` writes nothing and the stream is omitted from the manifest
entirely — a manifest entry with no file would make ``load_run`` raise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from embodied_sync.core.sample import Sample
from embodied_sync.datasets.io import (
    FORMAT_VERSION,
    jsonify_payload,
    sample_to_record,
)
from embodied_sync.session.config import StreamConfig

__all__ = ["MANIFEST_NAME", "SESSION_QUALITY_NAME", "STREAMS_DIR", "RunRecorder"]

MANIFEST_NAME = "manifest.json"
STREAMS_DIR = "streams"
#: Final ``quality()`` snapshot, written next to the manifest at close.
SESSION_QUALITY_NAME = "session_quality.json"

#: ``(stream_name, payload) -> JSON-able`` hook for ``persist="full"`` streams.
PayloadSerializer = Callable[[str, Any], Any]


class RunRecorder:
    """Append-as-you-go writer for one session's run directory.

    Not thread-safe on its own: the session serialises appends for a
    given stream under that stream's lock, and each stream has its own
    file handle, so two streams never contend.
    """

    __slots__ = ("_counts", "_files", "_persist", "_run_dir", "_serialize")

    def __init__(
        self,
        run_dir: str | Path,
        configs: Mapping[str, StreamConfig],
        *,
        serialize: PayloadSerializer | None = None,
    ) -> None:
        self._run_dir = Path(run_dir)
        streams_dir = self._run_dir / STREAMS_DIR
        streams_dir.mkdir(parents=True, exist_ok=True)
        self._serialize = serialize
        self._persist: dict[str, str] = {}
        self._files: dict[str, TextIO] = {}
        self._counts: dict[str, int] = {}
        for name, config in configs.items():
            self._persist[name] = config.persist
            if config.persist == "off":
                continue
            path = streams_dir / f"{name}.jsonl"
            if path.exists() and path.stat().st_size:
                raise FileExistsError(
                    f"refusing to append to an existing recorded stream: {path}"
                )
            self._files[name] = path.open("w", encoding="utf-8")
            self._counts[name] = 0

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def append(self, stream: str, sample: Sample) -> None:
        """Write one sample's record, honouring the stream's persist mode."""
        handle = self._files.get(stream)
        if handle is None:
            return
        record = sample_to_record(sample)
        if self._persist[stream] == "metadata":
            record["payload"] = None
        elif self._serialize is not None:
            record["payload"] = jsonify_payload(
                self._serialize(stream, sample.payload)
            )
        try:
            line = json.dumps(record, separators=(",", ":"))
        except TypeError as exc:
            raise TypeError(
                f"stream {stream!r}: payload of type "
                f"{type(sample.payload).__name__} is not JSON-serializable, so "
                f"persist='full' cannot record it. Pass a serialize=(stream, "
                f"payload) -> JSON-able hook to init()/SyncSession, or set "
                f"persist='metadata' for this stream."
            ) from exc
        handle.write(line)
        handle.write("\n")
        self._counts[stream] += 1

    def counts(self) -> dict[str, int]:
        """Records written so far, per persisted stream."""
        return dict(self._counts)

    def flush(self, manifest: dict[str, Any]) -> None:
        """Flush every stream file, then rewrite ``manifest.json``.

        Order matters: the manifest's ``sample_count`` must never claim
        more than is durable on disk, or ``load_run`` raises on a run
        that is actually fine.
        """
        for handle in self._files.values():
            handle.flush()
        self.write_manifest(manifest)

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        """Write ``manifest.json`` with run-format-v0 reserved keys enforced."""
        payload = dict(manifest)
        payload["format_version"] = FORMAT_VERSION
        with (self._run_dir / MANIFEST_NAME).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    def write_sidecar(self, name: str, data: dict[str, Any]) -> Path:
        """Write a JSON sidecar (e.g. ``session_quality.json``) into the run dir."""
        path = self._run_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        return path

    def close(self) -> None:
        """Close every stream file. Idempotent."""
        for handle in self._files.values():
            if not handle.closed:
                handle.close()
