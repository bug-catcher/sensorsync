"""Canonical types for sync-quality reports.

A :class:`SyncReport` (alias of :class:`SyncQualityReport`) is the
structured, machine-readable summary of an :class:`Episode`: per-stream
skew/missing statistics plus a format version. Rendering (HTML, JSON,
...) lives in ``embodied_sync.reports``; the data shape lives here so
downstream tools can consume it without pulling the reports subpackage
(D-0024).

``REPORT_FORMAT_VERSION`` is the on-disk schema version of the JSON
summary produced by ``report_summary_dict``; bump it whenever the
summary shape changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


REPORT_FORMAT_VERSION = 0


@dataclass(frozen=True, slots=True)
class StreamStats:
    """Per-stream summary derived from an aligned episode.

    ``median_skew_ns`` / ``median_abs_skew_ns`` / ``median_confidence``
    are ``None`` when the stream had zero non-missing frames. ``method``
    is the alignment policy string picked up from the frames' metadata
    (``None`` when no non-missing frame was observed).
    """

    name: str
    frame_count: int
    missing_count: int
    missing_rate: float
    median_skew_ns: int | None
    median_abs_skew_ns: int | None
    median_confidence: float | None
    method: str | None
    ground_truth_missing_count: int


@dataclass(frozen=True, slots=True)
class SyncQualityReport:
    """Structured sync-quality summary of an aligned episode."""

    frame_count: int
    streams: tuple[StreamStats, ...]


SyncReport: TypeAlias = SyncQualityReport
"""Canonical discoverable name for a structured sync-quality summary.

Alias of :class:`SyncQualityReport`. New code should prefer
``SyncReport``; the historical name continues to work because they are
the same class (D-0024)."""
