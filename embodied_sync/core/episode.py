"""Canonical types for aligned episodes.

An :class:`Episode` is the result of aligning a run to a fixed-rate
target grid: a sequence of :class:`AlignedFrame`s plus an
:class:`AlignmentReport` summarising missing samples and (optionally) a
cross-check against ground-truth drops.

These types live in ``core/`` — not in ``embodied_sync.align`` — so the
canonical shape sits with the rest of the data model. The engine that
produces them (:func:`embodied_sync.align.align_run`) imports these
types; ``Episode`` is a :data:`~typing.TypeAlias` for :class:`AlignedRun`
so the discoverable name and the historical name refer to the same class
without duplicating fields (D-0024).

See ARCHITECTURE.md ("Alignment") and D-0020 / D-0022 for the semantics
that populate these types (nearest-neighbor and zero-order-hold policies,
per-stream tolerance, skew convention, ground-truth cross-check).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from embodied_sync.core.sample import Sample


@dataclass(frozen=True, slots=True)
class AlignedSampleMetadata:
    """Per-stream, per-frame alignment metadata.

    ``source_time_ns`` and ``skew_ns`` are ``None`` when ``missing`` is
    true. ``method`` records which policy selected the sample (``"nearest_
    neighbor"`` or ``"zoh"`` in v0). ``confidence`` is a scalar in
    ``[0, 1]``, monotonically decreasing in ``|skew|``.
    """

    source_time_ns: int | None
    skew_ns: int | None
    method: str
    missing: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class AlignedFrame:
    """One aligned policy frame.

    ``target_time_ns`` is the world-time-grid target; ``samples[name]``
    is the chosen :class:`Sample` for that stream at that target (or
    ``None`` when missing), and ``metadata[name]`` carries the timing
    metadata for the pick.
    """

    target_time_ns: int
    samples: dict[str, Sample | None]
    metadata: dict[str, AlignedSampleMetadata]


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """Summary of alignment outcomes and ground-truth cross-check.

    ``missing_count[name]`` is the number of frames where ``name`` was
    marked missing. ``ground_truth_missing_count[name]`` counts drops
    from a corruption ground-truth sidecar that fell inside the aligned
    window; it's empty when no ground truth was supplied.
    ``median_skew_ns[name]`` is the signed nanosecond median of
    ``skew_ns`` across every non-missing frame for that stream (or
    ``None`` when every frame for a stream is missing). Same value the
    sync-quality report's "Median skew" column carries and the
    aligned-episode manifest echoes, exposed on the report so a
    downstream tool that only holds an :class:`AlignedRun` can read
    the signed direction of skew per stream without recomputing.
    ``alignment_policy`` is an optional manifest echo of the requested
    alignment policy. It is intentionally excluded from equality so old
    episodes missing the additive manifest key still compare by their
    frame/report outcomes.
    """

    missing_count: dict[str, int]
    ground_truth_missing_count: dict[str, int] = field(default_factory=dict)
    median_skew_ns: dict[str, int | None] = field(default_factory=dict)
    alignment_policy: object | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class AlignedRun:
    """Aligned frames plus report."""

    frames: list[AlignedFrame]
    report: AlignmentReport


Episode: TypeAlias = AlignedRun
"""Canonical discoverable name for an aligned run of samples.

Alias of :class:`AlignedRun`. New code should prefer ``Episode``; the
historical name ``AlignedRun`` continues to work because they are the
same class (D-0024)."""
