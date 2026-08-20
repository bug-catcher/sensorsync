"""``SyncBundle`` — what one :meth:`SyncSession.get` call returns.

A plain ``dict[str, Any]`` is the tempting return type and the wrong
one: it makes a good match and a bad match look identical. The frame
that was 2 ms off and the frame that fell back to a 400 ms-stale hold
both arrive as "the camera frame", and the researcher has no way to
tell without a second API. So the payloads stay one keystroke away
(``bundle["camera"]``) while the *evidence* rides along beside them
(``bundle.items["camera"].skew_ns``).

The one bundle-level quality number is :attr:`SyncBundle.span_ns`: the
spread between the earliest and latest matched acquisition time. It is
the quantity ROS's ``ApproximateTime`` policy minimises, and unlike a
per-stream skew it answers the question a multi-sensor consumer
actually has — *how far apart in time are the things I am about to
treat as simultaneous?*

Honesty note (§2.6 of the design): every number here measures
*timestamp consistency*, not physical simultaneity. Two cameras can
agree on timestamps to the microsecond while exposing 30 ms apart.
Establishing the physical relationship is
:mod:`embodied_sync.calibrate`'s job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from embodied_sync.core.sample import Sample

__all__ = ["BundleItem", "SyncBundle"]


@dataclass(frozen=True, slots=True)
class BundleItem:
    """One stream's entry in a :class:`SyncBundle`.

    ``payload`` is the caller's *own* object — the frame the SDK handed
    over, unmodified — or a list of them for a ``window`` stream, or
    ``None`` when nothing matched. ``sample`` is the corresponding
    :class:`~embodied_sync.core.sample.Sample` (list for ``window``).

    ``missing`` means no usable sample was picked. ``within_tolerance``
    means a sample was picked *and* its skew is inside the stream's
    configured tolerance; the two are not simply each other's negation,
    because a session can be asked for a bundle where a stream matched
    nothing at all (``missing=True``) versus matched something too old
    to trust (also ``missing=True``, but with ``skew_ns`` populated so
    the caller can see how stale). ``method`` records which picker ran.

    For a stream whose clock domain is foreign to the session,
    ``confidence`` has already been lowered by
    :func:`~embodied_sync.time.alignment.cross_domain_confidence_factor`.
    """

    payload: Any | None
    sample: Sample | list[Sample] | None
    skew_ns: int | None
    within_tolerance: bool
    missing: bool
    confidence: float
    method: str


@dataclass(frozen=True, slots=True)
class SyncBundle:
    """A synchronised set of per-stream payloads at one target time.

    ``items`` preserves the session's stream-configuration order.
    ``ok`` is ``True`` only when every configured stream is present and
    within its own tolerance — the single boolean a control loop can
    branch on without deciding what "good enough" means twice.
    """

    target_time_ns: int
    items: dict[str, BundleItem] = field(default_factory=dict)
    ok: bool = False
    span_ns: int | None = None

    def __getitem__(self, stream: str) -> Any:
        """``bundle["camera"]`` → that stream's payload (``KeyError`` if unknown)."""
        return self.items[stream].payload

    def __contains__(self, stream: object) -> bool:
        return stream in self.items

    def __iter__(self) -> Iterator[str]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def payloads(self) -> dict[str, Any]:
        """The plain-dict shape, for code that has already checked :attr:`ok`."""
        return {name: item.payload for name, item in self.items.items()}

    def missing_streams(self) -> list[str]:
        """Names of streams that matched nothing, in configuration order."""
        return [name for name, item in self.items.items() if item.missing]

    def out_of_tolerance_streams(self) -> list[str]:
        """Names of streams that are present but outside tolerance."""
        return [
            name
            for name, item in self.items.items()
            if not item.missing and not item.within_tolerance
        ]
