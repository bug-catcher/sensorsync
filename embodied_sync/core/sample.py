"""The canonical unit of data: a timestamped sample from one stream.

Contracts (see ARCHITECTURE.md and DECISIONS.md D-0002/D-0003):

- All timestamps are integer nanoseconds. Never floats.
- ``acquisition_time_ns`` is when the sensor observed the world, expressed in
  ``source_clock_domain``. ``receive_time_ns`` is when the host received the
  sample. In clean data ``receive_time_ns >= acquisition_time_ns``.
- Adapters must preserve imported timestamps exactly; they never resample or
  re-timestamp.
- Clock domains are never mixed silently. Cross-domain comparisons require an
  explicit mapping (``embodied_sync.time``, Milestone 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Modality(str, Enum):
    """Coarse sensor/data modality of a stream."""

    CAMERA = "camera"
    ROBOT_STATE = "robot_state"
    TACTILE = "tactile"
    AUDIO = "audio"
    ACTION = "action"
    EVENT = "event"
    OTHER = "other"


#: Well-known quality flags. Adapters/corruptors may add their own strings,
#: but should prefer these.
QUALITY_SYNTHETIC = "synthetic"
QUALITY_DUPLICATE = "duplicate"
QUALITY_NON_MONOTONIC = "non_monotonic"
QUALITY_GAP_BEFORE = "gap_before"
QUALITY_INTERPOLATED = "interpolated"
#: Live-capture flags (D-0037). ``receive_timestamped``: the SDK gave no
#: device timestamp, so ``acquisition_time_ns`` is really the host receive
#: time and carries the transport latency inside it. ``clock_mapped``: the
#: acquisition time was translated out of a foreign clock domain through a
#: registered ``LatencyEstimate``. ``unmapped_clock_domain``: the stream is
#: in a foreign domain with *no* mapping, so it was matched on receive time
#: — a degraded match that must never look like a clean one.
QUALITY_RECEIVE_TIMESTAMPED = "receive_timestamped"
QUALITY_CLOCK_MAPPED = "clock_mapped"
QUALITY_UNMAPPED_CLOCK_DOMAIN = "unmapped_clock_domain"


@dataclass(frozen=True, slots=True)
class Sample:
    """One timestamped sample from one stream.

    ``payload`` holds small inline data; ``payload_ref`` references bulk data
    (e.g. a video frame) by path/key. At least one of the two should be set
    for non-event samples; both may be set.
    """

    stream_name: str
    modality: Modality
    sequence_id: int
    acquisition_time_ns: int
    receive_time_ns: int
    source_clock_domain: str
    payload: Any = None
    payload_ref: str | None = None
    quality_flags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Integer-ns contract (D-0002). bool is an int subclass; reject it too.
        for name in ("acquisition_time_ns", "receive_time_ns", "sequence_id"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be int, got {type(value).__name__}")

    @property
    def transport_latency_ns(self) -> int:
        """receive - acquisition. Negative values indicate a timing anomaly
        (or an unmapped clock-domain offset) and should be flagged upstream."""
        return self.receive_time_ns - self.acquisition_time_ns
