"""``StreamManifest`` — static per-stream metadata (Milestone 2, D-0029).

A run bundles per-stream *samples* (list of :class:`Sample`) with
*static metadata* about the stream itself: its modality, nominal rate,
clock domain, and expected transport latency. The synthetic-truth
harness has always carried this shape as
:class:`~embodied_sync.streams.synthetic.SyntheticStreamSpec`; this
type lifts the same shape into the canonical data model so adapters
in Milestones 4–9 can produce identical metadata without importing
from ``streams/``.

Contract
--------
- ``rate_hz`` is ``None`` for irregular streams (events, markers).
- ``transport_latency_ns`` is the *nominal* transport latency the
  stream is expected to exhibit under clean conditions (not a
  measurement). Corruption profiles add noise on top of it; adapters
  reading real data may set it to ``0`` when unknown.
- ``clock_domain`` is a :class:`~embodied_sync.time.ClockDomain` value
  — the typed lift from the free-string ``Sample.source_clock_domain``
  field. Adapters that don't yet know the mapping should use
  :func:`~embodied_sync.time.resolve_clock_domain` on their free-string
  domain name.
- ``payload_dim`` is the length of a numeric payload vector (or
  ``None`` when payloads are scheme-specific — camera images, event
  markers).
"""

from __future__ import annotations

from dataclasses import dataclass

from embodied_sync.core.sample import Modality
from embodied_sync.time.clock_domain import ClockDomain

__all__ = ["StreamManifest"]


@dataclass(frozen=True, slots=True)
class StreamManifest:
    """Static per-stream description.

    Parallel to :class:`~embodied_sync.streams.synthetic.SyntheticStreamSpec`
    but living in ``core/`` so adapters and the alignment engine can
    consume it without depending on the synthetic-truth subpackage.
    """

    name: str
    modality: Modality
    rate_hz: float | None
    transport_latency_ns: int
    clock_domain: ClockDomain
    payload_dim: int | None = None
