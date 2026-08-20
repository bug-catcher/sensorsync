"""Clock-domain and cross-domain alignment types (Milestone 2, D-0029)."""

from embodied_sync.time.alignment import cross_domain_confidence_factor
from embodied_sync.time.clock_domain import (
    INITIAL_EPOCH,
    KNOWN_DOMAINS,
    ClockDomain,
    ClockEpoch,
    ClockEpochError,
    ClockEpochRegistry,
    ClockKind,
    LatencyEstimate,
    latency_estimate_to_dict,
    require_same_epoch,
    resolve_clock_domain,
    translate_ns,
)

__all__ = [
    "INITIAL_EPOCH",
    "KNOWN_DOMAINS",
    "ClockDomain",
    "ClockEpoch",
    "ClockEpochError",
    "ClockEpochRegistry",
    "ClockKind",
    "LatencyEstimate",
    "cross_domain_confidence_factor",
    "latency_estimate_to_dict",
    "require_same_epoch",
    "resolve_clock_domain",
    "translate_ns",
]
