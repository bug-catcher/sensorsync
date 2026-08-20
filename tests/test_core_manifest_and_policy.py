"""Milestone 2: canonical StreamManifest and AlignmentPolicy types."""

from __future__ import annotations

import pytest

from embodied_sync.core import AlignmentPolicy, Modality, StreamManifest
from embodied_sync.time import KNOWN_DOMAINS, ClockDomain, ClockKind


def test_stream_manifest_holds_typed_clock_domain() -> None:
    manifest = StreamManifest(
        name="cam_front",
        modality=Modality.CAMERA,
        rate_hz=30.0,
        transport_latency_ns=12_000_000,
        clock_domain=KNOWN_DOMAINS["host_mono"],
    )
    assert manifest.name == "cam_front"
    assert manifest.clock_domain.kind is ClockKind.MONOTONIC


def test_stream_manifest_irregular_rate_is_none() -> None:
    manifest = StreamManifest(
        name="events",
        modality=Modality.EVENT,
        rate_hz=None,
        transport_latency_ns=100_000,
        clock_domain=ClockDomain("custom", ClockKind.UNKNOWN),
    )
    assert manifest.rate_hz is None
    assert manifest.payload_dim is None


def test_alignment_policy_defaults() -> None:
    policy = AlignmentPolicy()
    assert policy.method == "nearest_neighbor"
    assert policy.tolerance_ns is None
    assert policy.deadline_ns == 0
    assert policy.clock_domain is None


def test_alignment_policy_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        AlignmentPolicy(method="lstm")


def test_alignment_policy_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError):
        AlignmentPolicy(tolerance_ns=-1)


def test_alignment_policy_rejects_negative_deadline() -> None:
    with pytest.raises(ValueError):
        AlignmentPolicy(deadline_ns=-1)


def test_alignment_policy_carries_clock_domain() -> None:
    policy = AlignmentPolicy(
        method="zoh",
        tolerance_ns=5_000_000,
        deadline_ns=1_000_000,
        clock_domain=KNOWN_DOMAINS["lsl_local_clock"],
    )
    assert policy.method == "zoh"
    assert policy.tolerance_ns == 5_000_000
    assert policy.deadline_ns == 1_000_000
    assert policy.clock_domain is KNOWN_DOMAINS["lsl_local_clock"]
