"""Milestone 2: typed ClockDomain / LatencyEstimate + cross-domain helpers."""

from __future__ import annotations

import warnings

import pytest

from embodied_sync.time import (
    KNOWN_DOMAINS,
    ClockDomain,
    ClockKind,
    LatencyEstimate,
    cross_domain_confidence_factor,
    resolve_clock_domain,
    translate_ns,
)


def test_known_domains_cover_builtin_adapter_strings() -> None:
    for name in (
        "host_mono",
        "host_wall",
        "lsl",
        "lsl_local_clock",
        "ros2_steady",
        "mcap_publish_time",
        "mcap_log_time",
        "unknown",
    ):
        assert name in KNOWN_DOMAINS
        assert KNOWN_DOMAINS[name].name == name


def test_resolve_hit_returns_the_registered_value() -> None:
    domain = resolve_clock_domain("host_mono")
    assert domain is KNOWN_DOMAINS["host_mono"]
    assert domain.kind is ClockKind.MONOTONIC


def test_resolve_miss_warns_once_and_returns_unknown_kind() -> None:
    from embodied_sync.time import clock_domain as cd_mod

    # Reset the warned set to make this test independent of any earlier
    # resolves in the same process.
    cd_mod._WARNED_UNKNOWN.discard("brand_new_clock")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = resolve_clock_domain("brand_new_clock")
        second = resolve_clock_domain("brand_new_clock")
    assert first.kind is ClockKind.UNKNOWN
    assert first.name == "brand_new_clock"
    assert first == second
    # Second call must not warn again (one-shot policy).
    assert sum(1 for w in caught if "brand_new_clock" in str(w.message)) == 1


def test_latency_estimate_rejects_float_offsets() -> None:
    src = ClockDomain("a", ClockKind.MONOTONIC)
    dst = ClockDomain("b", ClockKind.MONOTONIC)
    with pytest.raises(TypeError):
        LatencyEstimate(source=src, target=dst, offset_ns=1.5)  # type: ignore[arg-type]


def test_latency_estimate_rejects_negative_variance() -> None:
    src = ClockDomain("a", ClockKind.MONOTONIC)
    dst = ClockDomain("b", ClockKind.MONOTONIC)
    with pytest.raises(ValueError):
        LatencyEstimate(source=src, target=dst, offset_ns=0, variance_ns=-1)


def test_translate_ns_applies_offset_and_drift() -> None:
    src = ClockDomain("a", ClockKind.MONOTONIC)
    dst = ClockDomain("b", ClockKind.MONOTONIC)
    # 100 ppb source→target: 1e9 ns elapsed → +100 ns
    est = LatencyEstimate(
        source=src, target=dst, offset_ns=42, drift_ppb=100, anchor_time_ns=0
    )
    assert translate_ns(0, est) == 42
    assert translate_ns(1_000_000_000, est) == 42 + 1_000_000_000 + 100
    assert translate_ns(-1_000_000_000, est) == 42 - 1_000_000_000 - 100


def test_translate_ns_zero_drift_is_pure_offset() -> None:
    src = ClockDomain("a", ClockKind.MONOTONIC)
    dst = ClockDomain("b", ClockKind.MONOTONIC)
    est = LatencyEstimate(source=src, target=dst, offset_ns=-500)
    assert translate_ns(1000, est) == 500
    assert translate_ns(0, est) == -500


def test_cross_domain_confidence_factor_bounds() -> None:
    src = ClockDomain("a", ClockKind.MONOTONIC)
    dst = ClockDomain("b", ClockKind.MONOTONIC)
    zero_variance = LatencyEstimate(source=src, target=dst, offset_ns=0, variance_ns=0)
    assert cross_domain_confidence_factor(zero_variance, tolerance_ns=1000) == 1.0

    equal_variance = LatencyEstimate(
        source=src, target=dst, offset_ns=0, variance_ns=1000
    )
    assert cross_domain_confidence_factor(equal_variance, tolerance_ns=1000) == 0.5

    huge_variance = LatencyEstimate(
        source=src, target=dst, offset_ns=0, variance_ns=1_000_000
    )
    assert cross_domain_confidence_factor(huge_variance, tolerance_ns=1000) < 0.01

    # tolerance <= 0 rejects everything.
    assert cross_domain_confidence_factor(zero_variance, tolerance_ns=0) == 0.0
