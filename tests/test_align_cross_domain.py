"""Milestone 3: cross-domain confidence lowering in ``align_run``.

A single ``clock_map`` argument threads
:class:`~embodied_sync.time.LatencyEstimate` values into the aligner:

- Source acquisition timestamps are translated into the target domain
  before scoring, so a stream offset in its own domain still aligns
  against the world-time grid.
- Confidence is multiplied by the domain factor (a function of the
  mapping's ``variance_ns`` and the aligner's tolerance) — the same
  ``skew`` in a high-variance mapping is *less trusted* than in a
  zero-variance one.
- Skew is measured in the target domain, so downstream tooling can
  compare across streams without knowing which mapping applied.
"""

from __future__ import annotations

import pytest

from embodied_sync.align import align_run
from embodied_sync.core.sample import Modality, Sample
from embodied_sync.time import KNOWN_DOMAINS, LatencyEstimate


def _s(name: str, seq: int, acq: int, domain: str = "host_mono") -> Sample:
    return Sample(
        stream_name=name,
        modality=Modality.ROBOT_STATE,
        sequence_id=seq,
        acquisition_time_ns=acq,
        receive_time_ns=acq,
        source_clock_domain=domain,
        payload=[float(seq)],
    )


def _basic_run() -> dict[str, list[Sample]]:
    return {
        "a": [_s("a", i, i * 10_000_000) for i in range(35)],
        "b": [_s("b", i, i * 10_000_000, domain="lsl") for i in range(35)],
    }


def test_clock_map_shifts_source_times_into_target_domain() -> None:
    run = _basic_run()
    # Stream "b" is 50 ms behind the world clock; the mapping recovers
    # the world-time.
    lsl = KNOWN_DOMAINS["lsl"]
    host = KNOWN_DOMAINS["host_mono"]
    est = LatencyEstimate(
        source=lsl, target=host, offset_ns=50_000_000, variance_ns=0
    )
    aligned = align_run(
        run,
        target_rate_hz=10.0,
        clock_map={"b": est},
    )
    assert aligned.frames
    for frame in aligned.frames:
        meta_b = frame.metadata["b"]
        if meta_b.missing:
            continue
        # Source time is now expressed in the target domain — the pick
        # for stream "b" at target T lines up with T minus a small skew.
        assert meta_b.source_time_ns is not None
        assert abs(meta_b.source_time_ns - frame.target_time_ns) <= 5_000_000


def test_high_variance_mapping_lowers_confidence() -> None:
    run = _basic_run()
    domain = KNOWN_DOMAINS["host_mono"]
    # Same identity mapping, but variance_ns = tolerance / 2 → factor = 2/3.
    zero_variance = LatencyEstimate(
        source=domain, target=domain, offset_ns=0, variance_ns=0
    )
    high_variance = LatencyEstimate(
        source=domain, target=domain, offset_ns=0, variance_ns=5_000_000
    )
    baseline = align_run(
        run,
        target_rate_hz=10.0,
        clock_map={"a": zero_variance},
    )
    lowered = align_run(
        run,
        target_rate_hz=10.0,
        clock_map={"a": high_variance},
    )
    assert len(baseline.frames) == len(lowered.frames) > 0
    for base_frame, low_frame in zip(baseline.frames, lowered.frames):
        base_a = base_frame.metadata["a"]
        low_a = low_frame.metadata["a"]
        if base_a.missing or low_a.missing:
            continue
        assert low_a.confidence < base_a.confidence
        # Skew is a domain-invariant integer — identity mapping so it
        # equals the un-mapped skew.
        assert base_a.skew_ns == low_a.skew_ns


def test_clock_map_unknown_stream_rejected() -> None:
    run = _basic_run()
    domain = KNOWN_DOMAINS["host_mono"]
    est = LatencyEstimate(source=domain, target=domain, offset_ns=0)
    with pytest.raises(ValueError, match="clock_map references unknown streams"):
        align_run(
            run,
            target_rate_hz=10.0,
            clock_map={"nope": est},
        )


def test_streams_without_mapping_pass_through_unchanged() -> None:
    run = _basic_run()
    domain = KNOWN_DOMAINS["host_mono"]
    aligned_none = align_run(run, target_rate_hz=10.0)
    aligned_partial = align_run(
        run,
        target_rate_hz=10.0,
        clock_map={"a": LatencyEstimate(
            source=domain, target=domain, offset_ns=0, variance_ns=0
        )},
    )
    # Stream "b" has no mapping — its metadata should match the no-map run.
    for f1, f2 in zip(aligned_none.frames, aligned_partial.frames):
        assert f1.metadata["b"] == f2.metadata["b"]
