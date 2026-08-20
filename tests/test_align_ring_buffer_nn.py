"""Milestone 3: deadline-aware nearest-neighbor picks on ``StreamRingBuffer``.

The base ring-buffer (D-0026, session 9) is ZoH-only; deadline-0 ZoH
is the safe default for a strictly-causal policy tick. Deadline-aware
NN is the *lookahead* variant: given a non-zero deadline, pick the
sample with the smallest ``|skew|`` — the sample may sit slightly in
the future of the target as long as its ``receive_time_ns`` fits in
the deadline window. This test module pins the eligibility rule, the
tolerance-gated miss, and the confidence-decay shape.
"""

from __future__ import annotations

import pytest

from embodied_sync.align.ring_buffer import StreamRingBuffer
from embodied_sync.core.sample import Modality, Sample


def _sample(seq: int, acq: int, recv: int | None = None) -> Sample:
    return Sample(
        stream_name="s",
        modality=Modality.OTHER,
        sequence_id=seq,
        acquisition_time_ns=acq,
        receive_time_ns=recv if recv is not None else acq,
        source_clock_domain="host_mono",
        payload=None,
    )


def test_nn_prefers_future_sample_when_closer() -> None:
    buf = StreamRingBuffer(capacity=8, tolerance_ns=1_000)
    # target=100: past sample at 90 is 10 away, future at 105 is 5 away.
    buf.push(_sample(0, 90, recv=95))
    buf.push(_sample(1, 105, recv=110))
    sample, meta = buf.get_nearest_neighbor(target_ns=100, deadline_ns=20)
    assert sample is not None
    assert sample.sequence_id == 1
    assert meta.skew_ns == 5
    assert not meta.missing


def test_nn_falls_back_to_past_when_future_exceeds_deadline() -> None:
    buf = StreamRingBuffer(capacity=8, tolerance_ns=1_000)
    buf.push(_sample(0, 90, recv=95))
    buf.push(_sample(1, 105, recv=200))  # future sample, but arrived too late
    sample, meta = buf.get_nearest_neighbor(target_ns=100, deadline_ns=20)
    assert sample is not None
    assert sample.sequence_id == 0
    assert meta.skew_ns == -10
    assert not meta.missing


def test_nn_missing_when_no_sample_eligible() -> None:
    buf = StreamRingBuffer(capacity=8, tolerance_ns=1_000)
    buf.push(_sample(0, 90, recv=200))  # arrived after deadline
    sample, meta = buf.get_nearest_neighbor(target_ns=100, deadline_ns=20)
    assert sample is None
    assert meta.missing
    assert meta.method == "nearest_neighbor"
    assert meta.source_time_ns is None


def test_nn_missing_when_best_skew_exceeds_tolerance() -> None:
    buf = StreamRingBuffer(capacity=8, tolerance_ns=100)
    # Best eligible sample is 200 ns away from target — over tolerance.
    buf.push(_sample(0, 300, recv=305))
    sample, meta = buf.get_nearest_neighbor(target_ns=100, deadline_ns=1_000)
    assert sample is None
    assert meta.missing
    assert meta.source_time_ns == 300
    assert meta.skew_ns == 200


def test_nn_negative_deadline_rejected() -> None:
    buf = StreamRingBuffer(capacity=4, tolerance_ns=100)
    with pytest.raises(ValueError):
        buf.get_nearest_neighbor(target_ns=0, deadline_ns=-1)


def test_nn_confidence_decays_with_skew() -> None:
    buf = StreamRingBuffer(capacity=8, tolerance_ns=1000)
    buf.push(_sample(0, 500, recv=505))
    # At target=0, |skew|=500, confidence = 1 - 500/1000 = 0.5.
    sample, meta = buf.get_nearest_neighbor(target_ns=0, deadline_ns=1_000)
    assert sample is not None
    assert meta.confidence == pytest.approx(0.5)
