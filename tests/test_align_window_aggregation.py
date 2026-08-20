"""Milestone 3: window aggregation helper."""

from __future__ import annotations

import pytest

from embodied_sync.align import aggregate_window
from embodied_sync.core.sample import Modality, Sample


def _s(seq: int, acq: int, payload: object) -> Sample:
    return Sample(
        stream_name="s",
        modality=Modality.ROBOT_STATE,
        sequence_id=seq,
        acquisition_time_ns=acq,
        receive_time_ns=acq,
        source_clock_domain="host_mono",
        payload=payload,
    )


def test_mean_of_two_samples_in_window() -> None:
    samples = [
        _s(0, 100, [1.0, 2.0]),
        _s(1, 200, [3.0, 4.0]),
        _s(2, 400, [10.0, 20.0]),
    ]
    payload, count = aggregate_window(
        samples, target_ns=250, window_ns=200, reducer="mean"
    )
    assert count == 2
    assert payload == [2.0, 3.0]


def test_median_of_three_samples() -> None:
    samples = [_s(i, i * 10, [float(i)]) for i in range(5)]
    payload, count = aggregate_window(
        samples, target_ns=40, window_ns=40, reducer="median"
    )
    assert count == 5
    assert payload == [2.0]


def test_last_reducer_returns_newest_payload() -> None:
    samples = [_s(i, i * 10, [float(i)]) for i in range(5)]
    payload, count = aggregate_window(
        samples, target_ns=40, window_ns=1000, reducer="last"
    )
    assert count == 5
    assert payload == [4.0]


def test_no_eligible_samples_returns_none() -> None:
    samples = [_s(0, 100, [1.0])]
    payload, count = aggregate_window(
        samples, target_ns=1000, window_ns=100, reducer="mean"
    )
    assert payload is None
    assert count == 0


def test_scalar_payload_treated_as_singleton() -> None:
    samples = [_s(0, 100, 5.0), _s(1, 200, 15.0)]
    payload, count = aggregate_window(
        samples, target_ns=250, window_ns=200, reducer="mean"
    )
    assert payload == [10.0]
    assert count == 2


def test_non_numeric_samples_skipped() -> None:
    samples = [
        _s(0, 100, {"note": "hello"}),  # non-numeric — skipped
        _s(1, 200, [3.0]),
        _s(2, 300, [7.0]),
    ]
    payload, count = aggregate_window(
        samples, target_ns=350, window_ns=300, reducer="mean"
    )
    assert count == 2
    assert payload == [5.0]


def test_heterogeneous_dim_dropped() -> None:
    samples = [
        _s(0, 100, [1.0, 2.0]),  # dim 2 — establishes the shape
        _s(1, 200, [3.0]),  # dim 1 — dropped
        _s(2, 300, [5.0, 6.0]),
    ]
    payload, count = aggregate_window(
        samples, target_ns=350, window_ns=300, reducer="mean"
    )
    assert count == 2
    assert payload == [3.0, 4.0]


def test_zero_window_rejected() -> None:
    with pytest.raises(ValueError):
        aggregate_window([], target_ns=0, window_ns=0)


def test_unknown_reducer_rejected() -> None:
    with pytest.raises(ValueError):
        aggregate_window(
            [_s(0, 0, [1.0])], target_ns=0, window_ns=100, reducer="max"  # type: ignore[arg-type]
        )
