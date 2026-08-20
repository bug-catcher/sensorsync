"""Tests for the linear-interpolation alignment policy (D-0025).

Covers:

* successful interpolation of numeric-vector payloads at intermediate targets;
* scalar payloads (single float);
* skew and confidence semantics for interpolated frames vs ZoH fallback;
* per-stream fallback to ZoH (with warning) when payloads are non-numeric;
* per-frame fallback to ZoH when no bracket exists (target at/past the last
  sample) or when an anchor is beyond tolerance;
* the ``interpolated`` quality flag is present on synthesized samples and
  absent on fallback picks;
* method-selection surface (``LINEAR_INTERPOLATION`` constant, CLI choice).
"""

from __future__ import annotations

import warnings

import pytest

from embodied_sync.align import LINEAR_INTERPOLATION, align_run
from embodied_sync.core.sample import (
    QUALITY_INTERPOLATED,
    Modality,
    Sample,
)


def _sample(
    *,
    stream: str = "robot_state",
    modality: Modality = Modality.ROBOT_STATE,
    seq: int = 0,
    acq: int = 0,
    payload: object = None,
    flags: frozenset[str] = frozenset(),
    domain: str = "host_mono",
) -> Sample:
    return Sample(
        stream_name=stream,
        modality=modality,
        sequence_id=seq,
        acquisition_time_ns=acq,
        receive_time_ns=acq,
        source_clock_domain=domain,
        payload=payload,
        quality_flags=flags,
    )


def _linear_stream(
    *,
    stream: str = "robot_state",
    modality: Modality = Modality.ROBOT_STATE,
    n: int = 10,
    step_ns: int = 10_000_000,
    values: list[list[float]] | None = None,
) -> list[Sample]:
    """Regular stream at ``step_ns`` intervals; ``values[i]`` is payload i."""
    if values is None:
        values = [[float(i)] for i in range(n)]
    assert len(values) == n
    return [
        _sample(stream=stream, modality=modality, seq=i, acq=i * step_ns, payload=values[i])
        for i in range(n)
    ]


def test_linear_interp_synthesizes_a_sample_at_the_target(
    recwarn: pytest.WarningsRecorder,
) -> None:
    run = {"robot_state": _linear_stream(n=11, step_ns=10_000_000)}
    aligned = align_run(run, target_rate_hz=200.0, method=LINEAR_INTERPOLATION)

    # 200Hz targets fall halfway between the 100Hz stream samples in the
    # interior of the window (5ms offset from each anchor).
    interior = [
        f for f in aligned.frames if f.target_time_ns % 10_000_000 == 5_000_000
    ]
    assert interior
    for frame in interior:
        md = frame.metadata["robot_state"]
        sample = frame.samples["robot_state"]
        assert sample is not None
        assert md.method == LINEAR_INTERPOLATION
        assert md.missing is False
        assert md.skew_ns == 0
        assert md.source_time_ns == frame.target_time_ns
        # confidence = 1 - min_gap / tolerance; halfway → min_gap = tolerance
        # → confidence 0.
        assert md.confidence == pytest.approx(0.0)
        # Interpolated payload: 0.5 * (i + (i+1)) = i + 0.5 for target
        # sitting halfway between i and i+1.
        expected = frame.target_time_ns / 10_000_000
        assert sample.payload == [pytest.approx(expected)]
        assert QUALITY_INTERPOLATED in sample.quality_flags
    # No warnings on a fully-numeric stream.
    assert not [w for w in recwarn.list if "linear_interp" in str(w.message)]


def test_linear_interp_on_grid_matches_the_anchor_exactly() -> None:
    run = {"robot_state": _linear_stream(n=11, step_ns=10_000_000)}
    aligned = align_run(run, target_rate_hz=100.0, method=LINEAR_INTERPOLATION)

    # Targets sit exactly on anchor times inside the window. The last
    # anchor has no right partner → ZoH fallback; every other target
    # interpolates with t=0 and confidence=1.
    last_anchor_ns = 10 * 10_000_000
    on_grid_frames = [f for f in aligned.frames if f.target_time_ns < last_anchor_ns]
    assert on_grid_frames
    for frame in on_grid_frames:
        md = frame.metadata["robot_state"]
        sample = frame.samples["robot_state"]
        assert sample is not None
        assert md.method == LINEAR_INTERPOLATION
        assert md.missing is False
        assert md.skew_ns == 0
        # Interpolated payload equals the left anchor's payload exactly.
        expected = frame.target_time_ns / 10_000_000
        assert sample.payload == [pytest.approx(expected)]
        assert QUALITY_INTERPOLATED in sample.quality_flags
        # min_gap = 0 on the anchor → confidence 1.
        assert md.confidence == pytest.approx(1.0)


def test_linear_interp_interpolates_multidimensional_vectors() -> None:
    values = [[float(i), float(2 * i), float(-i)] for i in range(10)]
    run = {"robot_state": _linear_stream(n=10, step_ns=10_000_000, values=values)}
    aligned = align_run(run, target_rate_hz=200.0, method=LINEAR_INTERPOLATION)

    halfway = [f for f in aligned.frames if f.target_time_ns % 10_000_000 == 5_000_000]
    assert halfway
    frame = halfway[0]
    sample = frame.samples["robot_state"]
    assert sample is not None
    i = frame.target_time_ns // 10_000_000
    assert sample.payload == [
        pytest.approx(i + 0.5),
        pytest.approx(2 * i + 1.0),
        pytest.approx(-(i + 0.5)),
    ]


def test_linear_interp_falls_back_to_zoh_for_non_numeric_payloads() -> None:
    # Camera-like payload: a dict, not a numeric vector.
    stream = [
        _sample(
            stream="cam",
            modality=Modality.CAMERA,
            seq=i,
            acq=i * 33_000_000,
            payload={"frame_index": i, "signature": [i]},
        )
        for i in range(6)
    ]
    run = {"cam": stream}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aligned = align_run(run, target_rate_hz=30.0, method=LINEAR_INTERPOLATION)

    matches = [w for w in caught if "linear_interp" in str(w.message)]
    assert len(matches) == 1
    assert "cam" in str(matches[0].message)

    # Every frame that produced a sample got the ZoH pick, and the sample
    # is the original — no interpolated flag.
    non_missing = [f for f in aligned.frames if not f.metadata["cam"].missing]
    assert non_missing
    for frame in non_missing:
        sample = frame.samples["cam"]
        assert sample is not None
        assert QUALITY_INTERPOLATED not in sample.quality_flags
        # ZoH fallback: skew is nonpositive; method still records the request.
        assert frame.metadata["cam"].skew_ns is not None
        assert frame.metadata["cam"].skew_ns <= 0


def test_linear_interp_target_equal_to_last_anchor_falls_back_to_zoh() -> None:
    """No right anchor to bracket the target → per-frame ZoH fallback."""
    run = {"robot_state": _linear_stream(n=11, step_ns=10_000_000)}
    aligned = align_run(run, target_rate_hz=100.0, method=LINEAR_INTERPOLATION)

    last_frame = aligned.frames[-1]
    assert last_frame.target_time_ns == 100_000_000

    sample = last_frame.samples["robot_state"]
    assert sample is not None
    # ZoH picked the last anchor exactly (skew=0), and the sample carries
    # no interpolated flag — the flag is the honest "did we interpolate?"
    # signal, and this frame didn't.
    assert QUALITY_INTERPOLATED not in sample.quality_flags
    md = last_frame.metadata["robot_state"]
    assert md.skew_ns == 0
    assert md.confidence == pytest.approx(1.0)


def test_linear_interp_per_frame_fallback_when_bracket_gap_exceeds_interval() -> None:
    """A bracket wider than the median inter-sample interval falls back to ZoH.

    Drop two consecutive middle samples so the bracket across the hole is
    3x the median, tripping the ``max_gap <= interp_max_gap`` gate.
    """
    step_ns = 100_000_000
    n = 6
    values = [[float(i)] for i in range(n)]
    stream = _linear_stream(n=n, step_ns=step_ns, values=values)
    # Samples were at 0, 100, 200, 300, 400, 500 ms; remove indices 2 & 3.
    del stream[2:4]
    run = {"robot_state": stream}
    # 5 Hz targets: 0, 200 ms, 400 ms. Target=200 ms sits inside the hole.
    aligned = align_run(run, target_rate_hz=5.0, method=LINEAR_INTERPOLATION)

    mid = [f for f in aligned.frames if f.target_time_ns == 200_000_000]
    assert mid, "expected a frame at 200 ms across the dropped window"
    md = mid[0].metadata["robot_state"]
    sample = mid[0].samples["robot_state"]
    # Fallback path: either ZoH picked something without the interpolated
    # flag, or ZoH itself flagged missing (stale beyond tolerance).
    if sample is not None:
        assert QUALITY_INTERPOLATED not in sample.quality_flags
    else:
        assert md.missing is True

    # Targets that land on surviving anchors interpolate normally.
    edge = [f for f in aligned.frames if f.target_time_ns == 0]
    assert edge and edge[0].samples["robot_state"] is not None
    assert QUALITY_INTERPOLATED in edge[0].samples["robot_state"].quality_flags


def test_linear_interp_metadata_method_stays_linear_interp_for_the_stream() -> None:
    run = {"robot_state": _linear_stream(n=11, step_ns=10_000_000)}
    aligned = align_run(run, target_rate_hz=200.0, method=LINEAR_INTERPOLATION)

    for frame in aligned.frames:
        assert frame.metadata["robot_state"].method == LINEAR_INTERPOLATION


def test_linear_interp_rejects_unknown_method_string() -> None:
    run = {"robot_state": _linear_stream(n=4, step_ns=10_000_000)}
    with pytest.raises(ValueError, match="unknown alignment method"):
        align_run(run, target_rate_hz=100.0, method="not_a_method")  # type: ignore[arg-type]


def test_linear_interp_flag_is_absent_when_stream_is_empty() -> None:
    run: dict[str, list[Sample]] = {"empty": []}
    aligned = align_run(run, target_rate_hz=10.0, method=LINEAR_INTERPOLATION)
    # An empty run has no frames — but the missing_count entry must exist.
    assert aligned.frames == []
    assert aligned.report.missing_count == {"empty": 0}


def test_scalar_numeric_payload_is_interpolable() -> None:
    stream = [
        _sample(seq=i, acq=i * 10_000_000, payload=float(i)) for i in range(6)
    ]
    aligned = align_run(
        {"robot_state": stream}, target_rate_hz=200.0, method=LINEAR_INTERPOLATION
    )
    halfway = [f for f in aligned.frames if f.target_time_ns % 10_000_000 == 5_000_000]
    assert halfway
    for frame in halfway:
        sample = frame.samples["robot_state"]
        assert sample is not None
        # Scalar payload becomes a one-element list after interpolation.
        assert isinstance(sample.payload, list)
        assert len(sample.payload) == 1


def test_boolean_payload_is_not_treated_as_numeric() -> None:
    stream = [
        _sample(seq=i, acq=i * 10_000_000, payload=[True, False]) for i in range(6)
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aligned = align_run(
            {"robot_state": stream}, target_rate_hz=100.0, method=LINEAR_INTERPOLATION
        )
    fallback_warnings = [w for w in caught if "linear_interp" in str(w.message)]
    assert fallback_warnings, "bool payloads must trigger the non-numeric fallback"
    # Fallback delivers the original samples (no interpolated flag).
    non_missing = [f for f in aligned.frames if not f.metadata["robot_state"].missing]
    for frame in non_missing:
        sample = frame.samples["robot_state"]
        assert sample is not None
        assert QUALITY_INTERPOLATED not in sample.quality_flags
