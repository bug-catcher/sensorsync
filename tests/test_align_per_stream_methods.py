"""Milestone 3: per-stream method selection in ``align_run``.

The single-string ``method`` shape stays supported; passing a
``dict[str, method | AlignmentPolicy]`` picks per stream. This test
module pins:

- string-per-stream selection (mixed NN and ZoH on one run);
- :class:`AlignmentPolicy`-per-stream selection (also overrides
  tolerance);
- unknown-method and unknown-stream errors fail loudly.

Uses the synth mini fixture so the per-stream selection composes with
the real grid + interval math.
"""

from __future__ import annotations

import pytest

from embodied_sync.align import align_run
from embodied_sync.align.engine import LINEAR_INTERPOLATION, NEAREST_NEIGHBOR, ZERO_ORDER_HOLD
from embodied_sync.core import AlignmentPolicy
from embodied_sync.streams.synthetic import generate_synthetic_run


def test_string_method_still_applies_to_every_stream() -> None:
    run = generate_synthetic_run(duration_s=1.0, seed=0)
    aligned = align_run(run, target_rate_hz=10.0, method="zoh")
    for frame in aligned.frames:
        for meta in frame.metadata.values():
            if not meta.missing:
                assert meta.method == "zoh"


def test_string_per_stream_dict_selects_per_stream() -> None:
    run = generate_synthetic_run(duration_s=1.0, seed=0)
    methods: dict[str, str] = {
        "cam_front": NEAREST_NEIGHBOR,
        "cam_wrist": ZERO_ORDER_HOLD,
        "robot_state": ZERO_ORDER_HOLD,
        "tactile": NEAREST_NEIGHBOR,
        "audio": ZERO_ORDER_HOLD,
        "actions": NEAREST_NEIGHBOR,
        "events": NEAREST_NEIGHBOR,
    }
    aligned = align_run(run, target_rate_hz=10.0, method=methods)

    seen: dict[str, set[str]] = {name: set() for name in methods}
    for frame in aligned.frames:
        for name, meta in frame.metadata.items():
            if not meta.missing:
                seen[name].add(meta.method)
    for name, expected in methods.items():
        assert seen[name] == {expected} or seen[name] == set(), (
            f"stream {name!r}: expected only {expected}, saw {seen[name]}"
        )


def test_alignment_policy_dict_overrides_tolerance() -> None:
    from embodied_sync.core.sample import Modality, Sample

    def _s(name: str, seq: int, acq: int) -> Sample:
        return Sample(
            stream_name=name,
            modality=Modality.ROBOT_STATE,
            sequence_id=seq,
            acquisition_time_ns=acq,
            receive_time_ns=acq,
            source_clock_domain="host_mono",
            payload=[float(seq)],
        )

    # Hand-built run: "a" step (30_000_007 ns) is chosen coprime with
    # the 100 ms target period so no sample lands on the grid; ZoH picks
    # are always a few ms stale. Default tolerance for the 30 ms period
    # is ~15 ms; the 1 µs override forces every "a" pick to miss.
    step = 30_000_007
    # Offset "a" by 5 ns so its first sample doesn't land on the world-
    # time origin either.
    run = {
        "a": [_s("a", i, 5 + i * step) for i in range(35)],
        "b": [_s("b", i, i * 30_000_000) for i in range(35)],
    }

    tight = align_run(
        run,
        target_rate_hz=10.0,
        method={"a": AlignmentPolicy(method="zoh", tolerance_ns=1_000)},
    )
    permissive = align_run(
        run,
        target_rate_hz=10.0,
        method={"a": AlignmentPolicy(method="zoh", tolerance_ns=100_000_000)},
    )

    assert len(tight.frames) == len(permissive.frames)
    total = len(tight.frames)
    assert total > 0
    # Tight override → every "a" frame missing; permissive → none missing.
    assert tight.report.missing_count["a"] == total
    assert permissive.report.missing_count["a"] == 0
    # "b" defaulted to NN and should be non-missing regardless of "a".
    assert tight.report.missing_count["b"] == 0
    assert permissive.report.missing_count["b"] == 0


def test_linear_interp_per_stream_only_where_requested() -> None:
    run = generate_synthetic_run(duration_s=1.0, seed=0)
    methods: dict[str, str] = {
        "robot_state": LINEAR_INTERPOLATION,
    }
    aligned = align_run(run, target_rate_hz=10.0, method=methods)
    robot_methods = {
        m.method for m in (f.metadata["robot_state"] for f in aligned.frames) if not m.missing
    }
    assert robot_methods == {LINEAR_INTERPOLATION}
    cam_methods = {
        m.method for m in (f.metadata["cam_front"] for f in aligned.frames) if not m.missing
    }
    # cam_front defaulted to nearest_neighbor.
    assert cam_methods == {NEAREST_NEIGHBOR}


def test_unknown_method_string_rejected() -> None:
    run = generate_synthetic_run(duration_s=0.2, seed=0)
    with pytest.raises(ValueError, match="unknown alignment method"):
        align_run(run, target_rate_hz=10.0, method="lstm")  # type: ignore[arg-type]


def test_unknown_stream_in_method_dict_rejected() -> None:
    run = generate_synthetic_run(duration_s=0.2, seed=0)
    with pytest.raises(ValueError, match="unknown streams"):
        align_run(
            run,
            target_rate_hz=10.0,
            method={"not_a_stream": "zoh"},  # type: ignore[arg-type]
        )


def test_per_stream_dict_omits_streams_default_to_nearest_neighbor() -> None:
    run = generate_synthetic_run(duration_s=1.0, seed=0)
    aligned = align_run(
        run,
        target_rate_hz=10.0,
        method={"cam_front": ZERO_ORDER_HOLD},
    )
    seen: dict[str, set[str]] = {name: set() for name in run}
    for frame in aligned.frames:
        for name, meta in frame.metadata.items():
            if not meta.missing:
                seen[name].add(meta.method)
    assert seen["cam_front"] == {ZERO_ORDER_HOLD}
    for name in run:
        if name == "cam_front":
            continue
        assert seen[name] <= {NEAREST_NEIGHBOR}
