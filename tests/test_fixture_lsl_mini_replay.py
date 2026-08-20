"""Committed LSL/XDF replay fixture + alignment regression.

Milestone 7 continuation: the committed :data:`FIXTURE_PATH` holds a
tiny two-stream LSL/XDF-shape replay (a 100 Hz ``eeg`` stream and a
~10 Hz irregular ``marker`` stream) with distinct
``xdf_clock_offset_ns`` metadata per stream. Loading via
:func:`load_lsl_replay` returns integer-ns samples with the offset
folded into ``payload["xdf_clock_offset_ns"]``. This test then aligns
the replay to 10 Hz and pins two properties:

1. Alignment succeeds across the two different-rate streams.
2. Every selected non-missing frame preserves the per-stream
   ``xdf_clock_offset_ns`` inside its payload — the adapter/aligner
   pair does not drop the clock-offset metadata on the way through.

Fixture regeneration is not automated: this file is small, diffable,
and hand-authored (unlike ``synth_mini``, which round-trips through
the generator). If the LSL replay schema ever changes, edit the JSON
directly.
"""

from __future__ import annotations

from pathlib import Path

from embodied_sync.adapters.lsl import load_lsl_replay
from embodied_sync.align import align_run

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "fixtures"
    / "lsl_mini_replay"
    / "replay.json"
)


def test_fixture_loads_two_different_rate_streams() -> None:
    run = load_lsl_replay(FIXTURE_PATH)
    assert set(run) == {"eeg", "marker"}
    assert len(run["eeg"]) == 31
    assert len(run["marker"]) == 4
    # eeg is 100 Hz (10 ms period); marker is ~10 Hz irregular.
    eeg_intervals = {
        run["eeg"][i + 1].acquisition_time_ns - run["eeg"][i].acquisition_time_ns
        for i in range(len(run["eeg"]) - 1)
    }
    assert eeg_intervals == {10_000_000}


def test_fixture_preserves_clock_offset_per_stream() -> None:
    run = load_lsl_replay(FIXTURE_PATH)
    # xdf_clock_offset_ns is folded into each sample's payload dict.
    for sample in run["eeg"]:
        assert sample.payload["xdf_clock_offset_ns"] == -125_000
    for sample in run["marker"]:
        assert sample.payload["xdf_clock_offset_ns"] == 250


def test_replay_aligned_at_10hz_preserves_clock_offset_metadata() -> None:
    run = load_lsl_replay(FIXTURE_PATH)
    aligned = align_run(run, target_rate_hz=10.0)

    # World-time grid at 10 Hz clipped to [max first_acq, min last_acq]
    # = [0, 300_000_000] → frames at 0, 100e6, 200e6, 300e6.
    assert [f.target_time_ns for f in aligned.frames] == [
        0,
        100_000_000,
        200_000_000,
        300_000_000,
    ]

    for frame in aligned.frames:
        for name in ("eeg", "marker"):
            sample = frame.samples[name]
            assert sample is not None, f"{name}@{frame.target_time_ns} unexpectedly missing"
            offset = sample.payload["xdf_clock_offset_ns"]
            expected = -125_000 if name == "eeg" else 250
            assert offset == expected, (
                f"stream {name!r} lost xdf_clock_offset_ns after alignment: "
                f"got {offset!r}, expected {expected!r}"
            )
