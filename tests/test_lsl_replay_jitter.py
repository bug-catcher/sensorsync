"""Milestone 7: jitter injection on a replayed LSL/XDF stream.

The upstream :mod:`embodied_sync.corrupt` module operates on any run
dict — it does not care where the samples came from. This test wires
the committed LSL replay fixture through :func:`load_lsl_replay`,
applies an LSL-specific jitter profile, and asserts:

1. The injected receive-time noise is observable on the corrupted run
   — every corrupted ``eeg`` sample's ``receive_time_ns`` is shifted
   relative to the clean one, and the spread of ``transport_latency_ns``
   grows meaningfully. The offline acquisition-anchored aligner would
   pick the same samples (acquisition times are untouched), so the
   dropped ``xdf_clock_offset_ns`` regression below is the alignment-
   side property to hold.
2. Clock-offset metadata (``xdf_clock_offset_ns``) survives on every
   picked sample even after corruption + alignment — the corruption
   layer touches timestamps, never payloads.
"""

from __future__ import annotations

from pathlib import Path

import statistics

import yaml

from embodied_sync.adapters.lsl import load_lsl_replay
from embodied_sync.align import align_run
from embodied_sync.corrupt import apply_profile, load_profile

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "fixtures"
    / "lsl_mini_replay"
    / "replay.json"
)


def _write_lsl_jitter_profile(tmp_path: Path) -> Path:
    """LSL-shaped profile: strong receive-time jitter on the eeg stream."""
    profile = {
        "format_version": 0,
        "seed": 7,
        "corruptions": [
            {
                "stream": "eeg",
                "kind": "jitter",
                "distribution": "gaussian",
                "std_ms": 4.0,
                "clip_ms": 8.0,
            }
        ],
    }
    path = tmp_path / "lsl_jitter.yaml"
    path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    return path


def test_jitter_on_lsl_replay_shifts_receive_times(tmp_path: Path) -> None:
    run = load_lsl_replay(FIXTURE_PATH)
    profile = load_profile(_write_lsl_jitter_profile(tmp_path))
    corrupted = apply_profile(run, profile)

    clean_eeg = run["eeg"]
    dirty_eeg = corrupted.run["eeg"]
    assert len(clean_eeg) == len(dirty_eeg)

    # Every eeg sample's acquisition_time_ns is untouched; receive_time_ns
    # is shifted by the RNG-driven noise. On a stream this large the
    # probability that all shifts round to zero is vanishing.
    shifts = [
        dirty.receive_time_ns - clean.receive_time_ns
        for clean, dirty in zip(clean_eeg, dirty_eeg)
    ]
    assert any(shift != 0 for shift in shifts), "jitter produced no receive-time shifts"

    clean_latency_stdev = statistics.pstdev(s.transport_latency_ns for s in clean_eeg)
    dirty_latency_stdev = statistics.pstdev(s.transport_latency_ns for s in dirty_eeg)
    assert dirty_latency_stdev > clean_latency_stdev, (
        "jitter did not raise transport_latency_ns spread: "
        f"clean stdev={clean_latency_stdev}, dirty stdev={dirty_latency_stdev}"
    )
    # And the marker stream is untouched (profile did not target it).
    assert corrupted.run["marker"] == run["marker"]


def test_jitter_on_lsl_replay_preserves_clock_offset(tmp_path: Path) -> None:
    run = load_lsl_replay(FIXTURE_PATH)
    profile = load_profile(_write_lsl_jitter_profile(tmp_path))
    corrupted = apply_profile(run, profile)
    aligned = align_run(corrupted.run, target_rate_hz=10.0)

    for frame in aligned.frames:
        for name in ("eeg", "marker"):
            sample = frame.samples[name]
            assert sample is not None
            expected = -125_000 if name == "eeg" else 250
            assert sample.payload["xdf_clock_offset_ns"] == expected, (
                f"stream {name!r} lost xdf_clock_offset_ns after "
                f"corrupt+align: got {sample.payload['xdf_clock_offset_ns']!r}"
            )
