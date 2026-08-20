"""Kitchen-sink corruption regression on SurgSync-shaped streams.

The committed SurgSync mini fixture is only 200 ms long, while
``configs/corrupt_kitchen_sink.yaml`` was authored for the 2 s synthetic
harness. This regression keeps the committed SurgSync stream names,
modalities, clock domains, and cadences, extends those regular streams in
memory to the profile envelope, and remaps only the profile's stream targets.

That pins ``corrupt.apply_profile`` to the run's actual stream mapping and
guards against hidden assumptions about the synthetic harness's stream count
or names.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from embodied_sync.adapters.surg_sync import load_surg_sync_run
from embodied_sync.core.sample import (
    QUALITY_DUPLICATE,
    QUALITY_GAP_BEFORE,
    QUALITY_NON_MONOTONIC,
    Sample,
)
from embodied_sync.corrupt.apply import CorruptionResult, apply_profile
from embodied_sync.corrupt.profile import CorruptionProfile, load_profile

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "fixtures"
    / "surg_sync_mini"
    / "run.json"
)
PROFILE_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "corrupt_kitchen_sink.yaml"
)
PROFILE_ENVELOPE_NS = 2_000_000_000
STREAM_MAP = {
    "cam_wrist": "endoscope",
    "cam_front": "external_camera",
    "robot_state": "kinematics",
    "tactile": "force_sensor",
    "audio": "phase_events",
}


def _extend_regular_stream(samples: list[Sample], duration_ns: int) -> list[Sample]:
    period_ns = samples[1].acquisition_time_ns - samples[0].acquisition_time_ns
    receive_latency_ns = samples[0].receive_time_ns - samples[0].acquisition_time_ns
    extended: list[Sample] = []
    sequence_id = 0
    while sequence_id * period_ns <= duration_ns:
        acquisition_time_ns = sequence_id * period_ns
        extended.append(
            replace(
                samples[sequence_id % len(samples)],
                sequence_id=sequence_id,
                acquisition_time_ns=acquisition_time_ns,
                receive_time_ns=acquisition_time_ns + receive_latency_ns,
                quality_flags=frozenset(),
            )
        )
        sequence_id += 1
    return extended


def _load_surg_sync_shape_run() -> dict[str, list[Sample]]:
    run = load_surg_sync_run(FIXTURE_PATH)
    return {
        name: (
            samples
            if name == "phase_events"
            else _extend_regular_stream(samples, PROFILE_ENVELOPE_NS)
        )
        for name, samples in run.items()
    }


def _load_surg_sync_kitchen_sink_profile() -> CorruptionProfile:
    profile = load_profile(PROFILE_PATH)
    return replace(
        profile,
        corruptions=tuple(
            replace(corruption, stream=STREAM_MAP[corruption.stream])
            for corruption in profile.corruptions
        ),
    )


@pytest.fixture(scope="module")
def pipeline() -> dict[str, object]:
    clean = _load_surg_sync_shape_run()
    result = apply_profile(clean, _load_surg_sync_kitchen_sink_profile())
    return {"clean": clean, "result": result, "corrupted": result.run}


def test_surg_sync_shape_uses_only_surgical_stream_names(
    pipeline: dict[str, object],
) -> None:
    """The kitchen-sink kinds land on five SurgSync streams, not synth names."""
    corrupted = pipeline["corrupted"]
    assert isinstance(corrupted, dict)
    assert set(corrupted) == {
        "endoscope",
        "external_camera",
        "kinematics",
        "force_sensor",
        "phase_events",
    }


def test_receive_time_perturbations_land_on_surg_sync_streams(
    pipeline: dict[str, object],
) -> None:
    """Latency, jitter, clock drift, burst stall, and non-monotonic shift receives."""
    clean = pipeline["clean"]
    corrupted = pipeline["corrupted"]
    assert isinstance(clean, dict)
    assert isinstance(corrupted, dict)

    endoscope_by_seq = {s.sequence_id: s for s in clean["endoscope"]}
    endoscope_offsets = [
        sample.receive_time_ns - endoscope_by_seq[sample.sequence_id].receive_time_ns
        for sample in corrupted["endoscope"]
    ]
    assert min(endoscope_offsets) >= 15_000_000
    assert max(endoscope_offsets) > 15_000_000

    external_by_seq = {s.sequence_id: s for s in clean["external_camera"]}
    assert any(
        sample.receive_time_ns != external_by_seq[sample.sequence_id].receive_time_ns
        for sample in corrupted["external_camera"]
        if sample.sequence_id in external_by_seq
    )

    kinematics_by_seq = {s.sequence_id: s for s in clean["kinematics"]}
    late_kinematics = corrupted["kinematics"][-1]
    assert (
        late_kinematics.receive_time_ns
        != kinematics_by_seq[late_kinematics.sequence_id].receive_time_ns
    )

    assert any(
        QUALITY_NON_MONOTONIC in sample.quality_flags
        for sample in corrupted["phase_events"]
    )


def test_dropped_ground_truth_lands_on_surg_sync_streams(
    pipeline: dict[str, object],
) -> None:
    """Random frame drops and interval drops record surgical stream names."""
    clean = pipeline["clean"]
    result = pipeline["result"]
    corrupted = pipeline["corrupted"]
    assert isinstance(clean, dict)
    assert isinstance(result, CorruptionResult)
    assert isinstance(corrupted, dict)

    dropped = result.dropped
    assert len(dropped["external_camera"]) == 2
    assert len(corrupted["external_camera"]) == len(clean["external_camera"]) - 2

    assert [sample.sequence_id for sample in dropped["kinematics"]] == [10, 11, 12, 13]
    assert len(corrupted["kinematics"]) == len(clean["kinematics"]) - 4


def test_quality_flags_land_on_surg_sync_streams(pipeline: dict[str, object]) -> None:
    """Duplicate, gap-before, and receive-order flags survive SurgSync shape."""
    corrupted = pipeline["corrupted"]
    assert isinstance(corrupted, dict)

    duplicate_sequences = [
        sample.sequence_id
        for sample in corrupted["force_sensor"]
        if QUALITY_DUPLICATE in sample.quality_flags
    ]
    assert duplicate_sequences == [14, 23, 38]

    gap_sequences = [
        sample.sequence_id
        for sample in corrupted["kinematics"]
        if QUALITY_GAP_BEFORE in sample.quality_flags
    ]
    assert gap_sequences == [14]

    non_monotonic_sequences = [
        sample.sequence_id
        for sample in corrupted["phase_events"]
        if QUALITY_NON_MONOTONIC in sample.quality_flags
    ]
    assert non_monotonic_sequences == [3]
