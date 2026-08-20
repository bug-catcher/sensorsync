"""Clock-drift corruption (D-0014).

Clock drift is a deterministic linear offset on ``receive_time_ns`` anchored
at the target stream's first acquisition timestamp. It preserves acquisition
timestamps and does not create dropped-sample ground truth.
"""

from __future__ import annotations

from embodied_sync.core.sample import Sample
from embodied_sync.corrupt import (
    ClockDriftCorruption,
    CorruptionProfile,
    FixedLatencyCorruption,
    apply_clock_drift,
    apply_profile,
)
from embodied_sync.streams.synthetic import generate_synthetic_run


def _run() -> dict[str, list[Sample]]:
    return generate_synthetic_run(duration_s=0.2, seed=0)


class TestClockDrift:
    def test_exact_offsets_injected_from_first_acquisition_anchor(self) -> None:
        run = _run()
        profile = CorruptionProfile(
            seed=7,
            corruptions=(ClockDriftCorruption(stream="robot_state", drift_ppb=1_000_000),),
        )
        result = apply_profile(run, profile)

        assert result.dropped == {}
        anchor_ns = run["robot_state"][0].acquisition_time_ns
        for original, drifted in zip(run["robot_state"], result.run["robot_state"]):
            elapsed_ns = original.acquisition_time_ns - anchor_ns
            expected_offset_ns = round(elapsed_ns * 1_000_000 / 1_000_000_000)
            assert drifted.receive_time_ns == original.receive_time_ns + expected_offset_ns
            assert drifted.acquisition_time_ns == original.acquisition_time_ns
            assert drifted.sequence_id == original.sequence_id
            assert drifted.payload == original.payload
            assert drifted.quality_flags == original.quality_flags

        for name in run:
            if name != "robot_state":
                assert result.run[name] == run[name]
        assert run == _run(), "apply_profile must not mutate the input run"

    def test_negative_drift_makes_later_samples_earlier(self) -> None:
        samples = _run()["robot_state"][:4]
        drifted = apply_clock_drift(samples, drift_ppb=-1_000_000)
        offsets = [
            changed.receive_time_ns - original.receive_time_ns
            for original, changed in zip(samples, drifted)
        ]
        assert offsets == [0, -4_000, -8_000, -12_000]

    def test_drift_stacks_after_existing_receive_time_offsets(self) -> None:
        samples = _run()["actions"][:3]
        profile = CorruptionProfile(
            seed=0,
            corruptions=(
                FixedLatencyCorruption(stream="actions", offset_ns=10_000),
                ClockDriftCorruption(stream="actions", drift_ppb=1_000_000),
            ),
        )
        result = apply_profile({"actions": samples}, profile)

        assert [
            changed.receive_time_ns - original.receive_time_ns
            for original, changed in zip(samples, result.run["actions"])
        ] == [10_000, 110_000]
