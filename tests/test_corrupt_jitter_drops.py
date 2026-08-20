"""Jitter + dropped-frames corruptions (NEXT_TASKS #4, D-0010).

Jitter: deterministic per-profile-seed gaussian noise on ``receive_time_ns``
only, clipped when the profile says so. Drops: samples are removed, survivors
keep their original ``sequence_id`` (the observable symptom), the first
survivor after each removed block carries ``gap_before``, and the exact
removed samples come back as ground truth in ``CorruptionResult.dropped``.
"""

from __future__ import annotations

import pytest

from embodied_sync.core.sample import QUALITY_GAP_BEFORE
from embodied_sync.corrupt import (
    CorruptionProfile,
    DroppedFramesCorruption,
    FixedLatencyCorruption,
    JitterCorruption,
    apply_profile,
)
from embodied_sync.streams.synthetic import generate_synthetic_run

SEED = 99


def _run() -> dict:
    return generate_synthetic_run(duration_s=1.0, seed=0)


def _jitter_profile(seed: int, clip_ns: int | None = 30_000_000) -> CorruptionProfile:
    return CorruptionProfile(
        seed=seed,
        corruptions=(
            JitterCorruption(
                stream="cam_front", distribution="gaussian", std_ns=8_000_000, clip_ns=clip_ns
            ),
        ),
    )


class TestJitter:
    def test_same_profile_seed_identical_output(self) -> None:
        run = _run()
        a = apply_profile(run, _jitter_profile(SEED))
        b = apply_profile(run, _jitter_profile(SEED))
        assert a.run == b.run

    def test_different_profile_seed_changes_noise(self) -> None:
        run = _run()
        a = apply_profile(run, _jitter_profile(SEED))
        b = apply_profile(run, _jitter_profile(SEED + 1))
        assert [s.receive_time_ns for s in a.run["cam_front"]] != [
            s.receive_time_ns for s in b.run["cam_front"]
        ]

    def test_only_receive_time_changes_and_noise_is_nonzero(self) -> None:
        run = _run()
        result = apply_profile(run, _jitter_profile(SEED))
        offsets = []
        for original, jittered in zip(run["cam_front"], result.run["cam_front"]):
            assert jittered.acquisition_time_ns == original.acquisition_time_ns
            assert jittered.sequence_id == original.sequence_id
            assert jittered.payload == original.payload
            assert jittered.quality_flags == original.quality_flags
            offsets.append(jittered.receive_time_ns - original.receive_time_ns)
        assert any(offset != 0 for offset in offsets), "jitter must actually inject noise"
        for name in run:
            if name != "cam_front":
                assert result.run[name] == run[name]

    def test_clip_bounds_respected_exactly(self) -> None:
        run = _run()
        clip_ns = 1_000_000  # clip far below std so clipping certainly kicks in
        profile = _jitter_profile(SEED, clip_ns=clip_ns)
        result = apply_profile(run, profile)
        offsets = [
            jittered.receive_time_ns - original.receive_time_ns
            for original, jittered in zip(run["cam_front"], result.run["cam_front"])
        ]
        assert all(-clip_ns <= offset <= clip_ns for offset in offsets)
        assert clip_ns in offsets or -clip_ns in offsets, (
            "with std >> clip, some offsets must sit exactly on the clip bound"
        )


class TestDroppedFrames:
    def _profile(self, seed: int, probability: float = 0.1) -> CorruptionProfile:
        return CorruptionProfile(
            seed=seed,
            corruptions=(
                DroppedFramesCorruption(stream="robot_state", probability=probability),
            ),
        )

    def test_same_profile_seed_identical_output(self) -> None:
        run = _run()
        a = apply_profile(run, self._profile(SEED))
        b = apply_profile(run, self._profile(SEED))
        assert a.run == b.run
        assert a.dropped == b.dropped

    def test_ground_truth_records_exactly_what_was_removed(self) -> None:
        run = _run()
        result = apply_profile(run, self._profile(SEED))
        survivors = result.run["robot_state"]
        removed = result.dropped["robot_state"]
        assert removed, "with p=0.1 over 250 samples the seed must drop something"
        assert len(survivors) + len(removed) == len(run["robot_state"])
        # Removed samples are the original samples, untouched, in order.
        removed_ids = [s.sequence_id for s in removed]
        assert removed_ids == sorted(removed_ids)
        by_id = {s.sequence_id: s for s in run["robot_state"]}
        for s in removed:
            assert s == by_id[s.sequence_id]

    def test_survivors_keep_sequence_ids_and_gap_flags_mark_removed_blocks(self) -> None:
        run = _run()
        result = apply_profile(run, self._profile(SEED))
        survivors = result.run["robot_state"]
        previous_id = -1
        for survivor in survivors:
            expected_gap = survivor.sequence_id > previous_id + 1
            assert (QUALITY_GAP_BEFORE in survivor.quality_flags) == expected_gap, (
                f"sequence_id {survivor.sequence_id} after {previous_id}: "
                f"gap_before must mark exactly the survivors following a removed block"
            )
            previous_id = survivor.sequence_id
        assert any(QUALITY_GAP_BEFORE in s.quality_flags for s in survivors)

    @pytest.mark.parametrize(("probability", "expected_survivors"), [(0.0, 250), (1.0, 0)])
    def test_probability_extremes(self, probability: float, expected_survivors: int) -> None:
        run = _run()
        result = apply_profile(run, self._profile(SEED, probability))
        assert len(result.run["robot_state"]) == expected_survivors
        assert len(result.dropped["robot_state"]) == 250 - expected_survivors


class TestFullExampleStyleProfile:
    def test_stacked_corruptions_compose_in_order(self) -> None:
        run = _run()
        profile = CorruptionProfile(
            seed=SEED,
            corruptions=(
                JitterCorruption(
                    stream="cam_front",
                    distribution="gaussian",
                    std_ns=8_000_000,
                    clip_ns=30_000_000,
                ),
                DroppedFramesCorruption(stream="cam_front", probability=0.1),
                FixedLatencyCorruption(stream="cam_wrist", offset_ns=45_000_000),
            ),
        )
        a = apply_profile(run, profile)
        b = apply_profile(run, profile)
        assert a.run == b.run and a.dropped == b.dropped

        # cam_wrist got exactly the fixed offset, nothing else.
        for original, delayed in zip(run["cam_wrist"], a.run["cam_wrist"]):
            assert delayed.receive_time_ns == original.receive_time_ns + 45_000_000
        # cam_front lost the recorded samples and kept everything else jittered.
        assert len(a.run["cam_front"]) + len(a.dropped["cam_front"]) == len(run["cam_front"])
        # Dropped ground truth carries the *jittered* samples (drops ran second).
        original_receive = {
            s.sequence_id: s.receive_time_ns for s in run["cam_front"]
        }
        assert any(
            s.receive_time_ns != original_receive[s.sequence_id] for s in a.dropped["cam_front"]
        )
