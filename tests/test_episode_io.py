"""Aligned-episode on-disk format v0 (D-0021).

Round-trip contract: ``load_episode(save_episode(a, d, ...)) == a`` for
any :class:`AlignedRun` produced by :func:`align_run`. All timestamps
survive as integer ns; missing samples land as JSON ``null``; report
counts (missing + ground-truth) survive verbatim.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from embodied_sync.align import (
    AlignedFrame,
    AlignedRun,
    AlignedSampleMetadata,
    AlignmentReport,
    align_run,
)
from embodied_sync.core import AlignmentPolicy
from embodied_sync.corrupt import (
    CorruptionProfile,
    DroppedFramesCorruption,
    FixedLatencyCorruption,
    apply_profile,
)
from embodied_sync.datasets.io import load_episode, load_run, save_episode
from embodied_sync.streams.synthetic import generate_synthetic_run

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"


def _clean_aligned() -> tuple[AlignedRun, float]:
    run = generate_synthetic_run(duration_s=1.0, seed=0)
    return align_run(run, target_rate_hz=10.0), 10.0


def _corrupted_aligned_with_ground_truth() -> tuple[AlignedRun, float]:
    run = generate_synthetic_run(duration_s=1.0, seed=0)
    profile = CorruptionProfile(
        seed=0,
        corruptions=(DroppedFramesCorruption(stream="cam_front", probability=0.5),),
    )
    corr = apply_profile(run, profile)
    return align_run(corr.run, target_rate_hz=10.0, ground_truth=corr.dropped), 10.0


class TestRoundTrip:
    def test_clean_aligned_run_round_trip(self, tmp_path: Path) -> None:
        aligned, rate = _clean_aligned()
        episode_dir = tmp_path / "episode"
        save_episode(aligned, episode_dir, target_rate_hz=rate)
        loaded = load_episode(episode_dir)
        assert loaded == aligned

    def test_corrupted_run_with_ground_truth_round_trip(self, tmp_path: Path) -> None:
        aligned, rate = _corrupted_aligned_with_ground_truth()
        episode_dir = tmp_path / "episode"
        save_episode(aligned, episode_dir, target_rate_hz=rate)
        loaded = load_episode(episode_dir)
        assert loaded == aligned
        assert loaded.report.ground_truth_missing_count == aligned.report.ground_truth_missing_count

    def test_missing_samples_land_as_null(self, tmp_path: Path) -> None:
        aligned, rate = _corrupted_aligned_with_ground_truth()
        episode_dir = tmp_path / "episode"
        save_episode(aligned, episode_dir, target_rate_hz=rate)
        frames_path = episode_dir / "frames.jsonl"
        # At least one line must have a JSON null for cam_front.
        found_null = False
        for line in frames_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["samples"]["cam_front"] is None:
                found_null = True
                assert record["metadata"]["cam_front"]["missing"] is True
                break
        assert found_null, "expected at least one missing cam_front frame after 50% drops"


class TestManifest:
    def test_manifest_records_target_rate_and_derived_period(self, tmp_path: Path) -> None:
        aligned, rate = _clean_aligned()
        episode_dir = tmp_path / "episode"
        save_episode(aligned, episode_dir, target_rate_hz=rate)
        manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["format_version"] == 0
        assert manifest["type"] == "aligned_episode"
        assert manifest["target_rate_hz"] == rate
        assert manifest["target_period_ns"] == 100_000_000
        assert manifest["frame_count"] == len(aligned.frames)
        assert set(manifest["streams"]) == set(aligned.frames[0].samples.keys())

    def test_manifest_records_missing_counts(self, tmp_path: Path) -> None:
        aligned, rate = _corrupted_aligned_with_ground_truth()
        episode_dir = tmp_path / "episode"
        save_episode(aligned, episode_dir, target_rate_hz=rate)
        manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["missing_count"] == dict(aligned.report.missing_count)
        assert manifest["ground_truth_missing_count"] == dict(
            aligned.report.ground_truth_missing_count
        )

    def test_manifest_records_median_skew_per_stream(self, tmp_path: Path) -> None:
        aligned, rate = _clean_aligned()
        episode_dir = tmp_path / "episode"
        save_episode(aligned, episode_dir, target_rate_hz=rate)
        manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
        assert "median_skew_ns" in manifest
        assert set(manifest["median_skew_ns"].keys()) == set(
            aligned.frames[0].samples.keys()
        )
        # Regular streams on-grid → exact zero median skew.
        regular = {"cam_front", "cam_wrist", "robot_state", "tactile", "audio", "actions"}
        for name in regular:
            assert manifest["median_skew_ns"][name] == 0

    def test_median_skew_matches_statistics_median(self, tmp_path: Path) -> None:
        aligned, rate = _corrupted_aligned_with_ground_truth()
        episode_dir = tmp_path / "episode"
        save_episode(aligned, episode_dir, target_rate_hz=rate)
        manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
        for name in aligned.frames[0].samples.keys():
            skews = [
                frame.metadata[name].skew_ns
                for frame in aligned.frames
                if not frame.metadata[name].missing
                and frame.metadata[name].skew_ns is not None
            ]
            expected = int(statistics.median(skews)) if skews else None
            assert manifest["median_skew_ns"][name] == expected

    def test_median_skew_reflects_signed_direction(self, tmp_path: Path) -> None:
        # A fixed positive latency on receive time doesn't move the
        # acquisition-time skew of nearest-neighbor by itself, so use ZoH
        # instead: the returned samples still sit on-grid, so use a jitter
        # profile that shifts acquisition-time via missing-interval-adjacent
        # sequencing… simpler: assert sign matches direct computation, no
        # matter the magnitude. The point is that the manifest field
        # preserves sign, not that it's the same for every profile.
        run = generate_synthetic_run(duration_s=1.0, seed=0)
        profile = CorruptionProfile(
            seed=0,
            corruptions=(
                DroppedFramesCorruption(stream="cam_front", probability=0.4),
                FixedLatencyCorruption(stream="cam_front", offset_ns=5_000_000),
            ),
        )
        corr = apply_profile(run, profile)
        aligned = align_run(corr.run, target_rate_hz=10.0)
        episode_dir = tmp_path / "episode"
        save_episode(aligned, episode_dir, target_rate_hz=10.0)
        manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
        skews = [
            frame.metadata["cam_front"].skew_ns
            for frame in aligned.frames
            if not frame.metadata["cam_front"].missing
            and frame.metadata["cam_front"].skew_ns is not None
        ]
        expected = int(statistics.median(skews)) if skews else None
        assert manifest["median_skew_ns"]["cam_front"] == expected

    def test_report_median_skew_round_trips_as_typed_attribute(
        self, tmp_path: Path
    ) -> None:
        """``AlignmentReport.median_skew_ns`` survives a save/load cycle.

        Session 9 taught ``save_episode`` to write ``median_skew_ns`` into
        the manifest, but ``load_episode`` used to drop the value on
        read — downstream tools had to re-parse the manifest by hand.
        NEXT_TASKS #4 lifts the field onto the typed report so callers
        can read ``aligned.report.median_skew_ns[name]`` directly, and
        the value must round-trip byte-for-byte through save/load.
        """
        aligned, rate = _corrupted_aligned_with_ground_truth()
        # The engine now populates ``median_skew_ns`` on every fresh
        # :class:`AlignedRun` — this is what the manifest echoes.
        assert aligned.report.median_skew_ns  # non-empty for a run with frames
        assert set(aligned.report.median_skew_ns.keys()) == set(
            aligned.frames[0].samples.keys()
        )

        episode_dir = tmp_path / "episode"
        save_episode(aligned, episode_dir, target_rate_hz=rate)
        loaded = load_episode(episode_dir)
        assert loaded.report.median_skew_ns == aligned.report.median_skew_ns
        # The manifest and the typed attribute must agree — downstream
        # readers pick either one.
        manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
        assert loaded.report.median_skew_ns == manifest["median_skew_ns"]

    def test_report_median_skew_matches_frame_medians(self, tmp_path: Path) -> None:
        """The engine-computed report field matches ``statistics.median`` on frames."""
        aligned, rate = _corrupted_aligned_with_ground_truth()
        for name in aligned.frames[0].samples.keys():
            skews = [
                frame.metadata[name].skew_ns
                for frame in aligned.frames
                if not frame.metadata[name].missing
                and frame.metadata[name].skew_ns is not None
            ]
            expected: int | None = int(statistics.median(skews)) if skews else None
            assert aligned.report.median_skew_ns[name] == expected

    def test_median_skew_none_when_every_frame_missing(self, tmp_path: Path) -> None:
        run = generate_synthetic_run(duration_s=1.0, seed=0)
        run["cam_front"] = []
        aligned = align_run(run, target_rate_hz=10.0)
        episode_dir = tmp_path / "episode"
        save_episode(aligned, episode_dir, target_rate_hz=10.0)
        manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["median_skew_ns"]["cam_front"] is None

    def test_extra_manifest_survives_but_reserved_keys_win(self, tmp_path: Path) -> None:
        aligned, rate = _clean_aligned()
        episode_dir = tmp_path / "episode"
        save_episode(
            aligned,
            episode_dir,
            target_rate_hz=rate,
            extra_manifest={"note": "hello", "type": "should_be_overridden"},
        )
        manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["note"] == "hello"
        # Reserved key: our writer must not honour the caller's override.
        assert manifest["type"] == "aligned_episode"

    def test_alignment_policy_round_trips_through_manifest(
        self, tmp_path: Path
    ) -> None:
        run = generate_synthetic_run(duration_s=1.0, seed=0)
        policy = {
            "cam_front": "zoh",
            "robot_state": AlignmentPolicy(
                method="linear_interp",
                tolerance_ns=5_000_000,
            ),
        }
        aligned = align_run(run, target_rate_hz=10.0, method=policy)
        episode_dir = tmp_path / "episode"
        save_episode(
            aligned,
            episode_dir,
            target_rate_hz=10.0,
            alignment_policy=policy,
        )

        expected_policy = {
            "cam_front": "zoh",
            "robot_state": {
                "method": "linear_interp",
                "tolerance_ns": 5_000_000,
            },
        }
        manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["format_version"] == 0
        assert manifest["alignment_policy"] == expected_policy

        loaded = load_episode(episode_dir)
        assert loaded == aligned
        assert loaded.report.alignment_policy == expected_policy

    def test_missing_alignment_policy_loads_as_none_on_old_fixture(self) -> None:
        fixture_dir = FIXTURES_DIR / "synth_mini_aligned"
        manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
        assert "alignment_policy" not in manifest

        loaded = load_episode(fixture_dir)
        expected = align_run(
            load_run(FIXTURES_DIR / "synth_mini"),
            target_rate_hz=10.0,
        )
        assert loaded == expected
        assert loaded.report.alignment_policy is None


class TestFailureModes:
    def test_refuses_non_empty_out_dir(self, tmp_path: Path) -> None:
        aligned, rate = _clean_aligned()
        episode_dir = tmp_path / "episode"
        episode_dir.mkdir()
        (episode_dir / "existing.txt").write_text("x", encoding="utf-8")
        with pytest.raises(FileExistsError, match="non-empty"):
            save_episode(aligned, episode_dir, target_rate_hz=rate)

    def test_load_from_non_episode_dir_fails(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not an episode"):
            load_episode(tmp_path / "does_not_exist")

    def test_load_rejects_wrong_type(self, tmp_path: Path) -> None:
        episode_dir = tmp_path / "episode"
        episode_dir.mkdir()
        (episode_dir / "manifest.json").write_text(
            json.dumps({"format_version": 0, "type": "not_an_episode"}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="episode type"):
            load_episode(episode_dir)

    def test_load_rejects_wrong_format_version(self, tmp_path: Path) -> None:
        episode_dir = tmp_path / "episode"
        episode_dir.mkdir()
        (episode_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "format_version": 999,
                    "type": "aligned_episode",
                    "streams": [],
                    "frame_count": 0,
                    "missing_count": {},
                    "target_rate_hz": 10.0,
                    "target_period_ns": 100_000_000,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="format_version"):
            load_episode(episode_dir)

    def test_load_rejects_frame_count_mismatch(self, tmp_path: Path) -> None:
        aligned, rate = _clean_aligned()
        episode_dir = tmp_path / "episode"
        save_episode(aligned, episode_dir, target_rate_hz=rate)
        manifest_path = episode_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["frame_count"] = manifest["frame_count"] + 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="frame_count|frames"):
            load_episode(episode_dir)


class TestReconstructedShape:
    def test_reconstructed_frames_are_frozen(self, tmp_path: Path) -> None:
        aligned, rate = _clean_aligned()
        episode_dir = tmp_path / "episode"
        save_episode(aligned, episode_dir, target_rate_hz=rate)
        loaded = load_episode(episode_dir)
        assert isinstance(loaded, AlignedRun)
        assert isinstance(loaded.frames[0], AlignedFrame)
        assert isinstance(loaded.report, AlignmentReport)
        first_md = next(iter(loaded.frames[0].metadata.values()))
        assert isinstance(first_md, AlignedSampleMetadata)
