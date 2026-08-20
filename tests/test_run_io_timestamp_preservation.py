"""Milestone 1 TDD red tests: run save/load preserves timestamps exactly.

EXPECTED TO FAIL (NotImplementedError) until datasets/io.py is implemented.
See DECISIONS.md D-0004/D-0005.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodied_sync.core.sample import Modality, Sample
from embodied_sync.datasets.io import (
    CORRUPTION_GROUND_TRUTH_NAME,
    FORMAT_VERSION,
    load_corruption_ground_truth,
    load_run,
    save_corruption_ground_truth,
    save_run,
)


def _tiny_run() -> dict[str, list[Sample]]:
    """Hand-authored run with adversarial timestamps (large, odd, ns-precise).

    Values chosen to break any float path: 2**53 + 1 is the first integer a
    float64 cannot represent.
    """
    mk = lambda i, acq, rcv: Sample(  # noqa: E731
        stream_name="robot_state",
        modality=Modality.ROBOT_STATE,
        sequence_id=i,
        acquisition_time_ns=acq,
        receive_time_ns=rcv,
        source_clock_domain="host_mono",
        payload=[0.1 * i, -1.5, 3.0],
        quality_flags=frozenset({"synthetic"}),
    )
    return {
        "robot_state": [
            mk(0, 2**53 + 1, 2**53 + 1_000_001),
            mk(1, 1_699_999_999_123_456_789, 1_699_999_999_123_456_790),
            mk(2, 1_699_999_999_127_456_789, 1_699_999_999_127_456_791),
        ],
        "events": [
            Sample(
                stream_name="events",
                modality=Modality.EVENT,
                sequence_id=0,
                acquisition_time_ns=1_699_999_999_125_000_003,
                receive_time_ns=1_699_999_999_125_100_003,
                source_clock_domain="host_mono",
                payload={"marker": "contact"},
                quality_flags=frozenset({"synthetic"}),
            )
        ],
    }


class TestRoundTrip:
    def test_exact_round_trip(self, tmp_path: Path) -> None:
        run = _tiny_run()
        save_run(run, tmp_path / "run")
        loaded = load_run(tmp_path / "run")
        assert loaded == run

    def test_timestamps_are_ints_after_load(self, tmp_path: Path) -> None:
        save_run(_tiny_run(), tmp_path / "run")
        loaded = load_run(tmp_path / "run")
        for samples in loaded.values():
            for s in samples:
                assert type(s.acquisition_time_ns) is int
                assert type(s.receive_time_ns) is int

    def test_float_unrepresentable_timestamp_survives(self, tmp_path: Path) -> None:
        save_run(_tiny_run(), tmp_path / "run")
        loaded = load_run(tmp_path / "run")
        assert loaded["robot_state"][0].acquisition_time_ns == 2**53 + 1

    def test_sample_order_preserved(self, tmp_path: Path) -> None:
        run = _tiny_run()
        save_run(run, tmp_path / "run")
        loaded = load_run(tmp_path / "run")
        assert [s.sequence_id for s in loaded["robot_state"]] == [0, 1, 2]


class TestLayout:
    def test_run_dir_layout_and_manifest(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        save_run(_tiny_run(), run_dir, extra_manifest={"seed": 1234})
        assert (run_dir / "manifest.json").is_file()
        assert (run_dir / "streams" / "robot_state.jsonl").is_file()
        assert (run_dir / "streams" / "events.jsonl").is_file()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["format_version"] == FORMAT_VERSION
        assert manifest["seed"] == 1234
        assert set(manifest["streams"]) == {"robot_state", "events"}
        assert manifest["streams"]["robot_state"]["sample_count"] == 3

    def test_refuses_nonempty_target(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "junk.txt").write_text("x")
        with pytest.raises((FileExistsError, ValueError)):
            save_run(_tiny_run(), run_dir)


class TestCorruptionGroundTruthSidecar:
    def test_exact_round_trip(self, tmp_path: Path) -> None:
        dropped = {"robot_state": tuple(_tiny_run()["robot_state"][:2])}
        save_run(_tiny_run(), tmp_path / "run")
        path = save_corruption_ground_truth(
            dropped,
            tmp_path / "run",
            extra_metadata={"profile_seed": 1234},
        )

        assert path.name == CORRUPTION_GROUND_TRUTH_NAME
        metadata = json.loads(path.read_text(encoding="utf-8"))
        assert metadata["format_version"] == FORMAT_VERSION
        assert metadata["type"] == "corruption_ground_truth"
        assert metadata["profile_seed"] == 1234
        assert load_corruption_ground_truth(tmp_path / "run") == dropped
