from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodied_sync.cli.main import main
from embodied_sync.datasets.io import load_run
from embodied_sync.ingest import (
    DatasetImportAgent,
    DatasetProfile,
    Evidence,
    ImportPlan,
    execute_import_plan,
    infer_import,
    inspect_dataset,
    load_import_plan,
    save_json_document,
)


def _write_indexed_fixture(root: Path) -> None:
    for episode_id, start in (("0000", 1_000.0), ("0001", 2_000.0)):
        episode = root / "episodes" / episode_id
        video = episode / "video"
        video.mkdir(parents=True)
        (video / "side.mp4").write_bytes(b"")
        rows = [
            {"time": [start], "robot_q": [1.0, 2.0], "gripper_action": 0.0},
            {"time": [start + 128.0], "robot_q": [3.0, 4.0], "gripper_action": 1.0},
            {"time": [start + 251.0], "robot_q": [5.0, 6.0], "gripper_action": 0.0},
        ]
        (episode / "state.json").write_text(json.dumps(rows), encoding="utf-8")


def _row_plan() -> ImportPlan:
    return ImportPlan(
        executor="indexed_episode",
        confidence=0.91,
        parameters={
            "episode_glob": "episodes/*",
            "row_file": "state.json",
            "state_streams": {
                "robot_q": "robot_state",
                "gripper_action": "action",
            },
            "source_time_field": "time",
            "camera": {"source": "video", "glob": "video/*.mp4"},
            "clock": {"strategy": "row_index", "rate_hz": 10.0},
            "source_clock_domain": "fixture.indexed",
        },
        evidence=(Evidence("reviewed", "fixture plan", 0.91),),
    )


def _strong_profile(root: Path) -> DatasetProfile:
    return DatasetProfile(
        root=str(root),
        path_kind="directory",
        signatures=(),
        facts={
            "indexed_episode": {
                "episode_glob": "episodes/*",
                "row_file": "state.json",
                "episode_ids": ["0000", "0001"],
                "episode_count": 2,
                "row_counts": {"0000": 3, "0001": 3},
                "total_rows": 6,
                "common_fields": ["time", "robot_q", "gripper_action"],
                "timestamp_fields": [
                    {
                        "field": "time",
                        "episode_spans": [251.0, 251.0],
                        "median_delta": 125.5,
                        "mean_delta": 125.5,
                        "min_delta": 123.0,
                        "max_delta": 128.0,
                        "monotonic_fraction": 1.0,
                    }
                ],
                "hdf5": {
                    "path": "images.h5",
                    "camera_names": ["side", "top", "wrist"],
                    "count_matches": 6,
                    "count_checks": 6,
                    "count_match_ratio": 1.0,
                },
                "videos": {
                    "rate_hz": 10.0,
                    "video_glob": "video/*.mp4",
                    "frame_count_match_ratio": 1.0,
                },
            }
        },
    )


def test_import_plan_json_round_trip(tmp_path: Path) -> None:
    path = save_json_document(_row_plan(), tmp_path / "plan.json")
    assert load_import_plan(path) == _row_plan()


def test_probe_weak_layout_stops_at_ambiguity(tmp_path: Path) -> None:
    _write_indexed_fixture(tmp_path)

    profile = inspect_dataset(tmp_path)
    result = DatasetImportAgent().analyze(tmp_path, rate_hz=10.0)

    assert profile.facts["indexed_episode"]["episode_count"] == 2
    assert result.candidates
    assert result.selected is None
    assert "confidence" in result.decision or "margin" in result.decision


def test_inference_selects_row_clock_when_media_counts_disagree_with_timestamps(
    tmp_path: Path,
) -> None:
    result = infer_import(_strong_profile(tmp_path))

    assert result.selected is not None
    assert result.selected.executor == "indexed_episode"
    assert result.selected.parameters["clock"]["strategy"] == "row_index"
    assert result.selected.confidence == pytest.approx(0.95)
    assert result.candidates[1].parameters["clock"]["strategy"] == "timestamp_field"
    assert result.selected.confidence > result.candidates[1].confidence


def test_execute_reviewed_row_plan_uses_fixed_rate_and_preserves_source_time(
    tmp_path: Path,
) -> None:
    _write_indexed_fixture(tmp_path)

    run, info = execute_import_plan(tmp_path, _row_plan())

    assert info["imported_episodes"] == 2
    assert [sample.acquisition_time_ns for sample in run["robot_q"]] == [
        0,
        100_000_000,
        200_000_000,
        300_000_000,
        400_000_000,
        500_000_000,
    ]
    assert run["camera.side"][1].payload == {
        "episode_id": "0000",
        "frame_index": 1,
        "source_time": 1128.0,
    }
    assert run["camera.side"][1].payload_ref == (
        "episodes/0000/video/side.mp4#frame=1"
    )


def test_execute_rejects_plan_path_escape(tmp_path: Path) -> None:
    _write_indexed_fixture(tmp_path)
    plan = ImportPlan(
        executor="indexed_episode",
        confidence=0.9,
        parameters={**_row_plan().parameters, "episode_glob": "../*"},
        evidence=(),
    )
    with pytest.raises(ValueError, match="dataset root"):
        execute_import_plan(tmp_path, plan)


def test_known_lerobot_signature_selects_specialized_adapter(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "fps": 30}), encoding="utf-8"
    )

    result = DatasetImportAgent().analyze(tmp_path)

    assert result.selected is not None
    assert result.selected.executor == "lerobot_v3"
    assert result.selected.parameters["source_rate_hz"] == 30.0


def test_import_auto_cli_executes_reviewed_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "dataset"
    _write_indexed_fixture(dataset)
    plan_path = save_json_document(_row_plan(), tmp_path / "plan.json")
    out = tmp_path / "run"

    assert (
        main(
            [
                "import-auto",
                str(dataset),
                "--plan",
                str(plan_path),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    run = load_run(out)
    assert len(run["robot_q"]) == 6
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_rate_hz"] == 10.0
    assert manifest["auto_import"]["import_plan"]["executor"] == "indexed_episode"
    assert "indexed_episode/row_index confidence=0.910" in capsys.readouterr().out


def test_import_auto_cli_refuses_ambiguous_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "dataset"
    _write_indexed_fixture(dataset)

    assert (
        main(
            [
                "import-auto",
                str(dataset),
                "--rate-hz",
                "10",
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 1
    )
    assert "review `embsync infer-import`" in capsys.readouterr().err
