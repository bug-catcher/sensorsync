from __future__ import annotations

import json

import pytest
from conftest import external_data_path

from embodied_sync.adapters.qut import load_qut_dataset
from embodied_sync.align import align_run
from embodied_sync.ingest import DatasetImportAgent, execute_import_plan


@pytest.mark.external_data
def test_external_qut_images_h5_validates_row_aligned_sync() -> None:
    """Real QUT example: HDF5 image rows are the camera/state alignment oracle."""

    h5py = pytest.importorskip("h5py")
    root = external_data_path("qut") / "dataset_example"
    image_h5 = root / "images.h5"
    if not image_h5.is_file():
        pytest.skip(f"external QUT dataset skipped: {image_h5} is missing")

    run, info = load_qut_dataset(root)

    assert {"camera.side", "camera.top", "camera.wrist"}.issubset(run)
    expected_frame_count = sum(int(episode["length"]) for episode in info["episodes"])
    with h5py.File(image_h5, "r") as h5:
        for episode in info["episodes"]:
            episode_id = str(episode["episode_id"])
            state_rows = json.loads(
                (root / "episodes" / episode_id / "state.json").read_text(
                    encoding="utf-8"
                )
            )
            for camera in ("side", "top", "wrist"):
                assert h5[episode_id][camera].shape == (
                    len(state_rows),
                    240,
                    320,
                    3,
                )

    robot_seq_to_row: list[tuple[str, int]] = []
    for episode in info["episodes"]:
        episode_id = str(episode["episode_id"])
        for frame_index in range(int(episode["length"])):
            robot_seq_to_row.append((episode_id, frame_index))

    aligned = align_run(run, target_rate_hz=float(info["source_rate_hz"]))
    assert len(aligned.frames) == expected_frame_count
    assert set(aligned.report.missing_count.values()) == {0}

    checked = 0
    for frame in aligned.frames:
        robot = frame.samples["robot_q"]
        assert robot is not None
        expected_episode, expected_frame = robot_seq_to_row[robot.sequence_id]
        for stream in ("camera.side", "camera.top", "camera.wrist"):
            camera = frame.samples[stream]
            assert camera is not None
            assert camera.payload["episode_id"] == expected_episode
            assert camera.payload["frame_index"] == expected_frame
            assert camera.payload_ref == (
                f"images.h5:/{expected_episode}/{stream.removeprefix('camera.')}["
                f"{expected_frame}]"
            )
            checked += 1

    assert checked == expected_frame_count * 3


@pytest.mark.external_data
def test_external_qut_auto_import_selects_and_validates_row_clock() -> None:
    """The generic agent reaches the QUT interpretation without a named adapter."""

    pytest.importorskip("h5py")
    root = external_data_path("qut") / "dataset_example"
    if not (root / "images.h5").is_file():
        pytest.skip(f"external QUT dataset skipped: {root / 'images.h5'} is missing")

    inference = DatasetImportAgent().analyze(root)
    assert inference.selected is not None
    assert inference.selected.executor == "indexed_episode"
    assert inference.selected.parameters["clock"] == {
        "strategy": "row_index",
        "rate_hz": 10.0,
    }
    assert inference.selected.confidence >= 0.9

    run, _ = execute_import_plan(root, inference.selected, max_episodes=3)
    aligned = align_run(run, target_rate_hz=10.0)
    assert len(aligned.frames) == sum(
        len(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted((root / "episodes").glob("*/state.json"))[:3]
    )
    assert set(aligned.report.missing_count.values()) == {0}
