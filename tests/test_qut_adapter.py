from __future__ import annotations

import json
from pathlib import Path

from embodied_sync.adapters.qut import load_qut_dataset
from embodied_sync.core.sample import Modality


def _write_qut_fixture(root: Path) -> None:
    ep = root / "episodes" / "0000"
    (ep / "video").mkdir(parents=True)
    for name in ("top.mp4", "side.mp4", "wrist.mp4"):
        (ep / "video" / name).write_bytes(b"")
    rows = [
        {
            "time": [1000.0],
            "robot_q": [1.0, 2.0],
            "gello_q": [3.0, 4.0],
            "gripper_action": 0.0,
            "q": [1.0, 2.0],
        },
        {
            "time": [1128.0],
            "robot_q": [5.0, 6.0],
            "gello_q": [7.0, 8.0],
            "gripper_action": 1.0,
            "q": [5.0, 6.0],
        },
    ]
    (ep / "state.json").write_text(json.dumps(rows), encoding="utf-8")


def test_load_qut_dataset_maps_state_and_camera_streams(tmp_path) -> None:
    _write_qut_fixture(tmp_path)

    run, info = load_qut_dataset(tmp_path)

    assert info["imported_episodes"] == 1
    assert info["source_rate_hz"] == 10.0
    assert run["robot_q"][1].acquisition_time_ns == 100_000_000
    assert run["robot_q"][1].payload == [5.0, 6.0]
    assert run["robot_q"][0].modality is Modality.ROBOT_STATE

    camera = run["camera.side"][1]
    assert camera.acquisition_time_ns == 100_000_000
    assert camera.payload["source_time_ms"] == 1128.0
    assert camera.payload_ref == "episodes/0000/video/side.mp4#frame=1"
    assert camera.modality is Modality.CAMERA
