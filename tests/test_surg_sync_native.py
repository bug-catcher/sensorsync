"""Native SurgSync v1.0 reader tests (D-0035)."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import pytest
from conftest import external_data_path


def _write_native_surgsync_fixture(root: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    for partition, clip, deltas in [
        ("online_data", "1", [5_000_000, 6_000_000, None]),
        ("offline_data", "2", [100_000, -200_000, 300_000]),
    ]:
        ep = root / partition / "episodes" / "suturing" / clip
        ep.mkdir(parents=True)
        (root / "meta").mkdir(exist_ok=True)
        (ep / "video_raw").mkdir()
        (ep / "episode_meta.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "episode_id": f"{partition}_{clip}_123",
                    "task": "suturing",
                    "length_frames": 3,
                    "duration_s": 0.3,
                    "master_t0_ns": 123,
                    "recorder_variant": partition.removesuffix("_data"),
                }
            ),
            encoding="utf-8",
        )
        (ep / "modalities.json").write_text(
            json.dumps(
                {
                    "video_raw": {
                        "stereo_left": {"present": True, "path": "stereo_left.mkv"},
                        "stereo_right": {"present": True, "path": "stereo_right.mkv"},
                        "side": {"present": False},
                    }
                }
            ),
            encoding="utf-8",
        )
        timestamp = pa.table(
            {
                "frame_index": [0, 1, 2],
                "source_frame_index": [0, 1, 2],
                "master_timestamp_ns": [0, 100_000_000, 200_000_000],
                "delta_to_master.image_right_ns": [1_000_000, 1_100_000, 900_000],
                "delta_to_master.PSM1.measured_cp_ns": deltas,
                "delta_to_master.PSM1.setpoint_cp_ns": deltas,
                "is_contiguous_to_prev": [False, True, False],
                "drop_count_since_prev": [0, 0, 2],
            }
        )
        pq.write_table(timestamp, ep / "timestamp.parquet")
        psm1 = pa.table(
            {
                "frame_index": [0, 1, 2],
                "master_timestamp_ns": [0, 100_000_000, 200_000_000],
                "measured_cp.position": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                "measured_cp.orientation": [[0.0, 0.0, 0.0, 1.0]] * 3,
                "setpoint_cp.position": [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
                "setpoint_cp.orientation": [[0.0, 0.0, 0.0, 1.0]] * 3,
            }
        )
        pq.write_table(psm1, ep / "PSM1.parquet")
        pq.write_table(
            pa.table(
                {
                    "frame_index": [0, 1, 2],
                    "master_timestamp_ns": [0, 100_000_000, 200_000_000],
                    "contact.PSM1": [0, 1, 0],
                    "contact.PSM2": [0, 0, 1],
                    "phase": ["a", "b", "c"],
                    "step": ["x", "y", "z"],
                }
            ),
            ep / "annotation.parquet",
        )


@pytest.mark.optional_dep
def test_load_surg_sync_dataset_fixture_maps_delta_to_skew(tmp_path) -> None:
    _write_native_surgsync_fixture(tmp_path)
    from embodied_sync.adapters.surg_sync import load_surg_sync_dataset
    from embodied_sync.core.sample import QUALITY_GAP_BEFORE, Modality

    run, info = load_surg_sync_dataset(tmp_path, partition="online_data")

    assert {"PSM1.measured_cp", "video_raw.stereo_left", "annotation"}.issubset(run)
    measured = run["PSM1.measured_cp"]
    assert len(measured) == 2
    assert measured[0].acquisition_time_ns == 5_000_000
    assert measured[0].receive_time_ns == 0
    assert measured[0].transport_latency_ns == -5_000_000
    assert measured[1].payload["measured_cp.position"] == [4.0, 5.0, 6.0]
    assert measured[0].modality is Modality.ROBOT_STATE
    assert QUALITY_GAP_BEFORE not in measured[0].quality_flags

    assert info["episodes"][0]["drop_count_total"] == 2
    dropped_rows = [
        row for row in info["drop_ground_truth"] if row["drop_count_since_prev"] > 0
    ]
    assert dropped_rows[0]["drop_count_since_prev"] == 2
    assert info["timestamp_mapping"]["receive_time_ns"] == "master_timestamp_ns"


def _surgsync_or_skip() -> Path:
    root = external_data_path("surg_sync") / "surgsync_subset"
    if not (root / "meta").is_dir():
        pytest.skip(f"external SurgSync skipped: {root} has no meta/ directory")
    return root


@pytest.mark.external_data
def test_external_surg_sync_online_offline_skew_difference() -> None:
    pytest.importorskip("pyarrow.parquet")
    from embodied_sync.adapters.surg_sync import load_surg_sync_dataset

    root = _surgsync_or_skip()
    online, online_info = load_surg_sync_dataset(root, partition="online_data", max_episodes=1)
    offline, offline_info = load_surg_sync_dataset(root, partition="offline_data", max_episodes=1)

    online_skew = [abs(s.transport_latency_ns) for s in online["PSM1.measured_cp"][:200]]
    offline_skew = [abs(s.transport_latency_ns) for s in offline["PSM1.measured_cp"][:200]]

    assert median(online_skew) > 1_000_000
    assert median(offline_skew) < 1_000_000
    assert online_info["episodes"][0]["partition"] == "online"
    assert offline_info["episodes"][0]["partition"] == "offline"


@pytest.mark.external_data
def test_external_surg_sync_interrupted_episode_surfaces_drop_ground_truth() -> None:
    pytest.importorskip("pyarrow.parquet")
    from embodied_sync.adapters.surg_sync import load_surg_sync_dataset

    root = _surgsync_or_skip()
    run, info = load_surg_sync_dataset(
        root / "online_data" / "episodes" / "single_interrupted_stitch" / "18",
        max_episodes=1,
    )

    assert info["episodes"][0]["non_contiguous_count"] == 183
    assert info["episodes"][0]["drop_count_total"] == 424
    assert len(info["drop_ground_truth"]) == 183
    assert "PSM1.measured_cp" in run
