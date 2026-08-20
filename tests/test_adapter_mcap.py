from __future__ import annotations

import json
import subprocess
import sys

import pytest

from embodied_sync.adapters.mcap import load_mcap_run
from embodied_sync.align import align_run
from embodied_sync.exporters.mcap import save_mcap_episode, save_mcap_run
from embodied_sync.streams.synthetic import generate_synthetic_run
from conftest import external_data_path


def test_load_mcap_run_preserves_timestamps_and_flags(tmp_path) -> None:
    run = generate_synthetic_run(duration_s=0.2, seed=0, start_time_ns=123)
    path = tmp_path / "run.mcap"

    save_mcap_run(run, path)

    loaded = load_mcap_run(path)
    assert loaded == run
    assert loaded["cam_front"][0].acquisition_time_ns == 123
    assert loaded["cam_front"][0].quality_flags == frozenset({"synthetic"})


def test_load_mcap_run_rejects_episode_document(tmp_path) -> None:
    run = generate_synthetic_run(duration_s=0.2, seed=0)
    episode = align_run(run, target_rate_hz=10.0)
    path = tmp_path / "episode.mcap"
    save_mcap_episode(episode, path)

    with pytest.raises(ValueError, match="expected MCAP run document"):
        load_mcap_run(path)


def test_load_mcap_run_rejects_empty_zip_placeholder(tmp_path) -> None:
    path = tmp_path / "placeholder.zip"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="empty MCAP/rosbag zip placeholder"):
        load_mcap_run(path)


def test_mcap_document_records_episode_report(tmp_path) -> None:
    run = generate_synthetic_run(duration_s=0.2, seed=0)
    episode = align_run(run, target_rate_hz=10.0)
    path = tmp_path / "episode.mcap"

    save_mcap_episode(episode, path)

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["type"] == "aligned_episode"
    assert document["report"]["median_skew_ns"] == episode.report.median_skew_ns


@pytest.mark.external_data
def test_external_jkkds02_zip_placeholder_is_reported_clearly() -> None:
    path = external_data_path("mcap") / "jkkds02.zip"
    if not path.exists():
        pytest.skip(f"jkkds02.zip absent at {path}")

    with pytest.raises(ValueError, match="empty MCAP/rosbag zip placeholder"):
        load_mcap_run(path)


@pytest.mark.optional_dep
@pytest.mark.external_data
def test_load_realsense_mcap_preserves_true_mcap_timestamps() -> None:
    path = (
        external_data_path("mcap")
        / "realsense_sample"
        / "rosbag2_2024_02_18-23_35_48"
        / "rosbag2_2024_02_18-23_35_48_0.mcap"
    )
    if path.exists() and not _optional_module_available("mcap"):
        pytest.skip("optional dependency skipped: mcap is not installed")

    script = (
        "import json, sys; "
        "from embodied_sync.adapters.mcap import load_mcap_run; "
        "run = load_mcap_run(sys.argv[1], topics=['/camera/color/image_raw']); "
        "samples = run['/camera/color/image_raw']; "
        "first = samples[0]; "
        "print(json.dumps({"
        "'count': len(samples), "
        "'acquisition_time_ns': first.acquisition_time_ns, "
        "'receive_time_ns': first.receive_time_ns, "
        "'source_clock_domain': first.source_clock_domain, "
        "'modality': first.modality.value, "
        "'schema_name': first.payload['schema_name'], "
        "'byte_length': first.payload['byte_length'], "
        "'has_payload_ref': first.payload_ref is not None"
        "}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    summary = json.loads(result.stdout)
    assert summary == {
        "count": 295,
        "acquisition_time_ns": 1_708_266_948_486_359_989,
        "receive_time_ns": 1_708_266_948_486_359_989,
        "source_clock_domain": "mcap_publish_time",
        "modality": "camera",
        "schema_name": "sensor_msgs/msg/Image",
        "byte_length": 2_764_872,
        "has_payload_ref": True,
    }


@pytest.mark.optional_dep
@pytest.mark.external_data
def test_load_realsense_rosbag_directory_resolves_storage_mcap() -> None:
    path = (
        external_data_path("mcap")
        / "realsense_sample"
        / "rosbag2_2024_02_18-23_35_48"
    )
    if path.exists() and not _optional_module_available("mcap"):
        pytest.skip("optional dependency skipped: mcap is not installed")

    script = (
        "import json, sys; "
        "from embodied_sync.adapters.mcap import load_mcap_run; "
        "run = load_mcap_run(sys.argv[1], topics=['/camera/color/image_raw']); "
        "samples = run['/camera/color/image_raw']; "
        "print(json.dumps({"
        "'count': len(samples), "
        "'first_acquisition_time_ns': samples[0].acquisition_time_ns, "
        "'payload_ref': samples[0].payload_ref"
        "}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    summary = json.loads(result.stdout)
    assert summary["count"] == 295
    assert summary["first_acquisition_time_ns"] == 1_708_266_948_486_359_989
    assert "rosbag2_2024_02_18-23_35_48_0.mcap" in summary["payload_ref"]


def _optional_module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None
