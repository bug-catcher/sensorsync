"""External true-MCAP tests on the HKisland01 SLAM recording.

The recording is a Livox LiDAR + Livox IMU + left camera + DJI OSDK
telemetry drone flight (~12.5 min, ~19 GB, 1.2M messages) from
`DapengFeng/MCAP`. The user drops the single-file bag at
``EMBODIED_SYNC_EXTERNAL_DATA_ROOT/mcap/HKisland01_0.mcap``.

Each test runs the ``mcap`` read in a subprocess so the parent pytest
process never imports the optional ``mcap`` package — that keeps
``tests/test_optional_deps.py`` green regardless of local install
state. Each test is small in wall-clock time because
:func:`load_mcap_run` now takes ``start_time_ns`` / ``end_time_ns``
window filters that hand through to ``mcap.reader.iter_messages``, so
no test iterates the full 19 GB file.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import pytest

from conftest import external_data_path

BAG_START_TIME_NS = 1_669_703_463_001_080_535
BAG_END_TIME_NS = 1_669_704_213_999_570_541
BAG_DURATION_NS = BAG_END_TIME_NS - BAG_START_TIME_NS

LIDAR_TOPIC = "/livox/lidar"
LIVOX_IMU_TOPIC = "/livox/imu"
DJI_IMU_TOPIC = "/dji_osdk_ros/imu"
CAMERA_TOPIC = "/left_camera/image/compressed"


def _hkisland_path():
    return external_data_path("mcap") / "HKisland01_0.mcap"


def _optional_module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_mcap_available(path) -> None:
    if not path.exists():
        pytest.skip(f"HKisland MCAP absent at {path}")
    if not _optional_module_available("mcap"):
        pytest.skip("optional dependency skipped: mcap is not installed")


def _run_subprocess(script: str, path) -> dict:
    """Run ``script`` in a subprocess passing ``path`` as sys.argv[1].

    ``script`` must print a single JSON document on stdout.
    """
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.optional_dep
@pytest.mark.external_data
def test_hkisland_summary_channels_match_expected_layout() -> None:
    """`mcap.reader.get_summary()` reveals the channel/topic layout
    without iterating messages — a fast survey of a 19 GB file."""
    path = _hkisland_path()
    _require_mcap_available(path)

    script = (
        "import json, sys\n"
        "from mcap.reader import make_reader\n"
        "with open(sys.argv[1], 'rb') as f:\n"
        "    summary = make_reader(f).get_summary()\n"
        "stats = summary.statistics\n"
        "channels = {\n"
        "    summary.channels[ch_id].topic: {\n"
        "        'schema': summary.schemas[summary.channels[ch_id].schema_id].name,\n"
        "        'count': stats.channel_message_counts[ch_id],\n"
        "    }\n"
        "    for ch_id in stats.channel_message_counts\n"
        "}\n"
        "print(json.dumps({\n"
        "    'message_count': stats.message_count,\n"
        "    'chunk_count': stats.chunk_count,\n"
        "    'message_start_time': stats.message_start_time,\n"
        "    'message_end_time': stats.message_end_time,\n"
        "    'channels': channels,\n"
        "}))\n"
    )
    summary = _run_subprocess(script, path)

    assert summary["message_count"] == 1_202_840
    assert summary["chunk_count"] == 14_490
    assert summary["message_start_time"] == BAG_START_TIME_NS
    assert summary["message_end_time"] == BAG_END_TIME_NS

    channels = summary["channels"]
    # Multi-modal, SLAM-shaped rig: LiDAR + camera + two IMUs + GPS/RTK.
    assert channels[LIDAR_TOPIC]["schema"] == "livox_ros_driver/msg/CustomMsg"
    assert channels[LIDAR_TOPIC]["count"] == 7_510  # ~10 Hz over 750 s
    assert channels[LIVOX_IMU_TOPIC]["schema"] == "sensor_msgs/msg/Imu"
    assert channels[LIVOX_IMU_TOPIC]["count"] == 156_658  # ~209 Hz
    assert channels[DJI_IMU_TOPIC]["schema"] == "sensor_msgs/msg/Imu"
    assert channels[DJI_IMU_TOPIC]["count"] == 300_414  # ~400 Hz
    assert channels[CAMERA_TOPIC]["schema"] == "sensor_msgs/msg/CompressedImage"
    assert channels[CAMERA_TOPIC]["count"] == 7_511  # ~10 Hz


@pytest.mark.optional_dep
@pytest.mark.external_data
def test_hkisland_bounded_slice_loads_camera_imu_lidar() -> None:
    """One-second slice with topic filter: LiDAR + camera at 10 Hz,
    IMU at 200+ Hz. Large payloads live behind ``payload_ref``;
    small IMU messages carry inline ``data_hex``."""
    path = _hkisland_path()
    _require_mcap_available(path)

    # 1-second slice starting at the first bag message.
    start = BAG_START_TIME_NS
    end = start + 1_000_000_000

    script = (
        "import json, sys\n"
        "from embodied_sync.adapters.mcap import load_mcap_run\n"
        "start, end = int(sys.argv[2]), int(sys.argv[3])\n"
        "run = load_mcap_run(\n"
        "    sys.argv[1],\n"
        "    topics=[sys.argv[4], sys.argv[5], sys.argv[6]],\n"
        "    start_time_ns=start,\n"
        "    end_time_ns=end,\n"
        ")\n"
        "out = {}\n"
        "for name, samples in run.items():\n"
        "    first = samples[0]\n"
        "    last = samples[-1]\n"
        "    out[name] = {\n"
        "        'count': len(samples),\n"
        "        'modality': first.modality.value,\n"
        "        'source_clock_domain': first.source_clock_domain,\n"
        "        'schema_name': first.payload['schema_name'],\n"
        "        'first_byte_length': first.payload['byte_length'],\n"
        "        'first_acquisition_time_ns': first.acquisition_time_ns,\n"
        "        'last_acquisition_time_ns': last.acquisition_time_ns,\n"
        "        'first_has_payload_ref': first.payload_ref is not None,\n"
        "        'first_has_data_hex': 'data_hex' in first.payload,\n"
        "    }\n"
        "print(json.dumps(out))\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(path),
            str(start),
            str(end),
            LIDAR_TOPIC,
            LIVOX_IMU_TOPIC,
            CAMERA_TOPIC,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    out = json.loads(result.stdout)

    # LiDAR: Livox CustomMsg, 10 Hz nominal → ~10 samples over 1 s.
    lidar = out[LIDAR_TOPIC]
    assert lidar["schema_name"] == "livox_ros_driver/msg/CustomMsg"
    assert 8 <= lidar["count"] <= 12
    assert lidar["first_byte_length"] > 100_000  # point clouds are large
    assert lidar["first_has_payload_ref"] is True
    assert lidar["first_has_data_hex"] is False
    assert lidar["source_clock_domain"] == "mcap_publish_time"

    # Camera: JPEG compressed, ~10 Hz.
    camera = out[CAMERA_TOPIC]
    assert camera["schema_name"] == "sensor_msgs/msg/CompressedImage"
    assert 8 <= camera["count"] <= 12
    assert camera["modality"] == "camera"
    assert camera["first_has_payload_ref"] is True

    # Livox IMU: sensor_msgs/msg/Imu, ~200 Hz. Small message → inline hex.
    imu = out[LIVOX_IMU_TOPIC]
    assert imu["schema_name"] == "sensor_msgs/msg/Imu"
    assert imu["modality"] == "tactile"  # infer_modality maps IMU → TACTILE
    assert 180 <= imu["count"] <= 240
    assert imu["first_has_payload_ref"] is False
    assert imu["first_has_data_hex"] is True

    # Slice bounds respected on both ends.
    for entry in out.values():
        assert start <= entry["first_acquisition_time_ns"] < end
        assert start <= entry["last_acquisition_time_ns"] < end


@pytest.mark.optional_dep
@pytest.mark.external_data
def test_hkisland_slam_style_multirate_alignment() -> None:
    """5-second slice aligned at 10 Hz with a SLAM-style per-stream
    policy: NN for LiDAR + camera (regular 10 Hz), ZoH with an
    explicit 50 ms tolerance for IMU (delivered in bursts, so the
    engine-derived tolerance would be too tight).

    Pins two properties:

    1. Every LiDAR/camera frame is non-missing at 10 Hz.
    2. IMU misses at most one frame under the SLAM-style ZoH policy;
       naive NN with the engine's derived tolerance would miss every
       frame (the burst-delivery gotcha).
    """
    path = _hkisland_path()
    _require_mcap_available(path)

    start = BAG_START_TIME_NS
    end = start + 5_000_000_000

    script = (
        "import json, sys\n"
        "from embodied_sync.adapters.mcap import load_mcap_run\n"
        "from embodied_sync.align import align_run\n"
        "from embodied_sync.core import AlignmentPolicy\n"
        "path, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])\n"
        "topics = [sys.argv[4], sys.argv[5], sys.argv[6]]\n"
        "run = load_mcap_run(path, topics=topics, start_time_ns=start, end_time_ns=end)\n"
        "\n"
        "# Naive default policy on IMU: it misses every frame.\n"
        "naive = align_run(run, target_rate_hz=10.0)\n"
        "\n"
        "# SLAM-style per-stream policy: ZoH+50 ms for IMU.\n"
        "slam = align_run(\n"
        "    run,\n"
        "    target_rate_hz=10.0,\n"
        "    method={\n"
        "        sys.argv[5]: AlignmentPolicy(method='zoh', tolerance_ns=50_000_000),\n"
        "        sys.argv[4]: 'nearest_neighbor',\n"
        "        sys.argv[6]: 'nearest_neighbor',\n"
        "    },\n"
        ")\n"
        "print(json.dumps({\n"
        "    'frames': len(slam.frames),\n"
        "    'naive_missing': naive.report.missing_count,\n"
        "    'slam_missing': slam.report.missing_count,\n"
        "    'slam_median_skew_ns': slam.report.median_skew_ns,\n"
        "}))\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(path),
            str(start),
            str(end),
            LIDAR_TOPIC,
            LIVOX_IMU_TOPIC,
            CAMERA_TOPIC,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    out = json.loads(result.stdout)

    frames = out["frames"]
    assert frames >= 40  # ~49 frames over 5 s at 10 Hz

    # 1. Under the naive default policy IMU misses every frame — that's
    #    the burst-delivery observable the aligner correctly surfaces.
    assert out["naive_missing"][LIVOX_IMU_TOPIC] == frames
    # LiDAR and camera are already regular at 10 Hz — naive NN finds them.
    assert out["naive_missing"][LIDAR_TOPIC] == 0
    assert out["naive_missing"][CAMERA_TOPIC] == 0

    # 2. Under SLAM-style per-stream policy IMU picks resolve — at
    #    most one frame missing under 50 ms tolerance ZoH.
    assert out["slam_missing"][LIVOX_IMU_TOPIC] <= 2
    assert out["slam_missing"][LIDAR_TOPIC] == 0
    assert out["slam_missing"][CAMERA_TOPIC] == 0

    # ZoH skew is always <= 0; the picked IMU sample is at most one
    # 100 ms grid period stale (usually much less).
    slam_skews = out["slam_median_skew_ns"]
    assert slam_skews[LIVOX_IMU_TOPIC] is not None
    assert -100_000_000 <= slam_skews[LIVOX_IMU_TOPIC] <= 0


@pytest.mark.optional_dep
@pytest.mark.external_data
def test_hkisland_bounded_slice_load_is_fast() -> None:
    """The 1-second slice loader must not scan the whole 19 GB bag.

    ``mcap.reader.iter_messages(start_time=..., end_time=...)`` uses
    the chunk index, so a bounded slice is O(chunks-in-window). We
    assert a generous 15 s ceiling that catches accidental full-file
    scans (empirical: ~1 s) while tolerating slow CI disks.
    """
    path = _hkisland_path()
    _require_mcap_available(path)

    start = BAG_START_TIME_NS
    end = start + 1_000_000_000

    script = (
        "import json, sys, time\n"
        "from embodied_sync.adapters.mcap import load_mcap_run\n"
        "path, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])\n"
        "topics = [sys.argv[4], sys.argv[5], sys.argv[6]]\n"
        "t0 = time.perf_counter()\n"
        "run = load_mcap_run(path, topics=topics, start_time_ns=start, end_time_ns=end)\n"
        "elapsed = time.perf_counter() - t0\n"
        "print(json.dumps({\n"
        "    'elapsed_s': elapsed,\n"
        "    'total_samples': sum(len(s) for s in run.values()),\n"
        "}))\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(path),
            str(start),
            str(end),
            LIDAR_TOPIC,
            LIVOX_IMU_TOPIC,
            CAMERA_TOPIC,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    out = json.loads(result.stdout)

    # Empirically ~1 s on WSL2; 15 s catches an accidental full scan
    # (which would take minutes on a 19 GB file).
    assert out["elapsed_s"] < 15.0, (
        f"bounded slice took {out['elapsed_s']:.2f} s — possible full-file scan"
    )
    # Sanity: the slice contains ~200 IMU + 10 LiDAR + 10 camera samples.
    assert out["total_samples"] > 100
