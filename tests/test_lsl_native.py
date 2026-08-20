"""Native XDF reader tests (D-0034)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from conftest import external_data_path


def _vli(value: int) -> bytes:
    if value < 256:
        return b"\x01" + struct.pack("B", value)
    return b"\x04" + struct.pack("<I", value)


def _chunk(tag: int, payload: bytes, stream_id: int | None = None) -> bytes:
    body = struct.pack("<H", tag)
    if stream_id is not None:
        body += struct.pack("<I", stream_id)
    body += payload
    return _vli(len(body)) + body


def _write_tiny_xdf(path: Path) -> None:
    header = b"<info><version>1.0</version></info>"
    stream_header = b"""
    <info>
      <name>robot_state</name>
      <type>joint_state</type>
      <channel_count>2</channel_count>
      <nominal_srate>100</nominal_srate>
      <channel_format>float32</channel_format>
      <source_id>fixture_robot</source_id>
      <uid>fixture_robot_uid</uid>
    </info>
    """
    samples = _vli(3)
    for stamp, values in [
        (1.000000001, (1.0, 2.0)),
        (1.010100001, (3.0, 4.0)),
        (1.020000001, (5.0, 6.0)),
    ]:
        samples += b"\x01" + struct.pack("<d", stamp)
        samples += struct.pack("<ff", *values)
    clock_offset = struct.pack("<dd", 1.0, -0.000003)
    path.write_bytes(
        b"XDF:"
        + _chunk(1, header)
        + _chunk(2, stream_header.strip(), stream_id=1)
        + _chunk(4, clock_offset, stream_id=1)
        + _chunk(3, samples, stream_id=1)
    )


@pytest.mark.optional_dep
def test_load_xdf_file_preserves_raw_jitter_and_clock_offsets(tmp_path) -> None:
    pytest.importorskip("pyxdf")
    from embodied_sync.adapters.lsl import load_xdf_file
    from embodied_sync.core.sample import Modality

    xdf = tmp_path / "tiny.xdf"
    _write_tiny_xdf(xdf)

    run, info = load_xdf_file(xdf)

    assert set(run) == {"robot_state"}
    samples = run["robot_state"]
    assert [s.acquisition_time_ns for s in samples] == [
        1_000_000_001,
        1_010_100_001,
        1_020_000_001,
    ]
    assert [samples[i + 1].acquisition_time_ns - samples[i].acquisition_time_ns for i in range(2)] == [
        10_100_000,
        9_900_000,
    ]
    assert samples[0].receive_time_ns == samples[0].acquisition_time_ns
    assert samples[0].modality is Modality.ROBOT_STATE
    assert samples[0].payload == [1.0, 2.0]
    assert info["timestamp_mode"] == {
        "synchronize_clocks": False,
        "dejitter_timestamps": False,
        "units": "integer_ns_from_xdf_seconds",
    }
    assert info["streams"][0]["clock_offsets"] == [
        {"time_ns": 1_000_000_000, "offset_ns": -3_000}
    ]


def _xdf_or_skip() -> Path:
    root = external_data_path("xdf")
    paths = sorted(root.glob("*.xdf"))
    if not paths:
        pytest.skip(f"external XDF skipped: no .xdf files under {root}")
    return paths[0]


@pytest.mark.external_data
def test_external_labrecorder_session_loads_raw_jitter_and_offsets() -> None:
    pytest.importorskip("pyxdf")
    from embodied_sync.adapters.lsl import load_xdf_file

    run, info = load_xdf_file(_xdf_or_skip())

    assert len(run) >= 6
    assert {"robot_state", "event_markers"}.issubset(run)
    robot = run["robot_state"]
    assert len(robot) > 100
    deltas = [
        robot[i + 1].acquisition_time_ns - robot[i].acquisition_time_ns
        for i in range(min(200, len(robot) - 1))
    ]
    assert len(set(deltas)) > 1
    assert any(stream["clock_offsets"] for stream in info["streams"])
