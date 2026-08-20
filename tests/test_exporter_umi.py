"""UMI / diffusion-policy Zarr replay-buffer exporter tests (D-0036)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap

import pytest

from embodied_sync.align import align_run
from embodied_sync.cli.main import main
from embodied_sync.datasets.io import save_episode
from embodied_sync.streams.synthetic import generate_synthetic_run


def _zarr_local_store_or_skip() -> None:
    pytest.importorskip("zarr")
    script = textwrap.dedent(
        """
        import tempfile
        import zarr

        path = tempfile.mkdtemp(suffix=".zarr")
        zarr.open_group(path, mode="w")
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        pytest.skip(
            "optional dependency skipped: zarr local-store open timed out "
            "in this execution sandbox"
        )
    if result.returncode != 0:
        pytest.skip(
            "optional dependency skipped: zarr local-store open is unavailable "
            "in this execution sandbox"
        )


@pytest.mark.optional_dep
def test_export_umi_zarr_round_trips_numeric_streams_and_missing_frames(tmp_path) -> None:
    _zarr_local_store_or_skip()
    import zarr

    from embodied_sync.exporters.umi import export_umi_zarr

    run = generate_synthetic_run(duration_s=1.0, seed=4)
    tactile = run["tactile"]
    gappy = tactile[: len(tactile) // 3] + tactile[2 * len(tactile) // 3 :]
    aligned = align_run(
        {"robot_state": run["robot_state"], "tactile": gappy},
        target_rate_hz=60.0,
    )
    assert any(frame.metadata["tactile"].missing for frame in aligned.frames)

    out = tmp_path / "umi_buffer.zarr"
    export_umi_zarr(aligned, out, target_rate_hz=60.0)

    root = zarr.open_group(out, mode="r")
    assert sorted(root["data"].array_keys()) == ["robot_state", "tactile"]
    assert root.attrs["format"] == "embodied_sync.umi_zarr.v0"
    assert root.attrs["target_rate_hz"] == 60.0
    assert root["meta"]["episode_ends"][:].tolist() == [len(aligned.frames)]

    robot = root["data"]["robot_state"][:]
    tactile_arr = root["data"]["tactile"][:]
    assert robot.shape == (len(aligned.frames), 7)
    assert tactile_arr.shape == (len(aligned.frames), 16)

    missing_rows = [
        i for i, frame in enumerate(aligned.frames) if frame.metadata["tactile"].missing
    ]
    assert missing_rows
    assert all(math.isnan(float(tactile_arr[i, 0])) for i in missing_rows)


@pytest.mark.optional_dep
def test_cli_export_umi_uses_episode_manifest_rate(tmp_path, capsys) -> None:
    _zarr_local_store_or_skip()
    import zarr

    run = generate_synthetic_run(duration_s=0.5, seed=2)
    aligned = align_run({"robot_state": run["robot_state"]}, target_rate_hz=20.0)
    episode_dir = tmp_path / "episode"
    save_episode(aligned, episode_dir, target_rate_hz=20.0)

    out = tmp_path / "cli_umi.zarr"
    assert main(["export-umi", str(episode_dir), "--out", str(out)]) == 0
    assert "exported" in capsys.readouterr().out

    root = zarr.open_group(out, mode="r")
    assert root.attrs["target_rate_hz"] == 20.0
    assert root["meta"]["episode_ends"][:].tolist() == [len(aligned.frames)]


@pytest.mark.optional_dep
def test_umi_zarr_metadata_is_plain_zarr_v2(tmp_path) -> None:
    """The exporter writes a normal directory store; metadata is inspectable
    without importing zarr, keeping the base suite deterministic."""
    pytest.importorskip("zarr")
    from embodied_sync.exporters.umi import export_umi_zarr

    run = generate_synthetic_run(duration_s=0.2, seed=1)
    aligned = align_run({"robot_state": run["robot_state"]}, target_rate_hz=10.0)
    out = tmp_path / "plain.zarr"
    export_umi_zarr(aligned, out, target_rate_hz=10.0)

    assert json.loads((out / ".zgroup").read_text(encoding="utf-8")) == {
        "zarr_format": 2
    }
    attrs = json.loads((out / ".zattrs").read_text(encoding="utf-8"))
    assert attrs["stream_names"] == {"robot_state": "robot_state"}
    zarray = json.loads((out / "data" / "robot_state" / ".zarray").read_text())
    assert zarray["shape"] == [len(aligned.frames), 7]
    assert (out / "meta" / "episode_ends" / "0").is_file()
