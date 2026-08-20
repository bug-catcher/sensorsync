"""Native LeRobot v3.0 reader/exporter tests (D-0033).

Three tiers in one file:

- plain tests: CLI behaviors that need no optional deps (rate fallback,
  report-on-run-dir error paths);
- ``optional_dep`` tests: exporter round-trips through real parquet via
  ``pyarrow`` on synthetic data;
- ``external_data`` tests: the real datasets under
  ``EMBODIED_SYNC_EXTERNAL_DATA_ROOT/lerobot/`` (pusht, pusht_image,
  aloha_*). They skip with a clear message when a dataset is absent.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest
from conftest import external_data_path

from embodied_sync.align import align_run
from embodied_sync.cli.main import main
from embodied_sync.datasets.io import load_episode, load_run
from embodied_sync.streams.synthetic import generate_synthetic_run


def _dataset_or_skip(name: str) -> Path:
    path = external_data_path("lerobot") / name
    if not (path / "meta" / "info.json").is_file():
        pytest.skip(
            f"external dataset skipped: {path} has no meta/info.json. "
            f"Place the LeRobot dataset under lerobot/{name}/ to enable."
        )
    return path


def _float32(value: float) -> float:
    """The float32 quantization LeRobot's timestamp column applies."""
    return struct.unpack("f", struct.pack("f", value))[0]


# ---------------------------------------------------------------- plain tier


def test_align_without_rate_errors_for_unrated_run(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    assert main(["synth", "--out", str(run_dir), "--duration-s", "0.2"]) == 0
    rc = main(["align", str(run_dir), "--out", str(tmp_path / "ep")])
    assert rc == 1
    assert "--target-rate-hz is required" in capsys.readouterr().err


def test_report_on_run_dir_with_explicit_rate(tmp_path) -> None:
    run_dir = tmp_path / "run"
    assert main(["synth", "--out", str(run_dir), "--duration-s", "0.5"]) == 0
    rc = main(["report", str(run_dir), "--target-rate-hz", "10.0"])
    assert rc == 0
    assert (run_dir / "sync_report.html").is_file()


def test_report_on_run_dir_uses_manifest_source_rate(tmp_path) -> None:
    run_dir = tmp_path / "run"
    assert main(["synth", "--out", str(run_dir), "--duration-s", "0.5"]) == 0
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_rate_hz"] = 10.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "r.html"
    assert main(["report", str(run_dir), "--out", str(out)]) == 0
    assert out.is_file()


# --------------------------------------------------------- optional_dep tier


@pytest.mark.optional_dep
def test_export_synthetic_aligned_run_as_lerobot_v3(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from embodied_sync.exporters.lerobot import export_lerobot_dataset

    run = generate_synthetic_run(duration_s=1.0, seed=3)
    aligned = align_run(
        {name: run[name] for name in ("robot_state", "actions")},
        target_rate_hz=10.0,
    )
    out = tmp_path / "exported"
    export_lerobot_dataset(aligned, out, target_rate_hz=10.0, task="synthetic test")

    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v3.0"
    assert info["fps"] == 10
    assert info["total_episodes"] == 1
    assert set(info["features"]) >= {"robot_state", "actions", "timestamp", "frame_index"}

    table = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
    assert table.num_rows == len(aligned.frames) == info["total_frames"]
    assert table.column("episode_index").to_pylist() == [0] * table.num_rows
    ts = table.column("timestamp").to_pylist()
    assert ts[0] == 0.0 and ts[1] == pytest.approx(0.1)
    assert isinstance(table.schema.field("robot_state").type, pa.FixedSizeListType)

    tasks = pq.read_table(out / "meta" / "tasks.parquet").to_pylist()
    assert tasks[0]["__index_level_0__"] == "synthetic test"
    episodes = pq.read_table(
        out / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pylist()
    assert episodes[0]["length"] == table.num_rows
    assert episodes[0]["dataset_to_index"] == table.num_rows


@pytest.mark.optional_dep
def test_export_writes_nan_for_missing_frames(tmp_path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    from embodied_sync.exporters.lerobot import export_lerobot_dataset

    run = generate_synthetic_run(duration_s=1.0, seed=3)
    # Carve a hole out of the middle of tactile so alignment has frames
    # where the stream is missing (gap far beyond the NN tolerance).
    tactile = run["tactile"]
    gappy = tactile[: len(tactile) // 3] + tactile[2 * len(tactile) // 3 :]
    aligned = align_run(
        {"robot_state": run["robot_state"], "tactile": gappy},
        target_rate_hz=60.0,
    )
    assert any(f.metadata["tactile"].missing for f in aligned.frames)
    out = tmp_path / "exported"
    export_lerobot_dataset(aligned, out, target_rate_hz=1000.0)
    table = pq.read_table(out / "data" / "chunk-000" / "file-000.parquet")
    missing_rows = [
        i for i, f in enumerate(aligned.frames) if f.metadata["tactile"].missing
    ]
    tactile = table.column("tactile").to_pylist()
    assert all(math.isnan(tactile[i][0]) for i in missing_rows)


# --------------------------------------------------------- external_data tier


@pytest.mark.external_data
def test_pusht_reader_metadata_and_timestamps() -> None:
    from embodied_sync.adapters.lerobot import load_lerobot_dataset

    dataset = _dataset_or_skip("pusht")
    run, info = load_lerobot_dataset(dataset, max_episodes=3)

    assert info["codebase_version"].startswith("v3")
    assert info["fps"] == 10.0
    assert info["tasks"] == {"0": "Push the T-shaped block onto the T-shaped target."}
    assert info["imported_episodes"] == 3
    assert set(run) == {"observation.image", "observation.state", "action", "next.reward"}

    # Timestamps: stored float32 seconds converted to ns exactly (row 1 of
    # episode 0 is float32(0.1) seconds — the quantization is preserved).
    state = run["observation.state"]
    assert state[0].acquisition_time_ns == 0
    assert state[1].acquisition_time_ns == round(_float32(0.1) * 1e9)
    # No invented transport latency.
    assert all(s.receive_time_ns == s.acquisition_time_ns for s in state[:100])
    # Contiguous per-stream sequence ids across episode boundaries.
    assert [s.sequence_id for s in state[:5]] == [0, 1, 2, 3, 4]


@pytest.mark.external_data
def test_pusht_episode_boundaries_and_video_refs() -> None:
    from embodied_sync.adapters.lerobot import load_lerobot_dataset

    dataset = _dataset_or_skip("pusht")
    run, info = load_lerobot_dataset(dataset, max_episodes=2)

    ep0, ep1 = info["episodes"]
    assert ep0["episode_index"] == 0 and ep0["start_time_ns"] == 0
    # Episode 1 starts on the frame-grid boundary after episode 0's frames.
    assert ep1["start_time_ns"] == round(1e9 * ep0["length"] / info["fps"])

    video = run["observation.image"]
    assert len(video) == ep0["length"] + ep1["length"]
    # Streams are globally monotonic across the episode boundary.
    acq = [s.acquisition_time_ns for s in video]
    assert all(a <= b for a, b in zip(acq, acq[1:]))
    assert video[ep0["length"]].acquisition_time_ns == ep1["start_time_ns"]

    # Video refs map frame indices into the concatenated per-file video:
    # episode 1 frame 0 seeks to its from_timestamp (= 16.1 s for pusht).
    first_ep1 = video[ep0["length"]]
    assert first_ep1.payload_ref is not None
    assert "observation.image" in first_ep1.payload_ref
    assert "episode_frame=0" in first_ep1.payload_ref
    seek_s = float(first_ep1.payload_ref.split("#t=")[1].split("&")[0])
    assert seek_s == pytest.approx(ep1["start_time_ns"] / 1e9, abs=1e-6)


@pytest.mark.external_data
def test_pusht_image_parquet_refs() -> None:
    from embodied_sync.adapters.lerobot import load_lerobot_dataset

    dataset = _dataset_or_skip("pusht_image")
    run, _ = load_lerobot_dataset(dataset, max_episodes=1)
    image = run["observation.image"]
    assert image[0].payload_ref == (
        "data/chunk-000/file-000.parquet#row=0&column=observation.image"
    )
    assert image[0].payload is None


@pytest.mark.external_data
@pytest.mark.parametrize(
    "name", ["aloha_static_coffee", "aloha_static_pingpong_test"]
)
def test_aloha_multi_camera_import(name: str) -> None:
    from embodied_sync.adapters.lerobot import load_lerobot_dataset

    dataset = _dataset_or_skip(name)
    run, info = load_lerobot_dataset(dataset, max_episodes=2)
    cameras = [n for n in run if n.startswith("observation.images.")]
    assert len(cameras) == 4
    assert info["fps"] == 50.0
    lengths = {len(samples) for samples in run.values()}
    assert len(lengths) == 1  # every stream covers the same rows


@pytest.mark.external_data
def test_cli_import_and_report_pusht(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    dataset = _dataset_or_skip("pusht")
    run_dir = tmp_path / "runs" / "lerobot_pusht"
    assert main(
        ["import-lerobot", str(dataset), "--out", str(run_dir), "--max-episodes", "5"]
    ) == 0

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_rate_hz"] == 10.0
    assert len(manifest["lerobot"]["episodes"]) == 5

    # `report` directly on the imported run dir (aligns at source_rate_hz).
    assert main(["report", str(run_dir)]) == 0
    assert (run_dir / "sync_report.html").is_file()


@pytest.mark.external_data
def test_cli_import_align_export_roundtrip_pusht_image(tmp_path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    from embodied_sync.adapters.lerobot import load_lerobot_dataset

    dataset = _dataset_or_skip("pusht_image")
    run_dir = tmp_path / "runs" / "pusht_image"
    episode_dir = tmp_path / "episodes" / "pusht_image_aligned"
    export_dir = tmp_path / "out" / "roundtrip"

    assert main(
        ["import-lerobot", str(dataset), "--out", str(run_dir), "--max-episodes", "3"]
    ) == 0
    # No --target-rate-hz: falls back to the manifest's source_rate_hz.
    assert main(["align", str(run_dir), "--out", str(episode_dir)]) == 0
    with pytest.warns(UserWarning, match="observation.image"):
        assert main(["export-lerobot", str(episode_dir), "--out", str(export_dir)]) == 0

    aligned = load_episode(episode_dir)
    table = pq.read_table(export_dir / "data" / "chunk-000" / "file-000.parquet")
    assert table.num_rows == len(aligned.frames)

    # The exported dataset re-imports through the same native reader, with
    # the source task string carried through the manifest chain.
    roundtrip, info = load_lerobot_dataset(export_dir)
    assert info["tasks"] == {"0": "Push the T-shaped block onto the T-shaped target."}
    assert set(roundtrip) == {"observation.state", "action", "next.reward"}
    original = load_run(run_dir)
    state_rt = roundtrip["observation.state"]
    state_orig = original["observation.state"]
    assert len(state_rt) == len(aligned.frames)
    # Value fidelity through align → export (float32) → import.
    for src, back in zip(state_orig[:50], state_rt[:50]):
        for a, b in zip(src.payload, back.payload):
            assert b == pytest.approx(a, abs=1e-3)
