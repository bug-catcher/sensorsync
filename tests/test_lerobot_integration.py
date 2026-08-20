from __future__ import annotations

from embodied_sync.adapters.lerobot import load_lerobot_run
from embodied_sync.exporters.lerobot import save_lerobot_episode, save_lerobot_run
from embodied_sync.align import align_run
from embodied_sync.streams.synthetic import generate_synthetic_run


def test_lerobot_run_round_trip_preserves_episode_streams(tmp_path) -> None:
    run = generate_synthetic_run(duration_s=0.2, seed=7)
    dataset_dir = tmp_path / "lerobot_dataset"

    save_lerobot_run(run, dataset_dir)

    assert load_lerobot_run(dataset_dir) == run


def test_lerobot_episode_export_records_aligned_frames(tmp_path) -> None:
    run = generate_synthetic_run(duration_s=0.2, seed=7)
    episode = align_run(run, target_rate_hz=10.0)
    dataset_dir = tmp_path / "lerobot_episode"

    save_lerobot_episode(episode, dataset_dir)

    assert (dataset_dir / "dataset.json").is_file()
    assert "aligned_episode" in (dataset_dir / "dataset.json").read_text(encoding="utf-8")
