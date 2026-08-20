from __future__ import annotations

from embodied_sync.adapters.mcap import load_mcap_run
from embodied_sync.exporters.mcap import save_mcap_run
from embodied_sync.streams.synthetic import generate_synthetic_run


def test_save_mcap_run_is_deterministic(tmp_path) -> None:
    run = generate_synthetic_run(duration_s=0.2, seed=4)
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"

    save_mcap_run(run, first)
    save_mcap_run(run, second)

    assert first.read_bytes() == second.read_bytes()


def test_timestamp_round_trip(tmp_path) -> None:
    run = generate_synthetic_run(duration_s=0.2, seed=4, start_time_ns=99_000_000_001)
    path = tmp_path / "run.mcap"

    save_mcap_run(run, path)

    loaded = load_mcap_run(path)
    assert loaded == run
    assert loaded["robot_state"][0].acquisition_time_ns == 99_000_000_001


def test_quality_flag_round_trip(tmp_path) -> None:
    run = generate_synthetic_run(duration_s=0.2, seed=4)
    path = tmp_path / "run.mcap"

    save_mcap_run(run, path)

    loaded = load_mcap_run(path)
    assert loaded["actions"][0].quality_flags == run["actions"][0].quality_flags
