"""Milestone 9: SurgSync-style synthetic contract integration.

The committed :data:`FIXTURE_PATH` is a small SurgSync-shape run:
endoscope + external camera at 30 Hz, kinematics at 50 Hz, force
sensor at 25 Hz, workflow phase events. It exercises multi-modal +
mixed clock-domain alignment on the base install without any external
dataset. External true-SurgSync data is loaded via
:func:`~tests.conftest.external_data_path` — see
``docs/user/manual_dataset_setup.md``.
"""

from __future__ import annotations

from pathlib import Path

from embodied_sync.adapters.surg_sync import load_surg_sync_run
from embodied_sync.align import align_run
from embodied_sync.reports import build_report

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "fixtures"
    / "surg_sync_mini"
    / "run.json"
)


def test_fixture_loads_five_streams() -> None:
    run = load_surg_sync_run(FIXTURE_PATH)
    assert set(run) == {
        "endoscope",
        "external_camera",
        "kinematics",
        "force_sensor",
        "phase_events",
    }
    # 30 Hz cameras, 50 Hz kinematics, 25 Hz force, 3 events.
    assert len(run["endoscope"]) == 7
    assert len(run["kinematics"]) == 11
    assert len(run["force_sensor"]) == 6
    assert len(run["phase_events"]) == 4


def test_fixture_preserves_clock_domain_per_stream() -> None:
    run = load_surg_sync_run(FIXTURE_PATH)
    domains = {name: {s.source_clock_domain for s in samples} for name, samples in run.items()}
    assert domains["endoscope"] == {"endoscope_hw"}
    assert domains["external_camera"] == {"host_mono"}
    assert domains["force_sensor"] == {"force_hw"}


def test_surg_sync_run_aligns_at_10hz() -> None:
    run = load_surg_sync_run(FIXTURE_PATH)
    aligned = align_run(run, target_rate_hz=10.0)
    # World-time targets on the 10 Hz grid clipped to the intersection
    # of the acquisition-time windows. Common window: [0, 200_000_000].
    assert [f.target_time_ns for f in aligned.frames] == [
        0, 100_000_000, 200_000_000,
    ]
    # Every stream should have a non-missing pick on every frame at this
    # tolerance since the windows are lined up on the 10 Hz grid.
    for frame in aligned.frames:
        for name in ("endoscope", "external_camera", "kinematics", "force_sensor"):
            assert not frame.metadata[name].missing, (
                f"stream {name!r} missing at target {frame.target_time_ns} ns"
            )


def test_surg_sync_report_has_zero_missing_for_grid_aligned_streams() -> None:
    run = load_surg_sync_run(FIXTURE_PATH)
    aligned = align_run(run, target_rate_hz=10.0)
    report = build_report(aligned)
    by_name = {s.name: s for s in report.streams}
    for name in ("endoscope", "external_camera", "kinematics", "force_sensor"):
        assert by_name[name].missing_rate == 0.0
