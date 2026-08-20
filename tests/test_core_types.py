"""Pin the D-0024 core-type surface: Episode / SyncReport as thin aliases.

The aligned/report dataclasses were moved to ``embodied_sync.core`` so
the canonical data model lives with the rest of ``core/``. The alignment
engine and reports subpackage re-export them for backward compatibility.
``Episode`` and ``SyncReport`` are :data:`~typing.TypeAlias` names for
:class:`AlignedRun` and :class:`SyncQualityReport` — thin wrappers over
the existing types rather than parallel hierarchies.
"""

from __future__ import annotations

from embodied_sync.align import align_run
from embodied_sync.core import (
    REPORT_FORMAT_VERSION as CORE_REPORT_FORMAT_VERSION,
    AlignedFrame as CoreAlignedFrame,
    AlignedRun as CoreAlignedRun,
    AlignedSampleMetadata as CoreAlignedSampleMetadata,
    AlignmentReport as CoreAlignmentReport,
    Episode,
    Sample,
    StreamStats as CoreStreamStats,
    SyncQualityReport as CoreSyncQualityReport,
    SyncReport,
)
from embodied_sync.reports import build_report


def _tiny_run() -> dict[str, list[Sample]]:
    return {
        "a": [
            Sample(
                stream_name="a",
                modality="robot_state",  # type: ignore[arg-type]
                sequence_id=i,
                acquisition_time_ns=i * 1_000_000,
                receive_time_ns=i * 1_000_000,
                source_clock_domain="host_mono",
                payload=None,
            )
            for i in range(50)
        ]
    }


def test_episode_is_aligned_run_alias() -> None:
    from embodied_sync.align import AlignedRun as AlignAlignedRun
    from embodied_sync.align.engine import AlignedRun as EngineAlignedRun

    assert Episode is CoreAlignedRun
    assert AlignAlignedRun is CoreAlignedRun
    assert EngineAlignedRun is CoreAlignedRun


def test_sync_report_is_sync_quality_report_alias() -> None:
    from embodied_sync.reports import SyncQualityReport as ReportsSyncQualityReport
    from embodied_sync.reports.sync_quality import SyncQualityReport as ModuleSyncQualityReport

    assert SyncReport is CoreSyncQualityReport
    assert ReportsSyncQualityReport is CoreSyncQualityReport
    assert ModuleSyncQualityReport is CoreSyncQualityReport


def test_aligned_frame_and_metadata_reachable_from_core() -> None:
    from embodied_sync.align import (
        AlignedFrame as AlignAlignedFrame,
        AlignedSampleMetadata as AlignAlignedSampleMetadata,
        AlignmentReport as AlignAlignmentReport,
    )

    assert AlignAlignedFrame is CoreAlignedFrame
    assert AlignAlignedSampleMetadata is CoreAlignedSampleMetadata
    assert AlignAlignmentReport is CoreAlignmentReport


def test_stream_stats_and_format_version_reachable_from_core() -> None:
    from embodied_sync.reports import (
        REPORT_FORMAT_VERSION as ReportsREPORT_FORMAT_VERSION,
        StreamStats as ReportsStreamStats,
    )

    assert ReportsStreamStats is CoreStreamStats
    assert ReportsREPORT_FORMAT_VERSION == CORE_REPORT_FORMAT_VERSION


def test_align_run_returns_a_core_episode_instance() -> None:
    aligned = align_run(_tiny_run(), target_rate_hz=100.0)

    assert isinstance(aligned, CoreAlignedRun)
    # Episode is an alias, so isinstance(aligned, Episode) is equivalent.
    assert isinstance(aligned, Episode)
    # Frames and metadata are the core types.
    assert all(isinstance(f, CoreAlignedFrame) for f in aligned.frames)
    assert all(
        isinstance(md, CoreAlignedSampleMetadata)
        for f in aligned.frames
        for md in f.metadata.values()
    )
    assert isinstance(aligned.report, CoreAlignmentReport)


def test_build_report_returns_a_core_sync_report_instance() -> None:
    aligned = align_run(_tiny_run(), target_rate_hz=100.0)
    report = build_report(aligned)

    assert isinstance(report, CoreSyncQualityReport)
    assert isinstance(report, SyncReport)
    assert all(isinstance(s, CoreStreamStats) for s in report.streams)


def test_core_all_lists_the_new_types() -> None:
    from embodied_sync import core

    for name in (
        "AlignedFrame",
        "AlignedRun",
        "AlignedSampleMetadata",
        "AlignmentReport",
        "Episode",
        "StreamStats",
        "SyncQualityReport",
        "SyncReport",
        "REPORT_FORMAT_VERSION",
    ):
        assert name in core.__all__
        assert hasattr(core, name)
