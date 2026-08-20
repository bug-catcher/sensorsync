"""Sync-quality reports (Milestone 1 CLI target `embsync report`).

The data types (:class:`StreamStats`, :class:`SyncQualityReport`,
:data:`REPORT_FORMAT_VERSION`) live in
:mod:`embodied_sync.core.sync_report`; they are re-exported here for
backward compatibility. :data:`SyncReport` is a type alias of
:class:`SyncQualityReport`.
"""

from embodied_sync.core.sync_report import SyncReport
from embodied_sync.reports.sync_quality import (
    REPORT_FORMAT_VERSION,
    StreamStats,
    SyncQualityReport,
    build_report,
    render_html,
    report_summary_dict,
    save_report_html,
)

__all__ = [
    "REPORT_FORMAT_VERSION",
    "StreamStats",
    "SyncQualityReport",
    "SyncReport",
    "build_report",
    "render_html",
    "report_summary_dict",
    "save_report_html",
]
