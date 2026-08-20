"""Canonical data model.

``core/`` imports from :mod:`embodied_sync.time` (for
:class:`~embodied_sync.time.ClockDomain`, which
:class:`~embodied_sync.core.StreamManifest` and
:class:`~embodied_sync.core.AlignmentPolicy` compose over) but from no
other ``embodied_sync`` subpackage. The dependency direction is
``time → core → {align, corrupt, datasets, reports, streams,
adapters, exporters}``.
"""

from embodied_sync.core.episode import (
    AlignedFrame,
    AlignedRun,
    AlignedSampleMetadata,
    AlignmentReport,
    Episode,
)
from embodied_sync.core.manifest import StreamManifest
from embodied_sync.core.policy import METHOD_ALIASES, AlignmentPolicy
from embodied_sync.core.sample import Modality, Sample
from embodied_sync.core.sync_report import (
    REPORT_FORMAT_VERSION,
    StreamStats,
    SyncQualityReport,
    SyncReport,
)

__all__ = [
    "METHOD_ALIASES",
    "REPORT_FORMAT_VERSION",
    "AlignedFrame",
    "AlignedRun",
    "AlignedSampleMetadata",
    "AlignmentPolicy",
    "AlignmentReport",
    "Episode",
    "Modality",
    "Sample",
    "StreamManifest",
    "StreamStats",
    "SyncQualityReport",
    "SyncReport",
]
