"""Live synchronisation sessions (D-0037).

``session/`` is the drop-in surface: wrap your vendor SDK's callbacks (or
poll it), then ask for a synchronised bundle. It is a *composition* of
machinery that already existed —
:class:`~embodied_sync.align.ring_buffer.StreamRingBuffer` for the picks,
:mod:`embodied_sync.datasets.io`'s run format v0 for recording,
:mod:`embodied_sync.time` for clock-domain mappings — with the two things
a live user needs that offline code never had to provide: a clock, and an
opinion about what to do when synchronisation degrades.

Start at :func:`~embodied_sync.session.session.init` /
:class:`~embodied_sync.session.session.SyncSession`; both are re-exported
at the top level (``embodied_sync.init``, ``embodied_sync.SyncSession``,
``embodied_sync.StreamConfig``) through a lazy ``__getattr__`` so
``import embodied_sync`` stays a stdlib-only import.

Importing this subpackage pulls no optional dependency.
"""

from embodied_sync.session.approximate import (
    APPROXIMATE_METHOD,
    DEFAULT_QUEUE_CAPACITY,
    ApproximateSet,
    ApproximateTimeBundler,
)
from embodied_sync.session.bundle import BundleItem, SyncBundle
from embodied_sync.session.config import (
    KNOWN_PERSIST_MODES,
    KNOWN_POLICIES,
    POLICY_APPROXIMATE,
    POLICY_LATEST_BEFORE,
    POLICY_NEAREST,
    POLICY_WINDOW,
    StreamConfig,
)
from embodied_sync.session.quality import LiveStreamQuality, MatchRecord
from embodied_sync.session.recorder import SESSION_QUALITY_NAME, RunRecorder
from embodied_sync.session.session import (
    DEFAULT_QUALITY_WINDOW_S,
    DEFAULT_TIME_CORRECTION_MAX_AGE_S,
    REFERENCE_METHOD,
    SESSION_CLOCK_DOMAIN,
    SyncSession,
    init,
    logger,
)
from embodied_sync.session.violations import (
    CLOCK_EPOCH_ADVANCED,
    DEFAULT_VIOLATION_INTERVAL_S,
    NON_MONOTONIC,
    NO_ELIGIBLE_BEFORE_DEADLINE,
    NO_SAMPLES,
    OUTSIDE_TOLERANCE,
    RATE_BELOW_EXPECTED,
    UNMAPPED_CLOCK_DOMAIN,
    VIOLATION_REASONS,
    RateLimiter,
    SyncToleranceError,
    SyncViolation,
    ViolationHandler,
)

__all__ = [
    "APPROXIMATE_METHOD",
    "CLOCK_EPOCH_ADVANCED",
    "DEFAULT_QUALITY_WINDOW_S",
    "DEFAULT_QUEUE_CAPACITY",
    "DEFAULT_TIME_CORRECTION_MAX_AGE_S",
    "DEFAULT_VIOLATION_INTERVAL_S",
    "KNOWN_PERSIST_MODES",
    "KNOWN_POLICIES",
    "NON_MONOTONIC",
    "NO_ELIGIBLE_BEFORE_DEADLINE",
    "NO_SAMPLES",
    "OUTSIDE_TOLERANCE",
    "POLICY_APPROXIMATE",
    "POLICY_LATEST_BEFORE",
    "POLICY_NEAREST",
    "POLICY_WINDOW",
    "RATE_BELOW_EXPECTED",
    "REFERENCE_METHOD",
    "SESSION_CLOCK_DOMAIN",
    "SESSION_QUALITY_NAME",
    "UNMAPPED_CLOCK_DOMAIN",
    "VIOLATION_REASONS",
    "ApproximateSet",
    "ApproximateTimeBundler",
    "BundleItem",
    "LiveStreamQuality",
    "MatchRecord",
    "RateLimiter",
    "RunRecorder",
    "StreamConfig",
    "SyncBundle",
    "SyncSession",
    "SyncToleranceError",
    "SyncViolation",
    "ViolationHandler",
    "init",
    "logger",
]
