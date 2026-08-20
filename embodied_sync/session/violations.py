"""Structured sync violations, their dispatch, and warning rate limiting.

A live session that notices a problem has exactly three honest options:
say so, raise, or hand the fact to the caller. It never has the option
of staying quiet — the whole point of this library is that silent
mis-synchronisation is the failure mode that reaches the training set.

So every detected problem becomes a :class:`SyncViolation`: a value with
a machine-readable ``reason`` (one of the constants below — strings in
*one* place, so a caller can branch on them without matching prose) plus
the numbers that justify it. Dispatch is the session's ``on_violation``
setting:

``"warn"``   rate-limited :func:`logging.Logger.warning` (default)
``"raise"``  :class:`SyncToleranceError` carrying the violation
``"ignore"`` counted, not reported
callable     receives the :class:`SyncViolation`

Rate limiting (``"warn"`` only)
-------------------------------
A disconnected camera emits the same violation every tick. Logging all
of them buries the *first* one — the only one that carries information —
under hundreds of repeats, and in a real control loop the logging itself
becomes the bottleneck. :class:`RateLimiter` therefore admits at most one
record per ``(stream, reason)`` per interval (default 1 s), counts the
rest, and the session reports those counts at ``close()``.

Only the ``"warn"`` path is limited. ``"raise"`` raises on the first
violation, so limiting is moot; a callable is the caller's own code and
must see every event — a session-level filter would silently decimate
whatever it is accumulating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

__all__ = [
    "CLOCK_EPOCH_ADVANCED",
    "DEFAULT_VIOLATION_INTERVAL_S",
    "NON_MONOTONIC",
    "NO_ELIGIBLE_BEFORE_DEADLINE",
    "NO_SAMPLES",
    "OUTSIDE_TOLERANCE",
    "RATE_BELOW_EXPECTED",
    "UNMAPPED_CLOCK_DOMAIN",
    "VIOLATION_REASONS",
    "RateLimiter",
    "SyncToleranceError",
    "SyncViolation",
    "ViolationHandler",
]

#: The stream's buffer held nothing at all at match time.
NO_SAMPLES = "no_samples"
#: A sample existed but its skew exceeded the stream's configured tolerance.
OUTSIDE_TOLERANCE = "outside_tolerance"
#: Samples existed but none had been *received* by ``target + deadline``.
NO_ELIGIBLE_BEFORE_DEADLINE = "no_eligible_before_deadline"
#: An arriving sample's acquisition time went backwards.
NON_MONOTONIC = "non_monotonic"
#: Observed arrival rate fell below the configured fraction of ``rate_hz``.
RATE_BELOW_EXPECTED = "rate_below_expected"
#: The stream declares a foreign clock domain and no mapping is registered.
UNMAPPED_CLOCK_DOMAIN = "unmapped_clock_domain"
#: A clock domain opened a new generation (device reconnect, clock reset), so
#: everything fitted against the previous one was discarded. Not an error —
#: it is the *correct* response to a reset — but it is a discontinuity in the
#: recording and a caller that ignores it will wonder where its mapping went.
CLOCK_EPOCH_ADVANCED = "clock_epoch_advanced"

#: Every reason a session can emit. Exhaustive by construction: the session
#: asserts membership before dispatch, so a typo fails loudly at the source
#: instead of producing an un-branchable violation downstream.
VIOLATION_REASONS: frozenset[str] = frozenset(
    {
        NO_SAMPLES,
        OUTSIDE_TOLERANCE,
        NO_ELIGIBLE_BEFORE_DEADLINE,
        NON_MONOTONIC,
        RATE_BELOW_EXPECTED,
        UNMAPPED_CLOCK_DOMAIN,
        CLOCK_EPOCH_ADVANCED,
    }
)

#: Default minimum interval between two logged warnings for one
#: ``(stream, reason)`` pair.
DEFAULT_VIOLATION_INTERVAL_S = 1.0


@dataclass(frozen=True, slots=True)
class SyncViolation:
    """One detected sync problem, with the numbers that justify it.

    ``target_time_ns``, ``skew_ns`` and ``tolerance_ns`` are ``None``
    when the reason does not have them: an ingest-time violation
    (``non_monotonic``, ``rate_below_expected``) has no match target,
    and a stream with nothing buffered has no skew.
    """

    stream: str
    reason: str
    target_time_ns: int | None
    skew_ns: int | None
    tolerance_ns: int | None
    message: str


class SyncToleranceError(RuntimeError):
    """Raised by a session configured with ``on_violation="raise"``.

    Carries the :class:`SyncViolation` on ``.violation`` so an ``except``
    block can inspect the numbers rather than re-parse the message.
    """

    def __init__(self, violation: SyncViolation) -> None:
        super().__init__(violation.message)
        self.violation = violation


#: A user-supplied ``on_violation`` callable.
ViolationHandler = Callable[[SyncViolation], None]


class RateLimiter:
    """Per-key minimum-interval admission control with suppression counts.

    The clock is injected (same rule as the session: the library reads
    no wall clock of its own), so a test can drive suppression windows
    deterministically instead of sleeping.

    The first occurrence of a key is always admitted — the point is to
    protect the tail, never the head.
    """

    __slots__ = ("_clock", "_interval_ns", "_last_ns", "_suppressed")

    def __init__(self, *, interval_ns: int, clock: Callable[[], int]) -> None:
        if interval_ns < 0:
            raise ValueError(f"interval_ns must be >= 0, got {interval_ns}")
        self._interval_ns = interval_ns
        self._clock = clock
        self._last_ns: dict[tuple[str, str], int] = {}
        self._suppressed: dict[tuple[str, str], int] = {}

    def allow(self, key: tuple[str, str]) -> bool:
        """Return ``True`` if ``key`` may be reported now; else count it."""
        now = self._clock()
        last = self._last_ns.get(key)
        if last is None or now - last >= self._interval_ns:
            self._last_ns[key] = now
            return True
        self._suppressed[key] = self._suppressed.get(key, 0) + 1
        return False

    @property
    def suppressed(self) -> dict[tuple[str, str], int]:
        """Copy of the ``(stream, reason)`` → suppressed-count mapping."""
        return dict(self._suppressed)

    def total_suppressed(self) -> int:
        return sum(self._suppressed.values())
