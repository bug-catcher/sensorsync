"""Live per-stream quality over a trailing window (design §2.6).

What this deliberately does *not* have is a composite headline score.
"98% synced" is meaningless without saying synced *to what tolerance*,
and a single number that folds rate, jitter and skew together can be
made to look healthy by any one of its terms. The only defensible
summary is :attr:`LiveStreamQuality.within_tolerance_rate` — the
fraction of matches that met the tolerance *the user configured*.

Alongside it, :attr:`LiveStreamQuality.problems` carries the named
failed predicates: ``["observed_rate_hz 21.2 < 0.8x expected 30.0"]``
rather than ``health=0.7``. An empty list means healthy. The shape is
modelled on ffsubsync's ``assess_alignment_quality`` — say which check
failed and with which numbers, so the message is the diagnosis.

Cost contract: the session keeps bounded deques of receive times and
match records per stream, so a :func:`quality` call is O(window) and
the memory does not grow with session length. Nothing is retained
outside those deques — a 10-hour session costs the same as a 10-second
one.

Statistics, stated so hand-checks are exact:

- ``observed_rate_hz`` — ``(n - 1) / span_s`` over the receive times
  inside the window, where ``span_s`` is last-minus-first. This is the
  interval-count rate, not ``n / window_s``: it does not undercount a
  stream whose window happens to start mid-interval. ``None`` for
  fewer than two receives.
- ``receive_jitter_ms`` — median absolute deviation of the inter-receive
  deltas (median of ``|d_i - median(d)|``), not standard deviation: one
  stalled delivery must not dominate the number. ``None`` for fewer
  than three receives (two deltas are not a distribution).
- ``p95_abs_skew_ms`` — nearest-rank percentile: sorted index
  ``ceil(0.95 * n) - 1``. No interpolation, so the value is always an
  observed skew.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

__all__ = [
    "MAX_MISSING_RATE",
    "MIN_RATE_FRACTION",
    "MIN_WITHIN_TOLERANCE_RATE",
    "LiveStreamQuality",
    "MatchRecord",
    "median",
    "nearest_rank",
]

#: An observed rate below this fraction of the configured ``rate_hz`` is a
#: problem. 0.8 is the same threshold the ingest-time rate check uses, so a
#: stream cannot be "healthy" in one report and warned about in the other.
MIN_RATE_FRACTION = 0.8
#: More than this fraction of matches missing is a problem.
MAX_MISSING_RATE = 0.05
#: Fewer than this fraction of matches within tolerance is a problem.
MIN_WITHIN_TOLERANCE_RATE = 0.95


@dataclass(frozen=True, slots=True)
class MatchRecord:
    """One stream's outcome from one :meth:`SyncSession.get` call."""

    target_time_ns: int
    skew_ns: int | None
    missing: bool
    within_tolerance: bool


@dataclass(frozen=True, slots=True)
class LiveStreamQuality:
    """Trailing-window quality for one stream.

    The first three fields describe *arrival* (what the stream did on
    its own); the rest describe *matching* (how it fared against
    ``get()`` targets). ``None`` means "not enough data in the window
    to say", which is different from zero and is never rendered as
    zero.
    """

    stream: str
    window_s: float
    observed_rate_hz: float | None
    expected_rate_hz: float | None
    receive_jitter_ms: float | None
    match_count: int
    missing_rate: float | None
    median_abs_skew_ms: float | None
    p95_abs_skew_ms: float | None
    within_tolerance_rate: float | None
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """JSON-ready mapping, used for ``session_quality.json`` at close."""
        return {
            "stream": self.stream,
            "window_s": self.window_s,
            "observed_rate_hz": self.observed_rate_hz,
            "expected_rate_hz": self.expected_rate_hz,
            "receive_jitter_ms": self.receive_jitter_ms,
            "match_count": self.match_count,
            "missing_rate": self.missing_rate,
            "median_abs_skew_ms": self.median_abs_skew_ms,
            "p95_abs_skew_ms": self.p95_abs_skew_ms,
            "within_tolerance_rate": self.within_tolerance_rate,
            "problems": list(self.problems),
        }


def median(values: Sequence[float]) -> float:
    """Median of a non-empty sequence (even counts average the two middles)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile of a non-empty sequence, no interpolation."""
    ordered = sorted(values)
    index = math.ceil(percentile * len(ordered)) - 1
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def compute_stream_quality(
    *,
    stream: str,
    window_s: float,
    receive_times_ns: Sequence[int],
    matches: Sequence[MatchRecord],
    expected_rate_hz: float | None,
    tolerance_ns: int,
) -> LiveStreamQuality:
    """Build a :class:`LiveStreamQuality` from already-windowed inputs.

    The caller is responsible for restricting ``receive_times_ns`` and
    ``matches`` to the trailing window — this function is pure so the
    arithmetic can be unit-tested against hand-computed values without
    constructing a session.
    """
    problems: list[str] = []

    observed_rate_hz: float | None = None
    receive_jitter_ms: float | None = None
    if len(receive_times_ns) >= 2:
        span_ns = receive_times_ns[-1] - receive_times_ns[0]
        if span_ns > 0:
            observed_rate_hz = (len(receive_times_ns) - 1) / (span_ns / 1e9)
    if len(receive_times_ns) >= 3:
        deltas = [
            float(b - a) for a, b in zip(receive_times_ns, receive_times_ns[1:])
        ]
        centre = median(deltas)
        receive_jitter_ms = median([abs(d - centre) for d in deltas]) / 1e6

    if not receive_times_ns:
        problems.append(f"no samples received in the last {window_s:g} s")
    elif expected_rate_hz is not None and observed_rate_hz is not None:
        if observed_rate_hz < MIN_RATE_FRACTION * expected_rate_hz:
            problems.append(
                f"observed_rate_hz {observed_rate_hz:.1f} < "
                f"{MIN_RATE_FRACTION:g}x expected {expected_rate_hz:.1f}"
            )

    match_count = len(matches)
    missing_rate: float | None = None
    within_tolerance_rate: float | None = None
    median_abs_skew_ms: float | None = None
    p95_abs_skew_ms: float | None = None
    if match_count:
        missing = sum(1 for m in matches if m.missing)
        missing_rate = missing / match_count
        within = sum(1 for m in matches if m.within_tolerance)
        within_tolerance_rate = within / match_count
        abs_skews = [abs(m.skew_ns) for m in matches if m.skew_ns is not None]
        if abs_skews:
            median_abs_skew_ms = median(abs_skews) / 1e6
            p95_abs_skew_ms = nearest_rank(abs_skews, 0.95) / 1e6
        if missing_rate > MAX_MISSING_RATE:
            problems.append(
                f"missing_rate {missing_rate:.2f} > {MAX_MISSING_RATE:g}"
            )
        if within_tolerance_rate < MIN_WITHIN_TOLERANCE_RATE:
            problems.append(
                f"within_tolerance_rate {within_tolerance_rate:.2f} < "
                f"{MIN_WITHIN_TOLERANCE_RATE:g}"
            )
        if (
            p95_abs_skew_ms is not None
            and tolerance_ns > 0
            and p95_abs_skew_ms > tolerance_ns / 1e6
        ):
            problems.append(
                f"p95_abs_skew_ms {p95_abs_skew_ms:.2f} > "
                f"tolerance {tolerance_ns / 1e6:.2f} ms"
            )

    return LiveStreamQuality(
        stream=stream,
        window_s=window_s,
        observed_rate_hz=observed_rate_hz,
        expected_rate_hz=expected_rate_hz,
        receive_jitter_ms=receive_jitter_ms,
        match_count=match_count,
        missing_rate=missing_rate,
        median_abs_skew_ms=median_abs_skew_ms,
        p95_abs_skew_ms=p95_abs_skew_ms,
        within_tolerance_rate=within_tolerance_rate,
        problems=problems,
    )
