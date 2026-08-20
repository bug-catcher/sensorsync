"""Event-train matching: two lists of times, one clock mapping.

This is the shared engine behind clap alignment *and* semantic-event
alignment. The difference between "two microphones heard the same clap"
and "the wrist camera and the force sensor both saw the gripper close"
is entirely in who detected the events. Once you have two arrays of
integer-nanosecond event times that partly correspond, the problem is
the same one, and it is a metrology problem rather than a perception
one — which is exactly where this library's boundary sits.

The hard part is that the correspondence is unknown *and* imperfect:
train A has events train B missed, train B has spurious detections, and
the two clocks differ by an unknown offset plus an unknown drift. So:

1. **Coarse scan.** Histogram every pairwise difference ``b_j − a_i``
   that falls inside ``±max_offset_ms``. For sparse event trains this
   *is* the cross-correlation of the binned trains, computed directly
   from the events instead of from a dense buffer — the true offset
   shows up as a spike because every genuine pair contributes to the
   same bin, while spurious pairs spread out.
2. **Greedy one-to-one matching** inside a gate around the coarse
   offset, best-residual first. One-to-one matters: without it, a
   cluster of spurious detections can all match the same real event and
   outvote the honest pairs.
3. **Robust fit** (:func:`~embodied_sync.calibrate.estimator.fit_clock_mapping`).
4. **Re-match and re-fit once**, now with the drift applied. The first
   pass has to use a gate wide enough to survive the drift it hasn't
   measured yet; the second pass can be tight, which recovers matches
   at the ends of a long recording where the drift error is largest.

Unmatched and spurious events are handled by construction — they simply
never enter the fit — and are reported as ``matched_fraction_a`` /
``matched_fraction_b`` rather than being silently dropped. A result with
95% of A matched is a different claim from one with 30%, and the caller
must be able to tell them apart.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray

from embodied_sync.calibrate.estimator import (
    AMBIGUITY_EXCLUSION_BINS,
    ClockMappingFit,
    fit_clock_mapping,
    score_confidence,
)
from embodied_sync.time.clock_domain import ClockDomain, translate_ns

__all__ = ["EventTrainAlignment", "match_event_trains"]

#: Target number of bins across the whole ``±max_offset`` search range, so
#: the coarse peak is resolved finely enough to seed the fit even when the
#: matching gate is wide. Odd, so one bin is centred on zero offset.
_COARSE_BINS = 401

#: Default matching gate as a fraction of the median inter-event interval.
#: A gate wider than this starts pairing an event with its *neighbour*
#: rather than its counterpart.
_DEFAULT_GATE_FRACTION = 0.1
_MIN_GATE_NS = 1_000_000  # 1 ms

#: Guard on the coarse scan's pair enumeration. Two 10k-event trains that
#: overlap entirely would enumerate 1e8 differences; refuse rather than
#: quietly allocating gigabytes.
_MAX_COARSE_PAIRS = 4_000_000

_PPM_TO_PPB = 1_000

#: Theil-Sen's offset is a *median* of residuals, whose asymptotic standard
#: error is sqrt(pi/2)*sigma/sqrt(N) for Gaussian noise. Phase 0 measured the
#: estimator sitting essentially on this bound, so it is a fair predictor of
#: the offset error rather than an optimistic one.
_MEDIAN_EFFICIENCY = 1.2533141373155003


@dataclass(frozen=True, slots=True)
class EventTrainAlignment:
    """Result of matching two event trains.

    ``matched`` holds ``(index_into_a, index_into_b)`` pairs, sorted by
    ``index_into_a`` — index pairs rather than times so the caller can
    trace a suspicious residual back to the event that produced it.

    ``residual_p95_ns`` is the nearest-rank 95th percentile of
    ``|residual|`` over matched pairs: the "how wrong is this mapping,
    excluding the worst 5%" number that belongs next to a tolerance.
    """

    fit: ClockMappingFit
    matched: list[tuple[int, int]]
    matched_fraction_a: float
    matched_fraction_b: float
    residual_p95_ns: int
    confidence: float
    #: Predicted 1-sigma error of ``offset_ns``, in ns. This is the number
    #: to compare against a tolerance; ``confidence`` is the same evidence
    #: squashed to [0, 1] against the matching gate.
    offset_stderr_ns: int = 0

    @property
    def problems(self) -> tuple[str, ...]:
        """Named reasons the fit cannot support part of what it returns."""
        return self.fit.problems

    @property
    def offset_ns(self) -> int:
        """Convenience: the fitted offset at the anchor."""
        return self.fit.mapping.offset_ns

    @property
    def drift_ppb(self) -> int:
        """Convenience: the fitted drift in parts per billion."""
        return self.fit.mapping.drift_ppb


def _sorted_int_array(values: Sequence[int], name: str) -> NDArray[np.int64]:
    array = np.asarray(list(values), dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D sequence, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one event")
    return np.sort(array)


def _median_interval_ns(events: NDArray[np.int64]) -> int:
    if events.size < 2:
        return 0
    return int(np.median(np.diff(events)))


def _pairwise_differences(
    a: NDArray[np.int64], b: NDArray[np.int64], max_offset_ns: int
) -> NDArray[np.int64]:
    """All ``b_j − a_i`` within ``±max_offset_ns``, using sorted-range lookup."""
    lo = np.searchsorted(b, a - max_offset_ns, side="left")
    hi = np.searchsorted(b, a + max_offset_ns, side="right")
    total = int(np.sum(hi - lo))
    if total > _MAX_COARSE_PAIRS:
        raise ValueError(
            f"coarse scan would enumerate {total} candidate pairs (limit "
            f"{_MAX_COARSE_PAIRS}); narrow max_offset_ms or thin the event "
            f"trains before calling match_event_trains"
        )
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    chunks = [b[lo[i] : hi[i]] - a[i] for i in range(a.size) if hi[i] > lo[i]]
    return np.concatenate(chunks)


class _CoarseScan(NamedTuple):
    """Outcome of the binned cross-correlation scan."""

    offset_ns: int
    scores: NDArray[np.float64]
    peak_index: int
    exclusion_bins: int


def _coarse_offset(
    a: NDArray[np.int64],
    b: NDArray[np.int64],
    max_offset_ns: int,
    drift_slack_ns: int,
) -> _CoarseScan:
    """Scan ``±max_offset_ns`` for the offset that lines the trains up.

    The raw histogram bins ``b_j − a_i`` *without* compensating for
    drift, so a real drift smears the peak: at 120 ppm across 60 s the
    true differences spread over 7.2 ms, which at fine binning becomes
    a dozen bins of one count each instead of one bin of twelve. Left
    alone that reads as "no peak" — the smear's own bins become each
    other's runner-up and the ambiguity penalty zeroes a perfect match.

    So the curve is smoothed with a moving sum whose width is exactly
    the drift-induced spread the caller declared possible
    (``max_drift_ppm × span``). The smear re-accumulates into one
    position, the background stays finely sampled, and with
    ``max_drift_ppm=0`` the window is one bin and nothing changes.

    The reported offset is the smoothed peak's centre, which sits mid-
    smear rather than at the anchor — off by up to half the spread. The
    first-pass matching gate carries the same slack, so the pairing
    still lands, and the fit corrects the offset properly.

    Smoothing has a second consequence that must be handled with it: a
    moving sum of width ``w`` makes every position within ``w`` of the
    peak share samples with it, producing a plateau rather than a
    spike. Those positions are the peak, not rivals to it, so the
    returned ``exclusion_bins`` widens to ``w`` — otherwise the
    ambiguity penalty compares the peak against itself and scores a
    flawless match at zero.
    """
    bin_ns = max(1, (2 * max_offset_ns) // _COARSE_BINS)
    edges = np.arange(-max_offset_ns, max_offset_ns + bin_ns + 1, bin_ns)
    diffs = _pairwise_differences(a, b, max_offset_ns)
    if diffs.size == 0:
        empty = np.zeros(max(1, edges.size - 1), dtype=np.float64)
        return _CoarseScan(0, empty, 0, AMBIGUITY_EXCLUSION_BINS)
    counts, _ = np.histogram(diffs, bins=edges)
    scores = counts.astype(np.float64)
    width = max(1, int(round(drift_slack_ns / bin_ns)))
    if width % 2 == 0:
        width += 1
    if width > 1 and width < scores.size:
        scores = np.convolve(scores, np.ones(width), mode="same")
    else:
        width = 1
    peak_index = int(np.argmax(scores))
    centre = int(edges[peak_index]) + bin_ns // 2
    return _CoarseScan(
        centre, scores, peak_index, max(AMBIGUITY_EXCLUSION_BINS, width)
    )


def _greedy_match(
    a: NDArray[np.int64],
    predicted: NDArray[np.int64],
    b: NDArray[np.int64],
    gate_ns: int,
) -> list[tuple[int, int]]:
    """One-to-one match ``predicted[i]`` against ``b[j]`` within ``gate_ns``.

    Candidates are consumed best-residual first; ties break on
    ``(residual, i, j)`` so the result is a pure function of the inputs.
    """
    lo = np.searchsorted(b, predicted - gate_ns, side="left")
    hi = np.searchsorted(b, predicted + gate_ns, side="right")
    candidates: list[tuple[int, int, int]] = []
    for i in range(predicted.size):
        for j in range(int(lo[i]), int(hi[i])):
            candidates.append((abs(int(b[j]) - int(predicted[i])), i, j))
    candidates.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _residual, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j))
    matched.sort()
    return matched


def _nearest_rank_abs(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(abs(v) for v in values)
    index = int(np.ceil(percentile * len(ordered))) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def match_event_trains(
    events_a_ns: Sequence[int],
    events_b_ns: Sequence[int],
    *,
    max_offset_ms: float,
    max_drift_ppm: float = 500.0,
    match_tolerance_ms: float | None = None,
    source_domain: ClockDomain | str | None = None,
    target_domain: ClockDomain | str | None = None,
) -> EventTrainAlignment:
    """Align two event trains and fit the clock mapping ``a → b``.

    :param events_a_ns: source-clock event times (integer ns, any order).
    :param events_b_ns: target-clock event times (integer ns, any order).
    :param max_offset_ms: half-width of the coarse offset search. Make it
        comfortably larger than the offset you expect; too small and the
        true peak is simply outside the scan.
    :param max_drift_ppm: the caller's prior on how badly the two clocks
        can differ in *rate*. It widens the first-pass matching gate by
        ``max_drift · span`` (the mismatch an unmeasured drift can
        produce across the recording), and a fitted drift that exceeds
        it raises a :class:`UserWarning` — the prior was wrong, and the
        caller should know rather than receive a quietly clamped number.
    :param match_tolerance_ms: matching gate. Defaults to 10% of the
        median inter-event interval (floored at 1 ms), which is the
        scale at which "the next event" stops being a plausible partner.

    Fitting needs at least one matched pair; zero matches is a
    :class:`ValueError` naming ``max_offset_ms`` as the likely cause,
    because that is what it usually is.
    """
    a = _sorted_int_array(events_a_ns, "events_a_ns")
    b = _sorted_int_array(events_b_ns, "events_b_ns")
    if max_offset_ms <= 0:
        raise ValueError(f"max_offset_ms must be > 0, got {max_offset_ms}")
    if max_drift_ppm < 0:
        raise ValueError(f"max_drift_ppm must be >= 0, got {max_drift_ppm}")
    max_offset_ns = round(max_offset_ms * 1e6)

    if match_tolerance_ms is not None:
        if match_tolerance_ms <= 0:
            raise ValueError(
                f"match_tolerance_ms must be > 0, got {match_tolerance_ms}"
            )
        gate_ns = round(match_tolerance_ms * 1e6)
    elif min(a.size, b.size) < 2:
        # A10. The default gate is a fraction of the median inter-event
        # interval — "the scale at which the next event stops being a
        # plausible partner". With a single event on one side there is no
        # next event and no interval, and the old code fell through to the
        # 1 ms floor. That floor is narrower than the coarse scan's own bin
        # (5 ms at the default max_offset), so the scan would locate the
        # peak and the gate would then reject it: a lone, perfectly good
        # pair raised "no events matched". With no interval to measure,
        # the honest gate is the whole search window the caller declared.
        gate_ns = max_offset_ns
    else:
        interval_ns = _median_interval_ns(a) or _median_interval_ns(b)
        gate_ns = max(
            _MIN_GATE_NS, int(round(_DEFAULT_GATE_FRACTION * interval_ns))
        )
    gate_ns = min(gate_ns, max_offset_ns)

    span_ns = int(a[-1] - a[0])
    drift_slack_ns = int(round(max_drift_ppm * 1e-6 * span_ns))
    first_gate_ns = min(gate_ns + drift_slack_ns, max_offset_ns)

    scan = _coarse_offset(a, b, max_offset_ns, drift_slack_ns)
    coarse_ns = scan.offset_ns

    matched = _greedy_match(a, a + coarse_ns, b, first_gate_ns)
    if not matched:
        raise ValueError(
            f"no events matched within {first_gate_ns / 1e6:.3f} ms of the "
            f"coarse offset {coarse_ns / 1e6:.3f} ms; the true offset is "
            f"probably outside max_offset_ms={max_offset_ms:g}"
        )

    anchor_ns = int(a[0])
    fit = fit_clock_mapping(
        [int(a[i]) for i, _ in matched],
        [int(b[j]) for _, j in matched],
        anchor_ns=anchor_ns,
        source_domain=source_domain,
        target_domain=target_domain,
    )

    # Second pass: the drift is now known, so the gate can tighten back to
    # the caller's tolerance and pick up pairs the wide first gate mis-paired.
    predicted = np.asarray(
        [translate_ns(int(t), fit.mapping) for t in a.tolist()], dtype=np.int64
    )
    rematched = _greedy_match(a, predicted, b, gate_ns)
    if rematched:
        matched = rematched
        fit = fit_clock_mapping(
            [int(a[i]) for i, _ in matched],
            [int(b[j]) for _, j in matched],
            anchor_ns=anchor_ns,
            source_domain=source_domain,
            target_domain=target_domain,
        )

    if abs(fit.mapping.drift_ppb) > max_drift_ppm * _PPM_TO_PPB:
        warnings.warn(
            f"fitted drift {fit.mapping.drift_ppb / _PPM_TO_PPB:.1f} ppm exceeds "
            f"max_drift_ppm={max_drift_ppm:g}; returning it unclamped — either "
            f"the prior is too tight or the match is wrong",
            stacklevel=2,
        )

    # Confidence has two independent failure modes and needs both halves.
    #
    # The coarse peak answers "is this alignment distinctive?" — it catches
    # a periodic signal aliasing against itself. But on sparse event trains
    # its ambiguity penalty (1 - runner_up/peak) collapses to exactly zero
    # whenever any other histogram bin ties the peak, which with integer
    # counts happens constantly. Measured on real data, that made confidence
    # a *gate* rather than a *ranking*: trials scoring 0 had ten times the
    # error of the rest, but among non-zero scores it barely ordered them.
    #
    # So it is multiplied by the fit's own precision: the predicted standard
    # error of the offset, compared against the gate the events were matched
    # at. An underdetermined fit carries a one-second variance floor, which
    # drives this to ~0 — which is the point. A tight fit over many pairs
    # drives it toward 1.
    peak_confidence = score_confidence(
        scan.scores, scan.peak_index, exclusion_bins=scan.exclusion_bins
    )
    n_matched = max(1, len(matched))
    offset_stderr_ns = int(
        round(_MEDIAN_EFFICIENCY * fit.mapping.variance_ns / np.sqrt(n_matched))
    )
    # Explained share: the geometric mean of how much of each train the
    # mapping accounts for. This carries the weight, and the reason is
    # uncomfortable but important — *residual-based signals do not predict
    # this estimator's error at all*. Across validation trials, correlation
    # against absolute offset error was weak for the offset standard error,
    # residual p95, and inlier fraction. The reason is structural. When the
    # coarse scan aliases onto the wrong pairing, it fits a *consistent*
    # line through the wrong pairs, so the residuals are small and the fit
    # looks excellent while the answer is wrong by a whole event interval.
    # Residuals can only see whether the chosen pairs agree with each
    # other, never whether they were the right pairs.
    #
    # What does predict error is how much of the evidence the hypothesis
    # explains (-0.29 and -0.66 respectively): a wrong alignment pairs a
    # subset, the true one pairs most of what is there. That is the same
    # quantity RANSAC maximises, not an arbitrary feature fished out of the
    # benchmark.
    explained = float(
        np.sqrt(
            (len(matched) / int(a.size)) * (len(matched) / int(b.size))
        )
    )
    return EventTrainAlignment(
        fit=fit,
        matched=matched,
        matched_fraction_a=len(matched) / int(a.size),
        matched_fraction_b=len(matched) / int(b.size),
        residual_p95_ns=_nearest_rank_abs(fit.residuals_ns, 0.95),
        confidence=peak_confidence * explained,
        offset_stderr_ns=offset_stderr_ns,
    )
