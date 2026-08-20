"""Robust clock-mapping estimator — the shared numeric core of `calibrate/`.

Every calibrator in this subpackage ends here, and every one of them
outputs the same type: a
:class:`~embodied_sync.time.clock_domain.LatencyEstimate`. That is the
coherence spine of the design — *calibration produces the mapping,
`session` and `align` consume it* via
:func:`~embodied_sync.time.clock_domain.translate_ns`. A clap, a QR
code on a monitor, and a semantic event detector differ only in who
produced the paired times; the fit is one routine.

The model
---------
``target ≈ source + offset + drift·(source − anchor)``, fitted with
Theil–Sen: the slope is the median of pairwise slopes and the offset is
the median residual. Theil–Sen rather than least squares because
calibration data is exactly where outliers live — one mis-detected clap,
one dropped frame, one spurious contact event. Least squares lets a
single bad pair rotate the whole line; Theil–Sen tolerates up to ~29%
bad pairs before it does.

One clap gives you an offset. Many claps give you an offset *and* a
drift, and expose which claps disagree. That distinction is not a
nicety: an offset-only calibration applied across a 20-minute session
with a 100 ppm crystal mismatch is 120 ms wrong by the end.

Units and rounding
------------------
Timestamps are integer nanoseconds everywhere (D-0002). Fitting happens
in float64 *after* subtracting the anchor, because raw epoch-ns values
(~1e18) exceed float64's exact-integer range (2^53 ≈ 9e15) — subtracting
first keeps every intermediate exact. The reported ``residuals_ns`` are
then recomputed against the *rounded* integer mapping via
``translate_ns``, so they describe what the returned mapping actually
does rather than what an unrounded intermediate would have done.

``LatencyEstimate.variance_ns`` holds a robust **scale** in nanoseconds
(1.4826·MAD, the consistent-for-Gaussian estimator of σ), not a squared
quantity — that is the unit
:func:`~embodied_sync.time.alignment.cross_domain_confidence_factor`
compares against a tolerance.

Confidence, once, for everyone
------------------------------
:func:`score_confidence` is the shared metric so a clap result and a QR
result are directly comparable. It is the peak's :func:`standard_score`
(z-score against the rest of the score curve — audio-offset-finder's
Apache-2.0 idea) squashed into ``[0, 1]``, multiplied by an ambiguity
penalty derived from the runner-up peak (audalign's margin-to-second-best).
A tall peak in a flat curve scores high; a tall peak with an almost-as-tall
rival scores low, which is the honest answer — that is what a periodic
signal aliasing against itself looks like.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from embodied_sync.time.clock_domain import (
    INITIAL_EPOCH,
    KNOWN_DOMAINS,
    ClockDomain,
    LatencyEstimate,
    require_same_epoch,
    resolve_clock_domain,
    translate_ns,
)

__all__ = [
    "AMBIGUITY_EXCLUSION_BINS",
    "SINGLE_PAIR_VARIANCE_NS",
    "ClockMappingFit",
    "fit_clock_mapping",
    "score_confidence",
    "standard_score",
]

#: Robust-scale constant: ``1.4826 * MAD`` is a consistent estimator of the
#: standard deviation for Gaussian data.
_MAD_TO_SIGMA = 1.4826

#: Pairwise-slope budget for Theil–Sen. Above this the pairs are subsampled
#: with a fixed-seed generator, so the result stays deterministic — a
#: calibration that changed between runs on the same data would be useless
#: as evidence.
_MAX_SLOPE_PAIRS = 20_000
_SUBSAMPLE_SEED = 0

#: Point count above which the full pair set is not even enumerated.
_FULL_PAIRS_LIMIT = 200

#: ``variance_ns`` for a single-pair fit. A single pair determines an offset
#: and says *nothing* about drift; the honest variance is infinite, and
#: ``variance_ns`` is an ``int``, so this 1-second floor is the encoding.
#: Any sane tolerance drives
#: ``cross_domain_confidence_factor`` to ~0 against it, which is the intent:
#: usable as an offset, never mistakable for a verified mapping.
SINGLE_PAIR_VARIANCE_NS = 1_000_000_000

#: Bins on each side of the peak excluded when looking for the runner-up.
#: A real peak spreads over its neighbours; counting that spread as a rival
#: would penalise every good match.
AMBIGUITY_EXCLUSION_BINS = 2

#: z at which the squashed confidence reaches 0.5. Same saturating shape as
#: ``cross_domain_confidence_factor``: ``x / (x + k)``.
_Z_HALF = 6.0

#: Stand-in z for a peak rising out of a *perfectly flat* background, where
#: the true z-score is undefined (division by a zero standard deviation).
#: Sparse event trains hit this constantly — every genuine pair lands in one
#: bin and the rest of the curve is exactly zero — and it is the strongest
#: evidence the scan can produce, not the weakest. Returning 0 there (the
#: naive reading of "no scale to measure against") would report a perfect
#: match as no match at all.
_FLAT_BACKGROUND_Z = 50.0

_PPB = 1_000_000_000

#: Pairs needed before a drift term is reported at all. Two points define a
#: line exactly, so their residuals are zero by construction and every
#: residual-derived quality signal reports perfection — the fit looks
#: *better* the less data supports it (Phase 0, gate G0). Three points
#: leave a single degree of freedom, which is enough to compute a residual
#: and not enough to trust one, so A7's wording covers both: "with 2-3
#: events the fit is underdetermined". Four is the first count that can
#: actually contradict itself. The physical protocol's "3-5 claps" is
#: therefore a floor for *offset*; drift wants the upper half of that range.
MIN_DRIFT_PAIRS = 4

#: A drift must exceed this many standard errors of its own slope estimate
#: before it is reported. Two sigma is the usual significance bar, and
#: without it short recordings return confident nonsense: Phase 1 measured
#: a median 551 ppm recovered against an injected 10 ppm on ~6.5 s clips,
#: where the true accumulated skew is 0.07 ms — far below detection jitter.
DRIFT_SNR = 2.0

#: Problem strings attached to a fit that cannot support what it returns.
PROBLEM_UNDERDETERMINED = "underdetermined_fit"
PROBLEM_DRIFT_UNRESOLVABLE = "drift_unresolvable"
PROBLEM_DRIFT_TOO_FEW_PAIRS = "drift_needs_more_pairs"


@dataclass(frozen=True, slots=True)
class ClockMappingFit:
    """A fitted mapping plus the evidence for it.

    ``residuals_ns[i]`` is ``target_i - translate_ns(source_i, mapping)``:
    what the returned mapping leaves unexplained, in integer ns.
    ``inlier_fraction`` is the share of pairs within 3σ of the residual
    median — the number that says "12 of 14 claps agree, look at the
    other two" rather than hiding them in an average.
    """

    mapping: LatencyEstimate
    residuals_ns: list[int]
    n_pairs: int
    inlier_fraction: float
    #: Named reasons the fit cannot support part of what it returns —
    #: empty when nothing is wrong. A caller that only reads ``mapping``
    #: still gets a usable answer; one that reads ``problems`` learns that
    #: the drift was refused, or that the residuals are zero because the
    #: fit is underdetermined rather than because it is good.
    problems: tuple[str, ...] = ()

    @property
    def drift_reported(self) -> bool:
        """False when the drift term was refused as unsupportable."""
        return not any(
            p.startswith((PROBLEM_DRIFT_UNRESOLVABLE, PROBLEM_DRIFT_TOO_FEW_PAIRS))
            for p in self.problems
        )

    @property
    def residual_scale_ns(self) -> int:
        """Robust residual scale (``mapping.variance_ns``), in ns."""
        return self.mapping.variance_ns


def _as_int_array(values: Sequence[int], name: str) -> NDArray[np.int64]:
    array = np.asarray(list(values), dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D sequence, got shape {array.shape}")
    return array


def _resolve_domain(value: ClockDomain | str | None) -> ClockDomain:
    if value is None:
        return KNOWN_DOMAINS["unknown"]
    if isinstance(value, ClockDomain):
        return value
    return resolve_clock_domain(value)


def _median_abs_deviation(residuals: NDArray[np.float64]) -> float:
    centre = float(np.median(residuals))
    return float(np.median(np.abs(residuals - centre)))


def _theil_sen_slope(x: NDArray[np.float64], d: NDArray[np.float64]) -> float:
    """Median of pairwise slopes ``Δd/Δx``, subsampled above the pair budget."""
    n = x.size
    total_pairs = n * (n - 1) // 2
    if n <= _FULL_PAIRS_LIMIT and total_pairs <= _MAX_SLOPE_PAIRS:
        i, j = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(_SUBSAMPLE_SEED)
        i = rng.integers(0, n, size=_MAX_SLOPE_PAIRS)
        j = rng.integers(0, n, size=_MAX_SLOPE_PAIRS)
    dx = x[j] - x[i]
    valid = dx != 0
    if not np.any(valid):
        # Every source time is identical (or every sampled pair collided):
        # the data cannot see a slope. Offset-only is the honest answer.
        return 0.0
    slopes = (d[j][valid] - d[i][valid]) / dx[valid]
    return float(np.median(slopes))


def fit_clock_mapping(
    times_source_ns: Sequence[int],
    times_target_ns: Sequence[int],
    *,
    anchor_ns: int | None = None,
    method: str = "theil_sen",
    source_domain: ClockDomain | str | None = None,
    target_domain: ClockDomain | str | None = None,
    epochs: Sequence[int] | int | None = None,
) -> ClockMappingFit:
    """Fit ``target ≈ source + offset + drift·(source − anchor)`` robustly.

    ``times_source_ns[i]`` and ``times_target_ns[i]`` are the *same
    physical event* as timed by two clocks — the pairing is the caller's
    (or :func:`~embodied_sync.calibrate.events.match_event_trains`'s)
    job, not this function's.

    ``anchor_ns`` defaults to the first source time. The anchor only
    moves where the offset is exact; it does not change the fit's
    quality. Both parameters are integers (ns and ppb), so re-anchoring
    the *same* mapping re-rounds ``offset_ns`` and can shift a
    translated timestamp by ±1 ns — exactness is available at one
    anchor, not at all of them. ``method`` accepts ``"theil_sen"`` today — the argument
    exists so a future least-squares or RANSAC variant is a value, not a
    new function.

    ``source_domain`` / ``target_domain`` name the clocks. They default
    to ``"unknown"``, which is deliberately useless for a session:
    :meth:`SyncSession.register_clock_mapping
    <embodied_sync.session.SyncSession.register_clock_mapping>` requires
    a mapping that actually lands in the session domain, so a caller
    wiring calibration into a live session must say which clocks these
    were. Naming a clock is the point of the exercise.

    ``epochs`` is the source domain's generation for each pair (or one
    generation for all of them). Supplying it makes the fitter refuse a
    batch that straddles a clock reset with
    :class:`~embodied_sync.time.clock_domain.ClockEpochError`, and
    stamps the returned mapping with the generation it was fitted in.
    Two clean segments either side of a reconnect otherwise fit a single
    confident line *through* the discontinuity — small residuals, high
    inlier fraction, and a drift that is pure artefact. Omit it and the
    mapping carries
    :data:`~embodied_sync.time.clock_domain.INITIAL_EPOCH`, which reads
    as "the first (or only) generation".

    A single pair yields an offset-only mapping (``drift_ppb=0``) with
    :data:`SINGLE_PAIR_VARIANCE_NS`. **Two** pairs yield a drift that
    fits both exactly and residuals that are therefore identically zero:
    ``variance_ns`` will be ``0``, which reflects the algebra and not
    any verification. Treat ``n_pairs < 3`` as an unverified drift.
    """
    if method != "theil_sen":
        raise ValueError(f"unknown method {method!r}; known methods: ['theil_sen']")
    source = _as_int_array(times_source_ns, "times_source_ns")
    target = _as_int_array(times_target_ns, "times_target_ns")
    if source.size != target.size:
        raise ValueError(
            f"times_source_ns and times_target_ns must be the same length, "
            f"got {source.size} and {target.size}"
        )
    if source.size == 0:
        raise ValueError("fit_clock_mapping needs at least one paired time")

    anchor = int(source[0]) if anchor_ns is None else int(anchor_ns)
    domains = {
        "source": _resolve_domain(source_domain),
        "target": _resolve_domain(target_domain),
    }
    if epochs is None:
        epoch = INITIAL_EPOCH
    elif isinstance(epochs, bool):
        raise TypeError("epochs must be an int or a sequence of ints, got bool")
    elif isinstance(epochs, int):
        epoch = epochs
    else:
        supplied = list(epochs)
        if len(supplied) != source.size:
            raise ValueError(
                f"epochs must hold one generation per pair (or a single int), "
                f"got {len(supplied)} for {source.size} pairs"
            )
        epoch = require_same_epoch(supplied, context="calibration pairs")

    if source.size == 1:
        mapping = LatencyEstimate(
            source=domains["source"],
            target=domains["target"],
            offset_ns=int(target[0]) - int(source[0]),
            drift_ppb=0,
            anchor_time_ns=anchor,
            variance_ns=SINGLE_PAIR_VARIANCE_NS,
            epoch=epoch,
        )
        return ClockMappingFit(
            mapping=mapping,
            residuals_ns=[int(target[0]) - translate_ns(int(source[0]), mapping)],
            n_pairs=1,
            inlier_fraction=1.0,
            problems=(
                f"{PROBLEM_DRIFT_TOO_FEW_PAIRS}: 1 pair supports an offset only",
                f"{PROBLEM_UNDERDETERMINED}: variance_ns is a floor, not a measurement",
            ),
        )

    # Subtract in integer space first: raw ns values exceed float64's exact
    # integer range, the anchored differences do not.
    x = (source - anchor).astype(np.float64)
    d = (target - source).astype(np.float64)
    n = int(source.size)

    def _evaluate(drift_ppb: int) -> tuple[int, list[int], float]:
        """Offset, residuals and robust scale for a given (rounded) drift."""
        offset = int(round(float(np.median(d - (drift_ppb / _PPB) * x))))
        probe = LatencyEstimate(
            source=domains["source"],
            target=domains["target"],
            offset_ns=offset,
            drift_ppb=drift_ppb,
            anchor_time_ns=anchor,
            variance_ns=0,
            epoch=epoch,
        )
        res = [
            int(t) - translate_ns(int(s_), probe)
            for s_, t in zip(source.tolist(), target.tolist())
        ]
        scale = _MAD_TO_SIGMA * _median_abs_deviation(
            np.asarray(res, dtype=np.float64)
        )
        return offset, res, scale

    slope = _theil_sen_slope(x, d)
    drift_ppb = int(round(slope * _PPB))
    offset_ns, residuals, sigma = _evaluate(drift_ppb)

    problems: list[str] = []

    # --- A7/A9: is the drift term supportable at all? --------------------
    #
    # Two questions, and both must be answered yes. First, are there enough
    # pairs that the residuals mean anything (with n == 2 the line passes
    # through both points and every residual is zero by construction).
    # Second, is the fitted slope bigger than its own uncertainty? The
    # standard error of a slope is sigma / sqrt(Sxx); a drift smaller than a
    # couple of those is indistinguishable from zero, and reporting it is
    # inventing a number the data cannot see.
    spread = float(np.sum((x - float(np.mean(x))) ** 2))
    drift_sigma_ppb = (
        (sigma / np.sqrt(spread)) * _PPB if spread > 0.0 and sigma > 0.0 else 0.0
    )
    if n < MIN_DRIFT_PAIRS:
        problems.append(
            f"{PROBLEM_DRIFT_TOO_FEW_PAIRS}: {n} pairs, need {MIN_DRIFT_PAIRS}"
        )
        drift_ppb = 0
        offset_ns, residuals, sigma = _evaluate(0)
    elif drift_sigma_ppb > 0.0 and abs(drift_ppb) < DRIFT_SNR * drift_sigma_ppb:
        span_s = (float(x.max()) - float(x.min())) / 1e9
        problems.append(
            f"{PROBLEM_DRIFT_UNRESOLVABLE}: fitted {drift_ppb / 1000:.1f} ppm is "
            f"below {DRIFT_SNR:g}x its own standard error "
            f"({drift_sigma_ppb / 1000:.1f} ppm) over a {span_s:.1f} s baseline; "
            f"returning offset-only"
        )
        drift_ppb = 0
        offset_ns, residuals, sigma = _evaluate(0)

    # --- A8: variance must reflect degrees of freedom --------------------
    #
    # A robust scale computed from residuals alone reports perfection
    # whenever the fit is underdetermined, because the residuals really are
    # zero. What is missing is not spread but *evidence*: n - p degrees of
    # freedom. Below one, no scale is measurable and the honest output is
    # the same ignorance floor a single pair gets.
    parameters = 1 if drift_ppb == 0 else 2
    dof = n - parameters
    if dof <= 0:
        problems.append(
            f"{PROBLEM_UNDERDETERMINED}: {n} pairs fit {parameters} parameters, so "
            f"residuals are zero by construction; variance_ns is a floor"
        )
        variance_ns = SINGLE_PAIR_VARIANCE_NS
    else:
        # Small-sample inflation: with few degrees of freedom the residual
        # scale systematically understates the true spread.
        variance_ns = max(0, int(round(sigma * np.sqrt(n / dof))))

    mapping = LatencyEstimate(
        source=domains["source"],
        target=domains["target"],
        offset_ns=offset_ns,
        drift_ppb=drift_ppb,
        anchor_time_ns=anchor,
        variance_ns=variance_ns,
        epoch=epoch,
    )

    residual_array = np.asarray(residuals, dtype=np.float64)
    centre = float(np.median(residual_array))
    deviations = np.abs(residual_array - centre)
    if sigma > 0:
        inliers = int(np.count_nonzero(deviations <= 3.0 * sigma))
    else:
        inliers = int(np.count_nonzero(deviations == 0.0))
    return ClockMappingFit(
        mapping=mapping,
        residuals_ns=residuals,
        n_pairs=n,
        inlier_fraction=inliers / n,
        problems=tuple(problems),
    )


def standard_score(
    scores: NDArray[np.float64] | Sequence[float],
    peak_index: int,
    *,
    exclusion_bins: int = AMBIGUITY_EXCLUSION_BINS,
) -> float:
    """z-score of ``scores[peak_index]`` against the rest of the curve.

    The peak and its ``exclusion_bins`` neighbours on each side are left
    out of the background statistics — a genuine peak leaks into its
    neighbours, and letting it inflate its own background would
    systematically understate every good match.

    A background with zero spread is not "no evidence": a spike standing
    in an exactly flat field is the cleanest match a sparse event scan
    can produce, so it scores :data:`_FLAT_BACKGROUND_Z`. Only a peak
    that fails to exceed its background scores ``0.0``.
    """
    curve = np.asarray(scores, dtype=np.float64)
    if curve.size == 0:
        return 0.0
    low = max(0, peak_index - exclusion_bins)
    high = min(curve.size, peak_index + exclusion_bins + 1)
    background = np.concatenate([curve[:low], curve[high:]])
    if background.size == 0:
        return 0.0
    mean = float(background.mean())
    excess = float(curve[peak_index]) - mean
    sd = float(background.std())
    if sd == 0.0:
        return _FLAT_BACKGROUND_Z if excess > 0.0 else 0.0
    return excess / sd


def score_confidence(
    scores: NDArray[np.float64] | Sequence[float],
    peak_index: int,
    *,
    exclusion_bins: int = AMBIGUITY_EXCLUSION_BINS,
) -> float:
    """Shared ``[0, 1]`` confidence for a peak in a score curve.

    ``squash(standard_score) × ambiguity_penalty``, where the squash is
    ``z / (z + 6)`` (so z=6 → 0.5) and the penalty is
    ``1 − runner_up / peak`` clipped to ``[0, 1]``. Both halves are
    necessary: a peak can be tall against the background *and* have a
    near-twin somewhere else, which is precisely the periodic-signal
    failure a bare z-score calls a confident match.

    The penalty going hard to zero on a *tie* is deliberate and was
    checked against real data rather than assumed: trials scoring exactly
    zero here carried ten times the offset error of the rest (1.68 versus
    representative video-frame accuracy in validation trials). Two alignments that
    explain the events equally well really are ambiguous, and softening
    this to degrade smoothly makes the metric agree with a near-twin peak
    that it should be rejecting. What that zero cannot do is *rank* the
    surviving cases, which is why the caller-facing confidence in
    :func:`~embodied_sync.calibrate.events.match_event_trains` multiplies
    this by the share of events the mapping actually explains.
    """
    curve = np.asarray(scores, dtype=np.float64)
    if curve.size == 0:
        return 0.0
    peak = float(curve[peak_index])
    if peak <= 0.0:
        return 0.0
    z = standard_score(curve, peak_index, exclusion_bins=exclusion_bins)
    if z <= 0.0:
        return 0.0
    base = z / (z + _Z_HALF)
    low = max(0, peak_index - exclusion_bins)
    high = min(curve.size, peak_index + exclusion_bins + 1)
    background = np.concatenate([curve[:low], curve[high:]])
    if background.size == 0:
        return base
    runner_up = float(background.max())
    penalty = min(1.0, max(0.0, 1.0 - runner_up / peak))
    return base * penalty
