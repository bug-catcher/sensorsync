"""Ground-truth recovery for the robust clock-mapping estimator (D-0038).

Every test plants a known ``(offset, drift, jitter)`` and asks whether it
comes back. Bounds are stated in terms of the planted noise rather than
as magic numbers, and each is justified in a comment — an estimator test
whose tolerances nobody can derive is not evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from embodied_sync.calibrate import (
    SINGLE_PAIR_VARIANCE_NS,
    fit_clock_mapping,
    score_confidence,
    standard_score,
)
from embodied_sync.time import ClockDomain, ClockKind, translate_ns

MS = 1_000_000
S = 1_000_000_000


def _plant(
    *,
    seed: int,
    n: int = 20,
    span_ns: int = 60 * S,
    offset_ns: int = 12_345_000,
    drift_ppb: int = 120_000,
    jitter_ns: int = 100_000,
) -> tuple[list[int], list[int]]:
    """Build a paired (source, target) event set with known parameters."""
    rng = np.random.default_rng(seed)
    source = np.sort(rng.integers(0, span_ns, size=n)).astype(np.int64)
    anchor = int(source[0])
    target = np.array(
        [
            int(t) + offset_ns + round((int(t) - anchor) * drift_ppb / 1e9)
            for t in source
        ],
        dtype=np.int64,
    )
    if jitter_ns:
        target = target + rng.normal(0, jitter_ns, size=n).round().astype(np.int64)
    return source.tolist(), target.tolist()


class TestGroundTruthRecovery:
    def test_noiseless_recovery_is_exact(self) -> None:
        source, target = _plant(seed=0, jitter_ns=0)
        fit = fit_clock_mapping(source, target)
        assert fit.mapping.offset_ns == 12_345_000
        assert fit.mapping.drift_ppb == 120_000
        assert fit.residuals_ns == [0] * len(source)
        assert fit.mapping.variance_ns == 0
        assert fit.inlier_fraction == 1.0
        assert fit.n_pairs == len(source)

    @pytest.mark.parametrize("seed", range(10))
    def test_jittered_recovery_is_within_the_planted_noise(self, seed: int) -> None:
        jitter_ns = 100_000  # 100 us
        source, target = _plant(seed=seed, jitter_ns=jitter_ns)
        fit = fit_clock_mapping(source, target)
        # A median-based offset over n=20 has standard error ~1.25*sigma/sqrt(n)
        # = 28 us; 2x the planted sigma is a comfortable, honest bound.
        assert abs(fit.mapping.offset_ns - 12_345_000) <= 2 * jitter_ns
        # Theil-Sen slope error over a 60 s baseline with 20 points is a few
        # ppm at this jitter; the design's acceptance figure is 2 ppm and 3
        # is the observed worst case across seeds.
        assert abs(fit.mapping.drift_ppb - 120_000) <= 3_000
        # variance_ns is a robust scale, so it should land near the planted
        # sigma rather than at zero.
        assert 0.3 * jitter_ns <= fit.mapping.variance_ns <= 3 * jitter_ns

    def test_translate_ns_round_trips_the_fit(self) -> None:
        source, target = _plant(seed=3, jitter_ns=0)
        fit = fit_clock_mapping(source, target)
        # The loop closes: applying the fitted mapping reproduces the target.
        assert [translate_ns(t, fit.mapping) for t in source] == target

    def test_residuals_are_measured_against_the_rounded_mapping(self) -> None:
        source, target = _plant(seed=4)
        fit = fit_clock_mapping(source, target)
        expected = [t - translate_ns(s, fit.mapping) for s, t in zip(source, target)]
        assert fit.residuals_ns == expected

    def test_outliers_do_not_drag_the_fit(self) -> None:
        source, target = _plant(seed=5, jitter_ns=0)
        corrupted = list(target)
        corrupted[3] += 5 * S  # one wildly mis-detected event
        corrupted[11] -= 3 * S
        fit = fit_clock_mapping(source, corrupted)
        # Theil-Sen shrugs off 2 of 20 bad pairs; least squares would not.
        assert fit.mapping.offset_ns == 12_345_000
        assert fit.mapping.drift_ppb == 120_000
        assert fit.inlier_fraction == pytest.approx(18 / 20)

    def test_zero_drift_is_recovered_as_zero(self) -> None:
        source, target = _plant(seed=6, drift_ppb=0, jitter_ns=0)
        fit = fit_clock_mapping(source, target)
        assert fit.mapping.drift_ppb == 0
        assert fit.mapping.offset_ns == 12_345_000

    def test_negative_offset_and_drift(self) -> None:
        source, target = _plant(
            seed=7, offset_ns=-45 * MS, drift_ppb=-250_000, jitter_ns=0
        )
        fit = fit_clock_mapping(source, target)
        assert fit.mapping.offset_ns == -45 * MS
        assert fit.mapping.drift_ppb == -250_000

    def test_large_epoch_timestamps_stay_exact(self) -> None:
        """Raw epoch-ns values exceed float64's exact-integer range."""
        base = 1_780_000_000_000_000_000  # ~2026 in epoch ns
        source = [base + i * S for i in range(10)]
        target = [t + 7_000_000 for t in source]
        fit = fit_clock_mapping(source, target)
        assert fit.mapping.offset_ns == 7_000_000
        assert fit.mapping.drift_ppb == 0
        assert fit.residuals_ns == [0] * 10


class TestDegenerateCases:
    def test_single_pair_is_offset_only_with_an_infinite_variance_floor(
        self,
    ) -> None:
        fit = fit_clock_mapping([1_000], [1_000 + 42 * MS])
        assert fit.n_pairs == 1
        assert fit.mapping.offset_ns == 42 * MS
        assert fit.mapping.drift_ppb == 0
        assert fit.mapping.variance_ns == SINGLE_PAIR_VARIANCE_NS
        assert fit.residuals_ns == [0]
        assert fit.inlier_fraction == 1.0

    def test_single_pair_still_round_trips(self) -> None:
        fit = fit_clock_mapping([1_000], [1_000 + 42 * MS])
        assert translate_ns(1_000, fit.mapping) == 1_000 + 42 * MS

    def test_identical_source_times_yield_offset_only(self) -> None:
        """No baseline means no slope; the fit must not invent one."""
        fit = fit_clock_mapping([5_000] * 4, [5_000 + 10 * MS] * 4)
        assert fit.mapping.drift_ppb == 0
        assert fit.mapping.offset_ns == 10 * MS

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one paired time"):
            fit_clock_mapping([], [])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            fit_clock_mapping([1, 2, 3], [1, 2])

    def test_unknown_method_names_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="known methods"):
            fit_clock_mapping([1], [2], method="ransac")


class TestAnchorAndDomains:
    def test_anchor_defaults_to_the_first_source_time(self) -> None:
        source, target = _plant(seed=8, jitter_ns=0)
        fit = fit_clock_mapping(source, target)
        assert fit.mapping.anchor_time_ns == source[0]

    def test_explicit_anchor_moves_where_the_offset_is_exact(self) -> None:
        source, target = _plant(seed=9, jitter_ns=0)
        fit = fit_clock_mapping(source, target, anchor_ns=0)
        assert fit.mapping.anchor_time_ns == 0
        # Same mapping, different parameterisation. The round-trip is exact
        # to +-1 ns rather than bit-exact: offset_ns is an integer defined at
        # the anchor, so re-anchoring re-rounds it. Anchoring on a data point
        # (the default) is what makes it exact there.
        recovered = [translate_ns(t, fit.mapping) for t in source]
        assert all(abs(r - t) <= 1 for r, t in zip(recovered, target))

    def test_domains_default_to_unknown(self) -> None:
        fit = fit_clock_mapping([0, S], [0, S])
        assert fit.mapping.source.kind is ClockKind.UNKNOWN
        assert fit.mapping.target.kind is ClockKind.UNKNOWN

    def test_domains_can_be_named(self) -> None:
        fit = fit_clock_mapping(
            [0, S],
            [0, S],
            source_domain=ClockDomain("cam_hw", ClockKind.HARDWARE),
            target_domain="host_mono",
        )
        assert fit.mapping.source.name == "cam_hw"
        assert fit.mapping.target.name == "host_mono"


class TestConfidenceMetric:
    def test_isolated_spike_in_a_flat_field_scores_high(self) -> None:
        scores = np.zeros(101)
        scores[50] = 20.0
        assert standard_score(scores, 50) > 10.0
        assert score_confidence(scores, 50) > 0.8

    def test_a_near_twin_peak_is_penalised(self) -> None:
        scores = np.zeros(101)
        scores[50] = 20.0
        scores[20] = 19.0  # an almost-as-good alternative offset
        confident = score_confidence(np.where(np.arange(101) == 50, 20.0, 0.0), 50)
        ambiguous = score_confidence(scores, 50)
        assert ambiguous < confident
        assert ambiguous < 0.2

    def test_flat_curve_has_no_evidence(self) -> None:
        assert standard_score(np.ones(50), 10) == 0.0
        assert score_confidence(np.ones(50), 10) == 0.0

    def test_non_positive_peak_scores_zero(self) -> None:
        assert score_confidence(np.zeros(10), 3) == 0.0

    def test_confidence_is_bounded(self) -> None:
        scores = np.zeros(201)
        scores[100] = 1e6
        assert 0.0 <= score_confidence(scores, 100) <= 1.0

    def test_peak_neighbours_are_excluded_from_the_background(self) -> None:
        """A peak that leaks into its neighbours must not penalise itself."""
        scores = np.zeros(101)
        scores[50] = 20.0
        scores[49] = scores[51] = 12.0
        assert score_confidence(scores, 50) > 0.5
