"""Event-train matching against synthetic ground truth (D-0038).

The scenarios escalate: clean trains, then dropout, then spurious
detections, then both — because "recovers the offset when nothing is
wrong" is not the claim that matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from embodied_sync.calibrate import match_event_trains
from embodied_sync.time import translate_ns

MS = 1_000_000
S = 1_000_000_000

OFFSET_NS = 12_345_000
DRIFT_PPB = 120_000  # 120 ppm
JITTER_NS = 100_000  # 100 us


def _trains(
    *,
    seed: int,
    n: int = 20,
    span_ns: int = 60 * S,
    dropout: float = 0.0,
    spurious: int = 0,
    jitter_ns: int = JITTER_NS,
    drift_ppb: int = DRIFT_PPB,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    a = np.sort(rng.integers(0, span_ns, size=n)).astype(np.int64)
    anchor = int(a[0])
    b = np.array(
        [
            int(t) + OFFSET_NS + round((int(t) - anchor) * drift_ppb / 1e9)
            for t in a
        ],
        dtype=np.int64,
    )
    if jitter_ns:
        b = b + rng.normal(0, jitter_ns, size=n).round().astype(np.int64)
    if dropout:
        b = b[rng.random(n) > dropout]
    if spurious:
        extra = rng.integers(0, span_ns, size=spurious).astype(np.int64)
        b = np.concatenate([b, extra])
    return a.tolist(), np.sort(b).tolist()


class TestCleanRecovery:
    @pytest.mark.parametrize("seed", range(8))
    def test_offset_and_drift_come_back(self, seed: int) -> None:
        a, b = _trains(seed=seed)
        result = match_event_trains(a, b, max_offset_ms=100.0)
        assert len(result.matched) == 20
        assert result.matched_fraction_a == 1.0
        assert result.matched_fraction_b == 1.0
        assert abs(result.offset_ns - OFFSET_NS) <= 2 * JITTER_NS
        assert abs(result.drift_ppb - DRIFT_PPB) <= 3_000  # 3 ppm
        assert result.confidence > 0.5

    def test_noiseless_recovery_is_exact(self) -> None:
        a, b = _trains(seed=0, jitter_ns=0)
        result = match_event_trains(a, b, max_offset_ms=100.0)
        assert result.offset_ns == OFFSET_NS
        assert result.drift_ppb == DRIFT_PPB
        assert result.residual_p95_ns == 0

    def test_the_loop_closes_through_translate_ns(self) -> None:
        a, b = _trains(seed=1, jitter_ns=0)
        result = match_event_trains(a, b, max_offset_ms=100.0)
        for i, j in result.matched:
            assert translate_ns(a[i], result.fit.mapping) == b[j]

    def test_matched_holds_index_pairs_in_order(self) -> None:
        a, b = _trains(seed=2, jitter_ns=0)
        result = match_event_trains(a, b, max_offset_ms=100.0)
        assert result.matched == [(i, i) for i in range(20)]

    def test_residual_p95_reflects_the_planted_jitter(self) -> None:
        a, b = _trains(seed=3)
        result = match_event_trains(a, b, max_offset_ms=100.0)
        # 95th percentile of |residual| for 100 us Gaussian noise sits
        # around 2 sigma; allow generous headroom, just not 10x.
        assert result.residual_p95_ns <= 5 * JITTER_NS


class TestImperfectTrains:
    @pytest.mark.parametrize("seed", range(8))
    def test_dropout_is_survived(self, seed: int) -> None:
        a, b = _trains(seed=seed, dropout=0.3)
        result = match_event_trains(a, b, max_offset_ms=100.0)
        assert abs(result.offset_ns - OFFSET_NS) <= 3 * JITTER_NS
        assert abs(result.drift_ppb - DRIFT_PPB) <= 5_000
        assert result.matched_fraction_a < 1.0
        assert result.matched_fraction_b == 1.0

    @pytest.mark.parametrize("seed", range(8))
    def test_spurious_events_are_survived(self, seed: int) -> None:
        a, b = _trains(seed=seed, spurious=8)
        result = match_event_trains(a, b, max_offset_ms=100.0)
        assert abs(result.offset_ns - OFFSET_NS) <= 3 * JITTER_NS
        assert abs(result.drift_ppb - DRIFT_PPB) <= 5_000
        # The spurious events stay unmatched rather than joining the fit.
        assert result.matched_fraction_b < 1.0

    @pytest.mark.parametrize("seed", range(8))
    def test_dropout_and_spurious_together(self, seed: int) -> None:
        a, b = _trains(seed=seed, dropout=0.25, spurious=5)
        result = match_event_trains(a, b, max_offset_ms=100.0)
        assert abs(result.offset_ns - OFFSET_NS) <= 3 * JITTER_NS
        assert abs(result.drift_ppb - DRIFT_PPB) <= 5_000
        assert result.confidence > 0.3

    def test_matching_is_one_to_one(self) -> None:
        a, b = _trains(seed=4, spurious=10)
        result = match_event_trains(a, b, max_offset_ms=100.0)
        assert len({i for i, _ in result.matched}) == len(result.matched)
        assert len({j for _, j in result.matched}) == len(result.matched)


class TestDegenerateAndErrors:
    def test_single_event_each_gives_an_offset_only_mapping(self) -> None:
        result = match_event_trains([1 * S], [1 * S + 40 * MS], max_offset_ms=100.0)
        assert len(result.matched) == 1
        assert result.offset_ns == 40 * MS
        assert result.drift_ppb == 0
        assert result.fit.n_pairs == 1
        assert result.matched_fraction_a == 1.0

    def test_offset_outside_the_search_range_is_a_clear_error(self) -> None:
        a, b = _trains(seed=5, jitter_ns=0)
        with pytest.raises(ValueError, match="max_offset_ms"):
            match_event_trains(a, b, max_offset_ms=1.0)

    def test_empty_train_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one event"):
            match_event_trains([], [1], max_offset_ms=10.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_offset_ms": 0.0},
            {"max_offset_ms": 10.0, "max_drift_ppm": -1.0},
            {"max_offset_ms": 10.0, "match_tolerance_ms": 0.0},
        ],
    )
    def test_argument_validation(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            match_event_trains([0, S], [0, S], **kwargs)  # type: ignore[arg-type]

    def test_unsorted_input_is_accepted(self) -> None:
        a, b = _trains(seed=6, jitter_ns=0)
        shuffled_a = list(reversed(a))
        shuffled_b = list(reversed(b))
        result = match_event_trains(shuffled_a, shuffled_b, max_offset_ms=100.0)
        assert result.offset_ns == OFFSET_NS

    def test_drift_beyond_the_stated_prior_warns(self) -> None:
        a, b = _trains(seed=7, drift_ppb=800_000, jitter_ns=0)  # 800 ppm
        with pytest.warns(UserWarning, match="exceeds max_drift_ppm"):
            result = match_event_trains(a, b, max_offset_ms=200.0, max_drift_ppm=100.0)
        # Reported unclamped: it is evidence, not a decision.
        assert result.drift_ppb == pytest.approx(800_000, rel=0.01)


class TestDeterminism:
    def test_same_input_gives_identical_output(self) -> None:
        a, b = _trains(seed=8, dropout=0.2, spurious=6)
        first = match_event_trains(a, b, max_offset_ms=100.0)
        second = match_event_trains(a, b, max_offset_ms=100.0)
        assert first == second
