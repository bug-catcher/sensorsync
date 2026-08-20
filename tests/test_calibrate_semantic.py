"""Experimental semantic-event alignment (D-0038)."""

from __future__ import annotations

import numpy as np
import pytest

from embodied_sync.calibrate import align_semantic_events
from embodied_sync.calibrate import semantic as semantic_module
from embodied_sync.time import translate_ns

MS = 1_000_000
S = 1_000_000_000


@pytest.fixture(autouse=True)
def _reset_experimental_warning() -> None:
    """The warning is one-shot per process; reset it so each test sees it."""
    semantic_module._EXPERIMENTAL_WARNED = False


def _events(
    *, seed: int, n: int = 15, offset_ns: int = 40 * MS, jitter_ns: int = 15 * MS
) -> tuple[list[int], list[int]]:
    """Semantic events: sparse, and localisable only to ~a video frame."""
    rng = np.random.default_rng(seed)
    a = np.sort(rng.integers(0, 120 * S, size=n)).astype(np.int64)
    b = a + offset_ns + rng.normal(0, jitter_ns, size=n).round().astype(np.int64)
    return a.tolist(), np.sort(b).tolist()


class TestAlignSemanticEvents:
    def test_recovers_a_known_offset(self) -> None:
        a, b = _events(seed=0)
        with pytest.warns(UserWarning, match="experimental"):
            result = align_semantic_events(
                a, b, max_offset_ms=200.0, match_tolerance_ms=60.0
            )
        assert abs(result.offset_ns - 40 * MS) <= 15 * MS
        assert result.matched_count == 15
        assert result.n_events_a == 15
        assert result.n_events_b == 15
        assert result.matched_fraction_a == 1.0
        assert result.confidence > 0.0

    def test_result_is_evidence_not_a_verdict(self) -> None:
        a, b = _events(seed=1)
        with pytest.warns(UserWarning):
            result = align_semantic_events(
                a, b, max_offset_ms=200.0, match_tolerance_ms=60.0
            )
        # There is deliberately no "aligned" boolean: the caller owns the
        # tolerance, so the caller makes the call.
        assert not hasattr(result, "aligned")
        assert result.residual_p95_ns > 0  # the events are not crisp
        assert result.mapping.variance_ns > 0
        assert result.alignment.matched

    def test_mapping_feeds_translate_ns(self) -> None:
        a = [0, 30 * S, 60 * S]
        b = [t + 25 * MS for t in a]
        with pytest.warns(UserWarning):
            result = align_semantic_events(a, b, max_offset_ms=100.0)
        assert [translate_ns(t, result.mapping) for t in a] == b
        assert result.drift_ppb == 0

    def test_experimental_warning_is_one_shot(self) -> None:
        a, b = _events(seed=2)
        with pytest.warns(UserWarning, match="experimental"):
            align_semantic_events(a, b, max_offset_ms=200.0, match_tolerance_ms=60.0)
        # A second call in the same process stays quiet: an experimental
        # API must be noticeable, not a per-iteration flood.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            align_semantic_events(
                a, b, max_offset_ms=200.0, match_tolerance_ms=60.0
            )

    def test_dropout_is_reported_in_the_fractions(self) -> None:
        a, b = _events(seed=3, n=20)
        with pytest.warns(UserWarning):
            result = align_semantic_events(
                a, b[:12], max_offset_ms=200.0, match_tolerance_ms=60.0
            )
        assert result.matched_fraction_a < 1.0
        assert result.n_events_a == 20
        assert result.n_events_b == 12
