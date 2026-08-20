"""ApproximateTime bundling: the three guarantees as executable contracts (A1).

ROS states three properties for its ``ApproximateTime`` policy — each
message used at most once, sets published in order, span minimised — and
states them in prose. Prose is not a contract. These tests make each one
a property checked against generated push sequences, plus hand-computed
scenarios where the optimal answer is small enough to write down.

The bundler is clock-free and deterministic, so nothing here sleeps or
depends on arrival wall time: a "push order" is just a list.
"""

from __future__ import annotations

import itertools
import threading
from typing import Iterable

import pytest

from embodied_sync.core.sample import Modality, Sample
from embodied_sync.session import (
    APPROXIMATE_METHOD,
    ApproximateSet,
    ApproximateTimeBundler,
    StreamConfig,
    SyncSession,
)

MS = 1_000_000


class FakeClock:
    """Deterministic monotonic clock in integer nanoseconds."""

    def __init__(self, start_ns: int = 1_000_000_000) -> None:
        self._now = start_ns
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self._now

    def advance(self, ns: int) -> int:
        with self._lock:
            self._now += ns
            return self._now

    def set(self, ns: int) -> None:
        with self._lock:
            self._now = ns


def _sample(stream: str, t_ns: int, sequence_id: int = 0) -> Sample:
    return Sample(
        stream_name=stream,
        modality=Modality.OTHER,
        sequence_id=sequence_id,
        acquisition_time_ns=t_ns,
        receive_time_ns=t_ns,
        source_clock_domain="host_mono",
        payload={"t": t_ns},
    )


def _feed(
    bundler: ApproximateTimeBundler, pushes: Iterable[tuple[str, int]]
) -> list[ApproximateSet]:
    emitted: list[ApproximateSet] = []
    for index, (stream, t_ns) in enumerate(pushes):
        emitted.extend(bundler.push(_sample(stream, t_ns, index)))
    return emitted


def _interleaved(streams: dict[str, list[int]]) -> list[tuple[str, int]]:
    """Merge per-stream schedules into one globally time-ordered push list."""
    pushes = [(name, t) for name, times in streams.items() for t in times]
    pushes.sort(key=lambda item: (item[1], item[0]))
    return pushes


class TestGuaranteeUsedAtMostOnce:
    def test_no_sample_appears_in_two_sets(self) -> None:
        bundler = ApproximateTimeBundler(["a", "b", "c"])
        pushes = _interleaved(
            {
                "a": [i * 10 * MS for i in range(40)],
                "b": [3 * MS + i * 13 * MS for i in range(30)],
                "c": [7 * MS + i * 31 * MS for i in range(13)],
            }
        )
        emitted = _feed(bundler, pushes)
        assert emitted, "expected the bundler to emit something"
        seen: set[tuple[str, int]] = set()
        for result in emitted:
            for name, sample in result.samples.items():
                key = (name, sample.sequence_id)
                assert key not in seen, f"{key} was reused across sets"
                seen.add(key)

    def test_flush_does_not_re_emit_already_used_samples(self) -> None:
        bundler = ApproximateTimeBundler(["a", "b"])
        emitted = _feed(
            bundler,
            _interleaved(
                {"a": [i * 10 * MS for i in range(8)], "b": [i * 10 * MS for i in range(8)]}
            ),
        )
        used = {
            (name, s.sequence_id) for r in emitted for name, s in r.samples.items()
        }
        for result in bundler.flush():
            for name, sample in result.samples.items():
                assert (name, sample.sequence_id) not in used

    def test_counters_account_for_every_pushed_sample(self) -> None:
        bundler = ApproximateTimeBundler(["a", "b"])
        pushes = _interleaved(
            {"a": [i * 5 * MS for i in range(50)], "b": [i * 17 * MS for i in range(15)]}
        )
        emitted = _feed(bundler, pushes)
        emitted.extend(bundler.flush())
        stats = bundler.stats()
        used = sum(len(r.samples) for r in emitted)
        accounted = used + stats["superseded"] + stats["overflowed"] + stats["pending"]
        assert accounted == len(pushes)


class TestGuaranteeEmittedInOrder:
    def test_pivot_times_are_non_decreasing(self) -> None:
        bundler = ApproximateTimeBundler(["a", "b", "c"])
        emitted = _feed(
            bundler,
            _interleaved(
                {
                    "a": [i * 11 * MS for i in range(35)],
                    "b": [2 * MS + i * 7 * MS for i in range(55)],
                    "c": [5 * MS + i * 23 * MS for i in range(17)],
                }
            ),
        )
        pivots = [r.pivot_time_ns for r in emitted]
        assert pivots == sorted(pivots)

    def test_flushed_sets_continue_the_order(self) -> None:
        bundler = ApproximateTimeBundler(["a", "b"])
        emitted = _feed(
            bundler,
            _interleaved(
                {"a": [i * 10 * MS for i in range(9)], "b": [i * 14 * MS for i in range(7)]}
            ),
        )
        pivots = [r.pivot_time_ns for r in emitted] + [
            r.pivot_time_ns for r in bundler.flush()
        ]
        assert pivots == sorted(pivots)

    def test_sets_do_not_overlap_in_sample_order(self) -> None:
        """Set k's members all precede set k+1's, per stream."""
        bundler = ApproximateTimeBundler(["a", "b"])
        emitted = _feed(
            bundler,
            _interleaved(
                {"a": [i * 9 * MS for i in range(30)], "b": [i * 12 * MS for i in range(23)]}
            ),
        )
        for first, second in itertools.pairwise(emitted):
            for name in ("a", "b"):
                assert (
                    first.samples[name].sequence_id < second.samples[name].sequence_id
                )


class TestGuaranteeSpanMinimised:
    def test_hand_computed_optimum(self) -> None:
        """Two streams, one obvious best pairing, written out by hand.

        ``a`` fires at 0/100/200 ms, ``b`` at 95/205 ms. For a pivot at
        b=95 the closest ``a`` at or before it is 0 (span 95) — but ``a``
        at 100 is *after* the pivot, so it cannot join that set. The
        algorithm's first set is therefore (a=0, b=95) only if no better
        pivot is available earlier, which it is not.
        """
        bundler = ApproximateTimeBundler(["a", "b"])
        emitted = _feed(
            bundler,
            [
                ("a", 0),
                ("b", 95 * MS),
                ("a", 100 * MS),
                ("a", 200 * MS),
                ("b", 205 * MS),
            ],
        )
        assert [r.pivot_time_ns for r in emitted] == [95 * MS]
        first = emitted[0]
        assert first.samples["a"].acquisition_time_ns == 0
        assert first.span_ns == 95 * MS

    def test_stale_samples_are_superseded_not_bundled(self) -> None:
        """A backlogged fast stream must not pair its *oldest* sample."""
        bundler = ApproximateTimeBundler(["fast", "slow"])
        emitted = _feed(
            bundler,
            [("fast", i * 10 * MS) for i in range(11)]
            + [("slow", 100 * MS), ("slow", 200 * MS)],
        )
        assert len(emitted) == 1
        # The best `fast` member for a pivot at 100 ms is the one at 100 ms,
        # not the one at 0 ms that happened to be at the head of the queue.
        assert emitted[0].samples["fast"].acquisition_time_ns == 100 * MS
        assert emitted[0].span_ns == 0
        assert bundler.superseded == 10

    def test_span_is_never_beaten_by_an_alternative_from_the_same_data(self) -> None:
        """Brute force: no unused-at-the-time member gives a smaller span."""
        schedule = {
            "a": [i * 10 * MS for i in range(25)],
            "b": [4 * MS + i * 15 * MS for i in range(17)],
        }
        bundler = ApproximateTimeBundler(["a", "b"])
        emitted = _feed(bundler, _interleaved(schedule))
        for result in emitted:
            pivot = result.pivot_time_ns
            for name, sample in result.samples.items():
                # The chosen member is the latest one at or before the pivot.
                candidates = [t for t in schedule[name] if t <= pivot]
                assert sample.acquisition_time_ns == max(candidates)

    def test_perfectly_aligned_streams_emit_with_zero_span(self) -> None:
        bundler = ApproximateTimeBundler(["a", "b"])
        emitted = _feed(
            bundler,
            _interleaved(
                {"a": [i * 10 * MS for i in range(5)], "b": [i * 10 * MS for i in range(5)]}
            ),
        )
        assert emitted
        assert all(r.span_ns == 0 for r in emitted)


class TestLatencyAndProof:
    def test_nothing_is_emitted_until_every_stream_is_settled(self) -> None:
        """The inherent ~one-slowest-period wait, made explicit.

        With ``a`` at 0 and ``b`` at 5 ms the pivot is ``b``, and ``a``'s
        head at 0 might still be beaten by an ``a`` arriving at, say,
        4 ms. Nothing can be emitted until ``a`` produces a sample
        *after* the pivot, which is what proves 0 was its best member —
        and that wait is one ``a``-period long, by construction.
        """
        bundler = ApproximateTimeBundler(["a", "b"])
        assert _feed(bundler, [("a", 0), ("b", 5 * MS)]) == []
        emitted = bundler.push(_sample("a", 10 * MS, 2))
        assert len(emitted) == 1
        assert emitted[0].pivot_time_ns == 5 * MS
        assert emitted[0].samples["a"].acquisition_time_ns == 0

    def test_a_closer_late_arrival_is_preferred_over_the_queue_head(self) -> None:
        """The wait is not bureaucratic: it changes the answer."""
        bundler = ApproximateTimeBundler(["a", "b"])
        assert _feed(bundler, [("a", 0), ("b", 5 * MS)]) == []
        # `a` produces something between the old head and the pivot.
        emitted = _feed(bundler, [("a", 4 * MS), ("a", 10 * MS)])
        assert len(emitted) == 1
        assert emitted[0].samples["a"].acquisition_time_ns == 4 * MS
        assert emitted[0].span_ns == MS

    def test_a_head_sitting_on_the_pivot_needs_no_further_wait(self) -> None:
        bundler = ApproximateTimeBundler(["a", "b"])
        emitted = _feed(bundler, [("a", 7 * MS), ("b", 7 * MS)])
        assert len(emitted) == 1
        assert emitted[0].span_ns == 0

    def test_normal_sets_are_provable_and_flushed_sets_are_not(self) -> None:
        bundler = ApproximateTimeBundler(["a", "b"])
        emitted = _feed(bundler, [("a", 0), ("b", 0), ("a", 10 * MS), ("b", 10 * MS)])
        assert len(emitted) == 2
        assert all(r.provable for r in emitted)
        # Leave a genuinely unsettled tail: `a` at 20 ms is the pivot and
        # `b` at 15 ms has nothing after it, so the normal path waits.
        assert _feed(bundler, [("b", 15 * MS), ("a", 20 * MS)]) == []
        flushed = bundler.flush()
        assert len(flushed) == 1
        assert flushed[0].provable is False
        assert flushed[0].pivot_time_ns == 20 * MS


class TestBoundsAndValidation:
    def test_overflow_is_counted_not_hidden(self) -> None:
        bundler = ApproximateTimeBundler(["a", "b"], queue_capacity=4)
        _feed(bundler, [("a", i * MS) for i in range(20)])
        assert bundler.overflowed == 16
        assert bundler.stats()["overflowed"] == 16

    def test_backwards_samples_are_refused_and_counted(self) -> None:
        bundler = ApproximateTimeBundler(["a", "b"])
        _feed(bundler, [("a", 10 * MS), ("a", 5 * MS)])
        assert bundler.out_of_order == 1
        assert bundler.pending()["a"] == 1

    def test_unknown_stream_raises(self) -> None:
        bundler = ApproximateTimeBundler(["a", "b"])
        with pytest.raises(KeyError, match="not part of the approximate set"):
            bundler.push(_sample("c", 0))

    def test_a_single_stream_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two streams"):
            ApproximateTimeBundler(["only"])

    def test_capacity_below_two_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= 2"):
            ApproximateTimeBundler(["a", "b"], queue_capacity=1)

    def test_duplicate_stream_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate stream names"):
            ApproximateTimeBundler(["a", "a"])


class TestSessionIntegration:
    def _session(self, clock: FakeClock, **kwargs: object) -> SyncSession:
        return SyncSession(
            streams={
                "camera": StreamConfig(
                    rate_hz=10, tolerance_ms=60.0, policy="approximate"
                ),
                "robot": StreamConfig(
                    rate_hz=20, tolerance_ms=60.0, policy="approximate"
                ),
            },
            clock=clock,
            on_violation="ignore",
            **kwargs,  # type: ignore[arg-type]
        )

    def test_bundles_arrive_through_poll_bundles(self) -> None:
        clock = FakeClock()
        session = self._session(clock)
        assert session.poll_bundles() == []
        for i in range(6):
            t = 1_000_000_000 + i * 50 * MS
            clock.set(t)
            session.push("camera", {"frame": i}, t_ns=t)
            session.push("robot", {"q": i}, t_ns=t + 5 * MS)
        bundles = session.poll_bundles()
        assert bundles
        assert all(b.ok for b in bundles)
        assert [b.target_time_ns for b in bundles] == sorted(
            b.target_time_ns for b in bundles
        )
        first = bundles[0]
        assert set(first.items) == {"camera", "robot"}
        assert first.items["camera"].method == APPROXIMATE_METHOD
        assert first.span_ns is not None
        assert first["camera"] == {"frame": 0}
        session.close()

    def test_polling_drains(self) -> None:
        clock = FakeClock()
        session = self._session(clock)
        for i in range(6):
            t = 1_000_000_000 + i * 50 * MS
            clock.set(t)
            session.push("camera", {}, t_ns=t)
            session.push("robot", {}, t_ns=t)
        assert session.pending_bundles() > 0
        session.poll_bundles()
        assert session.pending_bundles() == 0
        assert session.poll_bundles() == []
        session.close()

    def test_max_bundles_limits_the_drain(self) -> None:
        clock = FakeClock()
        session = self._session(clock)
        for i in range(8):
            t = 1_000_000_000 + i * 50 * MS
            clock.set(t)
            session.push("camera", {}, t_ns=t)
            session.push("robot", {}, t_ns=t)
        assert len(session.poll_bundles(max_bundles=2)) == 2
        with pytest.raises(ValueError, match="must be > 0"):
            session.poll_bundles(max_bundles=0)
        session.close()

    def test_span_beyond_tolerance_marks_the_bundle_not_ok(self) -> None:
        clock = FakeClock()
        session = SyncSession(
            streams={
                "camera": StreamConfig(
                    rate_hz=10, tolerance_ms=5.0, policy="approximate"
                ),
                "robot": StreamConfig(
                    rate_hz=10, tolerance_ms=5.0, policy="approximate"
                ),
            },
            clock=clock,
            on_violation="ignore",
        )
        for i in range(5):
            t = 1_000_000_000 + i * 100 * MS
            clock.set(t)
            session.push("camera", {}, t_ns=t)
            session.push("robot", {}, t_ns=t + 40 * MS)  # far beyond 5 ms
        bundles = session.poll_bundles()
        assert bundles
        assert not any(b.ok for b in bundles)
        session.close()

    def test_get_still_works_on_approximate_streams(self) -> None:
        clock = FakeClock()
        session = self._session(clock)
        t = 1_000_000_000
        clock.set(t)
        session.push("camera", {"frame": 0}, t_ns=t)
        session.push("robot", {"q": 0}, t_ns=t)
        bundle = session.get(at_ns=t)
        assert bundle.ok
        # nearest-neighbour, per the documented equivalence
        assert bundle.items["camera"].method == "nearest_neighbor"
        session.close()

    def test_close_flushes_the_tail(self) -> None:
        clock = FakeClock()
        session = self._session(clock)
        t = 1_000_000_000
        clock.set(t)
        session.push("camera", {}, t_ns=t)
        session.push("robot", {}, t_ns=t + MS)
        assert session.poll_bundles() == []  # unsettled: nothing provable yet
        session.close()
        assert len(session.poll_bundles()) == 1

    def test_one_approximate_stream_is_a_configuration_error(self) -> None:
        with pytest.raises(ValueError, match="at least two streams"):
            SyncSession(
                streams={
                    "camera": StreamConfig(
                        rate_hz=10, tolerance_ms=5.0, policy="approximate"
                    ),
                    "robot": StreamConfig(rate_hz=10, tolerance_ms=5.0),
                },
                clock=FakeClock(),
            )

    def test_sessions_without_an_approximate_set_poll_empty(self) -> None:
        session = SyncSession(
            streams={"camera": StreamConfig(rate_hz=10, tolerance_ms=5.0)},
            clock=FakeClock(),
        )
        assert session.poll_bundles() == []
        assert session.approximate_stats() == {}
        session.close()

    def test_manifest_records_the_approximate_block(self, tmp_path: object) -> None:
        import json
        from pathlib import Path

        assert isinstance(tmp_path, Path)
        clock = FakeClock()
        run_dir = tmp_path / "run"
        session = self._session(clock, run_dir=run_dir)
        for i in range(4):
            t = 1_000_000_000 + i * 50 * MS
            clock.set(t)
            session.push("camera", {}, t_ns=t)
            session.push("robot", {}, t_ns=t)
        session.close()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        block = manifest["session"]["approximate"]
        assert block["streams"] == ["camera", "robot"]
        assert block["emitted"] >= 1
        assert block["overflowed"] == 0

    def test_quality_counts_approximate_matches(self) -> None:
        clock = FakeClock()
        session = self._session(clock)
        for i in range(6):
            t = 1_000_000_000 + i * 50 * MS
            clock.set(t)
            session.push("camera", {}, t_ns=t)
            session.push("robot", {}, t_ns=t)
        quality = session.quality(window_s=100.0)
        assert quality["camera"].match_count > 0
        assert quality["camera"].within_tolerance_rate == 1.0
        session.close()
