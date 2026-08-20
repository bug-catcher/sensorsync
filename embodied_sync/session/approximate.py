"""Pivot-and-span-minimising ApproximateTime bundling (D-0040, Lane A / A1).

The session's default retrieval is a *pull*: you name a target time and
each stream's picker answers for it. That is the right shape for a
control loop, and the wrong one for recording. A recorder does not have
a target time — it has N streams arriving at N different rates and one
question: *which N samples, one per stream, belong together?*

ROS 2 answers that with ``ApproximateTime``, and it is worth being
precise about what is and is not available in Python today. The Python
``message_filters.ApproximateTimeSynchronizer`` is **not** this
algorithm: it is a fixed-``slop`` heuristic that accepts any set whose
spread fits inside a constant the user guessed. The real
pivot-and-span-minimising policy exists only in C++
(``message_filters/sync_policies/approximate_time.h``) and was never
ported. This module is that algorithm, in Python, with its guarantees
turned into tested contracts.

The algorithm
-------------
Each stream keeps a queue of unused samples, in arrival order, with
**non-decreasing** acquisition times (a sample that goes backwards is
refused and counted — see :attr:`ApproximateTimeBundler.out_of_order`).

One round:

1. If any queue is empty, stop: a set needs one sample per stream.
2. **Pivot.** ``pivot_ns = max`` over the queue heads. This is a lower
   bound on the pivot of *every* remaining set: each set draws one
   sample per stream, each such sample is at or after that stream's
   head, so the set's latest member is at or after the latest head.
   Emitting at this pivot is therefore emitting the earliest set that
   remains — which is what keeps output in order.
3. **Shrink.** For each stream, while the *second* queued sample is
   still ``<= pivot_ns``, drop the head. Dropping is safe because the
   sample behind it is at least as close to the pivot, so no set with
   this pivot is worsened; it is what actually minimises the span.
4. **Proof.** The head of each stream is now that stream's best member
   for this pivot *only if* no future arrival could be closer. Arrivals
   are non-decreasing, so a stream is settled once it holds a sample
   after ``pivot_ns`` — i.e. once its queue has two entries — or once
   its head sits exactly on the pivot, where nothing can beat it. Until
   every stream is settled, emit nothing.
5. **Emit.** Pop one sample per stream. Span is ``pivot_ns`` minus the
   earliest member.

The three guarantees, and exactly what they mean here
-----------------------------------------------------
- **Each sample is used at most once.** Samples leave a queue either by
  being emitted or by being superseded in step 3, and nothing is ever
  re-queued. "At most once", not "exactly once": a sample the algorithm
  can prove is not part of any better set is discarded, which is the
  point of step 3.
- **Sets are emitted in order.** Each round's pivot is the maximum over
  heads; after a round every head has advanced to a sample at or after
  the one popped, so the next pivot cannot be earlier. Pivot times are
  non-decreasing across emissions.
- **The span is minimised** — precisely: *the emitted set has the
  smallest possible span among all sets sharing its pivot, and its
  pivot is the earliest pivot any remaining set can have.* That second
  clause is the honest qualifier. A globally smaller span may exist at
  a later pivot; taking it would mean either emitting out of order or
  waiting unboundedly to find out. ROS makes the same trade and so does
  this, and the docstring says so rather than claiming a minimum the
  algorithm does not compute.

Latency
-------
Step 4 is a wait for the *next* sample on every stream after the pivot,
so a set is emitted roughly one period of the slowest stream after the
data that composes it was complete. **This latency is inherent, not an
implementation artefact**: "no future sample can improve this set"
cannot be established without seeing a future sample. A caller who
cannot pay it wants :meth:`~embodied_sync.session.SyncSession.get`,
which answers immediately from whatever has arrived and tells you how
stale that was. There is no third option that is both instant and
optimal.

Bounded memory
--------------
Each queue is capped. On overflow the oldest unused sample is dropped
and counted in :attr:`ApproximateTimeBundler.overflowed`. A non-zero
overflow count means one stream has stalled long enough that the others
outran their queues, and the sets emitted around that gap are *not*
covered by the optimality proof — their best members may have been
evicted. It is reported rather than hidden for exactly that reason.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping

from embodied_sync.core.sample import Sample

__all__ = [
    "APPROXIMATE_METHOD",
    "DEFAULT_QUEUE_CAPACITY",
    "ApproximateSet",
    "ApproximateTimeBundler",
]

#: ``BundleItem.method`` / ``AlignedSampleMetadata.method`` for a member of
#: an ApproximateTime set. Live-session only — the offline engine has no
#: approximate picker, so this name never appears on an offline frame.
APPROXIMATE_METHOD = "approximate"

#: Per-stream queue depth when the caller does not choose one.
DEFAULT_QUEUE_CAPACITY = 32


@dataclass(frozen=True, slots=True)
class ApproximateSet:
    """One emitted set: a sample per stream, with the evidence for it.

    ``pivot_stream`` / ``pivot_time_ns`` name the latest member — the
    sample the set is anchored on. ``span_ns`` is the spread between the
    earliest and latest member, the quantity the algorithm minimises and
    the one number a multi-sensor consumer actually needs.

    ``provable`` is ``True`` for sets emitted by the normal path, where
    optimality was established before emission, and ``False`` for a set
    produced by :meth:`ApproximateTimeBundler.flush` at shutdown, where
    the remaining samples are emitted without waiting for the proof that
    will now never arrive. A recorder should not silently present the
    two as the same claim.
    """

    pivot_stream: str
    pivot_time_ns: int
    samples: dict[str, Sample]
    span_ns: int
    provable: bool = True

    @property
    def earliest_time_ns(self) -> int:
        return self.pivot_time_ns - self.span_ns

    def skew_ns(self, stream: str) -> int:
        """``acquisition − pivot`` for ``stream``; always ``<= 0``."""
        return self.samples[stream].acquisition_time_ns - self.pivot_time_ns


class ApproximateTimeBundler:
    """Streaming ApproximateTime set builder. Clock-free and deterministic.

    Feed samples with :meth:`push`; it returns the sets that became
    provable as a result, which is usually none and occasionally one.
    The bundler reads no clock and holds no session state, so its whole
    behaviour is a pure function of the push sequence — which is what
    makes the three guarantees testable rather than merely asserted.

    Thread-safety: an internal lock covers every queue, because the
    algorithm is inherently cross-stream (a push on one stream is what
    settles another). Callers must **not** hold a per-stream lock across
    :meth:`push`, or the two lock orders can invert.
    """

    __slots__ = (
        "_capacity",
        "_emitted",
        "_last_key",
        "_last_pivot_ns",
        "_lock",
        "_out_of_order",
        "_overflowed",
        "_queues",
        "_superseded",
    )

    def __init__(
        self,
        streams: Iterable[str],
        *,
        queue_capacity: Mapping[str, int] | int = DEFAULT_QUEUE_CAPACITY,
    ) -> None:
        names = list(streams)
        if len(names) < 2:
            raise ValueError(
                f"ApproximateTime needs at least two streams to have anything "
                f"to approximate, got {names}"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate stream names: {names}")
        self._capacity: dict[str, int] = {}
        for name in names:
            capacity = (
                queue_capacity
                if isinstance(queue_capacity, int)
                else queue_capacity.get(name, DEFAULT_QUEUE_CAPACITY)
            )
            if capacity < 2:
                # One slot can never hold the "sample after the pivot" that
                # step 4 needs, so the bundler would never emit anything.
                raise ValueError(
                    f"queue capacity for {name!r} must be >= 2 (the proof step "
                    f"needs a sample after the pivot), got {capacity}"
                )
            self._capacity[name] = capacity
        self._lock = threading.Lock()
        self._queues: dict[str, deque[Sample]] = {n: deque() for n in names}
        self._last_key: dict[str, int] = {}
        self._last_pivot_ns: int | None = None
        self._emitted = 0
        self._superseded = 0
        self._overflowed = 0
        self._out_of_order = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def stream_names(self) -> tuple[str, ...]:
        return tuple(self._queues)

    @property
    def emitted(self) -> int:
        """Sets emitted so far."""
        return self._emitted

    @property
    def superseded(self) -> int:
        """Samples dropped by the shrink step as provably not the best member."""
        return self._superseded

    @property
    def overflowed(self) -> int:
        """Samples dropped because a queue was full. Non-zero invalidates the proof."""
        return self._overflowed

    @property
    def out_of_order(self) -> int:
        """Samples refused because their acquisition time went backwards."""
        return self._out_of_order

    @property
    def last_pivot_time_ns(self) -> int | None:
        """Pivot of the most recently emitted set, or ``None``."""
        return self._last_pivot_ns

    def pending(self) -> dict[str, int]:
        """Queued-but-unused sample count per stream."""
        with self._lock:
            return {name: len(q) for name, q in self._queues.items()}

    def stats(self) -> dict[str, int]:
        """Counters as a flat mapping — manifest- and log-ready."""
        with self._lock:
            return {
                "emitted": self._emitted,
                "superseded": self._superseded,
                "overflowed": self._overflowed,
                "out_of_order": self._out_of_order,
                "pending": sum(len(q) for q in self._queues.values()),
            }

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def push(self, sample: Sample) -> list[ApproximateSet]:
        """Queue ``sample`` and return whatever became provable.

        ``sample.stream_name`` must be a registered stream
        (:class:`KeyError` otherwise — the same rule
        :meth:`~embodied_sync.align.online.MultiStreamAligner.push` uses:
        a producer typo must never silently drop data).

        A sample whose acquisition time precedes the previous one on the
        same stream is **refused** and counted in :attr:`out_of_order`.
        The algorithm's correctness rests on per-stream monotonicity —
        every "no future sample can be closer" argument is an argument
        about a sorted sequence — so accepting a backwards sample would
        not degrade the result, it would invalidate the proof. The
        session separately reports the same event as a ``non_monotonic``
        violation, so the data is not lost from the diagnostics.
        """
        name = sample.stream_name
        with self._lock:
            queue = self._queues.get(name)
            if queue is None:
                raise KeyError(
                    f"sample stream {name!r} is not part of the approximate "
                    f"set; member streams: {list(self._queues)}"
                )
            key = sample.acquisition_time_ns
            previous = self._last_key.get(name)
            if previous is not None and key < previous:
                self._out_of_order += 1
                return []
            self._last_key[name] = key
            queue.append(sample)
            if len(queue) > self._capacity[name]:
                queue.popleft()
                self._overflowed += 1
            return self._drain_locked()

    def flush(self) -> list[ApproximateSet]:
        """Emit remaining complete sets without waiting for the proof.

        For shutdown only. Every set this returns has ``provable=False``:
        the samples that would have settled it are never going to
        arrive, so "best available" is the strongest claim left. Sets
        still come out in pivot order, and each sample is still used at
        most once — only guarantee three is relaxed, and only for these.
        """
        emitted: list[ApproximateSet] = []
        with self._lock:
            while True:
                result = self._emit_locked(require_proof=False)
                if result is None:
                    break
                emitted.append(result)
        return emitted

    # ------------------------------------------------------------------
    # Algorithm
    # ------------------------------------------------------------------

    def _drain_locked(self) -> list[ApproximateSet]:
        emitted: list[ApproximateSet] = []
        while True:
            result = self._emit_locked(require_proof=True)
            if result is None:
                return emitted
            emitted.append(result)

    def _emit_locked(self, *, require_proof: bool) -> ApproximateSet | None:
        """One round of the algorithm. Returns a set, or ``None`` to wait."""
        queues = self._queues
        if any(not queue for queue in queues.values()):
            return None

        # Step 2 — pivot. `max` keeps the first maximal element, so ties
        # break on stream-configuration order and the result is a pure
        # function of the push sequence.
        pivot_stream = max(queues, key=lambda n: queues[n][0].acquisition_time_ns)
        pivot_ns = queues[pivot_stream][0].acquisition_time_ns

        # Step 3 — shrink. Anything the sample behind it can replace without
        # passing the pivot is provably not this set's best member.
        for queue in queues.values():
            while len(queue) >= 2 and queue[1].acquisition_time_ns <= pivot_ns:
                queue.popleft()
                self._superseded += 1

        # Step 4 — proof.
        if require_proof:
            for name, queue in queues.items():
                if name == pivot_stream:
                    continue
                if len(queue) >= 2:
                    continue  # holds a sample after the pivot: settled
                if queue[0].acquisition_time_ns == pivot_ns:
                    continue  # sitting on the pivot: unbeatable, ties go causal
                return None

        # Step 5 — emit.
        picked = {name: queue.popleft() for name, queue in queues.items()}
        earliest = min(s.acquisition_time_ns for s in picked.values())
        self._emitted += 1
        self._last_pivot_ns = pivot_ns
        return ApproximateSet(
            pivot_stream=pivot_stream,
            pivot_time_ns=pivot_ns,
            samples=picked,
            span_ns=pivot_ns - earliest,
            provable=require_proof,
        )
