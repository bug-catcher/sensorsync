"""Online alignment ring buffer — first Milestone 3 online slice (D-0026).

The offline engine (``align_run`` in :mod:`embodied_sync.align.engine`)
sees every sample before picking any frame; the online counterpart sees
samples arriving one at a time and must answer alignment queries against
whatever has already been pushed. The causality and bounded-memory
invariants are laid out in
``docs/concepts/online_vs_offline_alignment.md``.

This module implements the first online slice: :class:`StreamRingBuffer`
— a bounded per-stream buffer with a zero-order-hold picker. Nearest-
neighbor and linear interpolation are offline-only per D-0025 and the
concept doc, so ZoH is the only correct policy for a deadline-0 online
loop and is the only policy implemented here.

Query contract
--------------
``get_aligned_observation(target_ns, deadline_ns=0)`` returns a
``(sample, metadata)`` pair. A sample is *eligible* iff both:

- ``acquisition_time_ns <= target_ns`` (source is not in the future
  relative to the target — the ZoH "hold last value" semantic); and
- ``receive_time_ns <= target_ns + deadline_ns`` (the sample was
  actually received by the deadline the caller is willing to wait for).

Among eligible samples the pick is the one with the largest
``acquisition_time_ns`` (standard ZoH). The pick is marked
``missing`` if:

- no sample is eligible, or
- the best eligible sample is more than ``tolerance_ns`` older than
  ``target_ns`` (staleness beyond tolerance is the same "held value
  is too old" failure as offline ZoH).

At ``deadline_ns == 0`` the receive-time condition collapses to
``receive_time_ns <= target_ns`` — the causality invariant from the
concept doc.

Skew / confidence follow the offline ZoH convention
(``skew_ns = source - target``, always ``<= 0``;
``confidence = max(0, 1 - staleness / tolerance)``) so a report built
from online frames uses the same tolerance-relative scale as offline.

Tie-breaking (D-0037)
---------------------
Low-resolution vendor clocks stamp several samples with the *same*
``acquisition_time_ns` routinely (a 1 ms-resolution clock feeding a
1 kHz stream, an SDK that stamps a whole burst at flush time). Every
picker here therefore states one rule explicitly: **among candidates
with equal ``acquisition_time_ns`` the last-pushed sample wins**, i.e.
push order breaks the tie, newest push first. That is MCAP's
position-based tie-break (its message index sorts by
``(log_time, position)``) applied to a live buffer, and it matches the
"most recent value" intent of ZoH — the sample that arrived later *is*
the later value even when the device clock cannot say so.

The rule only fires on *identical acquisition times*. Deadline-aware
nearest-neighbor keeps its existing preference for the earlier
candidate when two samples straddle the target with equal ``|skew|``
but different acquisition times: that is a genuine before/after
choice, and picking the causal side of it is the safer default.

A third query, :meth:`StreamRingBuffer.get_window`, returns *every*
sample inside ``±window_ns/2`` of the target rather than a single pick
— observation semantics for audio and other high-rate streams where
"the value at T" is less meaningful than "everything around T".
"""

from __future__ import annotations

from collections import deque
from typing import Iterator

from embodied_sync.align.engine import NEAREST_NEIGHBOR, ZERO_ORDER_HOLD
from embodied_sync.core.episode import AlignedSampleMetadata
from embodied_sync.core.sample import Sample

#: ``AlignedSampleMetadata.method`` value for :meth:`StreamRingBuffer.get_window`.
#: Not an :data:`embodied_sync.align.engine.Method` — the offline engine has no
#: window picker, so the name only ever appears on live bundles (D-0037).
WINDOW = "window"

__all__ = ["WINDOW", "StreamRingBuffer"]


class StreamRingBuffer:
    """Bounded per-stream ring buffer for online ZoH alignment.

    Samples are pushed in receive order (the natural online arrival
    order) and evicted FIFO once ``capacity`` is reached. ``capacity``
    caps memory; ``tolerance_ns`` caps how stale a returned sample may
    be. The caller sets both up front — the online engine cannot
    compute median inter-sample intervals from the future, so
    ``tolerance_ns`` is a fixed input, not a derived statistic.

    Thread-safety: none. Wrap externally if multiple producers push
    concurrently.
    """

    __slots__ = ("_buf", "_capacity", "_tolerance_ns")

    def __init__(self, *, capacity: int, tolerance_ns: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if tolerance_ns < 0:
            raise ValueError(f"tolerance_ns must be non-negative, got {tolerance_ns}")
        self._buf: deque[Sample] = deque(maxlen=capacity)
        self._capacity = capacity
        self._tolerance_ns = tolerance_ns

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def tolerance_ns(self) -> int:
        return self._tolerance_ns

    def __len__(self) -> int:
        return len(self._buf)

    def __iter__(self) -> Iterator[Sample]:
        # `Iterator`, not `Iterable`: this *is* the iterator, and typing it as
        # the weaker protocol made `for s in buffer` fail type-checking at
        # every call site while working fine at runtime.
        return iter(self._buf)

    def push(self, sample: Sample) -> None:
        """Insert ``sample``; evict the oldest push if capacity is reached."""
        self._buf.append(sample)

    def get_aligned_observation(
        self,
        target_ns: int,
        *,
        deadline_ns: int = 0,
    ) -> tuple[Sample | None, AlignedSampleMetadata]:
        """Return the ZoH pick + metadata for ``target_ns``.

        Eligibility: ``acquisition_time_ns <= target_ns`` and
        ``receive_time_ns <= target_ns + deadline_ns``. The returned
        sample is ``None`` (and ``metadata.missing`` is ``True``) when
        no eligible sample exists or the best eligible sample is more
        than ``tolerance_ns`` old. Equal ``acquisition_time_ns``
        candidates tie-break to the last-pushed one (module docstring).

        Skew, confidence, and method match the offline ZoH convention.
        """
        if deadline_ns < 0:
            raise ValueError(f"deadline_ns must be non-negative, got {deadline_ns}")
        receive_cutoff = target_ns + deadline_ns
        best: Sample | None = None
        for sample in self._buf:
            if sample.acquisition_time_ns > target_ns:
                continue
            if sample.receive_time_ns > receive_cutoff:
                continue
            # `>=` (not `>`): iteration follows push order, so the last
            # push among equal acquisition times wins the tie.
            if best is None or sample.acquisition_time_ns >= best.acquisition_time_ns:
                best = sample
        if best is None:
            return None, AlignedSampleMetadata(
                source_time_ns=None,
                skew_ns=None,
                method=ZERO_ORDER_HOLD,
                missing=True,
                confidence=0.0,
            )
        skew = best.acquisition_time_ns - target_ns
        staleness = -skew
        if staleness > self._tolerance_ns:
            return None, AlignedSampleMetadata(
                source_time_ns=best.acquisition_time_ns,
                skew_ns=skew,
                method=ZERO_ORDER_HOLD,
                missing=True,
                confidence=0.0,
            )
        confidence = (
            max(0.0, 1.0 - staleness / self._tolerance_ns)
            if self._tolerance_ns > 0
            else 1.0
        )
        return best, AlignedSampleMetadata(
            source_time_ns=best.acquisition_time_ns,
            skew_ns=skew,
            method=ZERO_ORDER_HOLD,
            missing=False,
            confidence=confidence,
        )

    def get_latest_policy_observation(
        self, now_ns: int
    ) -> tuple[Sample | None, AlignedSampleMetadata]:
        """Return the ZoH pick + metadata for target=``now_ns`` with deadline 0.

        Thin wrapper around
        ``get_aligned_observation(target_ns=now_ns, deadline_ns=0)`` — the
        deadline-zero ZoH surface the concept doc names as the safe
        default for a policy tick. ``now_ns`` is an explicit argument
        because the library does not read the wall clock; the caller
        supplies whatever wall-clock (or simulated-clock) value drives
        their control loop, so tests stay deterministic.
        """
        return self.get_aligned_observation(now_ns, deadline_ns=0)

    def get_window(
        self,
        target_ns: int,
        *,
        window_ns: int,
        deadline_ns: int = 0,
    ) -> tuple[list[Sample], AlignedSampleMetadata]:
        """Return every sample within ``±window_ns/2`` of ``target_ns``.

        The observation-semantics query: for audio, high-rate tactile,
        or event streams, "the one value at T" is the wrong question —
        the caller wants *everything that happened around* T. Returns a
        list in push order plus a single metadata record describing the
        window as a whole.

        Eligibility: ``|acquisition_time_ns - target_ns| <= window_ns //
        2`` **and** ``receive_time_ns <= target_ns + deadline_ns``. The
        receive-time bound is the same causality guard the single-pick
        queries use, so at ``deadline_ns == 0`` a window can never
        contain a sample the host had not yet received at ``target_ns``.
        Note the asymmetry that follows: the *acquisition* window is
        symmetric around the target, but the future half is only
        populated for samples that were received early — exactly the
        deadline trade-off :meth:`get_nearest_neighbor` makes. At
        ``deadline_ns == 0`` a window is therefore effectively
        one-sided. **To get the symmetric window the name promises,
        pass ``deadline_ns >= window_ns // 2`` plus whatever transport
        latency the stream has**; that is what
        :class:`~embodied_sync.session.SyncSession` does for a
        ``policy="window"`` stream. Keeping the guard (rather than
        dropping it for this one query) means ``deadline_ns`` has a
        single meaning across all three pickers.

        Metadata contract (the window is a set, not a pick, so the
        per-sample fields describe a *representative*):

        - ``source_time_ns`` / ``skew_ns`` — the eligible sample nearest
          the target (ties: last push), so callers keep a comparable
          skew number; ``None`` when the window is empty.
        - ``missing`` — ``True`` iff the window is empty. A non-empty
          window is never "outside tolerance": ``window_ns`` *is* the
          tolerance, and every member is inside it by construction.
        - ``confidence`` — ``1.0`` for a non-empty window, ``0.0``
          otherwise. Deliberately not a skew-scaled score: for
          observation semantics the honest question is presence, and a
          scaled score would invent a precision the query does not
          claim. ``tolerance_ns`` is not consulted.
        """
        if window_ns <= 0:
            raise ValueError(f"window_ns must be positive, got {window_ns}")
        if deadline_ns < 0:
            raise ValueError(f"deadline_ns must be non-negative, got {deadline_ns}")
        half = window_ns // 2
        receive_cutoff = target_ns + deadline_ns
        picked: list[Sample] = []
        nearest: Sample | None = None
        nearest_abs_skew: int | None = None
        for sample in self._buf:
            if sample.receive_time_ns > receive_cutoff:
                continue
            abs_skew = abs(sample.acquisition_time_ns - target_ns)
            if abs_skew > half:
                continue
            picked.append(sample)
            if nearest_abs_skew is None or abs_skew <= nearest_abs_skew:
                nearest = sample
                nearest_abs_skew = abs_skew
        if nearest is None:
            return [], AlignedSampleMetadata(
                source_time_ns=None,
                skew_ns=None,
                method=WINDOW,
                missing=True,
                confidence=0.0,
            )
        return picked, AlignedSampleMetadata(
            source_time_ns=nearest.acquisition_time_ns,
            skew_ns=nearest.acquisition_time_ns - target_ns,
            method=WINDOW,
            missing=False,
            confidence=1.0,
        )

    def get_nearest_neighbor(
        self,
        target_ns: int,
        *,
        deadline_ns: int,
    ) -> tuple[Sample | None, AlignedSampleMetadata]:
        """Return the deadline-aware NN pick for ``target_ns``.

        Deadline-aware NN is the online counterpart the offline
        engine's ``method="nearest_neighbor"`` picks in one shot: given
        a ``deadline_ns`` window past the target the caller is willing
        to wait for, pick the sample with the smallest
        ``|acquisition_time_ns - target_ns|`` among those already
        received. This is *only* correct when the caller is willing to
        let observations from the future (``acquisition_time_ns >
        target_ns``) enter the pick — for a policy tick that must be
        strictly causal, use :meth:`get_aligned_observation` (ZoH) with
        ``deadline_ns=0`` instead.

        Eligibility: ``receive_time_ns <= target_ns + deadline_ns``.
        Among eligible samples the pick is by smallest ``|skew|``.
        Missing if no sample is eligible or the best sample's
        ``|skew| > tolerance_ns``.

        Ties: candidates with the *same* ``acquisition_time_ns``
        tie-break to the last-pushed one; candidates with equal
        ``|skew|`` but different acquisition times (one before, one
        after the target) keep the earlier — the causal — one. See the
        module docstring.
        """
        if deadline_ns < 0:
            raise ValueError(f"deadline_ns must be non-negative, got {deadline_ns}")
        receive_cutoff = target_ns + deadline_ns
        best: Sample | None = None
        best_abs_skew: int | None = None
        for sample in self._buf:
            if sample.receive_time_ns > receive_cutoff:
                continue
            abs_skew = abs(sample.acquisition_time_ns - target_ns)
            if best_abs_skew is None or abs_skew < best_abs_skew:
                best = sample
                best_abs_skew = abs_skew
            elif (
                abs_skew == best_abs_skew
                and best is not None
                and sample.acquisition_time_ns == best.acquisition_time_ns
            ):
                best = sample
        if best is None:
            return None, AlignedSampleMetadata(
                source_time_ns=None,
                skew_ns=None,
                method=NEAREST_NEIGHBOR,
                missing=True,
                confidence=0.0,
            )
        skew = best.acquisition_time_ns - target_ns
        if abs(skew) > self._tolerance_ns:
            return None, AlignedSampleMetadata(
                source_time_ns=best.acquisition_time_ns,
                skew_ns=skew,
                method=NEAREST_NEIGHBOR,
                missing=True,
                confidence=0.0,
            )
        confidence = (
            max(0.0, 1.0 - abs(skew) / self._tolerance_ns)
            if self._tolerance_ns > 0
            else 1.0
        )
        return best, AlignedSampleMetadata(
            source_time_ns=best.acquisition_time_ns,
            skew_ns=skew,
            method=NEAREST_NEIGHBOR,
            missing=False,
            confidence=confidence,
        )
