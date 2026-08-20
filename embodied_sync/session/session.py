"""``SyncSession`` — the drop-in live synchronisation surface (D-0037).

The offline half of this library assumes you already have a run on disk.
Getting one is the part every lab reimplements badly: wrap the vendor
SDK's callback, stuff samples in a list, pick "the closest frame" with a
loop nobody reviewed, and discover at training time that the wrist camera
led the robot state by 40 ms all along. :class:`SyncSession` is that
layer, done once::

    import embodied_sync as embsync

    with embsync.init(
        run_dir="runs/experiment_001",
        streams={
            "camera": embsync.StreamConfig(rate_hz=30, tolerance_ms=20.0),
            "robot":  embsync.StreamConfig(rate_hz=250, tolerance_ms=4.0,
                                           policy="nearest", deadline_ms=2.0),
        },
        primary="camera",
    ) as sync:
        camera_sdk.on_frame(sync.attach("camera",
                                        timestamp=lambda f: f.device_ts_ns))
        while running:
            sync.push("robot", robot_sdk.read_state())
            bundle = sync.get()
            if bundle.ok:
                act(bundle["camera"], bundle["robot"])

Design commitments worth stating in one place:

**One clock, injected.** This module is the *only* place in the library
that reads a wall or monotonic clock. Engines stay clock-free, so tests
inject a fake ``clock`` and get bit-identical behaviour. ``clock``
returns integer nanoseconds and must be monotonic.

**Nothing is silent.** Every degraded outcome — a stale hold, a stream
that matched nothing, a device whose timestamps went backwards, a
foreign clock domain with no mapping — becomes a
:class:`~embodied_sync.session.violations.SyncViolation` and is
dispatched per ``on_violation``. The default logs (rate-limited); it
never swallows.

**Live and offline share one on-disk contract.** ``run_dir`` is a run
format v0 directory, so a recorded session is a valid ``embsync align``
input. See :mod:`embodied_sync.session.recorder`.

**Bounded memory.** Per-stream ring buffers, bounded deques for the
quality window, incremental JSONL append. A session's memory does not
grow with its length.

Threading model
---------------
SDK callbacks arrive on whatever thread the vendor chose. Each stream
owns a :class:`threading.Lock` covering its buffer, stats deques,
sequence counter and file append, so two producers never interleave a
record. :meth:`get` takes those locks one at a time to snapshot picks —
it does not hold a global lock, so a slow stream cannot stall an
unrelated producer. Violations are always dispatched *after* the lock is
released: an ``on_violation`` callable is user code and may legitimately
call back into the session, which would deadlock under a held
non-reentrant lock.

No asyncio in v1. A blocking ``wait_for``-style ``get`` is deferred
(design §4, increment 2).

Open choices this module settled (design left them open)
--------------------------------------------------------
- ``get(reference=...)`` on a stream that has received *nothing* does
  not raise: it emits a ``no_samples`` violation, anchors the bundle at
  ``clock()``, and returns an all-missing bundle. A control loop
  polling during start-up is the common case, and crashing on it would
  push every caller into a try/except that hides real errors.
- An **unmapped** foreign clock domain is scored as though its mapping
  variance equalled the stream's tolerance, i.e. a 0.5 confidence
  multiplier via
  :func:`~embodied_sync.time.alignment.cross_domain_confidence_factor`.
  Matching happens on receive time, so the true mapping error is
  unknown; assuming it is comparable to the tolerance budget is the
  least-arbitrary pessimism available.
- The bundle's ``Sample`` for a mapped foreign stream carries
  *session-domain* acquisition times (that is what the skew is measured
  against). The run-dir JSONL keeps the **raw** device timestamps and
  the stream's declared domain, because adapters must never re-timestamp
  what they record (D-0003).

Increment 2 additions (Lane A, A1–A3)
-------------------------------------
- ``policy="approximate"`` enrols a stream in a true
  pivot-and-span-minimising ApproximateTime set
  (:mod:`embodied_sync.session.approximate`). Its bundles arrive through
  :meth:`SyncSession.poll_bundles` when the data allows, not when a
  caller asks — that is what "recording-time mode" means, and it costs
  roughly one period of the slowest member in latency.
- :meth:`SyncSession.time_correction` returns the LSL-shaped correction
  for a stream: the offset you **add** to a stream-domain timestamp to
  reach the session domain, typed as a
  :class:`~embodied_sync.time.clock_domain.LatencyEstimate` rather than
  a bare float, and cached after the first (computing) call.
- :meth:`SyncSession.mark_clock_reset` /
  :meth:`SyncSession.was_clock_reset` generalise LSL's
  ``was_clock_reset()`` to a per-domain generation counter, so a device
  reconnect *invalidates* the mappings and buffers fitted against the
  old timeline instead of silently poisoning every later fit with a
  discontinuity.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Iterable, Mapping

from embodied_sync.align.ring_buffer import StreamRingBuffer
from embodied_sync.core.episode import AlignedSampleMetadata
from embodied_sync.core.sample import (
    QUALITY_CLOCK_MAPPED,
    QUALITY_NON_MONOTONIC,
    QUALITY_RECEIVE_TIMESTAMPED,
    QUALITY_UNMAPPED_CLOCK_DOMAIN,
    Sample,
)
from embodied_sync.session.approximate import (
    APPROXIMATE_METHOD,
    ApproximateSet,
    ApproximateTimeBundler,
)
from embodied_sync.session.bundle import BundleItem, SyncBundle
from embodied_sync.session.config import (
    POLICY_APPROXIMATE,
    POLICY_LATEST_BEFORE,
    POLICY_NEAREST,
    POLICY_WINDOW,
    StreamConfig,
)
from embodied_sync.session.quality import (
    MIN_RATE_FRACTION,
    LiveStreamQuality,
    MatchRecord,
    compute_stream_quality,
    median,
)
from embodied_sync.session.recorder import (
    SESSION_QUALITY_NAME,
    PayloadSerializer,
    RunRecorder,
)
from embodied_sync.session.violations import (
    CLOCK_EPOCH_ADVANCED,
    DEFAULT_VIOLATION_INTERVAL_S,
    NO_ELIGIBLE_BEFORE_DEADLINE,
    NO_SAMPLES,
    NON_MONOTONIC,
    OUTSIDE_TOLERANCE,
    RATE_BELOW_EXPECTED,
    UNMAPPED_CLOCK_DOMAIN,
    VIOLATION_REASONS,
    RateLimiter,
    SyncToleranceError,
    SyncViolation,
    ViolationHandler,
)
from embodied_sync.time.alignment import cross_domain_confidence_factor
from embodied_sync.time.clock_domain import (
    KNOWN_DOMAINS,
    ClockDomain,
    ClockEpochRegistry,
    ClockKind,
    LatencyEstimate,
    latency_estimate_to_dict,
    translate_ns,
)

__all__ = [
    "DEFAULT_QUALITY_WINDOW_S",
    "DEFAULT_TIME_CORRECTION_MAX_AGE_S",
    "REFERENCE_METHOD",
    "SESSION_CLOCK_DOMAIN",
    "SyncSession",
    "init",
    "logger",
]

#: The logger every session diagnostic goes through. Never ``print``.
logger = logging.getLogger("embodied_sync.session")

#: Clock domain the session matches in. Streams declaring anything else
#: are foreign and need a registered mapping (§2.7).
SESSION_CLOCK_DOMAIN = "host_mono"

#: ``BundleItem.method`` for the reference stream's own entry: it defines
#: the target time rather than being picked against it.
REFERENCE_METHOD = "reference"

DEFAULT_QUALITY_WINDOW_S = 10.0

#: How long a cached :meth:`SyncSession.time_correction` stays fresh. LSL's
#: inlets refresh their time correction on roughly this cadence; the number
#: is a trade between tracking a slowly drifting clock and making a
#: per-sample call cost a measurement.
DEFAULT_TIME_CORRECTION_MAX_AGE_S = 5.0

#: Matches kept per stream for ``quality()``. Bounded so a long session
#: costs the same as a short one; ~2.7 minutes of 100 Hz ``get()`` calls.
_MATCH_HISTORY = 16_384

#: Approximate bundles held for :meth:`SyncSession.poll_bundles`. Bounded for
#: the same reason every other buffer here is: a consumer that stops polling
#: must cost memory that stops growing, not a session that dies.
_READY_BUNDLES = 4_096

#: Minimum receives before the ingest-time rate check says anything —
#: below this a "rate" is noise, not a measurement.
_RATE_CHECK_MIN_SAMPLES = 8

#: Robust-scale constant (1.4826·MAD ≈ σ for Gaussian data), as used by
#: ``calibrate/estimator.py``. Repeated rather than imported: ``session``
#: consumes clock mappings and must not depend on the package that
#: produces them.
_MAD_TO_SIGMA = 1.4826


def _session_domain() -> ClockDomain:
    return KNOWN_DOMAINS[SESSION_CLOCK_DOMAIN]


def _declared_domain(name: str) -> ClockDomain:
    """Typed view of a user-declared ``clock_domain`` string, without warning.

    :func:`~embodied_sync.time.clock_domain.resolve_clock_domain` warns
    once per unknown name, which is right for an adapter meeting a new
    format and wrong here: the session's domain strings come from the
    caller's own ``StreamConfig``, so there is nothing to tell them they
    do not already know.
    """
    known = KNOWN_DOMAINS.get(name)
    return known if known is not None else ClockDomain(name, ClockKind.UNKNOWN)


@dataclass(frozen=True, slots=True)
class _CachedCorrection:
    """One memoised :meth:`SyncSession.time_correction` result."""

    estimate: LatencyEstimate
    computed_at_ns: int


class _StreamState:
    """Mutable per-stream state, guarded by :attr:`lock`."""

    __slots__ = (
        "buffer",
        "config",
        "lock",
        "last_acquisition_ns",
        "mapping",
        "matches",
        "name",
        "newest",
        "next_sequence_id",
        "raw_deltas",
        "receive_times",
    )

    def __init__(self, name: str, config: StreamConfig) -> None:
        self.name = name
        self.config = config
        self.lock = threading.Lock()
        self.buffer = StreamRingBuffer(
            capacity=config.capacity, tolerance_ns=config.tolerance_ns
        )
        self.next_sequence_id = 0
        self.newest: Sample | None = None
        self.last_acquisition_ns: int | None = None
        self.mapping: LatencyEstimate | None = None
        self.receive_times: deque[int] = deque(maxlen=config.capacity)
        self.matches: deque[MatchRecord] = deque(maxlen=_MATCH_HISTORY)
        #: ``receive_time_ns − acquisition_time_ns`` against the **raw device**
        #: timestamp, which the ring buffer cannot supply: it stores the
        #: session-domain *match* view, whose acquisition time is already the
        #: receive time for an unmapped foreign stream (so every delta there
        #: is trivially zero). :meth:`SyncSession.time_correction` needs the
        #: device's own numbers, so they are kept here.
        self.raw_deltas: deque[int] = deque(maxlen=config.capacity)


class SyncSession:
    """Live multi-stream synchronisation with recording and diagnostics.

    Construct via :func:`init` or directly; both take the same keyword
    arguments. Use as a context manager so :meth:`close` runs even on an
    exception path.

    :param streams: stream name → :class:`StreamConfig`. Iteration order
        is bundle order and manifest order.
    :param run_dir: run format v0 directory to record into, or ``None``
        for no persistence.
    :param primary: default reference stream for :meth:`get`.
    :param on_violation: ``"warn"`` (default, rate-limited logging),
        ``"raise"``, ``"ignore"``, or a callable taking a
        :class:`SyncViolation`.
    :param clock: monotonic integer-ns clock. Injected for tests.
    :param violation_interval_s: rate-limit interval for ``"warn"``.
    :param serialize: ``(stream, payload) -> JSON-able`` hook used by
        ``persist="full"`` streams.
    """

    def __init__(
        self,
        *,
        streams: Mapping[str, StreamConfig],
        run_dir: str | Path | None = None,
        primary: str | None = None,
        on_violation: str | ViolationHandler = "warn",
        clock: Callable[[], int] = time.monotonic_ns,
        violation_interval_s: float = DEFAULT_VIOLATION_INTERVAL_S,
        serialize: PayloadSerializer | None = None,
    ) -> None:
        if not streams:
            raise ValueError("streams must contain at least one StreamConfig")
        for name, config in streams.items():
            if not isinstance(config, StreamConfig):
                raise TypeError(
                    f"streams[{name!r}] must be a StreamConfig, got "
                    f"{type(config).__name__}"
                )
        if primary is not None and primary not in streams:
            raise ValueError(
                f"primary={primary!r} is not a configured stream; "
                f"known streams: {list(streams)}"
            )
        if isinstance(on_violation, str):
            if on_violation not in ("warn", "raise", "ignore"):
                raise ValueError(
                    f"on_violation must be 'warn', 'raise', 'ignore' or a "
                    f"callable, got {on_violation!r}"
                )
        elif not callable(on_violation):
            raise TypeError(
                f"on_violation must be a str or callable, got "
                f"{type(on_violation).__name__}"
            )

        self._configs: dict[str, StreamConfig] = dict(streams)
        approximate = [
            name
            for name, config in self._configs.items()
            if config.policy == POLICY_APPROXIMATE
        ]
        if len(approximate) == 1:
            raise ValueError(
                f"policy='approximate' is a set-level policy and needs at least "
                f"two streams to have anything to approximate; only "
                f"{approximate[0]!r} declares it. Mark the streams that should "
                f"be bundled together, or use policy='nearest' for a "
                f"single-stream pick."
            )
        self._states: dict[str, _StreamState] = {
            name: _StreamState(name, config) for name, config in self._configs.items()
        }
        self._approximate_streams: tuple[str, ...] = tuple(approximate)
        self._bundler: ApproximateTimeBundler | None = (
            ApproximateTimeBundler(
                approximate,
                queue_capacity={n: self._configs[n].capacity for n in approximate},
            )
            if approximate
            else None
        )
        self._ready: deque[SyncBundle] = deque(maxlen=_READY_BUNDLES)
        self._ready_lock = threading.Lock()
        self._dropped_bundles = 0
        self._epochs = ClockEpochRegistry()
        self._corrections: dict[str, _CachedCorrection] = {}
        self._corrections_lock = threading.Lock()
        self._seen_epoch: dict[str, int] = {}
        self._primary = primary
        self._on_violation = on_violation
        self._clock = clock
        self._serialize = serialize
        self._limiter = RateLimiter(
            interval_ns=round(violation_interval_s * 1e9), clock=clock
        )
        self._violation_counts: dict[tuple[str, str], int] = {}
        self._violation_lock = threading.Lock()
        self._mappings: dict[str, LatencyEstimate] = {}
        self._start_time_ns = clock()
        self._closed = False
        self._recorder: RunRecorder | None = None
        if run_dir is not None:
            self._recorder = RunRecorder(run_dir, self._configs, serialize=serialize)
            # An empty-but-valid manifest exists from the first moment, so a
            # run directory is never half-formed on disk.
            self._recorder.write_manifest(self._build_manifest())

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def stream_names(self) -> tuple[str, ...]:
        """Configured stream names, in bundle order."""
        return tuple(self._configs)

    @property
    def primary(self) -> str | None:
        return self._primary

    @property
    def run_dir(self) -> Path | None:
        return self._recorder.run_dir if self._recorder is not None else None

    @property
    def closed(self) -> bool:
        return self._closed

    def config(self, stream: str) -> StreamConfig:
        """Return ``stream``'s config (``KeyError`` naming known streams)."""
        return self._state(stream).config

    def buffer(self, stream: str) -> StreamRingBuffer:
        """Return ``stream``'s ring buffer, for tests and introspection."""
        return self._state(stream).buffer

    def violation_counts(self) -> dict[tuple[str, str], int]:
        """Total violations per ``(stream, reason)``, including suppressed ones."""
        return dict(self._violation_counts)

    def suppressed_counts(self) -> dict[tuple[str, str], int]:
        """Warnings withheld by the rate limiter, per ``(stream, reason)``."""
        return self._limiter.suppressed

    def _state(self, stream: str) -> _StreamState:
        try:
            return self._states[stream]
        except KeyError:
            raise KeyError(
                f"unknown stream {stream!r}; configured streams: "
                f"{list(self._configs)}"
            ) from None

    # ------------------------------------------------------------------
    # Clock domains
    # ------------------------------------------------------------------

    def register_clock_mapping(self, mapping: LatencyEstimate) -> None:
        """Register a foreign-domain → session-domain mapping.

        ``mapping.source.name`` must be the ``clock_domain`` of at least
        one configured stream and ``mapping.target.name`` must be the
        session domain (:data:`SESSION_CLOCK_DOMAIN`) — a mapping that
        lands somewhere else would silently do nothing, which is exactly
        the failure this library exists to prevent. Produced by
        :mod:`embodied_sync.calibrate`; consumed here via
        :func:`~embodied_sync.time.clock_domain.translate_ns` at push
        time. Applies to every stream declaring that source domain, and
        replaces any previous mapping for it (a recalibration is
        expected to supersede).

        ``mapping.epoch`` must be the source domain's **current**
        generation (:meth:`clock_epoch`). A mapping fitted before a
        reconnect describes a timeline that no longer exists, and
        accepting it is the exact silent-poisoning failure the epoch
        counter exists to prevent; use
        :meth:`~embodied_sync.time.clock_domain.LatencyEstimate.with_epoch`
        only when the fit really was taken in the current generation.
        """
        source = mapping.source.name
        target = mapping.target.name
        if target != SESSION_CLOCK_DOMAIN:
            raise ValueError(
                f"mapping target must be the session clock domain "
                f"{SESSION_CLOCK_DOMAIN!r}, got {target!r}"
            )
        users = [n for n, c in self._configs.items() if c.clock_domain == source]
        if not users:
            declared = sorted({c.clock_domain for c in self._configs.values()})
            raise ValueError(
                f"no configured stream declares clock_domain {source!r}; "
                f"declared domains: {declared}"
            )
        current = self._epochs.current(source)
        if mapping.epoch != current:
            raise ValueError(
                f"mapping for clock domain {source!r} was fitted in epoch "
                f"{mapping.epoch} but the domain is now in epoch {current}: a "
                f"clock reset or device reconnect happened in between, so this "
                f"mapping no longer describes this clock. Recalibrate and "
                f"register the new fit."
            )
        self._mappings[source] = mapping
        for name in users:
            state = self._states[name]
            with state.lock:
                state.mapping = mapping
        self._invalidate_corrections(users)

    # ------------------------------------------------------------------
    # Clock epochs (LSL's was_clock_reset(), generalised — A3)
    # ------------------------------------------------------------------

    def clock_epoch(self, stream: str) -> int:
        """Current generation of ``stream``'s clock domain.

        Starts at :data:`~embodied_sync.time.clock_domain.INITIAL_EPOCH`
        and increases by one per :meth:`mark_clock_reset`. Streams
        sharing a ``clock_domain`` share its epoch, because they share
        the clock that reset.
        """
        return self._epochs.current(self._state(stream).config.clock_domain)

    def clock_epochs(self) -> dict[str, int]:
        """Current generation per declared clock domain."""
        return {
            domain: self._epochs.current(domain)
            for domain in sorted({c.clock_domain for c in self._configs.values()})
        }

    def was_clock_reset(self, stream: str) -> bool:
        """Whether ``stream``'s clock reset since *this method* last said so.

        LSL's ``was_clock_reset()`` semantics, including its
        statefulness: the answer is consumed by reading it, so a polling
        loop sees each reset exactly once and the first call — which has
        no previous observation to compare against — is always ``False``.
        Ask :meth:`clock_epoch` instead when you want an idempotent read
        and want to hold the comparison point yourself; that spelling
        composes, this one matches the loop LSL users already write.
        """
        domain = self._state(stream).config.clock_domain
        current = self._epochs.current(domain)
        with self._corrections_lock:
            seen = self._seen_epoch.get(stream)
            self._seen_epoch[stream] = current
        return seen is not None and current > seen

    def mark_clock_reset(
        self, stream: str, *, reason: str = "device reconnect"
    ) -> int:
        """Declare that ``stream``'s clock domain restarted; return the new epoch.

        Call this the moment a device reconnects, a clock steps, or a
        sensor power-cycles. Everything the session holds about that
        domain's *old* timeline is then discarded rather than left to
        contaminate what comes next:

        - the registered clock mapping for the domain is dropped (it was
          fitted against a timeline that no longer exists);
        - each affected stream's ring buffer is replaced, because
          old-epoch acquisition times are not comparable to new-epoch
          ones and matching across the boundary produces confident
          nonsense;
        - the monotonicity tracker is reset, so the legitimate backwards
          jump of a restarting counter is not reported as a fault;
        - any cached :meth:`time_correction` for those streams is dropped.

        A ``clock_epoch_advanced`` violation is emitted per affected
        stream. The reset itself is not a fault — it is a fact about the
        hardware — but its *consequences* are a discontinuity in the
        recording, and a caller who does not hear about it will spend an
        afternoon wondering where their calibration went.

        Quality history and the run-dir JSONL are deliberately left
        alone: both are records of what happened, and what happened
        includes the reset.
        """
        config = self._state(stream).config
        domain = config.clock_domain
        record = self._epochs.advance(domain, reason=reason, at_ns=self._clock())
        affected = [n for n, c in self._configs.items() if c.clock_domain == domain]
        self._mappings.pop(domain, None)
        for name in affected:
            state = self._states[name]
            with state.lock:
                state.mapping = None
                state.buffer = StreamRingBuffer(
                    capacity=state.config.capacity,
                    tolerance_ns=state.config.tolerance_ns,
                )
                state.newest = None
                state.last_acquisition_ns = None
                state.raw_deltas.clear()
        self._invalidate_corrections(affected)
        for name in affected:
            self._emit(
                SyncViolation(
                    stream=name,
                    reason=CLOCK_EPOCH_ADVANCED,
                    target_time_ns=record.started_at_ns,
                    skew_ns=None,
                    tolerance_ns=None,
                    message=(
                        f"stream {name!r}: clock domain {domain!r} entered epoch "
                        f"{record.epoch} ({reason}); its clock mapping and "
                        f"buffered samples were discarded because they describe "
                        f"the previous timeline. Recalibrate before relying on "
                        f"cross-domain timing again."
                    ),
                )
            )
        return record.epoch

    def _invalidate_corrections(self, streams: Iterable[str]) -> None:
        with self._corrections_lock:
            for name in streams:
                self._corrections.pop(name, None)

    # ------------------------------------------------------------------
    # Time correction (LSL semantics — A2)
    # ------------------------------------------------------------------

    def time_correction(
        self,
        stream: str,
        *,
        max_age_s: float = DEFAULT_TIME_CORRECTION_MAX_AGE_S,
        force: bool = False,
    ) -> LatencyEstimate:
        """Return the correction from ``stream``'s clock to the session's.

        **Sign convention, stated once because getting it backwards is
        the entire class of bug this method exists to remove:** the
        result's ``offset_ns`` is what you *add* to a timestamp in the
        stream's domain to obtain the session-domain time of the same
        instant. That is LSL's ``time_correction()`` convention exactly
        (``remote + correction = local``), and
        :func:`~embodied_sync.time.clock_domain.translate_ns` already
        applies it — so ``translate_ns(t, session.time_correction(s))``
        is the whole usage.

        Unlike LSL this returns the repo's typed
        :class:`~embodied_sync.time.clock_domain.LatencyEstimate` rather
        than a float. A bare number carries no domains, no drift, no
        anchor and no uncertainty, and every one of those is needed to
        decide whether the correction may be trusted or applied at all.

        **Performance contract, also LSL's**: the first call for a
        stream computes, subsequent calls are cached reads. The cache is
        invalidated by age (``max_age_s``, default
        :data:`DEFAULT_TIME_CORRECTION_MAX_AGE_S`), by
        :meth:`register_clock_mapping`, and by :meth:`mark_clock_reset`
        — never silently retained across a reset. ``force=True``
        recomputes now.

        Where the number comes from, in order:

        1. **The stream is already in the session domain.** The
           correction is exactly zero with zero variance. There is
           nothing to measure; transport latency is a different quantity
           and :meth:`quality` is where it is reported.
        2. **A calibrated mapping is registered** for the domain
           (:meth:`register_clock_mapping`). That mapping *is* the
           correction — offset, drift, anchor and measured variance
           intact — and is returned unchanged.
        3. **Neither.** The correction is estimated from the stream's
           own arrivals as the median of ``receive_time_ns −
           acquisition_time_ns`` over the buffer. That is honest but
           weak, and ``variance_ns`` says so: a one-way observation
           cannot separate clock offset from transport delay (the
           identifiability problem that is why NTP and PTP measure
           *round* trips), so the true offset lies somewhere between
           this estimate and this estimate minus the full one-way
           latency. The reported variance is therefore at least the
           magnitude of the estimate itself, which drives
           :func:`~embodied_sync.time.alignment.cross_domain_confidence_factor`
           down accordingly. Calibrate with
           :mod:`embodied_sync.calibrate` to do better; no arithmetic
           extracts a clean offset from one-way data.

        Case 3 with an empty buffer raises :class:`ValueError`. LSL
        blocks until it has a measurement; this session never blocks, so
        refusing is the only alternative to inventing one.
        """
        state = self._state(stream)
        config = state.config
        now_ns = self._clock()
        if not force:
            with self._corrections_lock:
                cached = self._corrections.get(stream)
            if cached is not None and now_ns - cached.computed_at_ns < round(
                max_age_s * 1e9
            ):
                return cached.estimate
        estimate = self._compute_correction(state, config, now_ns)
        with self._corrections_lock:
            self._corrections[stream] = _CachedCorrection(
                estimate=estimate, computed_at_ns=now_ns
            )
        return estimate

    def _compute_correction(
        self, state: _StreamState, config: StreamConfig, now_ns: int
    ) -> LatencyEstimate:
        domain = config.clock_domain
        epoch = self._epochs.current(domain)
        if domain == SESSION_CLOCK_DOMAIN:
            return LatencyEstimate(
                source=_session_domain(),
                target=_session_domain(),
                offset_ns=0,
                drift_ppb=0,
                anchor_time_ns=now_ns,
                variance_ns=0,
                epoch=epoch,
            )
        with state.lock:
            mapping = state.mapping
            deltas = list(state.raw_deltas)
        if mapping is not None:
            return mapping
        if not deltas:
            raise ValueError(
                f"stream {state.name!r} declares clock domain {domain!r}, has no "
                f"registered mapping, and has buffered no samples, so there is "
                f"nothing to measure a time correction from. Push at least one "
                f"sample, or register a calibrated mapping with "
                f"register_clock_mapping()."
            )
        offset_ns = round(median([float(d) for d in deltas]))
        spread = (
            _MAD_TO_SIGMA * median([abs(float(d) - offset_ns) for d in deltas])
            if len(deltas) >= 2
            else 0.0
        )
        # The one-way ambiguity dominates: the true clock offset lies in
        # [offset - latency, offset], and `latency` is exactly what a one-way
        # observation cannot see. Claiming less uncertainty than the estimate's
        # own magnitude would be claiming to have solved that.
        variance_ns = max(round(spread), abs(offset_ns))
        self._emit(
            SyncViolation(
                stream=state.name,
                reason=UNMAPPED_CLOCK_DOMAIN,
                target_time_ns=now_ns,
                skew_ns=offset_ns,
                tolerance_ns=config.tolerance_ns,
                message=(
                    f"stream {state.name!r}: time_correction for clock domain "
                    f"{domain!r} was estimated from {len(deltas)} one-way "
                    f"arrival(s) as {offset_ns / 1e6:.3f} ms with variance "
                    f"{variance_ns / 1e6:.3f} ms, because no calibrated mapping "
                    f"is registered. One-way data cannot separate clock offset "
                    f"from transport latency; calibrate for a real mapping."
                ),
            )
        )
        return LatencyEstimate(
            source=_declared_domain(domain),
            target=_session_domain(),
            offset_ns=offset_ns,
            drift_ppb=0,
            anchor_time_ns=now_ns,
            variance_ns=variance_ns,
            epoch=epoch,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def attach(
        self,
        stream: str,
        callback: Callable[..., Any] | None = None,
        *,
        timestamp: Callable[..., int] | None = None,
        payload: Callable[..., Any] | None = None,
    ) -> Callable[..., Any]:
        """Wrap an SDK callback so every delivery lands in ``stream``.

        The returned wrapper records the receive time *first*, extracts
        the device timestamp with ``timestamp(*args, **kwargs)`` (falling
        back to the receive time plus a ``receive_timestamped`` quality
        flag), selects the payload with ``payload(*args, **kwargs)`` (or
        the single positional argument, or the whole args tuple), pushes
        the sample, and only then calls ``callback`` with the *original*
        arguments, returning its return value.

        Exceptions from ``callback`` propagate — they are the caller's
        own bug and must not be swallowed — but the sample is already
        secured by the time it can raise, so a crashing consumer never
        costs data.

        The stream name is validated now rather than at first delivery:
        a typo should fail at wiring time, not an hour into a recording.
        """
        self._state(stream)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            receive_ns = self._clock()
            t_ns = timestamp(*args, **kwargs) if timestamp is not None else None
            if payload is not None:
                value = payload(*args, **kwargs)
            elif len(args) == 1 and not kwargs:
                value = args[0]
            else:
                value = args
            self._ingest(stream, value, t_ns=t_ns, receive_ns=receive_ns)
            if callback is not None:
                return callback(*args, **kwargs)
            return None

        if callback is not None:
            functools.update_wrapper(wrapper, callback)
        return wrapper

    def push(self, stream: str, payload: Any, *, t_ns: int | None = None) -> Sample:
        """Poll-style ingestion: the same pipeline without the wrapper.

        Pull-based SDKs are first-class here — plenty of vendor APIs only
        offer ``read()``. ``t_ns`` is the device acquisition time in the
        stream's declared clock domain; omit it to timestamp on receipt
        (which flags the sample ``receive_timestamped``, because the
        transport latency is then baked into the acquisition time).

        Unknown stream → :class:`KeyError`, the same rule
        :meth:`~embodied_sync.align.online.MultiStreamAligner.push` uses.
        Returns the stored :class:`Sample`.
        """
        return self._ingest(stream, payload, t_ns=t_ns, receive_ns=self._clock())

    def _ingest(
        self,
        stream: str,
        payload: Any,
        *,
        t_ns: int | None,
        receive_ns: int,
    ) -> Sample:
        state = self._state(stream)
        config = state.config
        if t_ns is not None and (not isinstance(t_ns, int) or isinstance(t_ns, bool)):
            raise TypeError(
                f"stream {stream!r}: t_ns must be int nanoseconds, got "
                f"{type(t_ns).__name__}"
            )
        flags: set[str] = set()
        if t_ns is None:
            acquisition_ns = receive_ns
            flags.add(QUALITY_RECEIVE_TIMESTAMPED)
        else:
            acquisition_ns = t_ns

        pending: list[SyncViolation] = []
        with state.lock:
            sequence_id = state.next_sequence_id
            state.next_sequence_id = sequence_id + 1

            match_ns = acquisition_ns
            foreign = config.clock_domain != SESSION_CLOCK_DOMAIN
            if foreign:
                if state.mapping is not None:
                    match_ns = translate_ns(acquisition_ns, state.mapping)
                    flags.add(QUALITY_CLOCK_MAPPED)
                else:
                    match_ns = receive_ns
                    flags.add(QUALITY_UNMAPPED_CLOCK_DOMAIN)
                    pending.append(
                        SyncViolation(
                            stream=stream,
                            reason=UNMAPPED_CLOCK_DOMAIN,
                            target_time_ns=None,
                            skew_ns=None,
                            tolerance_ns=config.tolerance_ns,
                            message=(
                                f"stream {stream!r} declares clock domain "
                                f"{config.clock_domain!r} with no registered "
                                f"mapping to {SESSION_CLOCK_DOMAIN!r}; matching on "
                                f"receive time with lowered confidence. Call "
                                f"register_clock_mapping() with a LatencyEstimate "
                                f"(embodied_sync.calibrate produces one)."
                            ),
                        )
                    )

            previous = state.last_acquisition_ns
            if previous is not None and acquisition_ns < previous:
                flags.add(QUALITY_NON_MONOTONIC)
                pending.append(
                    SyncViolation(
                        stream=stream,
                        reason=NON_MONOTONIC,
                        target_time_ns=None,
                        skew_ns=acquisition_ns - previous,
                        tolerance_ns=None,
                        message=(
                            f"stream {stream!r}: acquisition time went backwards "
                            f"by {(previous - acquisition_ns) / 1e6:.3f} ms "
                            f"({previous} -> {acquisition_ns})"
                        ),
                    )
                )
            state.last_acquisition_ns = acquisition_ns

            record_sample = Sample(
                stream_name=stream,
                modality=config.modality_value,
                sequence_id=sequence_id,
                acquisition_time_ns=acquisition_ns,
                receive_time_ns=receive_ns,
                source_clock_domain=config.clock_domain,
                payload=payload,
                quality_flags=frozenset(flags),
            )
            if match_ns == acquisition_ns and not foreign:
                match_sample = record_sample
            else:
                # Session-domain view for matching; the recorded sample keeps
                # the device's own numbers (D-0003).
                match_sample = Sample(
                    stream_name=stream,
                    modality=config.modality_value,
                    sequence_id=sequence_id,
                    acquisition_time_ns=match_ns,
                    receive_time_ns=receive_ns,
                    source_clock_domain=SESSION_CLOCK_DOMAIN,
                    payload=payload,
                    quality_flags=frozenset(flags),
                )

            state.buffer.push(match_sample)
            if (
                state.newest is None
                or match_sample.acquisition_time_ns
                >= state.newest.acquisition_time_ns
            ):
                state.newest = match_sample
            state.receive_times.append(receive_ns)
            state.raw_deltas.append(receive_ns - acquisition_ns)
            rate_violation = self._rate_violation(state)
            if rate_violation is not None:
                pending.append(rate_violation)
            if self._recorder is not None:
                self._recorder.append(stream, record_sample)

        # Outside the stream lock, deliberately. The bundler is cross-stream
        # and holds its own lock; taking it while holding a per-stream lock
        # would give the two locks an order that a push on another stream can
        # invert. Violations are dispatched out here for the same reason
        # (user code may re-enter), so the rule is one rule: nothing that can
        # block on someone else happens under a stream lock.
        if self._bundler is not None and config.policy == POLICY_APPROXIMATE:
            self._absorb_sets(self._bundler.push(match_sample))
        for violation in pending:
            self._emit(violation)
        return record_sample

    # ------------------------------------------------------------------
    # ApproximateTime bundling (A1)
    # ------------------------------------------------------------------

    def _absorb_sets(self, sets: Iterable[ApproximateSet]) -> None:
        """Turn emitted :class:`ApproximateSet`s into queued bundles."""
        for emitted in sets:
            bundle = self._bundle_from_set(emitted)
            for name, item in bundle.items.items():
                state = self._states[name]
                with state.lock:
                    state.matches.append(
                        MatchRecord(
                            target_time_ns=bundle.target_time_ns,
                            skew_ns=item.skew_ns,
                            missing=item.missing,
                            within_tolerance=item.within_tolerance,
                        )
                    )
            with self._ready_lock:
                if len(self._ready) == self._ready.maxlen:
                    self._dropped_bundles += 1
                self._ready.append(bundle)

    def _bundle_from_set(self, emitted: ApproximateSet) -> SyncBundle:
        """Render one set as a :class:`SyncBundle` in configuration order.

        The bundle covers exactly the approximate streams — the set is
        what the algorithm computed, and padding it with picks for
        streams that were never part of the optimisation would present a
        different claim under the same type. Query the rest with
        ``get(at_ns=bundle.target_time_ns)`` when you want them.
        """
        items: dict[str, BundleItem] = {}
        for name in self._approximate_streams:
            sample = emitted.samples[name]
            config = self._configs[name]
            skew_ns = sample.acquisition_time_ns - emitted.pivot_time_ns
            within = abs(skew_ns) <= config.tolerance_ns
            confidence = (
                max(0.0, 1.0 - abs(skew_ns) / config.tolerance_ns)
                if config.tolerance_ns > 0
                else 1.0
            )
            items[name] = BundleItem(
                payload=sample.payload,
                sample=sample,
                skew_ns=skew_ns,
                within_tolerance=within,
                missing=False,
                confidence=self._scaled_confidence(
                    config, self._states[name].mapping, confidence
                ),
                method=APPROXIMATE_METHOD,
            )
        return SyncBundle(
            target_time_ns=emitted.pivot_time_ns,
            items=items,
            ok=all(item.within_tolerance for item in items.values()),
            span_ns=emitted.span_ns if len(items) >= 2 else None,
        )

    def poll_bundles(self, max_bundles: int | None = None) -> list[SyncBundle]:
        """Drain the ApproximateTime bundles that have become available.

        Non-blocking by construction: it returns what the algorithm has
        already proven optimal and nothing else, so an empty list means
        "not yet", never "no". Bundles come out in emission order, which
        is non-decreasing in ``target_time_ns`` (guarantee two).

        A session with no ``policy="approximate"`` stream always returns
        an empty list rather than raising — a consumer that polls
        unconditionally should not have to know how the session was
        configured to do so safely.
        """
        if max_bundles is not None and max_bundles <= 0:
            raise ValueError(f"max_bundles must be > 0 or None, got {max_bundles}")
        drained: list[SyncBundle] = []
        with self._ready_lock:
            while self._ready and (max_bundles is None or len(drained) < max_bundles):
                drained.append(self._ready.popleft())
        return drained

    def pending_bundles(self) -> int:
        """Bundles emitted but not yet polled."""
        with self._ready_lock:
            return len(self._ready)

    def approximate_stats(self) -> dict[str, int]:
        """Bundler counters, or an empty mapping when no set is configured.

        ``overflowed`` deserves attention: a non-zero value means a
        stream stalled long enough for the others to outrun their
        queues, and the sets around that gap are no longer covered by
        the optimality proof. ``dropped_unpolled`` counts bundles
        discarded because the caller stopped polling.
        """
        if self._bundler is None:
            return {}
        stats = self._bundler.stats()
        with self._ready_lock:
            stats["ready"] = len(self._ready)
            stats["dropped_unpolled"] = self._dropped_bundles
        return stats

    def _rate_violation(self, state: _StreamState) -> SyncViolation | None:
        """Cheap arrival-rate check, run under the stream lock.

        O(1): only the ends of the bounded receive deque are read. The
        threshold is :data:`~embodied_sync.session.quality.MIN_RATE_FRACTION`
        — the same one ``quality()``'s ``problems`` predicate uses, so a
        stream cannot be warned about here and called healthy there.
        """
        expected = state.config.rate_hz
        if expected is None or len(state.receive_times) < _RATE_CHECK_MIN_SAMPLES:
            return None
        span_ns = state.receive_times[-1] - state.receive_times[0]
        if span_ns <= 0:
            return None
        observed = (len(state.receive_times) - 1) / (span_ns / 1e9)
        if observed >= MIN_RATE_FRACTION * expected:
            return None
        return SyncViolation(
            stream=state.name,
            reason=RATE_BELOW_EXPECTED,
            target_time_ns=None,
            skew_ns=None,
            tolerance_ns=None,
            message=(
                f"stream {state.name!r}: observed rate {observed:.1f} Hz < "
                f"{MIN_RATE_FRACTION:g}x expected {expected:.1f} Hz over the "
                f"last {len(state.receive_times)} arrivals"
            ),
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(
        self,
        *,
        reference: str | None = None,
        at_ns: int | None = None,
    ) -> SyncBundle:
        """Return a :class:`SyncBundle` anchored at one target time.

        Exactly one anchor. ``reference="camera"`` anchors on that
        stream's newest sample (which then appears in the bundle with
        skew 0 — it *defines* the target); ``at_ns=t`` anchors on an
        explicit time and every stream, including the primary, is picked
        by its own policy. With neither, the configured ``primary`` is
        the reference; with no primary configured that is a
        :class:`ValueError` naming the fix.

        Every other stream is picked by *its own* configured
        policy/tolerance/deadline from *its own* ring buffer — the whole
        point of per-stream configuration is that a 250 Hz robot and a
        30 Hz camera do not want the same rule.
        """
        if reference is not None and at_ns is not None:
            raise ValueError(
                "pass exactly one anchor: reference= or at_ns=, not both"
            )
        if reference is None and at_ns is None:
            if self._primary is None:
                raise ValueError(
                    "no primary stream configured: call get(reference=...) or "
                    "get(at_ns=...), or construct the session with "
                    "primary='<stream>'"
                )
            reference = self._primary
        if reference is not None:
            self._state(reference)

        anchor_sample: Sample | None = None
        if at_ns is not None:
            target_ns = at_ns
        else:
            assert reference is not None  # narrowed above
            ref_state = self._states[reference]
            with ref_state.lock:
                anchor_sample = ref_state.newest
            if anchor_sample is None:
                target_ns = self._clock()
                self._emit(
                    SyncViolation(
                        stream=reference,
                        reason=NO_SAMPLES,
                        target_time_ns=target_ns,
                        skew_ns=None,
                        tolerance_ns=self._configs[reference].tolerance_ns,
                        message=(
                            f"reference stream {reference!r} has received no "
                            f"samples; anchoring the bundle at clock()="
                            f"{target_ns} and returning an all-missing bundle"
                        ),
                    )
                )
            else:
                target_ns = anchor_sample.acquisition_time_ns

        items: dict[str, BundleItem] = {}
        matched_times: list[int] = []
        pending: list[SyncViolation] = []
        for name, state in self._states.items():
            if anchor_sample is not None and name == reference:
                item = BundleItem(
                    payload=anchor_sample.payload,
                    sample=anchor_sample,
                    skew_ns=0,
                    within_tolerance=True,
                    missing=False,
                    confidence=1.0,
                    method=REFERENCE_METHOD,
                )
                matched_times.append(anchor_sample.acquisition_time_ns)
            else:
                item, violation, matched_ns = self._pick(state, target_ns)
                if violation is not None:
                    pending.append(violation)
                if matched_ns is not None:
                    matched_times.append(matched_ns)
            items[name] = item
            with state.lock:
                state.matches.append(
                    MatchRecord(
                        target_time_ns=target_ns,
                        skew_ns=item.skew_ns,
                        missing=item.missing,
                        within_tolerance=item.within_tolerance,
                    )
                )

        for violation in pending:
            self._emit(violation)

        span_ns = (
            max(matched_times) - min(matched_times) if len(matched_times) >= 2 else None
        )
        ok = all(
            item.within_tolerance and not item.missing for item in items.values()
        )
        return SyncBundle(
            target_time_ns=target_ns, items=items, ok=ok, span_ns=span_ns
        )

    def _pick(
        self, state: _StreamState, target_ns: int
    ) -> tuple[BundleItem, SyncViolation | None, int | None]:
        """Run ``state``'s configured policy at ``target_ns``.

        Returns ``(item, violation_or_None, matched_acquisition_ns_or_None)``.
        The violation is returned rather than emitted so the caller can
        dispatch it outside the stream lock.
        """
        config = state.config
        with state.lock:
            buffered = len(state.buffer)
            samples: Sample | list[Sample] | None
            if config.policy == POLICY_WINDOW:
                assert config.window_ns is not None  # enforced by StreamConfig
                # A symmetric window is inherently non-causal on its future
                # half: a sample acquired at target + window/2 cannot have
                # been *received* by the target. Grant exactly the slack the
                # window itself implies, so `window` means what it says;
                # `deadline_ms` then adds transport-latency slack on top.
                window, metadata = state.buffer.get_window(
                    target_ns,
                    window_ns=config.window_ns,
                    deadline_ns=config.deadline_ns + config.window_ns // 2,
                )
                samples = window if window else None
                payload: Any = [s.payload for s in window] if window else None
            elif config.policy in (POLICY_NEAREST, POLICY_APPROXIMATE):
                # `approximate` shares the nearest picker here on purpose:
                # ApproximateTime's per-stream choice for a given pivot *is*
                # the member nearest that pivot. The set-level surface
                # (poll_bundles) chooses the pivot; get() lets the caller
                # choose it. Same criterion, different chooser.
                pick, metadata = state.buffer.get_nearest_neighbor(
                    target_ns, deadline_ns=config.deadline_ns
                )
                samples = pick
                payload = pick.payload if pick is not None else None
            elif config.policy == POLICY_LATEST_BEFORE:
                pick, metadata = state.buffer.get_aligned_observation(
                    target_ns, deadline_ns=config.deadline_ns
                )
                samples = pick
                payload = pick.payload if pick is not None else None
            else:  # pragma: no cover - StreamConfig validates the vocabulary
                raise ValueError(f"unhandled policy {config.policy!r}")
            mapping = state.mapping

        missing = samples is None
        confidence = self._scaled_confidence(config, mapping, metadata.confidence)
        item = BundleItem(
            payload=payload,
            sample=samples,
            skew_ns=metadata.skew_ns,
            within_tolerance=not missing,
            missing=missing,
            confidence=confidence,
            method=metadata.method,
        )
        violation = (
            self._miss_violation(state, target_ns, metadata, buffered)
            if missing
            else None
        )
        matched_ns = metadata.source_time_ns if not missing else None
        return item, violation, matched_ns

    def _scaled_confidence(
        self,
        config: StreamConfig,
        mapping: LatencyEstimate | None,
        confidence: float,
    ) -> float:
        """Lower ``confidence`` when the stream lives in a foreign domain."""
        if config.clock_domain == SESSION_CLOCK_DOMAIN:
            return confidence
        if mapping is not None:
            return confidence * cross_domain_confidence_factor(
                mapping, config.tolerance_ns
            )
        # Unmapped: matched on receive time, mapping error unknown. Treat it
        # as "comparable to the tolerance budget" — a 0.5 multiplier.
        return confidence * 0.5

    def _miss_violation(
        self,
        state: _StreamState,
        target_ns: int,
        metadata: AlignedSampleMetadata,
        buffered: int,
    ) -> SyncViolation:
        """Classify *why* a pick came back empty into one reason constant."""
        config = state.config
        name = state.name
        if metadata.source_time_ns is not None:
            reason = OUTSIDE_TOLERANCE
            skew = metadata.skew_ns
            skew_ms = skew / 1e6 if skew is not None else float("nan")
            message = (
                f"stream {name!r}: best candidate is {skew_ms:.3f} ms from "
                f"target {target_ns} (tolerance "
                f"{config.tolerance_ns / 1e6:.3f} ms, policy {config.policy!r})"
            )
        elif buffered:
            reason = NO_ELIGIBLE_BEFORE_DEADLINE
            message = (
                f"stream {name!r}: {buffered} sample(s) buffered but none had "
                f"been received by target {target_ns} + deadline "
                f"{config.deadline_ns / 1e6:.3f} ms"
            )
        elif config.policy == POLICY_WINDOW:
            reason = NO_SAMPLES
            message = (
                f"stream {name!r}: no samples in the "
                f"{(config.window_ns or 0) / 1e6:.3f} ms window around target "
                f"{target_ns}"
            )
        else:
            reason = NO_SAMPLES
            message = f"stream {name!r}: no samples buffered at target {target_ns}"
        return SyncViolation(
            stream=name,
            reason=reason,
            target_time_ns=target_ns,
            skew_ns=metadata.skew_ns,
            tolerance_ns=config.tolerance_ns,
            message=message,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _emit(self, violation: SyncViolation) -> None:
        """Dispatch one violation per ``on_violation``. Never called under a lock.

        The bookkeeping (counters, rate limiter) happens under a small
        session-wide lock because ``d[k] = d.get(k, 0) + 1`` is not
        atomic and producer threads emit concurrently. The *dispatch*
        happens after that lock is released: a user callable is arbitrary
        code that may call back into the session, and holding a lock
        across it would deadlock.
        """
        if violation.reason not in VIOLATION_REASONS:  # pragma: no cover - guard
            raise ValueError(
                f"unknown violation reason {violation.reason!r}; known reasons: "
                f"{sorted(VIOLATION_REASONS)}"
            )
        key = (violation.stream, violation.reason)
        handler = self._on_violation
        with self._violation_lock:
            self._violation_counts[key] = self._violation_counts.get(key, 0) + 1
            allowed = handler == "warn" and self._limiter.allow(key)
        if callable(handler):
            handler(violation)
            return
        if handler == "ignore":
            return
        if handler == "raise":
            raise SyncToleranceError(violation)
        if allowed:
            logger.warning("%s: %s", violation.reason, violation.message)

    # ------------------------------------------------------------------
    # Live quality
    # ------------------------------------------------------------------

    def quality(
        self, window_s: float = DEFAULT_QUALITY_WINDOW_S
    ) -> dict[str, LiveStreamQuality]:
        """Per-stream quality over the trailing ``window_s`` seconds.

        Reads only the bounded per-stream deques, so the cost is
        O(window) and independent of session length. Note the honesty
        boundary restated in :mod:`embodied_sync.session.quality`: these
        numbers describe *timestamp consistency*, not physical
        simultaneity — two cameras agreeing on timestamps can still be
        exposing 30 ms apart. Establishing the physical relationship is
        :mod:`embodied_sync.calibrate`'s job.
        """
        if window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        cutoff_ns = self._clock() - round(window_s * 1e9)
        result: dict[str, LiveStreamQuality] = {}
        for name, state in self._states.items():
            with state.lock:
                receives = [t for t in state.receive_times if t >= cutoff_ns]
                matches = [
                    m for m in state.matches if m.target_time_ns >= cutoff_ns
                ]
            result[name] = compute_stream_quality(
                stream=name,
                window_s=window_s,
                receive_times_ns=receives,
                matches=matches,
                expected_rate_hz=state.config.rate_hz,
                tolerance_ns=state.config.tolerance_ns,
            )
        return result

    # ------------------------------------------------------------------
    # Persistence and lifecycle
    # ------------------------------------------------------------------

    def _build_manifest(self) -> dict[str, Any]:
        """Run-format-v0 manifest plus a ``session`` block of live metadata.

        ``streams`` carries exactly what ``load_run`` needs (modality,
        clock domains, sample count) so the directory stays a valid run;
        the extra per-stream keys are additive and ignored by the reader.
        Streams with ``persist="off"`` are omitted entirely — a manifest
        entry with no JSONL file would make ``load_run`` raise.
        """
        counts = self._recorder.counts() if self._recorder is not None else {}
        streams: dict[str, Any] = {}
        for name, config in self._configs.items():
            if config.persist == "off":
                continue
            streams[name] = {
                "modality": config.modality,
                "clock_domains": sorted({config.clock_domain}),
                "sample_count": counts.get(name, 0),
                "rate_hz": config.rate_hz,
                "tolerance_ns": config.tolerance_ns,
                "policy": config.policy,
                "deadline_ns": config.deadline_ns,
                "window_ns": config.window_ns,
                "persist": config.persist,
            }
        mappings = {
            domain: latency_estimate_to_dict(mapping)
            for domain, mapping in self._mappings.items()
        }
        session_block: dict[str, Any] = {
            "recorder": "embodied_sync.session.SyncSession",
            "primary": self._primary,
            "clock_domain": SESSION_CLOCK_DOMAIN,
            "start_time_ns": self._start_time_ns,
            "clock_mappings": mappings,
            # Which generation each declared domain is in, and how it got
            # there. A run whose camera reconnected twice mid-recording is a
            # different artefact from one that did not, and the manifest is
            # the only place that fact survives the session.
            "clock_epochs": self.clock_epochs(),
            "clock_epoch_history": {
                domain: [
                    {
                        "epoch": record.epoch,
                        "started_at_ns": record.started_at_ns,
                        "reason": record.reason,
                    }
                    for record in self._epochs.history(domain)
                ]
                for domain in self._epochs.domains()
            },
            "omitted_streams": [
                n for n, c in self._configs.items() if c.persist == "off"
            ],
        }
        if self._bundler is not None:
            session_block["approximate"] = {
                "streams": list(self._approximate_streams),
                **self.approximate_stats(),
            }
        return {"streams": streams, "session": session_block}

    def flush(self) -> None:
        """Flush recorded streams to disk and rewrite the manifest."""
        if self._recorder is not None:
            self._recorder.flush(self._build_manifest())

    def close(self) -> None:
        """Flush, write the final manifest and quality snapshot, summarise.

        Idempotent. The suppressed-warning summary is the other half of
        the rate limiter's bargain: warnings are throttled during the
        run, and the totals are stated once at the end, so a
        disconnected camera shows up as "1 warning + 412 suppressed"
        rather than 413 log lines or a single misleading one.
        """
        if self._closed:
            return
        self._closed = True
        if self._bundler is not None:
            # Best-effort tail: the samples that would have *proved* these
            # sets optimal are never arriving now, so they come out marked
            # `provable=False` rather than being silently dropped or
            # silently promoted.
            self._absorb_sets(self._bundler.flush())
        snapshot = self.quality(window_s=self._session_window_s())
        if self._recorder is not None:
            self._recorder.flush(self._build_manifest())
            self._recorder.write_sidecar(
                SESSION_QUALITY_NAME,
                {
                    "format_version": 0,
                    "type": "session_quality",
                    "window_s": self._session_window_s(),
                    "streams": {
                        name: q.to_dict() for name, q in snapshot.items()
                    },
                },
            )
            self._recorder.close()
        self._log_summary(snapshot)

    def _session_window_s(self) -> float:
        """Whole-session window for the closing snapshot (bounded by the deques)."""
        elapsed_ns = max(1, self._clock() - self._start_time_ns)
        return elapsed_ns / 1e9

    def _log_summary(self, snapshot: Mapping[str, LiveStreamQuality]) -> None:
        suppressed = self._limiter.suppressed
        if suppressed:
            total = sum(suppressed.values())
            detail = ", ".join(
                f"{stream}/{reason}={count}"
                for (stream, reason), count in sorted(suppressed.items())
            )
            logger.warning(
                "session close: %d suppressed warning(s) (%s)", total, detail
            )
        for name, quality in snapshot.items():
            if quality.problems:
                logger.warning(
                    "session close: stream %r problems: %s",
                    name,
                    "; ".join(quality.problems),
                )
        stats = self.approximate_stats()
        if stats:
            logger.info(
                "session close: approximate sets emitted=%d superseded=%d "
                "overflowed=%d out_of_order=%d ready=%d dropped_unpolled=%d",
                stats["emitted"],
                stats["superseded"],
                stats["overflowed"],
                stats["out_of_order"],
                stats["ready"],
                stats["dropped_unpolled"],
            )
            if stats["overflowed"]:
                logger.warning(
                    "session close: %d sample(s) were evicted from an "
                    "approximate queue; sets emitted around those gaps are not "
                    "covered by the span-minimality proof (raise "
                    "buffer_capacity on the affected stream)",
                    stats["overflowed"],
                )

    def __enter__(self) -> SyncSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def init(
    *,
    streams: Mapping[str, StreamConfig],
    run_dir: str | Path | None = None,
    primary: str | None = None,
    on_violation: str | ViolationHandler = "warn",
    clock: Callable[[], int] = time.monotonic_ns,
    violation_interval_s: float = DEFAULT_VIOLATION_INTERVAL_S,
    serialize: PayloadSerializer | None = None,
) -> SyncSession:
    """Construct a :class:`SyncSession`. No hidden global state.

    ``embsync.init(...)`` is the one-line spelling the design leads with;
    ``SyncSession(...)`` is the explicit one. They are the same call —
    this factory exists so the entry point reads like an entry point,
    not so it can cache a singleton somewhere.
    """
    return SyncSession(
        streams=streams,
        run_dir=run_dir,
        primary=primary,
        on_violation=on_violation,
        clock=clock,
        violation_interval_s=violation_interval_s,
        serialize=serialize,
    )
