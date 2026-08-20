"""Typed clock-domain values (Milestone 2, D-0029).

Free-string ``Sample.source_clock_domain`` stays on-disk data (run
format v0 unchanged). This module lifts the string into a typed
:class:`ClockDomain` value that adapters and the alignment engine can
reason about — enum kind, optional nominal resolution — and adds the
:class:`LatencyEstimate` type that describes the offset+drift mapping
between two domains.

Migration path mirrors what
``docs/developer/clock_domain_mapping.md`` documented as the plan:

- :data:`KNOWN_DOMAINS` maps the built-in adapter strings to typed
  values (``"host_mono"`` → monotonic, ``"lsl_local_clock"`` →
  network, ``"mcap_publish_time"`` → wall, ``"ros2_steady"`` →
  monotonic, ``"unknown"`` → unknown).
- :func:`resolve_clock_domain` looks up a string and emits a one-shot
  :class:`UserWarning` on a miss so the first appearance of a new
  adapter's clock name surfaces the exact string to add here.
- :func:`translate_ns` applies a :class:`LatencyEstimate` to a
  source-domain integer nanosecond timestamp, producing an integer
  target-domain nanosecond timestamp (never floats, D-0002).

Cross-domain alignment consumes these — see
:func:`embodied_sync.time.alignment.cross_domain_confidence_factor`
for how :class:`LatencyEstimate.variance_ns` lowers confidence per
D-0003's "never silently mix" rule.

Clock epochs (D-0041, Lane A / A3)
----------------------------------
A clock domain is not one timeline forever. A camera reconnects and its
hardware counter restarts at zero; an NTP step moves a wall clock by
half a second; a device power-cycles mid-session. Every one of those
events ends the timeline the current mapping was fitted against, and a
drift fit that keeps accumulating across the discontinuity is not merely
stale — it is *poisoned*, and it stays poisoned for the rest of the
recording while continuing to look like a measurement.

LSL exposes exactly one bit of this (``was_clock_reset()``, "did the
remote clock reset since you last asked"). The generalisation here is a
**monotonically increasing epoch (generation) counter per domain**:

- :class:`LatencyEstimate` carries an ``epoch``. A mapping is only valid
  for the generation it was fitted in; :meth:`LatencyEstimate.with_epoch`
  stamps one without re-fitting (translation is epoch-independent, so
  re-stamping is safe and cheap).
- :class:`ClockEpochRegistry` owns the counters. It only ever counts
  *up*, so an epoch is a total order and "which generation is this" is
  never ambiguous. It also keeps a short history, because the reason a
  device reset is usually the interesting part of a post-mortem.
- :func:`require_same_epoch` is the guard: any routine that fits or
  compares times from several observations calls it and refuses a mixed
  batch with :class:`ClockEpochError`. Refusing loudly is the whole
  point — silently fitting across a reset is the failure mode this
  machinery exists to make impossible.
- :func:`translate_ns` takes an optional ``epoch=`` and raises when the
  caller's generation is not the mapping's. It is opt-in so existing
  epoch-unaware callers keep working; epoch ``0`` is "the first (or
  only) generation", which is what every mapping fitted before this
  existed implicitly is.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, replace
from enum import Enum
from typing import Sequence

__all__ = [
    "INITIAL_EPOCH",
    "KNOWN_DOMAINS",
    "ClockDomain",
    "ClockEpoch",
    "ClockEpochError",
    "ClockEpochRegistry",
    "ClockKind",
    "LatencyEstimate",
    "latency_estimate_to_dict",
    "require_same_epoch",
    "resolve_clock_domain",
    "translate_ns",
]

_PPB_DENOMINATOR = 1_000_000_000

#: The generation every clock domain starts in, and the value a mapping
#: fitted by epoch-unaware code carries. Never negative: an epoch counts
#: resets, and "minus one reset" is not a thing.
INITIAL_EPOCH = 0


class ClockKind(str, Enum):
    """Coarse taxonomy for a clock's monotonicity and drift behaviour."""

    #: Sensor-side hardware clock — e.g. a camera or IMU's internal timer.
    HARDWARE = "hardware"
    #: Host monotonic clock (``CLOCK_MONOTONIC``, ``time.monotonic_ns``).
    MONOTONIC = "monotonic"
    #: Wall-clock time — jumps around at NTP corrections. Avoid.
    WALL = "wall"
    #: Network-time-provider clock (LSL's ``local_clock``, PTP, chrony).
    NETWORK = "network"
    #: Anything the caller can't classify; treated pessimistically.
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ClockDomain:
    """One named clock timebase.

    ``name`` is the string that appears on-disk as
    ``Sample.source_clock_domain``. ``kind`` tells the alignment engine
    how pessimistic to be about drift. ``resolution_ns`` is the nominal
    quantum of the clock (``None`` if unknown).
    """

    name: str
    kind: ClockKind
    resolution_ns: int | None = None


@dataclass(frozen=True, slots=True)
class LatencyEstimate:
    """Anchored offset+drift mapping between two clock domains.

    ``offset_ns`` is exact at ``anchor_time_ns`` (expressed in
    ``source``'s domain). ``drift_ppb`` is parts-per-billion, positive
    when the *source* ticks slower than the target — same convention
    D-0014 uses for the ``clock_drift`` corruption.

    ``variance_ns`` is the mapping's measurement uncertainty; the
    alignment engine lowers confidence in proportion to it via
    :func:`~embodied_sync.time.alignment.cross_domain_confidence_factor`.
    A perfectly known mapping (e.g. two streams sharing a hardware
    clock) sets ``variance_ns = 0`` and leaves confidence untouched.

    ``epoch`` is the ``source`` domain's generation the mapping was
    fitted in (see the module docstring). It defaults to
    :data:`INITIAL_EPOCH`, so epoch-unaware code keeps working and
    reads as "the first generation". A consumer that tracks epochs — a
    live :class:`~embodied_sync.session.SyncSession` does — must refuse
    a mapping whose epoch is not the domain's current one, because
    applying a pre-reconnect mapping to post-reconnect timestamps is
    precisely the silent poisoning this field exists to prevent.
    """

    source: ClockDomain
    target: ClockDomain
    offset_ns: int
    drift_ppb: int = 0
    anchor_time_ns: int = 0
    variance_ns: int = 0
    epoch: int = INITIAL_EPOCH

    def __post_init__(self) -> None:
        for name in (
            "offset_ns",
            "drift_ppb",
            "anchor_time_ns",
            "variance_ns",
            "epoch",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be int, got {type(value).__name__}")
        if self.variance_ns < 0:
            raise ValueError(f"variance_ns must be >= 0, got {self.variance_ns}")
        if self.epoch < INITIAL_EPOCH:
            raise ValueError(f"epoch must be >= {INITIAL_EPOCH}, got {self.epoch}")

    def with_epoch(self, epoch: int) -> LatencyEstimate:
        """Return a copy stamped with ``epoch``, leaving the fit untouched.

        Translation does not depend on the epoch, so re-stamping a
        mapping is arithmetically a no-op — which is exactly why it is
        safe to offer. It is the cheap path for a calibrator that does
        not know (or care) which generation it ran in: fit once, then
        stamp the result with the generation the session reports.

        It is *not* a way to launder a stale mapping into the current
        generation. Re-stamping a fit taken before a reconnect claims a
        measurement that was never made; refit instead.
        """
        return replace(self, epoch=epoch)


#: Built-in mapping from free-string ``source_clock_domain`` values to
#: typed :class:`ClockDomain`s. Adapters are expected to register their
#: domain string here (or accept a warning + ``UNKNOWN``).
KNOWN_DOMAINS: dict[str, ClockDomain] = {
    "host_mono": ClockDomain("host_mono", ClockKind.MONOTONIC, resolution_ns=1),
    "host_wall": ClockDomain("host_wall", ClockKind.WALL, resolution_ns=1),
    "lsl": ClockDomain("lsl", ClockKind.NETWORK, resolution_ns=1_000),
    "lsl_local_clock": ClockDomain(
        "lsl_local_clock", ClockKind.NETWORK, resolution_ns=1_000
    ),
    "ros2_steady": ClockDomain("ros2_steady", ClockKind.MONOTONIC, resolution_ns=1),
    "mcap_publish_time": ClockDomain(
        "mcap_publish_time", ClockKind.WALL, resolution_ns=1
    ),
    "mcap_log_time": ClockDomain("mcap_log_time", ClockKind.WALL, resolution_ns=1),
    "unknown": ClockDomain("unknown", ClockKind.UNKNOWN),
}


_WARNED_UNKNOWN: set[str] = set()


def resolve_clock_domain(name: str) -> ClockDomain:
    """Look ``name`` up in :data:`KNOWN_DOMAINS`, warn once on a miss.

    A miss returns a synthetic ``ClockDomain(name, ClockKind.UNKNOWN)``
    so downstream code has *something* to work with. The one-shot
    :class:`UserWarning` names ``name`` verbatim so a first-time
    integration sees exactly which entry to add.
    """
    hit = KNOWN_DOMAINS.get(name)
    if hit is not None:
        return hit
    if name not in _WARNED_UNKNOWN:
        _WARNED_UNKNOWN.add(name)
        warnings.warn(
            f"unknown clock domain {name!r}; treating as ClockKind.UNKNOWN. "
            f"Add an entry to embodied_sync.time.KNOWN_DOMAINS to silence.",
            stacklevel=2,
        )
    return ClockDomain(name, ClockKind.UNKNOWN)


class ClockEpochError(ValueError):
    """Raised when timestamps or mappings from different generations meet.

    Carries the offending ``epochs`` so a handler can report them
    without re-parsing the message.
    """

    def __init__(self, message: str, epochs: tuple[int, ...] = ()) -> None:
        super().__init__(message)
        self.epochs = epochs


@dataclass(frozen=True, slots=True)
class ClockEpoch:
    """One generation of a clock domain's timeline.

    ``epoch`` counts from :data:`INITIAL_EPOCH`. ``reason`` is free text
    naming what ended the *previous* generation ("device reconnect",
    "NTP step", "power cycle") — the first epoch of a domain has an
    empty reason because nothing preceded it. ``started_at_ns`` is the
    session-domain time the generation opened, or ``None`` when the
    caller had no clock to hand.
    """

    domain: str
    epoch: int
    started_at_ns: int | None = None
    reason: str = ""


class ClockEpochRegistry:
    """Monotonic per-domain generation counters with a short history.

    This is LSL's ``was_clock_reset()`` generalised from one boolean to
    a total order. A boolean can only answer "did something happen since
    I last looked", which forces every consumer to poll and gives a
    stored mapping no way to say which timeline it belongs to. A counter
    answers both: :meth:`current` is a value a mapping can be stamped
    with, and comparing two stamps is the whole validity check.

    Counters only ever increase, and only via :meth:`advance`. There is
    deliberately no way to set an epoch to an arbitrary value: an epoch
    that could go backwards would let a stale mapping be re-validated by
    accident, which is the failure this class exists to make impossible.

    Thread-safety: all methods take an internal lock, because device
    reconnects are detected on producer threads while mappings are read
    on consumer threads.
    """

    __slots__ = ("_current", "_history", "_history_limit", "_lock")

    #: Generations retained per domain. A reconnect loop should not grow
    #: memory without bound, and only the recent past is diagnostic.
    DEFAULT_HISTORY_LIMIT = 32

    def __init__(self, *, history_limit: int = DEFAULT_HISTORY_LIMIT) -> None:
        if history_limit < 1:
            raise ValueError(f"history_limit must be >= 1, got {history_limit}")
        self._lock = threading.Lock()
        self._current: dict[str, int] = {}
        self._history: dict[str, list[ClockEpoch]] = {}
        self._history_limit = history_limit

    def current(self, domain: str) -> int:
        """Current generation of ``domain``.

        A domain nobody has ever reset is in :data:`INITIAL_EPOCH`;
        asking about an unknown domain is not an error, because "never
        reset" is the honest answer for a clock that has simply been
        behaving.
        """
        with self._lock:
            return self._current.get(domain, INITIAL_EPOCH)

    def advance(
        self, domain: str, *, reason: str = "", at_ns: int | None = None
    ) -> ClockEpoch:
        """Open a new generation for ``domain`` and return it.

        Call this the moment a device reconnects, a clock steps, or any
        other event breaks the continuity of ``domain``'s timeline.
        Everything fitted against the previous generation is thereby
        marked invalid rather than left to keep quietly accumulating.
        """
        with self._lock:
            epoch = self._current.get(domain, INITIAL_EPOCH) + 1
            self._current[domain] = epoch
            record = ClockEpoch(
                domain=domain, epoch=epoch, started_at_ns=at_ns, reason=reason
            )
            history = self._history.setdefault(
                domain, [ClockEpoch(domain=domain, epoch=INITIAL_EPOCH)]
            )
            history.append(record)
            if len(history) > self._history_limit:
                del history[: len(history) - self._history_limit]
            return record

    def history(self, domain: str) -> tuple[ClockEpoch, ...]:
        """Retained generations of ``domain``, oldest first."""
        with self._lock:
            recorded = self._history.get(domain)
            if recorded is None:
                return (ClockEpoch(domain=domain, epoch=INITIAL_EPOCH),)
            return tuple(recorded)

    def domains(self) -> tuple[str, ...]:
        """Domains that have ever been advanced, sorted."""
        with self._lock:
            return tuple(sorted(self._current))

    def snapshot(self) -> dict[str, int]:
        """Current generation per advanced domain — manifest-ready."""
        with self._lock:
            return dict(self._current)

    def is_current(self, mapping: LatencyEstimate) -> bool:
        """Whether ``mapping`` belongs to its source domain's live generation."""
        return mapping.epoch == self.current(mapping.source.name)

    def was_reset(self, domain: str, since_epoch: int) -> bool:
        """Whether ``domain`` advanced past ``since_epoch``.

        The explicit-argument spelling of LSL's ``was_clock_reset()``.
        Stateless on purpose: the caller says which generation it last
        saw, so two independent consumers of the same domain cannot
        consume each other's notification.
        """
        return self.current(domain) > since_epoch


def require_same_epoch(
    epochs: Sequence[int], *, context: str = "observations"
) -> int:
    """Return the single epoch shared by ``epochs``, or raise.

    The guard every routine that fits or compares timestamps from
    multiple observations should run first. Mixing generations produces
    a fit that is not merely imprecise but *wrong in a way that looks
    precise*: two clean linear segments either side of a reset fit a
    single confident line through the discontinuity, with small
    residuals and a drift that is pure artefact.

    An empty sequence returns :data:`INITIAL_EPOCH` — there is nothing
    to mix, so there is nothing to refuse.
    """
    if not epochs:
        return INITIAL_EPOCH
    distinct = sorted(set(int(e) for e in epochs))
    if len(distinct) > 1:
        raise ClockEpochError(
            f"cannot combine {context} from different clock epochs {distinct}: "
            f"a clock reset or device reconnect separates them, so a fit "
            f"across the boundary would measure the discontinuity rather "
            f"than the clocks. Refit within one epoch.",
            tuple(distinct),
        )
    return distinct[0]


def latency_estimate_to_dict(mapping: LatencyEstimate) -> dict[str, object]:
    """JSON-ready mapping of a :class:`LatencyEstimate`.

    One serialiser, so a run manifest, a CLI calibration report, and any
    future sidecar all describe a mapping with the same keys. Domains
    are flattened to their names because that is what
    ``Sample.source_clock_domain`` stores on disk (D-0003).
    """
    return {
        "source": mapping.source.name,
        "target": mapping.target.name,
        "offset_ns": mapping.offset_ns,
        "drift_ppb": mapping.drift_ppb,
        "anchor_time_ns": mapping.anchor_time_ns,
        "variance_ns": mapping.variance_ns,
        "epoch": mapping.epoch,
    }


def translate_ns(
    source_ns: int, mapping: LatencyEstimate, *, epoch: int | None = None
) -> int:
    """Translate a source-domain timestamp to the target domain.

    ``target_ns = source_ns + offset + round((source_ns - anchor) * drift / 1e9)``.
    Integer-ns in, integer-ns out (D-0002).

    ``epoch`` is the generation ``source_ns`` was observed in. Pass it
    and a mismatch against ``mapping.epoch`` raises
    :class:`ClockEpochError` instead of returning a plausible-looking
    number computed from a mapping that no longer applies. It is
    optional because most callers work inside one generation and should
    not have to say so; when a caller *does* track epochs, saying so is
    the difference between a loud failure and a silent one.
    """
    if not isinstance(source_ns, int) or isinstance(source_ns, bool):
        raise TypeError(f"source_ns must be int, got {type(source_ns).__name__}")
    if epoch is not None and epoch != mapping.epoch:
        raise ClockEpochError(
            f"timestamp is from epoch {epoch} of clock domain "
            f"{mapping.source.name!r} but the mapping was fitted in epoch "
            f"{mapping.epoch}; the clock reset between them, so this mapping "
            f"does not describe this timestamp. Refit after the reset.",
            (mapping.epoch, epoch),
        )
    drift_shift = round(
        (source_ns - mapping.anchor_time_ns) * mapping.drift_ppb / _PPB_DENOMINATOR
    )
    return source_ns + mapping.offset_ns + int(drift_shift)
