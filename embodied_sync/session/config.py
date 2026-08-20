"""``StreamConfig`` — per-stream declaration for a live :class:`SyncSession`.

The config boundary is the one place in the library where timings are
*milliseconds as floats*: a researcher writes ``tolerance_ms=20.0``, not
``tolerance_ns=20_000_000``. Everything past construction is integer
nanoseconds (D-0002). This is exactly the convention
``corrupt/profile.py`` uses for YAML profiles — ms in, ns at parse time,
never a float on a stored timestamp — applied to a Python-level API
instead of a file format.

Policy vocabulary
-----------------
The session speaks the vocabulary a live user thinks in; the engine
speaks the vocabulary the offline aligner has always used. They are one
vocabulary (D-0037), related by
:data:`~embodied_sync.core.policy.METHOD_ALIASES`:

================= ==================== ======================================
session name      engine method        semantics
================= ==================== ======================================
``latest_before`` ``zoh``              strictly causal hold — safe for control
``nearest``       ``nearest_neighbor`` deadline-aware, may pick a "future" sample
``window``        (live only)          every sample within ``±window_ms/2``
``approximate``   (live only)          member of a span-minimising set (A1)
================= ==================== ======================================

Both spellings are accepted on input and normalised to the session
spelling, so ``StreamConfig(policy="zoh").policy == "latest_before"``.
``window`` and ``approximate`` are live-only: the offline engine has no
picker for either, so neither name is an ``AlignmentPolicy`` method.

``approximate`` is a set-level policy, not a picker
---------------------------------------------------
The other three names answer "given a target time, which sample?".
``approximate`` answers a different question — "which samples belong
together?" — and it answers it for the whole set at once. Marking a
stream ``approximate`` therefore enrols it in the session's
ApproximateTime set (:mod:`embodied_sync.session.approximate`), whose
bundles arrive through
:meth:`~embodied_sync.session.SyncSession.poll_bundles` as the data
allows, rather than when a caller asks. Fewer than two enrolled streams
is a configuration error: a set of one has nothing to approximate.

That leaves ``get()``, which still has to say *something* about an
enrolled stream. It picks by deadline-aware **nearest**, which is not a
compromise but the same rule: ApproximateTime's per-stream choice for a
given pivot is the member nearest that pivot. The two surfaces agree on
the criterion and differ only in who chooses the target time.

Where a tolerance comes from
----------------------------
``latest_before`` and ``nearest`` need a tolerance to decide whether a
pick is good enough. It comes from ``tolerance_ms`` if given, else from
half the nominal period implied by ``rate_hz`` (the same "half the
median inter-sample interval" default the offline engine derives from
the data, computed from the declared rate because a live session cannot
see the future). If neither is available the config is an error, named
as such — silently defaulting a tolerance is how "98% synced" gets
reported against a number nobody chose. ``window`` needs no tolerance:
``window_ms`` is the tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from embodied_sync.core.sample import Modality

__all__ = [
    "KNOWN_PERSIST_MODES",
    "KNOWN_POLICIES",
    "POLICY_APPROXIMATE",
    "POLICY_LATEST_BEFORE",
    "POLICY_NEAREST",
    "POLICY_WINDOW",
    "StreamConfig",
]

POLICY_LATEST_BEFORE = "latest_before"
POLICY_NEAREST = "nearest"
POLICY_WINDOW = "window"
POLICY_APPROXIMATE = "approximate"

#: Session-vocabulary policy names, in the order they appear in the docs.
KNOWN_POLICIES: tuple[str, ...] = (
    POLICY_LATEST_BEFORE,
    POLICY_NEAREST,
    POLICY_WINDOW,
    POLICY_APPROXIMATE,
)

#: Engine spellings accepted on input, mapped to the session spelling.
_POLICY_ALIASES: dict[str, str] = {
    "zoh": POLICY_LATEST_BEFORE,
    "nearest_neighbor": POLICY_NEAREST,
}

#: ``persist`` modes. ``metadata`` writes timing/sequence/flags with a null
#: payload (size-safe, always JSON-serialisable); ``full`` also writes the
#: payload; ``off`` writes nothing for the stream.
KNOWN_PERSIST_MODES: tuple[str, ...] = ("metadata", "full", "off")

#: Floor for a derived ring-buffer capacity, and the number of seconds of
#: samples a derived capacity must hold.
_MIN_CAPACITY = 64
_CAPACITY_SECONDS = 2.0

_NS_PER_MS = 1_000_000


def _ms_to_ns(value_ms: float) -> int:
    """Config-boundary conversion: ms float → integer ns (D-0002)."""
    return round(value_ms * _NS_PER_MS)


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """Declaration of one live stream.

    Fields are the user-facing ms/float knobs; the ``*_ns`` attributes
    beside them are the integer-nanosecond values the session actually
    uses, derived once at construction. ``capacity`` is the ring-buffer
    size, derived from ``rate_hz`` (at least ``_CAPACITY_SECONDS`` worth
    of samples, and at least the window width for ``window`` streams)
    unless given explicitly.
    """

    rate_hz: float | None = None
    tolerance_ms: float | None = None
    policy: str = POLICY_LATEST_BEFORE
    deadline_ms: float = 0.0
    window_ms: float | None = None
    modality: str = Modality.OTHER.value
    clock_domain: str = "host_mono"
    persist: str = "metadata"
    buffer_capacity: int | None = None

    # Derived, integer-ns. Never set by the caller.
    tolerance_ns: int = field(init=False, default=0)
    deadline_ns: int = field(init=False, default=0)
    window_ns: int | None = field(init=False, default=None)
    capacity: int = field(init=False, default=_MIN_CAPACITY)

    def __post_init__(self) -> None:
        policy = _POLICY_ALIASES.get(self.policy, self.policy)
        if policy not in KNOWN_POLICIES:
            raise ValueError(
                f"unknown policy {self.policy!r}; known policies: "
                f"{list(KNOWN_POLICIES)} (aliases: {sorted(_POLICY_ALIASES)})"
            )
        object.__setattr__(self, "policy", policy)

        if self.persist not in KNOWN_PERSIST_MODES:
            raise ValueError(
                f"unknown persist mode {self.persist!r}; "
                f"known modes: {list(KNOWN_PERSIST_MODES)}"
            )
        try:
            Modality(self.modality)
        except ValueError:
            known = [m.value for m in Modality]
            raise ValueError(
                f"unknown modality {self.modality!r}; known modalities: {known}"
            ) from None

        if self.rate_hz is not None and self.rate_hz <= 0:
            raise ValueError(f"rate_hz must be > 0 or None, got {self.rate_hz}")
        if self.tolerance_ms is not None and self.tolerance_ms < 0:
            raise ValueError(f"tolerance_ms must be >= 0 or None, got {self.tolerance_ms}")
        if self.deadline_ms < 0:
            raise ValueError(f"deadline_ms must be >= 0, got {self.deadline_ms}")
        object.__setattr__(self, "deadline_ns", _ms_to_ns(self.deadline_ms))

        if policy == POLICY_WINDOW:
            if self.window_ms is None:
                raise ValueError("policy='window' requires window_ms")
            if self.window_ms <= 0:
                raise ValueError(f"window_ms must be > 0, got {self.window_ms}")
            object.__setattr__(self, "window_ns", _ms_to_ns(self.window_ms))
            # A window stream needs no tolerance to *pick* — membership of the
            # window is the whole decision. It still needs one to *report*
            # against (quality()'s skew-vs-tolerance predicate), so an explicit
            # tolerance_ms is honoured and the window width is the fallback.
            object.__setattr__(
                self,
                "tolerance_ns",
                _ms_to_ns(self.tolerance_ms)
                if self.tolerance_ms is not None
                else self.window_ns,
            )
        else:
            if self.window_ms is not None:
                raise ValueError(
                    f"window_ms is only meaningful for policy='window', "
                    f"got policy={policy!r}"
                )
            if self.tolerance_ms is not None:
                object.__setattr__(self, "tolerance_ns", _ms_to_ns(self.tolerance_ms))
            elif self.rate_hz is not None:
                # Half the nominal period — the live analogue of the offline
                # engine's "half the median inter-sample interval".
                object.__setattr__(self, "tolerance_ns", round(0.5e9 / self.rate_hz))
            else:
                raise ValueError(
                    f"policy={policy!r} needs a tolerance: set tolerance_ms, or "
                    f"set rate_hz so it can be derived as half the nominal period"
                )

        if self.buffer_capacity is not None:
            if self.buffer_capacity <= 0:
                raise ValueError(
                    f"buffer_capacity must be > 0 or None, got {self.buffer_capacity}"
                )
            object.__setattr__(self, "capacity", self.buffer_capacity)
        else:
            object.__setattr__(self, "capacity", self._derived_capacity())

    def _derived_capacity(self) -> int:
        """At least ``_MIN_CAPACITY``, at least ``_CAPACITY_SECONDS`` of samples.

        A ``window`` stream also needs its whole window to fit, so the
        span is widened to the window width when that is longer.
        """
        if self.rate_hz is None:
            return _MIN_CAPACITY
        span_s = _CAPACITY_SECONDS
        if self.window_ns is not None:
            span_s = max(span_s, self.window_ns / 1e9)
        return max(_MIN_CAPACITY, math.ceil(self.rate_hz * span_s))

    @property
    def modality_value(self) -> Modality:
        """``modality`` as the canonical :class:`Modality` enum member."""
        return Modality(self.modality)

    @property
    def expected_period_ns(self) -> int | None:
        """Nominal inter-sample interval in ns, or ``None`` for irregular streams."""
        if self.rate_hz is None:
            return None
        return round(1e9 / self.rate_hz)
