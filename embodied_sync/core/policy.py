"""``AlignmentPolicy`` — per-stream alignment method + tolerance (Milestone 2, D-0029).

The offline engine's ``align_run`` grew from "one method for every
stream" (session 7) to "a method plus tolerance chosen per stream"
(this milestone). :class:`AlignmentPolicy` bundles the choice into a
canonical value so callers can express the mixed policy without a
positional-argument explosion:

- ``method`` is one of ``"nearest_neighbor"``, ``"zoh"``,
  ``"linear_interp"`` — same set the top-level engine accepts — plus
  ``"window"``, which only the live session
  (:mod:`embodied_sync.session`) implements.
- The live-session vocabulary spells two of those methods differently
  (``"latest_before"`` for ZoH, ``"nearest"`` for deadline-aware
  nearest-neighbor). :data:`METHOD_ALIASES` maps those spellings onto
  the engine constants and :class:`AlignmentPolicy` normalises them at
  construction, so ``AlignmentPolicy(method="latest_before").method``
  is ``"zoh"`` and the two vocabularies stay one vocabulary (D-0037).
- ``tolerance_ns`` overrides the default (half the median
  inter-sample interval) when the caller wants an explicit budget;
  ``None`` keeps the engine's derived default.
- ``deadline_ns`` is honoured by the online composite
  (:class:`~embodied_sync.align.online.MultiStreamAligner`) for
  deadline-aware NN and deadline-shifted ZoH; offline callers ignore
  it.
- ``clock_domain`` names the domain this stream lives in so cross-
  domain alignment can look up the right
  :class:`~embodied_sync.time.LatencyEstimate` before scoring.

The policy is a value type — pure data, safe to pass across process
boundaries via JSON — so a future manifest field can echo the actual
policy used at alignment time.
"""

from __future__ import annotations

from dataclasses import dataclass

from embodied_sync.time.clock_domain import ClockDomain

__all__ = ["METHOD_ALIASES", "AlignmentPolicy"]

#: Live-session spellings accepted for :attr:`AlignmentPolicy.method`,
#: mapped onto the canonical engine constants. ``"window"`` has no
#: engine equivalent and is therefore not an alias — it is a method in
#: its own right, accepted here and implemented only by
#: :mod:`embodied_sync.session`; passing it to the offline
#: :func:`~embodied_sync.align.align_run` raises with the engine's
#: known-method list, which is the intended loud failure.
METHOD_ALIASES: dict[str, str] = {
    "latest_before": "zoh",
    "nearest": "nearest_neighbor",
}

_KNOWN_METHODS = ("nearest_neighbor", "zoh", "linear_interp", "window")


@dataclass(frozen=True, slots=True)
class AlignmentPolicy:
    """Per-stream alignment choice."""

    method: str = "nearest_neighbor"
    tolerance_ns: int | None = None
    deadline_ns: int = 0
    clock_domain: ClockDomain | None = None

    def __post_init__(self) -> None:
        canonical = METHOD_ALIASES.get(self.method, self.method)
        if canonical not in _KNOWN_METHODS:
            raise ValueError(
                f"unknown method {self.method!r}; known methods: "
                f"{list(_KNOWN_METHODS)} (aliases: {sorted(METHOD_ALIASES)})"
            )
        if canonical != self.method:
            object.__setattr__(self, "method", canonical)
        if self.tolerance_ns is not None:
            if not isinstance(self.tolerance_ns, int) or isinstance(self.tolerance_ns, bool):
                raise TypeError(
                    f"tolerance_ns must be int or None, got {type(self.tolerance_ns).__name__}"
                )
            if self.tolerance_ns < 0:
                raise ValueError(f"tolerance_ns must be >= 0, got {self.tolerance_ns}")
        if not isinstance(self.deadline_ns, int) or isinstance(self.deadline_ns, bool):
            raise TypeError(
                f"deadline_ns must be int, got {type(self.deadline_ns).__name__}"
            )
        if self.deadline_ns < 0:
            raise ValueError(f"deadline_ns must be >= 0, got {self.deadline_ns}")
