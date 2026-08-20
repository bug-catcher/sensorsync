"""Semantic-event alignment — **experimental**.

Sometimes there is no clap and no QR code, only the recording: a hand
closing on an object, a tool touching a surface, a door shutting. If two
sensors both register that moment, the moment is a calibration event, and
the fit is the same one every other calibrator uses.

This module is intentionally thin, and its thinness is the design
position. The hard part of semantic alignment is *detection* — deciding
that frame 412 is where the gripper contacted the object — and detection
is domain-specific perception research that varies per task, per rig, per
dataset. It belongs in the application pipeline, not in a metrology
library that also has to be sane for a dVRK surgical rig and an LSL EEG
lab. What belongs here is what happens after: turning two arrays of event
times into an offset, a drift, and an honest statement of how well they
agree.

Why "experimental": the assumption underneath is that both sensors
register the event at the *same physical instant*, and semantic events
are far less crisp than a clap. A contact event smeared over three video
frames puts a frame-scale floor under the residuals, and no amount of
fitting removes it.

On deep audio-visual sync models (Synchformer and relatives): they belong
here eventually as a future ``[semantic]`` extra, but as **proposal
generators**, not estimators. Per the literature they classify offsets
into bins at or above frame size — useful for finding the right
neighbourhood, useless for the sub-frame number. The robust fit stays the
estimator either way; a proposal model would only seed
``max_offset_ms``.

The result type is evidence-first by construction. There is no
``aligned: bool`` field and there will not be one: "aligned" is a
judgement against a tolerance the caller owns, and a library that
announces it has hidden the tolerance somewhere the caller cannot see.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

from embodied_sync.calibrate.events import EventTrainAlignment, match_event_trains
from embodied_sync.time.clock_domain import ClockDomain, LatencyEstimate

__all__ = ["SemanticAlignment", "align_semantic_events"]

_EXPERIMENTAL_WARNED = False


@dataclass(frozen=True, slots=True)
class SemanticAlignment:
    """Evidence for a semantic-event alignment. No verdict, by design.

    ``mapping`` is the fitted clock mapping — the same
    :class:`~embodied_sync.time.clock_domain.LatencyEstimate` every
    calibrator produces, so it feeds ``translate_ns`` and
    ``register_clock_mapping`` unchanged. The rest is what the caller
    needs to decide whether to believe it: how many events actually
    matched out of how many were offered, how large the leftover
    disagreement is, and how distinctive the correlation peak was.
    """

    mapping: LatencyEstimate
    matched_count: int
    n_events_a: int
    n_events_b: int
    residual_p95_ns: int
    confidence: float
    alignment: EventTrainAlignment

    @property
    def offset_ns(self) -> int:
        return self.mapping.offset_ns

    @property
    def drift_ppb(self) -> int:
        return self.mapping.drift_ppb

    @property
    def matched_fraction_a(self) -> float:
        return self.alignment.matched_fraction_a

    @property
    def matched_fraction_b(self) -> float:
        return self.alignment.matched_fraction_b


def align_semantic_events(
    events_a_ns: Sequence[int],
    events_b_ns: Sequence[int],
    *,
    max_offset_ms: float,
    max_drift_ppm: float = 500.0,
    match_tolerance_ms: float | None = None,
    source_domain: ClockDomain | str | None = None,
    target_domain: ClockDomain | str | None = None,
) -> SemanticAlignment:
    """Fit a clock mapping from detected semantic events. Experimental.

    Emits a one-shot :class:`UserWarning` on first use (the same
    warn-once pattern
    :func:`~embodied_sync.time.clock_domain.resolve_clock_domain` uses)
    so an experimental API cannot end up load-bearing without anyone
    noticing, while a per-call warning does not flood a loop.

    Detection is the caller's: pass event times your own detector
    produced. Set ``match_tolerance_ms`` to the scale at which your
    events are actually localisable — for events detected on 30 fps
    video that is tens of milliseconds, and claiming tighter is claiming
    precision the detector does not have.
    """
    global _EXPERIMENTAL_WARNED
    if not _EXPERIMENTAL_WARNED:
        _EXPERIMENTAL_WARNED = True
        warnings.warn(
            "embodied_sync.calibrate.semantic is experimental: it assumes both "
            "detectors register the same physical instant, which semantic "
            "events satisfy far less well than a clap or a screen timestamp. "
            "Read SemanticAlignment.residual_p95_ns and .confidence before "
            "using the mapping.",
            stacklevel=2,
        )
    alignment = match_event_trains(
        events_a_ns,
        events_b_ns,
        max_offset_ms=max_offset_ms,
        max_drift_ppm=max_drift_ppm,
        match_tolerance_ms=match_tolerance_ms,
        source_domain=source_domain,
        target_domain=target_domain,
    )
    return SemanticAlignment(
        mapping=alignment.fit.mapping,
        matched_count=len(alignment.matched),
        n_events_a=len(events_a_ns),
        n_events_b=len(events_b_ns),
        residual_p95_ns=alignment.residual_p95_ns,
        confidence=alignment.confidence,
        alignment=alignment,
    )
