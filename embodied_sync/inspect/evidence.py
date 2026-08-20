"""The competing pairings behind one matched event, and the evidence for them.

Why this exists at all
----------------------
Every statistic an alignment reports is computed *downstream of the
pairing*: residual p95, matched fraction, offset stderr, coarse-scan
confidence. When the matcher pairs event *i* in stream A with event
*i±1* in stream B, the fit closes on that wrong correspondence and every
one of those numbers describes how self-consistent the wrong answer is.
This failure mode is upstream of those statistics, so no residual-based
statistic can reliably detect it on its own.

What does detect it is a comparison the numbers never make: rendering
the chosen pairing *beside the two the matcher rejected*, at the instants
those hypotheses predict, and asking whether the chosen one depicts the
same moment as the reference and the neighbours do not. That comparison
is what this module assembles. It is the core feature, not a
presentational extra — a page without the ±1 columns is a page that
cannot catch aliasing.

The three candidates are three mappings, not three events
---------------------------------------------------------
Each candidate is placed at the B-clock time *its own hypothesis*
predicts for the reference event: the chosen one at
``translate_ns(t_a, mapping)``, the neighbours at that time shifted by
``b[j∓1] − b[j]`` — the offset change adopting that pairing would imply.
Placing the neighbours at their raw event times instead would differ by
the chosen pairing's residual, a few milliseconds and invisible, but the
framing matters: these are competing answers to one question, not one
answer plus two nearby events.

Thresholds are defaults, not laws
---------------------------------
:data:`AMBIGUOUS_MARGIN` and :data:`INDISTINCT_GRAY_LEVELS` are parameters
everywhere they are used, because a rig with different imagery or a
different event rate has different numbers. They decide what the page
*flags*, never what it shows: every candidate is rendered regardless.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from embodied_sync.calibrate.estimator import fit_clock_mapping
from embodied_sync.calibrate.events import EventTrainAlignment
from embodied_sync.inspect.provider import (
    AudioClip,
    ChannelReading,
    EvidenceProvider,
    FrameImage,
)
from embodied_sync.time.clock_domain import LatencyEstimate, translate_ns

__all__ = [
    "AMBIGUOUS_MARGIN",
    "DEFAULT_MAX_EVENTS",
    "INDISTINCT_GRAY_LEVELS",
    "Candidate",
    "EventEvidence",
    "Inspection",
    "collect_evidence",
    "inspection_from_alignment",
    "perturb",
    "residuals_ns",
    "restrict_to_overlap",
]

_MS_NS = 1_000_000

#: Default cap on rendered events. Each event costs four images and up
#: to two audio excerpts, so
#: twelve keeps the page in the low megabytes and is about as much as
#: anyone will actually scroll.
DEFAULT_MAX_EVENTS = 12

#: A neighbour less than this many times further away than the accepted
#: pairing is not meaningfully rejected. Three is where the separation
#: exceeds typical detector scatter; it is a prompt to look harder, not a
#: verdict, and the
#: page names the events that tripped it rather than down-weighting them.
AMBIGUOUS_MARGIN = 3.0

#: Grayscale levels (0-255) below which two candidate frames cannot be
#: told apart however carefully anyone looks, as reported by
#: :meth:`~embodied_sync.inspect.provider.EvidenceProvider.visual_separation`.
#: This is a rendering heuristic, not a verdict; callers may override it
#: for different sensors and visual content.
INDISTINCT_GRAY_LEVELS = 8.0

#: Below this separation the reference instant and the mapped instant are
#: the same moment of wall-clock audio, and two excerpts would be the
#: same sound twice.
_AUDIO_DISTINCT_NS = 100 * _MS_NS

_CHOSEN = "chosen"
_STEPS: tuple[tuple[str, int], ...] = (("prev", -1), (_CHOSEN, 0), ("next", 1))


@dataclass(frozen=True, slots=True)
class Candidate:
    """One hypothesis about which B event the reference event corresponds to."""

    label: str
    #: Index into B's event list; ``None`` when the neighbour does not
    #: exist because this is the first or last event of the train.
    event_index: int | None = None
    event_time_ns: int | None = None
    #: B-clock time this hypothesis predicts for the reference event.
    predicted_ns: int | None = None
    #: ``b_candidate − predicted_by_the_chosen_mapping``. Small for the
    #: chosen pairing by construction; about one inter-event interval for
    #: a neighbour, unless the alignment is aliased.
    delta_ns: int | None = None
    frame: FrameImage | None = None
    channels: ChannelReading | None = None
    #: Difference between this candidate's frame and the *chosen* one, in
    #: gray levels. Near zero means the two hypotheses show the same
    #: picture and no amount of looking will separate them.
    visual_separation: float | None = None
    #: ``|corroboration_channel(candidate) − corroboration_channel(reference)|``.
    #: The reference instant's reading is a physical fact about that
    #: moment; a candidate landing on a different reading landed on a
    #: different moment.
    channel_gap: float | None = None


@dataclass(frozen=True, slots=True)
class EventEvidence:
    """Everything the page shows about one matched pair."""

    ordinal: int
    index_a: int
    index_b: int
    time_a_ns: int
    predicted_ns: int
    residual_ns: int
    reference_frame: FrameImage | None = None
    reference_channels: ChannelReading | None = None
    candidates: tuple[Candidate, ...] = ()
    audio_reference: AudioClip | None = None
    audio_mapped: AudioClip | None = None
    #: Typical sample-to-sample change in the corroboration channel around
    #: this instant. Two candidates whose gaps differ by less than this
    #: are tied, not ranked; see :meth:`corroboration_verdict`.
    channel_noise: float | None = None
    ambiguous_margin: float = AMBIGUOUS_MARGIN
    indistinct_gray_levels: float = INDISTINCT_GRAY_LEVELS

    @property
    def chosen(self) -> Candidate | None:
        return next((c for c in self.candidates if c.label == _CHOSEN), None)

    @property
    def best_neighbour_delta_ns(self) -> int | None:
        deltas = [
            abs(c.delta_ns)
            for c in self.candidates
            if c.label != _CHOSEN and c.delta_ns is not None
        ]
        return min(deltas) if deltas else None

    @property
    def neighbour_wins(self) -> bool:
        """A rejected neighbour sits closer than the accepted pairing.

        A clean greedy match cannot produce this — it takes the nearest
        free candidate. When it happens the mapping has moved away from
        the pairing that produced it, which is what aliasing looks like
        from the inside.
        """
        best = self.best_neighbour_delta_ns
        return best is not None and best < abs(self.residual_ns)

    @property
    def margin_ratio(self) -> float | None:
        """How many times further the nearest rejected neighbour sits."""
        best = self.best_neighbour_delta_ns
        if best is None:
            return None
        return best / max(abs(self.residual_ns), _MS_NS)

    @property
    def ambiguous(self) -> bool:
        ratio = self.margin_ratio
        return ratio is not None and ratio < self.ambiguous_margin

    @property
    def frames_indistinct(self) -> bool:
        """No neighbour's frame differs enough from the chosen one to judge.

        Not a fault in the alignment — a fault in this *event* as
        evidence. A robot holding still for a second produces three
        identical pictures, and quoting a comfortable numeric margin next
        to three identical pictures is the exact overconfidence this page
        exists to prevent.
        """
        separations = [
            c.visual_separation
            for c in self.candidates
            if c.label != _CHOSEN and c.visual_separation is not None
        ]
        return bool(separations) and max(separations) < self.indistinct_gray_levels

    @property
    def corroboration_verdict(self) -> str | None:
        """Which pairing the scalar channel prefers: chosen, neighbour or tie.

        ``None`` when the channel is unavailable anywhere in the row. This
        is the one comparison on the page that does not depend on pixels,
        so it is the one that still works when the frames are indistinct
        — which on real recordings is often.

        Three-valued rather than boolean because the gaps have a noise
        floor (:attr:`channel_noise`). Anything inside it is a tie,
        decided by the pictures instead; calling a difference smaller than
        the channel's own jitter a verdict would manufacture disagreement
        out of resampling noise.
        """
        gaps = {c.label: c.channel_gap for c in self.candidates if c.channel_gap is not None}
        if _CHOSEN not in gaps or len(gaps) < 2:
            return None
        chosen = gaps.pop(_CHOSEN)
        best = min(gaps.values())
        floor = self.channel_noise if self.channel_noise is not None else 0.0
        if chosen + floor < best:
            return _CHOSEN
        if best + floor < chosen:
            return "neighbour"
        return "tie"


@dataclass(frozen=True, slots=True)
class Inspection:
    """An alignment as rendered, including any deliberate perturbation.

    Carries the event trains it was fitted to, not just the pairs: the
    neighbouring candidates are events the *pairing* rejected, so they
    cannot be recovered from ``matched`` alone.
    """

    events_a_ns: tuple[int, ...]
    events_b_ns: tuple[int, ...]
    mapping: LatencyEstimate
    matched: tuple[tuple[int, int], ...]
    residuals_ns: tuple[int, ...]
    #: Coarse-scan confidence, when the caller's matcher reports one.
    confidence: float | None = None
    #: Named reasons the fit cannot support part of what it returns.
    problems: tuple[str, ...] = ()
    perturbation: str | None = None
    #: Warnings the matcher raised while producing this alignment,
    #: captured rather than left on stderr: the page is what gets kept and
    #: forwarded, and an alignment whose own estimator said "either the
    #: prior is too tight or the match is wrong" must not be able to
    #: present itself as unremarkable because a terminal scrolled.
    matcher_warnings: tuple[str, ...] = ()

    @property
    def residual_p95_ns(self) -> int:
        """Nearest-rank 95th percentile of ``|residual|`` over matched pairs."""
        if not self.residuals_ns:
            return 0
        ordered = sorted(abs(r) for r in self.residuals_ns)
        rank = max(0, int(np.ceil(0.95 * len(ordered))) - 1)
        return ordered[min(rank, len(ordered) - 1)]


def residuals_ns(
    events_a: Sequence[int],
    events_b: Sequence[int],
    matched: Sequence[tuple[int, int]],
    mapping: LatencyEstimate,
) -> tuple[int, ...]:
    """``b − translate_ns(a)`` per pair, the estimator's own convention."""
    return tuple(
        int(events_b[j]) - translate_ns(int(events_a[i]), mapping) for i, j in matched
    )


def restrict_to_overlap(
    events_a: Sequence[int], events_b: Sequence[int]
) -> tuple[list[int], list[int]]:
    """Keep only events from the window where both streams were running.

    An event from a stretch only one stream recorded can never be
    matched, so leaving it in deflates the matched fraction with events
    that never had a partner to find.
    """
    a = [int(t) for t in events_a]
    b = [int(t) for t in events_b]
    if not a or not b:
        return a, b
    lo = max(min(a), min(b))
    hi = min(max(a), max(b))
    return [t for t in a if lo <= t <= hi], [t for t in b if lo <= t <= hi]


def inspection_from_alignment(
    alignment: EventTrainAlignment,
    events_a: Sequence[int],
    events_b: Sequence[int],
    *,
    matcher_warnings: Sequence[str] = (),
) -> Inspection:
    """Adapt the library matcher's own result into the page's model."""
    matched = tuple((int(i), int(j)) for i, j in alignment.matched)
    mapping = alignment.fit.mapping
    return Inspection(
        events_a_ns=tuple(int(t) for t in events_a),
        events_b_ns=tuple(int(t) for t in events_b),
        mapping=mapping,
        matched=matched,
        residuals_ns=residuals_ns(events_a, events_b, matched, mapping),
        confidence=float(alignment.confidence),
        problems=tuple(alignment.problems),
        matcher_warnings=tuple(matcher_warnings),
    )


def perturb(
    inspection: Inspection, *, alias_shift: int = 0, force_offset_ms: float = 0.0
) -> Inspection:
    """Break the alignment on purpose, in one of two realistic ways.

    ``alias_shift`` re-pairs every matched event with B's event *N*
    positions along and **refits**, so the resulting mapping is a genuine
    least-cost fit to a wrong correspondence — what makes aliasing
    dangerous is precisely that everything downstream of the pairing is
    computed correctly. ``force_offset_ms`` leaves the pairing alone and
    moves the mapping: the cruder failure of a mis-transcribed constant.

    Both exist so the page's ability to *show* a bad alignment can be
    tested rather than assumed; a verification tool nobody has watched
    fail is not known to work. The coarse-scan confidence is carried over
    unchanged on purpose — it was computed from the unshifted scan and is
    now attached to a wrong mapping, which is the demonstration.
    """
    matched = inspection.matched
    mapping = inspection.mapping
    events_a = inspection.events_a_ns
    events_b = inspection.events_b_ns
    notes: list[str] = []
    if alias_shift:
        shifted = tuple(
            (i, j + alias_shift)
            for i, j in matched
            if 0 <= j + alias_shift < len(events_b)
        )
        if not shifted:
            raise ValueError(
                f"alias_shift {alias_shift:+d} leaves no pair inside stream B"
            )
        refit = fit_clock_mapping(
            [int(events_a[i]) for i, _ in shifted],
            [int(events_b[j]) for _, j in shifted],
            anchor_ns=int(min(events_a)),
        )
        matched = shifted
        mapping = refit.mapping
        notes.append(f"pairing shifted by {alias_shift:+d} events and refitted")
    if force_offset_ms:
        mapping = replace(
            mapping, offset_ns=mapping.offset_ns + round(force_offset_ms * 1e6)
        )
        notes.append(f"offset moved by {force_offset_ms:+.1f} ms")
    if not notes:
        return inspection
    return Inspection(
        events_a_ns=events_a,
        events_b_ns=events_b,
        mapping=mapping,
        matched=matched,
        residuals_ns=residuals_ns(events_a, events_b, matched, mapping),
        confidence=inspection.confidence,
        problems=inspection.problems,
        perturbation="; ".join(notes),
        matcher_warnings=inspection.matcher_warnings,
    )


def _spread(count: int, limit: int) -> list[int]:
    """Which matched pairs to render: spread, not the first ``limit``.

    Both aliasing and unmodelled drift are worst at the ends of a
    recording, so a page built from the first twelve pairs would
    systematically flatter the alignment.
    """
    if limit <= 0 or count <= limit:
        return list(range(count))
    picks = np.rint(np.linspace(0, count - 1, limit)).astype(np.int64)
    return sorted({int(p) for p in picks.tolist()})


def _frames_for(
    provider: EvidenceProvider, times_ns: list[int], *, label: str
) -> list[FrameImage | None]:
    """One batched decode, with the length contract enforced.

    A provider that returns fewer frames than were asked for would, if
    zipped against the request, put every remaining frame under the wrong
    caption — the page would look normal and be lying. Refusing is the
    only safe response.
    """
    if not times_ns:
        return []
    frames = list(provider.frames_at(times_ns))
    if len(frames) != len(times_ns):
        raise ValueError(
            f"{label} provider returned {len(frames)} frames for "
            f"{len(times_ns)} requested instants; frames_at must return one "
            f"entry per request (use None for instants it cannot cover)"
        )
    return frames


def collect_evidence(
    inspection: Inspection,
    reference: EvidenceProvider,
    candidate: EvidenceProvider,
    *,
    max_events: int = DEFAULT_MAX_EVENTS,
    ambiguous_margin: float = AMBIGUOUS_MARGIN,
    indistinct_gray_levels: float = INDISTINCT_GRAY_LEVELS,
) -> list[EventEvidence]:
    """Gather, for a sample of matched pairs, what a person needs to judge them.

    Two passes: decide what to show (pure arithmetic over the event
    trains), then ask each provider for the media in one batched call.
    The order matters for cost — see
    :meth:`~embodied_sync.inspect.provider.EvidenceProvider.frames_at` —
    and for honesty: nothing the providers return can change which
    candidates are displayed, so the page cannot quietly drop the
    hypothesis whose picture failed to decode.
    """
    events_a = inspection.events_a_ns
    events_b = inspection.events_b_ns
    channel = candidate.stream_info().corroboration_channel

    skeletons: list[EventEvidence] = []
    for ordinal, pick in enumerate(_spread(len(inspection.matched), max_events), start=1):
        i, j = inspection.matched[pick]
        time_a = int(events_a[i])
        predicted = translate_ns(time_a, inspection.mapping)
        reference_channels = reference.channels_at(time_a)
        hypotheses: list[Candidate] = []
        for label, step in _STEPS:
            k = j + step
            if not 0 <= k < len(events_b):
                hypotheses.append(Candidate(label=label))
                continue
            hypothesis_ns = predicted + (int(events_b[k]) - int(events_b[j]))
            hypotheses.append(
                Candidate(
                    label=label,
                    event_index=k,
                    event_time_ns=int(events_b[k]),
                    predicted_ns=hypothesis_ns,
                    delta_ns=int(events_b[k]) - predicted,
                    channels=candidate.channels_at(hypothesis_ns),
                )
            )
        skeletons.append(
            EventEvidence(
                ordinal=ordinal,
                index_a=i,
                index_b=j,
                time_a_ns=time_a,
                predicted_ns=predicted,
                residual_ns=int(events_b[j]) - predicted,
                reference_channels=reference_channels,
                candidates=tuple(
                    _with_channel_gap(c, reference_channels, channel) for c in hypotheses
                ),
                channel_noise=(
                    candidate.channel_noise(channel, predicted)
                    if channel is not None
                    else None
                ),
                ambiguous_margin=ambiguous_margin,
                indistinct_gray_levels=indistinct_gray_levels,
            )
        )

    reference_times = [e.time_a_ns for e in skeletons]
    candidate_times = [
        c.predicted_ns
        for e in skeletons
        for c in e.candidates
        if c.predicted_ns is not None
    ]
    reference_frames = _frames_for(reference, reference_times, label="reference")
    candidate_frames = iter(_frames_for(candidate, candidate_times, label="candidate"))

    evidence: list[EventEvidence] = []
    for skeleton, reference_frame in zip(skeletons, reference_frames):
        chosen_ns = next(
            (
                c.predicted_ns
                for c in skeleton.candidates
                if c.label == _CHOSEN and c.predicted_ns is not None
            ),
            None,
        )
        candidates = tuple(
            replace(
                c,
                frame=next(candidate_frames) if c.predicted_ns is not None else None,
                visual_separation=(
                    candidate.visual_separation(c.predicted_ns, chosen_ns)
                    if c.predicted_ns is not None
                    and chosen_ns is not None
                    and c.label != _CHOSEN
                    else None
                ),
            )
            for c in skeleton.candidates
        )
        mapped_audio = (
            candidate.audio_at(skeleton.predicted_ns)
            if abs(skeleton.predicted_ns - skeleton.time_a_ns) >= _AUDIO_DISTINCT_NS
            else None
        )
        evidence.append(
            replace(
                skeleton,
                reference_frame=reference_frame,
                candidates=candidates,
                audio_reference=reference.audio_at(skeleton.time_a_ns),
                audio_mapped=mapped_audio,
            )
        )
    return evidence


def _with_channel_gap(
    candidate: Candidate, reference: ChannelReading | None, channel: str | None
) -> Candidate:
    """Attach ``|candidate − reference|`` for the corroboration channel."""
    if channel is None or reference is None or candidate.channels is None:
        return candidate
    here = candidate.channels.values.get(channel)
    there = reference.values.get(channel)
    if here is None or there is None:
        return candidate
    return replace(candidate, channel_gap=abs(float(here) - float(there)))
