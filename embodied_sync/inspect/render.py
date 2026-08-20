"""The page itself: one self-contained HTML file, no references out.

Self-contained is a correctness property, not a packaging preference. A
verification artefact gets emailed, copied to a USB stick, opened on a
lab machine with no route to the internet, and archived next to the
results it justifies. A page whose evidence lives at a URL renders
differently depending on whether a CDN answered — and worse, its
*evidence* can change after it was signed off. So every image and every
audio excerpt is embedded as a ``data:`` URI, the CSS is inlined, the
plot is hand-written SVG, and nothing here emits an ``http`` anything.
:func:`build_page` returns a string; a test asserts the string contains
no external scheme.

What the page argues
--------------------
Top to bottom it is one argument. Banners state what is wrong before a
reader has scrolled anywhere, because a warning below the fold is a
warning nobody read. The fitted mapping panel gives the numbers *and*
says in the same panel that all of them are computed downstream of the
pairing, so a small residual is not evidence the pairing is right. The
residual plot shows every matched pair, not only the rendered ones, so
the sample below can be checked against the population it came from.
Then the events themselves: reference tile on the left, the three
competing pairings on the right, with the rejected neighbours rendered
identically to the chosen one — same size, same caption fields — because
a layout that makes the chosen column look authoritative is a layout
that answers the question for the reader.

Where the words come from
-------------------------
Every dataset-specific sentence arrives through :class:`PageContext` as
plain text and is escaped on the way in. That is deliberate: the caveats
worth printing ("this force channel was resampled onto the camera clock
by the dataset authors", "these two cameras see the scene from different
places") are things only the dataset binding knows, and a renderer that
guessed at them would print something false for the next dataset. The
renderer supplies the sentences that are true of *any* alignment, and
nothing else.
"""

from __future__ import annotations

import base64
import html
from dataclasses import dataclass

from embodied_sync.inspect.evidence import EventEvidence, Inspection
from embodied_sync.inspect.provider import AudioClip, ChannelReading, StreamInfo

__all__ = ["PageContext", "build_page"]

_MS_NS = 1_000_000

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 28px 64px; background: #f6f7f9; color: #16181d;
       font: 14px/1.5 ui-sans-serif, system-ui, "Segoe UI", Helvetica, Arial, sans-serif; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 28px 0 10px; text-transform: uppercase;
     letter-spacing: .06em; color: #5a6270; }
h3 { font-size: 14px; margin: 0 0 10px; }
.sub { color: #5a6270; margin: 0 0 18px; }
.panel { background: #fff; border: 1px solid #dfe3e8; border-radius: 8px;
         padding: 16px 18px; margin-bottom: 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 10px 18px; }
.kv .k { color: #5a6270; font-size: 12px; text-transform: uppercase;
         letter-spacing: .04em; }
.kv .v { font-size: 17px; font-variant-numeric: tabular-nums; }
.note { color: #5a6270; font-size: 12.5px; margin-top: 12px; }
.banner { border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;
          border: 1px solid; font-weight: 600; }
.banner.bad { background: #fdecea; border-color: #e0a3a0; color: #8a1c14; }
.banner.warn { background: #fff6e0; border-color: #e3c37a; color: #7a5400; }
.banner.ok { background: #eaf6ec; border-color: #a6cfb0; color: #1d5c2c; }
.event { background: #fff; border: 1px solid #dfe3e8; border-radius: 8px;
         padding: 14px 16px; margin-bottom: 18px; }
.event.flag { border-color: #d9534f; box-shadow: inset 4px 0 0 #d9534f; }
.strip { display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-start; }
figure { margin: 0; width: 320px; max-width: 100%; }
figure img { width: 100%; display: block; border-radius: 4px; background: #222; }
figure.slot { border: 2px solid transparent; border-radius: 6px; padding: 4px; }
figure.ref { border-color: #2f6fb0; }
figure.chosen { border-color: #2e8b57; }
figure.chosen.losing { border-color: #d9534f; }
figure.rival { border-color: #cfd4da; }
figure.rival.winning { border-color: #d9534f; border-style: dashed; }
figcaption { font-size: 12px; line-height: 1.45; margin-top: 6px;
             font-variant-numeric: tabular-nums; }
figcaption .tag { display: inline-block; font-weight: 700; letter-spacing: .04em;
                  text-transform: uppercase; font-size: 11px; margin-bottom: 2px; }
figcaption .muted { color: #6a7280; }
figcaption .big { font-size: 14px; font-weight: 600; }
.missing { width: 100%; aspect-ratio: 16 / 9; border-radius: 4px; background: #eceef1;
           border: 1px dashed #b9bfc7; display: flex; align-items: center;
           justify-content: center; color: #6a7280; font-size: 12px;
           text-align: center; padding: 8px; }
.audio { margin-top: 12px; display: flex; gap: 18px; flex-wrap: wrap; }
.audio div { font-size: 12px; color: #5a6270; }
audio { display: block; width: 320px; max-width: 100%; margin-top: 4px; }
.verdict { margin-top: 10px; font-size: 12.5px; }
.verdict b { font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
@media (prefers-color-scheme: dark) {
  body { background: #14161a; color: #e6e8eb; }
  .panel, .event { background: #1c1f24; border-color: #2e333a; }
  .sub, .note, .kv .k, figcaption .muted, .audio div { color: #98a1ad; }
  .missing { background: #23272d; border-color: #3a4048; color: #98a1ad; }
  figure.rival { border-color: #3a4048; }
  .banner.bad { background: #3a1a18; border-color: #7d3a34; color: #ffb4ac; }
  .banner.warn { background: #3a2f14; border-color: #7a6428; color: #ffd88a; }
  .banner.ok { background: #16301f; border-color: #2f6b42; color: #9fdcb1; }
}
"""


@dataclass(frozen=True, slots=True)
class PageContext:
    """Everything the renderer cannot know about the data it is rendering."""

    title: str
    reference: StreamInfo
    candidate: StreamInfo
    #: One line under the heading: what was aligned against what, and how.
    subtitle: str = ""
    #: Extra ``(name, value)`` cells for the summary panel — detector
    #: settings, scene statistics, whatever the binding measured.
    extra_rows: tuple[tuple[str, str], ...] = ()
    #: Caveat paragraphs printed above the events. Plain text, escaped.
    notes: tuple[str, ...] = ()
    #: Extra warning banners, for conditions only the binding can detect.
    warnings: tuple[str, ...] = ()
    #: The caller's prior on clock drift, if they had one. A fitted drift
    #: beyond it is flagged in the panel: the number was not measured
    #: within the assumption it was fitted under.
    drift_prior_ppm: float | None = None


# --- small formatters -----------------------------------------------------


def _data_uri(mime: str, payload: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def _ms(ns: int | None) -> str:
    return "—" if ns is None else f"{ns / 1e6:+.1f} ms"


def _in_frames(ns: int | None, period_ns: int | None) -> str:
    if ns is None or not period_ns:
        return ""
    return f"{ns / period_ns:+.2f} samples"


def _number(value: float) -> str:
    """Enough digits to compare two readings, few enough to scan a column."""
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _clock(ns: int, origin_ns: int) -> str:
    """Seconds since the first rendered event, plus the raw epoch ms.

    Both, because the relative number is what a reader reasons with and
    the absolute one is what they need to go back to the source files.
    """
    return (
        f"+{(ns - origin_ns) / 1e9:.3f} s "
        f"<span class='muted'>({ns // _MS_NS} ms)</span>"
    )


def _channel_text(reading: ChannelReading | None, *, absent: str) -> str:
    if reading is None or not reading.values:
        return f"<span class='muted'>{html.escape(absent)}</span>"
    bits = " · ".join(
        f"{html.escape(name)} {_number(value)}" for name, value in reading.values.items()
    )
    age = f"{reading.age_ns / 1e6:+.0f} ms"
    return f"<span class='muted'>{bits} (sample {age})</span>"


def _figure(*, css_class: str, tag: str, mime: str | None, payload: bytes | None,
            missing_reason: str, caption_html: str) -> str:
    if payload is None or mime is None:
        body = f"<div class='missing'>{html.escape(missing_reason)}</div>"
    else:
        body = f"<img alt='{html.escape(tag)}' src='{_data_uri(mime, payload)}'>"
    return (
        f"<figure class='slot {css_class}'>{body}"
        f"<figcaption><span class='tag'>{html.escape(tag)}</span><br>"
        f"{caption_html}</figcaption></figure>"
    )


def _audio_block(label: str, clip: AudioClip | None, origin_ns: int) -> str:
    if clip is None:
        return (
            f"<div><b>{html.escape(label)}</b><br>"
            "<span class='muted'>no audio covers this instant — reported as "
            "unavailable rather than substituted</span></div>"
        )
    clipped = " (window clipped at the end of the recording)" if clip.clipped else ""
    return (
        f"<div><b>{html.escape(label)}</b> · {clip.duration_s:.2f} s from "
        f"{_clock(clip.start_time_ns, origin_ns)}{html.escape(clipped)}"
        f"<audio controls preload='none' "
        f"src='{_data_uri(clip.mime, clip.payload)}'></audio></div>"
    )


# --- panels ---------------------------------------------------------------


def _summary_panel(inspection: Inspection, context: PageContext) -> str:
    mapping = inspection.mapping
    period = _display_period_ns(context)
    residual_p95 = inspection.residual_p95_ns
    drift = f"{mapping.drift_ppb / 1000.0:+.1f} ppm"
    if (
        context.drift_prior_ppm is not None
        and abs(mapping.drift_ppb) > context.drift_prior_ppm * 1000
    ):
        drift += "  ⚠ beyond prior"
    rows: list[tuple[str, str]] = [
        ("offset at anchor", f"{mapping.offset_ns / 1e6:+.2f} ms"),
        ("drift", drift),
        (
            "coarse-scan confidence",
            "not reported" if inspection.confidence is None else f"{inspection.confidence:.3f}",
        ),
        ("matched pairs", f"{len(inspection.matched)}"),
        (
            "matched fraction",
            (
                f"A {len(inspection.matched) / max(1, len(inspection.events_a_ns)):.0%} · "
                f"B {len(inspection.matched) / max(1, len(inspection.events_b_ns)):.0%}"
            ),
        ),
        (
            "residual p95",
            f"{residual_p95 / 1e6:.1f} ms"
            + (f" ({residual_p95 / period:.2f} samples)" if period else ""),
        ),
        (
            "events detected",
            f"A {len(inspection.events_a_ns)} · B {len(inspection.events_b_ns)}",
        ),
    ]
    periods = [
        (context.reference.label, context.reference.frame_period_ns),
        (context.candidate.label, context.candidate.frame_period_ns),
    ]
    known = [f"{label} {value / 1e6:.0f} ms" for label, value in periods if value]
    if known:
        rows.append(("sample period", " · ".join(known)))
    rows.extend(context.extra_rows)
    cells = "".join(
        f"<div class='kv'><div class='k'>{html.escape(k)}</div>"
        f"<div class='v'>{html.escape(v)}</div></div>"
        for k, v in rows
    )
    problems = ""
    if inspection.problems:
        problems = (
            "<p class='note'><b>Fit reported problems:</b> "
            + html.escape(", ".join(inspection.problems))
            + "</p>"
        )
    return (
        f"<section class='panel'><div class='grid'>{cells}</div>{problems}"
        "<p class='note'>Confidence, residual p95 and matched fraction are all "
        "computed <i>downstream of the pairing</i>. If the pairing is aliased "
        "they describe how self-consistent the wrong answer is, not whether it "
        "is right. That is what the frames below are for.</p></section>"
    )


def _display_period_ns(context: PageContext) -> int | None:
    """The slower stream's sampling period, or ``None`` if neither has one."""
    periods = [
        p
        for p in (context.reference.frame_period_ns, context.candidate.frame_period_ns)
        if p
    ]
    return max(periods) if periods else None


def _residual_svg(
    inspection: Inspection, evidence: list[EventEvidence], period_ns: int | None
) -> str:
    """Residual against event time, for every matched pair.

    Every pair, not only the rendered ones: the events below are a sample
    and this is the population they were drawn from, so a reader can see
    whether the sample is representative. A slope here is unmodelled
    drift, a step is a clock reset, and a fan is a detector losing the
    event rather than the clock moving.
    """
    if not inspection.residuals_ns:
        return "<p class='note'>No matched pairs to plot.</p>"
    width, height = 900, 230
    left, right, top, bottom = 62, 16, 16, 34
    events_a = inspection.events_a_ns
    matched = inspection.matched
    origin = int(events_a[matched[0][0]])
    xs = [(int(events_a[i]) - origin) / 1e9 for i, _ in matched]
    ys = [r / 1e6 for r in inspection.residuals_ns]
    span_x = (max(xs) - min(xs)) or 1.0
    floor_ms = (period_ns / 1e6) if period_ns else 1.0
    limit = max(max(abs(y) for y in ys), floor_ms) * 1.15
    plot_w = width - left - right
    plot_h = height - top - bottom

    def px(x: float) -> float:
        return left + (x - min(xs)) / span_x * plot_w

    def py(y: float) -> float:
        return top + (limit - y) / (2 * limit) * plot_h

    rendered = {e.time_a_ns for e in evidence}
    band = ""
    if period_ns:
        band = (
            f"<rect x='{left}' y='{py(floor_ms):.1f}' width='{plot_w}' "
            f"height='{abs(py(-floor_ms) - py(floor_ms)):.1f}' "
            "fill='#2f6fb0' opacity='0.10'></rect>"
        )
    ticks: list[str] = []
    for value in (-limit, -limit / 2, 0.0, limit / 2, limit):
        y = py(value)
        ticks.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{width - right}' y2='{y:.1f}' "
            f"stroke='#9aa3ad' stroke-width='{1 if value else 1.4}' "
            f"opacity='{0.9 if value == 0 else 0.25}'></line>"
            f"<text x='{left - 8}' y='{y + 4:.1f}' text-anchor='end' font-size='11' "
            f"fill='#6a7280'>{value:+.0f}</text>"
        )
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        seconds = min(xs) + fraction * span_x
        ticks.append(
            f"<text x='{px(seconds):.1f}' y='{height - 12}' text-anchor='middle' "
            f"font-size='11' fill='#6a7280'>{seconds:.0f}s</text>"
        )
    dots = "".join(
        f"<circle cx='{px(x):.1f}' cy='{py(y):.1f}' "
        f"r='{4.2 if int(events_a[i]) in rendered else 2.6}' "
        f"fill='{'#d9534f' if int(events_a[i]) in rendered else '#2f6fb0'}' "
        f"opacity='0.85'></circle>"
        for x, y, (i, _) in zip(xs, ys, matched)
    )
    caption = (
        "Red points are the events rendered below; blue points are the matched "
        "pairs not rendered."
    )
    if period_ns:
        caption = "Blue band is ±1 sampling period of the slower stream. " + caption
    return (
        f"<div class='scroll'><svg viewBox='0 0 {width} {height}' width='{width}' "
        f"height='{height}' role='img' aria-label='residual against event time'>"
        f"{band}{''.join(ticks)}{dots}"
        f"<text x='14' y='{top + plot_h / 2:.0f}' font-size='11' fill='#6a7280' "
        f"transform='rotate(-90 14 {top + plot_h / 2:.0f})' text-anchor='middle'>"
        f"residual (ms)</text></svg></div>"
        f"<p class='note'>{caption}</p>"
    )


# --- one event ------------------------------------------------------------


def _comparison_text(
    candidate_index: int, evidence: EventEvidence, channel: str | None
) -> str:
    """The two lines that let a reader check a tile without eyeballing it.

    Both are stated even when they disagree. A channel that prefers the
    chosen pairing while the frames cannot tell it from its neighbour is
    a real situation on real data, and averaging the two into one score
    would hide exactly the case a person needs to see.
    """
    candidate = evidence.candidates[candidate_index]
    parts: list[str] = []
    if candidate.label == "chosen":
        parts.append(
            "<span class='muted'>frames and channels below are measured against "
            "this tile</span>"
        )
    elif candidate.visual_separation is not None:
        if candidate.visual_separation < evidence.indistinct_gray_levels:
            parts.append(
                "<b>frame is indistinguishable from the chosen one</b> "
                f"<span class='muted'>(p95 pixel change "
                f"{candidate.visual_separation:.0f}/255)</span>"
            )
        else:
            parts.append(
                "frame differs visibly from chosen "
                f"<span class='muted'>(p95 pixel change "
                f"<b>{candidate.visual_separation:.0f}</b>/255)</span>"
            )
    if candidate.channel_gap is not None and channel is not None:
        best = min(c.channel_gap for c in evidence.candidates if c.channel_gap is not None)
        floor = evidence.channel_noise or 0.0
        marker = ""
        if candidate.channel_gap <= best and evidence.corroboration_verdict != "tie":
            marker = " ← closest to reference"
        elif candidate.channel_gap <= best + floor:
            marker = " ← tied for closest"
        parts.append(
            f"{html.escape(channel)} differs from reference by "
            f"<b>{_number(candidate.channel_gap)}</b>"
            f"<span class='muted'>{html.escape(marker)}</span>"
        )
    return "<br>".join(parts) if parts else "<span class='muted'>—</span>"


def _event_section(
    evidence: EventEvidence, context: PageContext, origin_ns: int
) -> str:
    period_b = context.candidate.frame_period_ns
    channel = context.candidate.corroboration_channel
    reference_frame = evidence.reference_frame
    reference_caption = f"event A[{evidence.index_a}] at {_clock(evidence.time_a_ns, origin_ns)}<br>"
    if reference_frame is not None:
        index = (
            f"#{reference_frame.index} " if reference_frame.index is not None else ""
        )
        reference_caption += (
            f"frame {index}at {_clock(reference_frame.time_ns, origin_ns)}<br>"
        )
    reference_caption += _channel_text(
        evidence.reference_channels, absent="no channel reading at this instant"
    )
    tiles = [
        _figure(
            css_class="ref",
            tag=f"reference · {context.reference.label}",
            mime=None if reference_frame is None else reference_frame.mime,
            payload=None if reference_frame is None else reference_frame.payload,
            missing_reason=(
                f"no frame available from {context.reference.label} at this instant"
            ),
            caption_html=reference_caption,
        )
    ]

    for position, candidate in enumerate(evidence.candidates):
        chosen = candidate.label == "chosen"
        if candidate.event_index is None:
            tiles.append(
                _figure(
                    css_class="rival",
                    tag=f"{candidate.label} candidate",
                    mime=None,
                    payload=None,
                    missing_reason=(
                        "no such event — this is the first or last event in "
                        "stream B, so this neighbour does not exist"
                    ),
                    caption_html="<span class='muted'>no candidate</span>",
                )
            )
            continue
        loses = chosen and evidence.neighbour_wins
        wins = (not chosen) and (
            candidate.delta_ns is not None
            and abs(candidate.delta_ns) < abs(evidence.residual_ns)
        )
        css = ("chosen losing" if loses else "chosen") if chosen else (
            "rival winning" if wins else "rival"
        )
        label = "chosen pairing" if chosen else f"{candidate.label} candidate"
        caption = (
            f"<span class='big'>{_ms(candidate.delta_ns)}</span> "
            f"<span class='muted'>{_in_frames(candidate.delta_ns, period_b)}</span><br>"
            f"{'residual' if chosen else 'offset change if adopted'}<br>"
            f"event B[{candidate.event_index}] at "
            f"{_clock(candidate.event_time_ns or 0, origin_ns)}<br>"
            f"rendered at {_clock(candidate.predicted_ns or 0, origin_ns)}<br>"
        )
        if candidate.frame is not None:
            index = f"#{candidate.frame.index} " if candidate.frame.index is not None else ""
            caption += (
                f"frame {index}at {_clock(candidate.frame.time_ns, origin_ns)}<br>"
            )
        caption += (
            _channel_text(candidate.channels, absent="no channel reading at this instant")
            + "<br>"
            + _comparison_text(position, evidence, channel)
        )
        tiles.append(
            _figure(
                css_class=css,
                tag=f"{label} · {context.candidate.label}",
                mime=None if candidate.frame is None else candidate.frame.mime,
                payload=None if candidate.frame is None else candidate.frame.payload,
                missing_reason=(
                    f"no frame available from {context.candidate.label} at this instant"
                ),
                caption_html=caption,
            )
        )

    audio_blocks = [
        _audio_block("audio at reference instant", evidence.audio_reference, origin_ns)
    ]
    if evidence.audio_mapped is not None or abs(
        evidence.predicted_ns - evidence.time_a_ns
    ) >= 100 * _MS_NS:
        audio_blocks.append(
            _audio_block("audio at mapped instant", evidence.audio_mapped, origin_ns)
        )
    audio = "".join(audio_blocks) if any(
        b is not None for b in (evidence.audio_reference, evidence.audio_mapped)
    ) else ""

    ratio = evidence.margin_ratio
    extra: list[str] = []
    if evidence.frames_indistinct:
        extra.append(
            "The neighbouring frames are visually indistinguishable from the "
            "chosen one — over the interval separating these hypotheses the "
            "scene barely changed, so <b>the pictures cannot decide this "
            "event</b>."
        )
    verdict_source = evidence.corroboration_verdict
    if channel is not None and verdict_source == "chosen":
        extra.append(
            f"{html.escape(channel)} at the chosen instant is closer to the "
            "reference instant's reading than either neighbour's is, by more "
            "than the channel's own sample-to-sample noise."
        )
    elif channel is not None and verdict_source == "neighbour":
        extra.append(
            f"<b>{html.escape(channel)} at a rejected neighbour is closer to the "
            "reference than the chosen pairing is.</b> The channel disagrees "
            "with this pairing."
        )
    elif channel is not None and verdict_source == "tie":
        extra.append(
            f"<span class='muted'>{html.escape(channel)} cannot separate the "
            "candidates here — the gaps are within the channel's own noise."
            "</span>"
        )
    if evidence.neighbour_wins:
        verdict = (
            "<span class='banner bad' style='display:inline-block;padding:6px 10px'>"
            "A REJECTED NEIGHBOUR IS CLOSER THAN THE CHOSEN PAIRING — "
            "this mapping does not agree with the events it was fitted to.</span>"
        )
    elif evidence.ambiguous and ratio is not None:
        verdict = (
            "<span class='banner warn' style='display:inline-block;padding:6px 10px'>"
            f"AMBIGUOUS — nearest rejected neighbour is only {ratio:.1f}× further "
            "away. Judge this one on the pictures, not the numbers.</span>"
        )
    elif ratio is not None:
        verdict = (
            f"Nearest rejected neighbour is <b>{ratio:.0f}×</b> further from the "
            "mapped instant than the chosen pairing."
        )
    else:
        verdict = (
            "<span class='muted'>No neighbouring candidate exists to compare "
            "against.</span>"
        )
    if extra:
        verdict += "<br>" + "<br>".join(extra)
    flag = " flag" if (evidence.neighbour_wins or verdict_source == "neighbour") else ""
    return (
        f"<section class='event{flag}'>"
        f"<h3>Event {evidence.ordinal} · A[{evidence.index_a}] ↔ B[{evidence.index_b}] "
        f"· residual {_ms(evidence.residual_ns)} "
        f"{_in_frames(evidence.residual_ns, period_b)}</h3>"
        f"<div class='strip'>{''.join(tiles)}</div>"
        f"<div class='audio'>{audio}</div>"
        f"<div class='verdict'>{verdict}</div>"
        f"</section>"
    )


# --- banners --------------------------------------------------------------


def _banners(
    inspection: Inspection, evidence: list[EventEvidence], context: PageContext
) -> str:
    banners: list[str] = []
    if inspection.perturbation:
        banners.append(
            "<div class='banner bad'>PERTURBED PAGE — NOT A MEASUREMENT. "
            f"{html.escape(inspection.perturbation)}. Any confidence figure below "
            "is carried over from the unperturbed fit on purpose, to show that a "
            "healthy-looking statistic can sit on top of a wrong mapping.</div>"
        )
    for message in inspection.matcher_warnings:
        banners.append(
            "<div class='banner bad'>The matcher warned while producing this "
            f"alignment: {html.escape(message)}</div>"
        )
    if not evidence:
        banners.append(
            "<div class='banner bad'>No matched pairs were rendered — there is "
            "nothing on this page to judge the alignment by.</div>"
        )
        for message in context.warnings:
            banners.append(f"<div class='banner warn'>{html.escape(message)}</div>")
        return "".join(banners)

    flagged = [e for e in evidence if e.neighbour_wins]
    ambiguous = [e for e in evidence if e.ambiguous and not e.neighbour_wins]
    indistinct = [e for e in evidence if e.frames_indistinct]
    margin = evidence[0].ambiguous_margin
    if flagged:
        banners.append(
            f"<div class='banner bad'>{len(flagged)} of {len(evidence)} rendered "
            "events have a rejected neighbour closer than the accepted pairing. "
            "That cannot happen when a mapping matches the events it was fitted "
            "to — treat this alignment as wrong until the frames say otherwise."
            "</div>"
        )
    elif ambiguous:
        banners.append(
            f"<div class='banner warn'>{len(ambiguous)} of {len(evidence)} rendered "
            f"events have a neighbour less than {margin:g}× further away than the "
            "chosen pairing. Those are decided by the pictures, not by the "
            "residual.</div>"
        )
    else:
        banners.append(
            f"<div class='banner ok'>All {len(evidence)} rendered events reject "
            f"their neighbours by more than {margin:g}×. That is a necessary "
            "condition, not a sufficient one — the pictures still have to agree."
            "</div>"
        )

    channel = context.candidate.corroboration_channel
    against = [e for e in evidence if e.corroboration_verdict == "neighbour"]
    supporting = [e for e in evidence if e.corroboration_verdict == "chosen"]
    if channel is not None and against and len(against) >= len(supporting):
        banners.append(
            f"<div class='banner bad'>{len(against)} of {len(evidence)} rendered "
            f"events have a rejected neighbour whose {html.escape(channel)} is "
            "closer to the reference instant than the chosen pairing's is. This "
            "is a channel independent of the pixels, so it is positive evidence "
            "against the mapping, not merely an absence of evidence for it.</div>"
        )
    elif channel is not None and against:
        banners.append(
            f"<div class='banner warn'>{html.escape(channel)} prefers the chosen "
            f"pairing on {len(supporting)} of {len(evidence)} rendered events but "
            f"prefers a rejected neighbour on {len(against)}. A minority of "
            "disagreements is what a real recording looks like — a detector "
            "occasionally fires on a different physical event in each stream — "
            "but the disagreeing events are flagged below and are the ones worth "
            "looking at first.</div>"
        )
    elif channel is not None and supporting:
        banners.append(
            f"<div class='banner ok'>{html.escape(channel)} picks the chosen "
            f"pairing over both neighbours in {len(supporting)} of "
            f"{len(evidence)} rendered events."
            + (
                " " + html.escape(context.candidate.channel_note)
                if context.candidate.channel_note
                else ""
            )
            + "</div>"
        )
    if indistinct:
        banners.append(
            f"<div class='banner warn'>{len(indistinct)} of {len(evidence)} "
            "rendered events show near-identical frames across the competing "
            "hypotheses: nothing moved over the interval separating them, so "
            "those events carry no visual evidence either way. Judge the "
            "alignment on the events that do.</div>"
        )
    if not any(e.reference_frame is not None for e in evidence) and not any(
        c.frame is not None for e in evidence for c in e.candidates
    ):
        banners.append(
            "<div class='banner warn'>No frames were available for any instant "
            "on this page, so it carries the arithmetic of the pairing and none "
            "of the pictures. The ±1 columns below still show which competing "
            "pairings the matcher rejected and by how much, but nothing here "
            "checks that the streams saw the same moment.</div>"
        )
    elif not any(e.audio_reference is not None or e.audio_mapped is not None for e in evidence):
        banners.append(
            "<div class='banner warn'>No audio covered any rendered instant — "
            "every clip is reported as unavailable rather than substituted.</div>"
        )
    for info in (context.reference, context.candidate):
        if info.missing:
            banners.append(
                f"<div class='banner warn'>{html.escape(info.label)}: missing "
                + html.escape(", ".join(info.missing))
                + ". The channels they carry read as unavailable.</div>"
            )
    for message in context.warnings:
        banners.append(f"<div class='banner warn'>{html.escape(message)}</div>")
    return "".join(banners)


def build_page(
    inspection: Inspection, evidence: list[EventEvidence], context: PageContext
) -> str:
    """The whole self-contained HTML document, as one string."""
    origin_ns = (
        int(inspection.events_a_ns[inspection.matched[0][0]])
        if inspection.matched
        else 0
    )
    notes = "".join(
        f"<p class='sub'>{html.escape(note)}</p>" for note in context.notes
    )
    sources = " · ".join(
        f"{info.label}: {info.source}"
        for info in (context.reference, context.candidate)
        if info.source
    )
    body = "".join(_event_section(e, context, origin_ns) for e in evidence)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(context.title)}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"<h1>{html.escape(context.title)}</h1>"
        + (f"<p class='sub'>{html.escape(context.subtitle)}</p>" if context.subtitle else "")
        + (f"<p class='sub'>{html.escape(sources)}</p>" if sources else "")
        + _banners(inspection, evidence, context)
        + "<h2>Fitted mapping</h2>"
        + _summary_panel(inspection, context)
        + "<h2>Residual against event time</h2><section class='panel'>"
        + _residual_svg(inspection, evidence, _display_period_ns(context))
        + "</section>"
        + f"<h2>Matched events ({len(evidence)} of {len(inspection.matched)} rendered)</h2>"
        + "<p class='sub'>Left tile is the reference: stream A at its own event "
        "time. The three right-hand tiles are stream B under three competing "
        "pairings, each drawn at the B-clock instant that pairing predicts for "
        "the reference. The chosen one should depict the same moment as the "
        "reference and the neighbours should not.</p>"
        + notes
        + body
        + "</body></html>"
    )
