"""Sync-quality report generator (D-0023).

Takes an :class:`~embodied_sync.align.AlignedRun` (typically loaded from
an aligned episode via :func:`~embodied_sync.datasets.io.load_episode`)
and produces a self-contained HTML report plus a machine-readable
summary dict.

Statistics per stream (over the frames that were emitted):

- ``frame_count``: total aligned frames in the episode
- ``missing_count``: frames where this stream was flagged missing
- ``missing_rate``: ``missing_count / frame_count``
- ``median_skew_ns``: median of non-missing frames' ``skew_ns``; ``None``
  if every frame is missing
- ``median_abs_skew_ns``: median of ``abs(skew_ns)`` (useful when both
  nearest-neighbor's signed skew and ZoH's non-positive skew appear in
  the same table)
- ``median_confidence``: median of non-missing frames' confidence;
  ``None`` if every frame is missing
- ``method``: the alignment method (from the first non-missing frame)
- ``ground_truth_missing_count``: verbatim from
  :class:`~embodied_sync.align.AlignmentReport` (may be ``0`` when the
  alignment ran without ground truth)

Statistics use plain-Python median (``statistics.median``) so the base
install stays numpy-only for math and skips the sort for numpy import.
Rendering is a small inline-styled HTML template — no external CSS or
JS, no data URIs — so the file drops into an email, a static server, or
a repo diff without breaking.
"""

from __future__ import annotations

import html
import statistics
from pathlib import Path
from typing import Any

from embodied_sync.core.episode import AlignedRun
from embodied_sync.core.sync_report import (
    REPORT_FORMAT_VERSION,
    StreamStats,
    SyncQualityReport,
)

__all__ = [
    "REPORT_FORMAT_VERSION",
    "StreamStats",
    "SyncQualityReport",
    "build_report",
    "render_html",
    "report_summary_dict",
    "save_report_html",
]


def build_report(aligned: AlignedRun) -> SyncQualityReport:
    """Compute per-stream sync-quality statistics from an :class:`AlignedRun`."""
    frame_count = len(aligned.frames)
    stream_names = (
        list(aligned.frames[0].samples.keys())
        if aligned.frames
        else list(aligned.report.missing_count.keys())
    )

    stats: list[StreamStats] = []
    for name in stream_names:
        missing = aligned.report.missing_count.get(name, 0)
        signed_skews: list[int] = []
        abs_skews: list[int] = []
        confidences: list[float] = []
        method: str | None = None
        for frame in aligned.frames:
            md = frame.metadata[name]
            if md.method is not None:
                method = md.method
            if md.missing or md.skew_ns is None:
                continue
            signed_skews.append(md.skew_ns)
            abs_skews.append(abs(md.skew_ns))
            confidences.append(md.confidence)
        stats.append(
            StreamStats(
                name=name,
                frame_count=frame_count,
                missing_count=missing,
                missing_rate=(missing / frame_count) if frame_count else 0.0,
                median_skew_ns=(
                    int(statistics.median(signed_skews)) if signed_skews else None
                ),
                median_abs_skew_ns=(
                    int(statistics.median(abs_skews)) if abs_skews else None
                ),
                median_confidence=(
                    float(statistics.median(confidences)) if confidences else None
                ),
                method=method,
                ground_truth_missing_count=aligned.report.ground_truth_missing_count.get(
                    name, 0
                ),
            )
        )
    return SyncQualityReport(frame_count=frame_count, streams=tuple(stats))


def _fmt_ns(value: int | None) -> str:
    if value is None:
        return "—"
    ms = value / 1_000_000
    return f"{ms:+.3f} ms" if value else "0.000 ms"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_conf(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


_HTML_STYLE = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
       Roboto, Helvetica, Arial, sans-serif; margin: 2rem auto;
       max-width: 60rem; color: #222; }
h1 { font-size: 1.5rem; border-bottom: 1px solid #ccc; padding-bottom: .25rem; }
h2 { font-size: 1.15rem; margin-top: 2rem; }
table { border-collapse: collapse; margin-top: .5rem; width: 100%; }
th, td { text-align: left; padding: .35rem .6rem; border-bottom: 1px solid #eee; }
th { background: #f6f6f6; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.bad { color: #b22; font-weight: 600; }
td.warn { color: #a60; }
.summary { color: #555; font-size: .95rem; margin-bottom: 1rem; }
"""


def _render_stream_row(stats: StreamStats, show_ground_truth: bool) -> str:
    missing_cls = "bad" if stats.missing_rate > 0.10 else "warn" if stats.missing_rate > 0 else "num"
    cells = [
        f"<td>{html.escape(stats.name)}</td>",
        f"<td class='num'>{stats.frame_count}</td>",
        (
            f"<td class='{missing_cls}'>{stats.missing_count} "
            f"({_fmt_pct(stats.missing_rate)})</td>"
        ),
        f"<td class='num'>{_fmt_ns(stats.median_skew_ns)}</td>",
        f"<td class='num'>{_fmt_ns(stats.median_abs_skew_ns)}</td>",
        f"<td class='num'>{_fmt_conf(stats.median_confidence)}</td>",
    ]
    if show_ground_truth:
        cells.append(
            f"<td class='num'>{stats.ground_truth_missing_count}</td>"
        )
    return "<tr>" + "".join(cells) + "</tr>"


def render_html(
    report: SyncQualityReport,
    *,
    title: str = "Sync-quality report",
    source_run: str | None = None,
    target_rate_hz: float | None = None,
) -> str:
    """Render a :class:`SyncQualityReport` as a self-contained HTML page."""
    show_ground_truth = any(s.ground_truth_missing_count > 0 for s in report.streams)
    method = next((s.method for s in report.streams if s.method), None)

    summary_bits: list[str] = [f"<strong>Frames:</strong> {report.frame_count}"]
    if method:
        summary_bits.append(f"<strong>Method:</strong> {html.escape(method)}")
    if target_rate_hz is not None:
        summary_bits.append(f"<strong>Target rate:</strong> {target_rate_hz:g} Hz")
    if source_run:
        summary_bits.append(f"<strong>Source run:</strong> {html.escape(source_run)}")
    summary_line = " &nbsp;·&nbsp; ".join(summary_bits)

    headers = [
        "Stream",
        "Frames",
        "Missing",
        "Median skew",
        "Median |skew|",
        "Median confidence",
    ]
    if show_ground_truth:
        headers.append("Ground truth drops in window")

    thead = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    tbody = "\n".join(_render_stream_row(s, show_ground_truth) for s in report.streams)

    ground_truth_note = (
        "<p class='summary'>Ground-truth drop counts come from the "
        "corruption sidecar loaded at align time.</p>"
        if show_ground_truth
        else ""
    )

    return (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        f"<meta charset='utf-8'><title>{html.escape(title)}</title>\n"
        f"<style>{_HTML_STYLE}</style>\n"
        "</head>\n<body>\n"
        f"<h1>{html.escape(title)}</h1>\n"
        f"<p class='summary'>{summary_line}</p>\n"
        f"<h2>Per-stream summary</h2>\n"
        f"<table>\n<thead>\n{thead}\n</thead>\n<tbody>\n{tbody}\n</tbody>\n</table>\n"
        f"{ground_truth_note}"
        "</body></html>\n"
    )


def save_report_html(
    aligned: AlignedRun,
    out_path: str | Path,
    *,
    title: str = "Sync-quality report",
    source_run: str | None = None,
    target_rate_hz: float | None = None,
) -> Path:
    """Compute a report from ``aligned`` and write HTML to ``out_path``."""
    report = build_report(aligned)
    html_text = render_html(
        report,
        title=title,
        source_run=source_run,
        target_rate_hz=target_rate_hz,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


def report_summary_dict(report: SyncQualityReport) -> dict[str, Any]:
    """Return the structured summary as a plain dict (for JSON / logging).

    This is *not* saved to disk by v0; it exists so callers who want to
    log or diff report contents don't have to reach into dataclass
    internals.
    """
    return {
        "format_version": REPORT_FORMAT_VERSION,
        "frame_count": report.frame_count,
        "streams": [
            {
                "name": s.name,
                "frame_count": s.frame_count,
                "missing_count": s.missing_count,
                "missing_rate": s.missing_rate,
                "median_skew_ns": s.median_skew_ns,
                "median_abs_skew_ns": s.median_abs_skew_ns,
                "median_confidence": s.median_confidence,
                "method": s.method,
                "ground_truth_missing_count": s.ground_truth_missing_count,
            }
            for s in report.streams
        ],
    }
