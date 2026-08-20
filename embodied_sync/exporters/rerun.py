"""Rerun exporter (Milestone 8).

Optional-dependency discipline: this module imports at base-install
time (no top-level ``import rerun``), so ``import
embodied_sync.exporters.rerun`` succeeds without ``pip install
embodied-sync[rerun]``. The concrete ``rerun.log`` calls happen inside
:func:`save_rerun_episode` behind a lazy import.

When Rerun is unavailable the exporter falls back to a self-contained
HTML page (``save_html_fallback``) that summarises the same aligned
episode — the roadmap's Milestone-8 acceptance criterion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from embodied_sync.core.episode import AlignedRun
from embodied_sync.reports import save_report_html

__all__ = ["is_rerun_available", "save_html_fallback", "save_rerun_episode"]


def is_rerun_available() -> bool:
    """True iff the optional ``rerun`` package can be imported.

    The check is lazy so importing this module does not pull ``rerun``
    into the base-install path. Used by :func:`save_rerun_episode` to
    decide whether to log through Rerun or fall back to HTML.
    """
    try:
        import importlib.util  # noqa: PLC0415

        return importlib.util.find_spec("rerun") is not None
    except ImportError:
        return False


def save_rerun_episode(
    episode: AlignedRun,
    path: str | Path,
    *,
    application_id: str = "embodied_sync",
    force_html: bool = False,
) -> str:
    """Export ``episode`` to Rerun ``.rrd`` (if available) or HTML.

    Returns the string ``"rerun"`` when the Rerun path was used and
    ``"html"`` when the fallback triggered. ``force_html=True`` skips the
    Rerun path entirely — useful for CI where we want the fallback tested
    even on machines that happen to have ``rerun`` installed.

    Rerun path: emits one entity per stream with a numeric-payload plot
    where the payload is numeric, plus a "missing" scalar (0/1) per
    frame. Non-numeric payloads become text log entries so the entity
    tree still contains every stream.
    """
    output = Path(path)
    if force_html or not is_rerun_available():
        html_path = _replace_extension(output, ".html")
        save_html_fallback(episode, html_path)
        return "html"

    import rerun as rr  # noqa: PLC0415

    output.parent.mkdir(parents=True, exist_ok=True)
    rec = rr.new_recording(application_id=application_id)
    for frame in episode.frames:
        rr.set_time_nanos("world_time_ns", frame.target_time_ns, recording=rec)
        for name, sample in frame.samples.items():
            meta = frame.metadata[name]
            rr.log(
                f"streams/{name}/missing",
                rr.Scalars(1.0 if meta.missing else 0.0),
                recording=rec,
            )
            if sample is None:
                continue
            payload = _numeric(sample.payload)
            if payload is not None:
                rr.log(
                    f"streams/{name}/payload",
                    rr.Scalars(payload),
                    recording=rec,
                )
            else:
                rr.log(
                    f"streams/{name}/payload",
                    rr.TextLog(str(sample.payload)),
                    recording=rec,
                )
    rr.save(str(output), recording=rec)
    return "rerun"


def save_html_fallback(episode: AlignedRun, path: str | Path) -> None:
    """Write the sync-quality HTML report at ``path``.

    The HTML fallback IS the sync-quality report — same
    self-contained page :func:`~embodied_sync.reports.save_report_html`
    produces from ``embsync report``. Keeps the two paths from diverging.
    """
    save_report_html(
        episode,
        path,
        title="Sync-quality report (Rerun fallback)",
    )


def _replace_extension(path: Path, new_ext: str) -> Path:
    if path.suffix:
        return path.with_suffix(new_ext)
    return path.parent / (path.name + new_ext)


def _numeric(payload: Any) -> list[float] | None:
    if isinstance(payload, bool):
        return None
    if isinstance(payload, (int, float)):
        return [float(payload)]
    if isinstance(payload, (list, tuple)):
        out: list[float] = []
        for v in payload:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            out.append(float(v))
        return out
    return None
