"""JSON shape for a calibration result (A5).

``embsync calibrate clap`` has to write something to disk, and what it
writes is a contract: another tool reads it, a human reviews it, and a
session consumes the mapping inside it. So the shape lives here rather
than in the CLI, next to the code that produces the numbers.

The organising principle is the one the rest of the repo uses for
reports: **the mapping and the evidence for it travel together.** A
bare offset is unfalsifiable. An offset accompanied by how many events
matched, what fraction of each train that was, how large the residuals
are, and how confident the coarse scan was, can be argued with — and a
calibration nobody can argue with is a calibration nobody should trust.

``residuals_ns`` and ``matched_pairs`` are included in full. They are
small (one entry per matched clap, and if you have thousands of claps
you have a different problem), and they are what turns "the fit is
2.1 ms off somewhere" into "clap 7 is the outlier, look at the video at
0:43".
"""

from __future__ import annotations

from typing import Any, Sequence

from embodied_sync.calibrate.events import EventTrainAlignment
from embodied_sync.time.clock_domain import latency_estimate_to_dict

__all__ = ["CALIBRATION_FORMAT_VERSION", "CLAP_REPORT_TYPE", "clap_report_dict"]

#: Bumped only on a breaking change to the document shape, matching the
#: ``format_version`` convention of run format v0 (D-0005).
CALIBRATION_FORMAT_VERSION = 0

#: ``type`` discriminator, so a reader can tell this document from a run
#: manifest or a sync-report summary without guessing from its keys.
CLAP_REPORT_TYPE = "clap_calibration"


def clap_report_dict(
    alignment: EventTrainAlignment,
    *,
    audio_onsets_ns: Sequence[int],
    visual_events_ns: Sequence[int],
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a clap calibration as a JSON-ready document.

    ``inputs`` is echoed verbatim under ``"inputs"`` so the document
    records what produced it (file paths, detector settings, whether
    refinement ran). A calibration whose provenance is missing is a
    number of unknown origin, and six months later that is the same as
    no number at all.
    """
    fit = alignment.fit
    return {
        "format_version": CALIBRATION_FORMAT_VERSION,
        "type": CLAP_REPORT_TYPE,
        "mapping": latency_estimate_to_dict(fit.mapping),
        "diagnostics": {
            "n_audio_onsets": len(audio_onsets_ns),
            "n_visual_events": len(visual_events_ns),
            "n_matched": len(alignment.matched),
            "matched_fraction_audio": alignment.matched_fraction_a,
            "matched_fraction_visual": alignment.matched_fraction_b,
            "residual_p95_ns": alignment.residual_p95_ns,
            "residual_scale_ns": fit.residual_scale_ns,
            "inlier_fraction": fit.inlier_fraction,
            "confidence": alignment.confidence,
            "residuals_ns": list(fit.residuals_ns),
            "matched_pairs": [list(pair) for pair in alignment.matched],
        },
        "events": {
            "audio_onsets_ns": list(audio_onsets_ns),
            "visual_events_ns": list(visual_events_ns),
        },
        "inputs": dict(inputs or {}),
    }
