"""Calibration: measuring the mapping between two clocks (D-0038).

`session/` and `align/` *consume* clock mappings; `calibrate/` is where
they come from. Everything in this subpackage — a clap, a QR code on a
monitor, a matched pair of semantic event trains — ends in the same
:class:`~embodied_sync.time.clock_domain.LatencyEstimate` that
:func:`~embodied_sync.time.clock_domain.translate_ns` already knows how
to apply and that
:meth:`SyncSession.register_clock_mapping
<embodied_sync.session.SyncSession.register_clock_mapping>` already knows
how to accept. One output type is what keeps the library coherent
instead of a bag of scripts.

The division of labour is deliberate: **detection is the caller's,
fitting is ours.** Knowing what a soldering iron sounds like, or which
frame the gripper made contact in, is domain research that differs for
every rig. Turning detected event times into an offset, a drift, and an
honest uncertainty is metrology, and it is the same arithmetic for a
surgical robot and a kitchen GoPro. The one exception is
:func:`~embodied_sync.calibrate.clap.detect_audio_onsets`, which is
generic enough (and cheap enough in numpy) to live here.

Dependencies: numpy only — no scipy, no OpenCV. Frame decoding for the
QR path is an increment-2 extra
(:func:`~embodied_sync.calibrate.visual_timestamp.decode_timestamp_frames`
is a documented stub). Importing this subpackage pulls no optional
dependency.

Also note what these numbers do *and do not* claim: `session/`'s
``quality()`` measures timestamp consistency, while this subpackage is
what measures physical simultaneity — two cameras can agree perfectly on
timestamps while exposing 30 ms apart, and only a calibration event can
tell you so.
"""

from embodied_sync.calibrate.audio_io import SUPPORTED_SUFFIXES, load_waveform
from embodied_sync.calibrate.clap import (
    align_clap_events,
    detect_audio_onsets,
    gcc_phat,
    refine_onsets,
)
from embodied_sync.calibrate.estimator import (
    SINGLE_PAIR_VARIANCE_NS,
    ClockMappingFit,
    fit_clock_mapping,
    score_confidence,
    standard_score,
)
from embodied_sync.calibrate.events import EventTrainAlignment, match_event_trains
from embodied_sync.calibrate.report import (
    CALIBRATION_FORMAT_VERSION,
    CLAP_REPORT_TYPE,
    clap_report_dict,
)
from embodied_sync.calibrate.semantic import SemanticAlignment, align_semantic_events
from embodied_sync.calibrate.visual_timestamp import (
    TimestampObservation,
    decode_timestamp_frames,
    fit_visual_timestamp,
)

__all__ = [
    "CALIBRATION_FORMAT_VERSION",
    "CLAP_REPORT_TYPE",
    "SINGLE_PAIR_VARIANCE_NS",
    "SUPPORTED_SUFFIXES",
    "ClockMappingFit",
    "EventTrainAlignment",
    "SemanticAlignment",
    "TimestampObservation",
    "align_clap_events",
    "align_semantic_events",
    "clap_report_dict",
    "decode_timestamp_frames",
    "detect_audio_onsets",
    "fit_clock_mapping",
    "fit_visual_timestamp",
    "gcc_phat",
    "load_waveform",
    "match_event_trains",
    "refine_onsets",
    "score_confidence",
    "standard_score",
]
