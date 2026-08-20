"""Screen-timestamp (QR / clock-on-a-monitor) calibration.

Point the camera at a display that renders the reference clock's current
time — as a QR code, a seven-segment readout, a binary strip. Each frame
then carries two times: the one the display *showed* and the one the
camera *stamped*. A sequence of those pairs is a clock mapping, fitted by
the same robust estimator every other calibrator uses.

This module is the **fit**. The decoder is increment-2 work behind the
``[calibrate-vision]`` extra (see :func:`decode_timestamp_frames`),
because decoding a QR code is OpenCV/pyzbar territory and this package's
base install stays at numpy + pyyaml.

What this method can and cannot measure
---------------------------------------
The honest error budget, because a QR calibration looks far more exact
than it is:

- **Display refresh quantisation.** The screen only updates on a refresh
  boundary: ~16.7 ms on a 60 Hz panel, ~8.3 ms at 120 Hz. The displayed
  time is therefore stale by 0–1 refresh intervals, uniformly. This is
  usually the dominant term and it is a *bias plus noise*, not just
  noise — the expected staleness is half a refresh interval.
- **Rolling shutter.** On a CMOS sensor each row is exposed at a
  different instant; a code near the bottom of the frame is sampled
  milliseconds after one near the top (full-frame readout is typically
  5–30 ms). Keep the code in a fixed, known screen region across a
  calibration, or the row time leaks into the residuals.
- **Exposure midpoint vs render time.** The camera's own timestamp may
  refer to exposure start, exposure midpoint, or readout completion —
  vendors differ and rarely document it — while the displayed value
  refers to when the *renderer* composed the frame, which precedes the
  photons by an unknown compositor latency.

Consequence: treat a QR fit's **offset** as good to roughly half a
refresh interval, and trust its **drift** far more than its offset, since
drift comes from the slope across a long baseline and the quantisation
error does not accumulate. The fit's ``variance_ns`` reports the
observed residual scale, which will show the quantisation as a floor —
that floor is real information, not a defect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from embodied_sync.calibrate.estimator import ClockMappingFit, fit_clock_mapping
from embodied_sync.time.clock_domain import ClockDomain

__all__ = [
    "TimestampObservation",
    "decode_timestamp_frames",
    "fit_visual_timestamp",
]

_EXTRA_HINT = (
    "decode_timestamp_frames is not implemented yet: frame decoding needs the "
    "'calibrate-vision' extra (opencv-python, pyzbar) and lands in increment 2. "
    "Install it with `pip install 'embodied-sync[calibrate-vision]'` once "
    "available. Until then, decode the codes with your own pipeline and pass "
    "the results to fit_visual_timestamp() as TimestampObservation values — "
    "the fit is the part this library owns."
)


@dataclass(frozen=True, slots=True)
class TimestampObservation:
    """One frame that saw a displayed reference time.

    ``displayed_time_ns`` is the reference-clock value read off the
    screen; ``frame_time_ns`` is the camera's own timestamp for the
    frame that saw it. Both integer nanoseconds (D-0002).
    """

    displayed_time_ns: int
    frame_time_ns: int


def fit_visual_timestamp(
    observations: Sequence[TimestampObservation],
    *,
    anchor_ns: int | None = None,
    source_domain: ClockDomain | str | None = None,
    target_domain: ClockDomain | str | None = None,
) -> ClockMappingFit:
    """Fit camera-clock → displayed-reference-clock from screen observations.

    Direction matters and is chosen deliberately: **source =
    ``frame_time_ns``, target = ``displayed_time_ns``**. The caller holds
    frames stamped in camera time and wants them restated in the
    reference clock, so ``translate_ns(frame_time, fit.mapping)`` is
    directly the useful operation, and the mapping can be handed to
    :meth:`SyncSession.register_clock_mapping
    <embodied_sync.session.SyncSession.register_clock_mapping>` when the
    reference clock *is* the session clock.

    A sequence yields offset **and** drift; a single observation yields
    offset only, with
    :data:`~embodied_sync.calibrate.estimator.SINGLE_PAIR_VARIANCE_NS`
    marking the drift as unmeasured. Read the module docstring before
    trusting the offset to better than half a display refresh interval.
    """
    if not observations:
        raise ValueError("fit_visual_timestamp needs at least one observation")
    return fit_clock_mapping(
        [observation.frame_time_ns for observation in observations],
        [observation.displayed_time_ns for observation in observations],
        anchor_ns=anchor_ns,
        source_domain=source_domain,
        target_domain=target_domain,
    )


def decode_timestamp_frames(
    frames: Sequence[Any],
    *,
    frame_times_ns: Sequence[int],
    decoder: str = "qr",
) -> list[TimestampObservation]:
    """Decode displayed timestamps out of image frames. **Not implemented.**

    Deliberately a stub rather than a half-decoder: reading a QR code
    needs OpenCV or pyzbar, both of which are heavier than this
    library's entire base install, and a wrong decode produces a
    *plausible* calibration, which is worse than no calibration.

    Raises :class:`NotImplementedError` with the guided message naming
    the ``[calibrate-vision]`` extra and the supported path in the
    meantime (decode yourself, then call :func:`fit_visual_timestamp`).
    """
    raise NotImplementedError(_EXTRA_HINT)
