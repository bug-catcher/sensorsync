"""QR / screen-timestamp calibration: the fit, and the honest stub (D-0038)."""

from __future__ import annotations

import numpy as np
import pytest

from embodied_sync.calibrate import (
    SINGLE_PAIR_VARIANCE_NS,
    TimestampObservation,
    decode_timestamp_frames,
    fit_visual_timestamp,
)
from embodied_sync.time import ClockDomain, ClockKind, translate_ns

MS = 1_000_000
S = 1_000_000_000


def _observations(
    *,
    seed: int = 0,
    n: int = 60,
    fps: float = 30.0,
    offset_ns: int = 7_500_000,
    drift_ppb: int = 80_000,
    refresh_hz: float | None = 60.0,
) -> list[TimestampObservation]:
    """Frames of a display, with realistic refresh quantisation.

    ``displayed_time_ns`` is what the screen *showed*, which is stale by
    however long ago the last refresh was — the dominant error term for
    this method (see the module docstring).
    """
    rng = np.random.default_rng(seed)
    frame_period_ns = round(1e9 / fps)
    observations = []
    for i in range(n):
        frame_time_ns = 1_000_000_000 + i * frame_period_ns
        true_displayed = (
            frame_time_ns + offset_ns + round(i * frame_period_ns * drift_ppb / 1e9)
        )
        if refresh_hz is not None:
            # Quantise down to the last refresh boundary, with the phase of
            # the display's own clock unrelated to the camera's.
            refresh_ns = round(1e9 / refresh_hz)
            phase = int(rng.integers(0, refresh_ns))
            true_displayed -= (true_displayed + phase) % refresh_ns
        observations.append(
            TimestampObservation(
                displayed_time_ns=int(true_displayed),
                frame_time_ns=int(frame_time_ns),
            )
        )
    return observations


class TestFit:
    def test_noiseless_fit_is_exact(self) -> None:
        observations = _observations(refresh_hz=None)
        fit = fit_visual_timestamp(observations)
        assert fit.mapping.offset_ns == 7_500_000
        assert fit.mapping.drift_ppb == 80_000
        assert fit.residuals_ns == [0] * len(observations)

    def test_direction_is_camera_to_reference(self) -> None:
        """source = frame_time_ns, target = displayed_time_ns."""
        observations = [
            TimestampObservation(displayed_time_ns=1_000 + 5 * MS, frame_time_ns=1_000)
        ]
        fit = fit_visual_timestamp(observations)
        assert translate_ns(1_000, fit.mapping) == 1_000 + 5 * MS

    @pytest.mark.parametrize("seed", range(6))
    def test_offset_recovery_is_floored_by_the_refresh_interval(
        self, seed: int
    ) -> None:
        observations = _observations(seed=seed, refresh_hz=60.0)
        fit = fit_visual_timestamp(observations)
        refresh_ns = round(1e9 / 60.0)  # 16.67 ms

        # Quantisation makes the displayed time stale by 0-1 refresh
        # intervals, so the offset carries a ~half-interval bias plus noise
        # of the same scale. One full interval is the honest bound, and no
        # amount of extra data tightens it -- see the module docstring.
        assert abs(fit.mapping.offset_ns - 7_500_000) <= refresh_ns
        # The residual scale reports that floor rather than hiding it:
        # refresh / sqrt(12) ~ 4.8 ms for a uniform quantisation error.
        assert 2 * MS <= fit.mapping.variance_ns <= 10 * MS

    @pytest.mark.parametrize("seed", range(6))
    def test_drift_recovery_needs_a_long_baseline(self, seed: int) -> None:
        """40 s of frames: quantisation noise does not accumulate, so the
        slope tightens with the baseline even though the offset never does."""
        fit = fit_visual_timestamp(
            _observations(seed=seed, n=1200, refresh_hz=60.0)
        )
        assert abs(fit.mapping.drift_ppb - 80_000) <= 50_000  # 50 ppm

    def test_a_short_baseline_cannot_measure_drift(self) -> None:
        """The counterpart claim, made explicit so nobody trusts a 2 s QR run.

        Same quantisation, 2 s of frames instead of 40: the slope error is
        an order of magnitude worse. Drift needs baseline, not frame count.
        """
        short_errors = [
            abs(
                fit_visual_timestamp(
                    _observations(seed=seed, n=60, refresh_hz=60.0)
                ).mapping.drift_ppb
                - 80_000
            )
            for seed in range(6)
        ]
        long_errors = [
            abs(
                fit_visual_timestamp(
                    _observations(seed=seed, n=1200, refresh_hz=60.0)
                ).mapping.drift_ppb
                - 80_000
            )
            for seed in range(6)
        ]
        # A short baseline now *refuses* the drift rather than returning a
        # worse estimate of it, so the old "short error is 10x long error"
        # comparison no longer applies: the short fits return exactly 0 and
        # carry a named problem. That is the improvement, not a regression
        # - a caller reading the slope learns it is unmeasurable here
        # instead of reading a number an order of magnitude off.
        assert max(short_errors) > max(long_errors)
        short_fits = [
            fit_visual_timestamp(_observations(seed=seed, n=60, refresh_hz=60.0))
            for seed in range(6)
        ]
        assert all(f.mapping.drift_ppb == 0 for f in short_fits)
        assert all(
            any("drift_unresolvable" in p for p in f.problems) for f in short_fits
        )

    def test_single_observation_is_offset_only(self) -> None:
        fit = fit_visual_timestamp(
            [TimestampObservation(displayed_time_ns=5 * S, frame_time_ns=4 * S)]
        )
        assert fit.mapping.offset_ns == 1 * S
        assert fit.mapping.drift_ppb == 0
        assert fit.mapping.variance_ns == SINGLE_PAIR_VARIANCE_NS
        assert fit.n_pairs == 1

    def test_empty_observations_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one observation"):
            fit_visual_timestamp([])

    def test_domains_can_be_named_for_session_handoff(self) -> None:
        fit = fit_visual_timestamp(
            _observations(n=10, refresh_hz=None),
            source_domain=ClockDomain("cam_hw", ClockKind.HARDWARE),
            target_domain="host_mono",
        )
        assert fit.mapping.source.name == "cam_hw"
        assert fit.mapping.target.name == "host_mono"

    def test_a_mis_decoded_frame_does_not_wreck_the_fit(self) -> None:
        observations = _observations(n=40, refresh_hz=None)
        corrupted = list(observations)
        # One QR read off by a whole second.
        corrupted[7] = TimestampObservation(
            displayed_time_ns=corrupted[7].displayed_time_ns + S,
            frame_time_ns=corrupted[7].frame_time_ns,
        )
        fit = fit_visual_timestamp(corrupted)
        assert fit.mapping.offset_ns == 7_500_000
        assert fit.inlier_fraction == pytest.approx(39 / 40)


class TestDecoderStub:
    def test_decoder_raises_with_a_guided_message(self) -> None:
        with pytest.raises(NotImplementedError) as excinfo:
            decode_timestamp_frames([object()], frame_times_ns=[0])
        message = str(excinfo.value)
        assert "calibrate-vision" in message
        assert "fit_visual_timestamp" in message

    def test_importing_the_module_needs_no_vision_extra(self) -> None:
        import sys

        import embodied_sync.calibrate.visual_timestamp  # noqa: F401

        assert "cv2" not in sys.modules
        assert "pyzbar" not in sys.modules
