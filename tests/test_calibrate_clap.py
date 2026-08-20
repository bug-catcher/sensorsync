"""Audio onset detection and clap alignment (D-0038, refinement A4).

The detector is checked against planted transients in synthetic noise —
both that it finds them and, just as importantly, that it does not
invent any.

``TestOnsetRefinement`` is the A4 half: the coarse detector's accuracy
is bounded by its frame length (~10 ms) and does not improve with more
data, so the refinement pass has to be demonstrated *quantitatively*
against the same planted ground truth rather than asserted. The tests
therefore compare refined error to coarse error on identical signals
and require an order-of-magnitude improvement, not merely "some".
"""

from __future__ import annotations

import numpy as np
import pytest

from embodied_sync.calibrate import (
    align_clap_events,
    detect_audio_onsets,
    gcc_phat,
    refine_onsets,
)

MS = 1_000_000
S = 1_000_000_000
SAMPLE_RATE = 16_000.0


def _noise_with_transients(
    *,
    seed: int,
    duration_s: float = 5.0,
    onsets_s: tuple[float, ...] = (0.7, 1.9, 3.3, 4.4),
    noise_sigma: float = 0.02,
    burst_gain: float = 0.6,
    decay_samples: int = 60,
    burst_samples: int = 400,
) -> np.ndarray:
    """Room tone plus sharp decaying bursts at known times."""
    rng = np.random.default_rng(seed)
    n = int(SAMPLE_RATE * duration_s)
    signal = rng.normal(0.0, noise_sigma, n)
    envelope = np.exp(-np.arange(burst_samples) / decay_samples)
    for onset_s in onsets_s:
        start = int(onset_s * SAMPLE_RATE)
        burst = burst_gain * envelope * rng.normal(0.0, 1.0, burst_samples)
        signal[start : start + burst_samples] += burst
    return signal


class TestOnsetDetection:
    @pytest.mark.parametrize("seed", range(6))
    def test_finds_every_planted_transient_and_nothing_else(self, seed: int) -> None:
        onsets_s = (0.7, 1.9, 3.3, 4.4)
        signal = _noise_with_transients(seed=seed, onsets_s=onsets_s)
        detected = detect_audio_onsets(signal, SAMPLE_RATE)

        assert len(detected) == len(onsets_s), (
            f"expected {len(onsets_s)} onsets, got "
            f"{[round(d / 1e6, 1) for d in detected]} ms"
        )
        for detected_ns, true_s in zip(detected, onsets_s):
            error_ms = abs(detected_ns - round(true_s * S)) / 1e6
            # Onsets are reported at the start of the frame in which the
            # energy rose, so the report is early by up to frame_ms (10 ms).
            assert error_ms <= 10.0, f"onset off by {error_ms:.1f} ms"

    def test_detection_peak_clears_the_noise_floor_by_6_db(self) -> None:
        """No false positives at a 6 dB margin: the weakest true peak is at
        least 2x the strongest non-onset excursion of the strength curve."""
        onsets_s = (0.7, 1.9, 3.3, 4.4)
        signal = _noise_with_transients(seed=0, onsets_s=onsets_s)
        detected = detect_audio_onsets(signal, SAMPLE_RATE)
        assert len(detected) == 4

        # Rebuild the strength curve the detector uses.
        frame_len = round(10.0 * SAMPLE_RATE / 1000.0)
        hop_len = round(2.5 * SAMPLE_RATE / 1000.0)
        frames = np.lib.stride_tricks.sliding_window_view(
            signal * signal, frame_len
        )[::hop_len]
        energy = frames.mean(axis=1)
        floor = max(float(energy.max()) * 1e-10, np.finfo(np.float64).tiny)
        strength = np.diff(np.log10(np.maximum(energy, floor)))

        onset_frames = {int(d / 1e9 * SAMPLE_RATE / hop_len) - 1 for d in detected}
        near_onset = set()
        for frame in onset_frames:
            near_onset.update(range(frame - 4, frame + 5))
        background = [
            strength[k] for k in range(strength.size) if k not in near_onset
        ]
        weakest_true = min(strength[k] for k in onset_frames)
        assert weakest_true >= 2.0 * max(background)

    def test_gain_invariance(self) -> None:
        signal = _noise_with_transients(seed=1)
        quiet = detect_audio_onsets(signal * 0.001, SAMPLE_RATE)
        loud = detect_audio_onsets(signal * 1000.0, SAMPLE_RATE)
        assert quiet == loud

    def test_start_time_offsets_the_result(self) -> None:
        signal = _noise_with_transients(seed=2)
        base = detect_audio_onsets(signal, SAMPLE_RATE)
        shifted = detect_audio_onsets(signal, SAMPLE_RATE, start_time_ns=10 * S)
        assert shifted == [t + 10 * S for t in base]

    def test_pure_noise_yields_no_onsets(self) -> None:
        rng = np.random.default_rng(3)
        signal = rng.normal(0.0, 0.02, int(SAMPLE_RATE * 3))
        assert detect_audio_onsets(signal, SAMPLE_RATE) == []

    def test_silence_yields_no_onsets(self) -> None:
        assert detect_audio_onsets(np.zeros(int(SAMPLE_RATE)), SAMPLE_RATE) == []

    def test_stereo_is_downmixed(self) -> None:
        signal = _noise_with_transients(seed=4)
        stereo = np.stack([signal, signal], axis=1)
        assert detect_audio_onsets(stereo, SAMPLE_RATE) == detect_audio_onsets(
            signal, SAMPLE_RATE
        )

    def test_min_separation_collapses_close_candidates(self) -> None:
        # Two bursts 20 ms apart, with a 200 ms separation floor.
        signal = _noise_with_transients(seed=5, onsets_s=(1.0, 1.02))
        detected = detect_audio_onsets(
            signal, SAMPLE_RATE, min_separation_ms=200.0
        )
        assert len(detected) == 1

    def test_short_input_returns_nothing(self) -> None:
        assert detect_audio_onsets(np.zeros(10), SAMPLE_RATE) == []

    def test_results_are_sorted_and_integral(self) -> None:
        signal = _noise_with_transients(seed=0)
        detected = detect_audio_onsets(signal, SAMPLE_RATE)
        assert detected == sorted(detected)
        assert all(isinstance(t, int) for t in detected)

    @pytest.mark.parametrize(
        "kwargs",
        [{"sample_rate_hz": 0.0}, {"frame_ms": 0.0}, {"hop_ms": -1.0}],
    )
    def test_argument_validation(self, kwargs: dict[str, float]) -> None:
        call: dict[str, float] = {"sample_rate_hz": SAMPLE_RATE}
        call.update(kwargs)
        rate = call.pop("sample_rate_hz")
        with pytest.raises(ValueError):
            detect_audio_onsets(np.zeros(16_000), rate, **call)  # type: ignore[arg-type]

    def test_1d_shape_is_required(self) -> None:
        with pytest.raises(ValueError, match="1-D or 2-D"):
            detect_audio_onsets(np.zeros((2, 2, 2)), SAMPLE_RATE)


def _errors_ms(detected: list[int], onsets_s: tuple[float, ...]) -> list[float]:
    assert len(detected) == len(onsets_s)
    return [abs(d - round(t * S)) / 1e6 for d, t in zip(detected, onsets_s)]


class TestOnsetRefinement:
    ONSETS = (0.7, 1.9, 3.3, 4.4)

    @pytest.mark.parametrize("seed", range(6))
    def test_refinement_is_sub_millisecond(self, seed: int) -> None:
        signal = _noise_with_transients(seed=seed, onsets_s=self.ONSETS)
        refined = detect_audio_onsets(signal, SAMPLE_RATE, refine=True)
        errors = _errors_ms(refined, self.ONSETS)
        assert max(errors) <= 1.0, f"refined onsets off by up to {max(errors):.3f} ms"

    @pytest.mark.parametrize("seed", range(6))
    def test_refinement_beats_the_coarse_detector_by_an_order_of_magnitude(
        self, seed: int
    ) -> None:
        """The quantitative claim, on identical input."""
        signal = _noise_with_transients(seed=seed, onsets_s=self.ONSETS)
        coarse = _errors_ms(detect_audio_onsets(signal, SAMPLE_RATE), self.ONSETS)
        refined = _errors_ms(
            detect_audio_onsets(signal, SAMPLE_RATE, refine=True), self.ONSETS
        )
        coarse_median = float(np.median(coarse))
        refined_median = float(np.median(refined))
        assert refined_median * 10.0 < coarse_median, (
            f"expected >10x improvement, got coarse={coarse_median:.3f} ms "
            f"refined={refined_median:.3f} ms"
        )

    def test_coarse_error_is_frame_quantised_and_refined_error_is_not(self) -> None:
        """Why averaging cannot fix the coarse detector: the error is a bias.

        Every coarse onset lands on a hop boundary, so its error is the
        same systematic value at every onset — averaging N of them
        removes nothing. Refined times are not on the hop grid at all.
        """
        signal = _noise_with_transients(seed=0, onsets_s=self.ONSETS)
        hop_ns = round(2.5 * 1e6)
        coarse = detect_audio_onsets(signal, SAMPLE_RATE)
        assert all(t % hop_ns == 0 for t in coarse)
        refined = detect_audio_onsets(signal, SAMPLE_RATE, refine=True)
        assert any(t % hop_ns != 0 for t in refined)

    def test_refinement_never_changes_the_number_of_onsets(self) -> None:
        signal = _noise_with_transients(seed=2, onsets_s=self.ONSETS)
        coarse = detect_audio_onsets(signal, SAMPLE_RATE)
        refined = detect_audio_onsets(signal, SAMPLE_RATE, refine=True)
        assert len(coarse) == len(refined) == len(self.ONSETS)

    def test_pure_noise_still_yields_no_onsets_when_refining(self) -> None:
        """The A4 non-regression: refinement must not create false positives."""
        rng = np.random.default_rng(3)
        signal = rng.normal(0.0, 0.02, int(SAMPLE_RATE * 3))
        assert detect_audio_onsets(signal, SAMPLE_RATE, refine=True) == []

    def test_silence_still_yields_no_onsets_when_refining(self) -> None:
        assert (
            detect_audio_onsets(np.zeros(int(SAMPLE_RATE)), SAMPLE_RATE, refine=True)
            == []
        )

    def test_refine_onsets_is_a_pure_retiming_pass(self) -> None:
        signal = _noise_with_transients(seed=0, onsets_s=self.ONSETS)
        coarse = detect_audio_onsets(signal, SAMPLE_RATE)
        refined = refine_onsets(signal, SAMPLE_RATE, coarse)
        assert len(refined) == len(coarse)
        assert refined == sorted(refined)
        assert all(isinstance(t, int) for t in refined)

    def test_refinement_respects_start_time(self) -> None:
        signal = _noise_with_transients(seed=1, onsets_s=self.ONSETS)
        base = detect_audio_onsets(signal, SAMPLE_RATE, refine=True)
        shifted = detect_audio_onsets(
            signal, SAMPLE_RATE, start_time_ns=10 * S, refine=True
        )
        assert shifted == [t + 10 * S for t in base]

    def test_empty_candidate_list_stays_empty(self) -> None:
        assert refine_onsets(np.zeros(1000), SAMPLE_RATE, []) == []

    def test_a_candidate_with_no_usable_window_is_returned_unchanged(self) -> None:
        # Window entirely outside a very short signal.
        assert refine_onsets(np.zeros(20), SAMPLE_RATE, [5 * S]) == [5 * S]

    @pytest.mark.parametrize(
        "kwargs", [{"search_ms": 0.0}, {"edge_ms": -1.0}, {"sample_rate_hz": 0.0}]
    )
    def test_argument_validation(self, kwargs: dict[str, float]) -> None:
        call = {"sample_rate_hz": SAMPLE_RATE, **kwargs}
        rate = call.pop("sample_rate_hz")
        with pytest.raises(ValueError):
            refine_onsets(np.zeros(16_000), rate, [0], **call)  # type: ignore[arg-type]


class TestGccPhat:
    def _shift(self, signal: np.ndarray, samples: float) -> np.ndarray:
        """Fractional-sample delay via the Fourier shift theorem."""
        n = signal.size
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n)
        return np.asarray(
            np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * samples), n)
        )

    def test_recovers_an_integer_sample_delay(self) -> None:
        signal = _noise_with_transients(seed=0)
        delayed = self._shift(signal, 200.0)
        estimate = gcc_phat(signal, delayed, SAMPLE_RATE, max_delay_ms=30.0)
        assert abs(estimate - 200.0 / SAMPLE_RATE) < 1e-5

    def test_recovers_a_fractional_sample_delay(self) -> None:
        """The sub-sample claim: parabolic interpolation beats the sample grid."""
        signal = _noise_with_transients(seed=0)
        true_samples = 137.4
        delayed = self._shift(signal, true_samples)
        estimate = gcc_phat(signal, delayed, SAMPLE_RATE, max_delay_ms=30.0)
        error_s = abs(estimate - true_samples / SAMPLE_RATE)
        one_sample_s = 1.0 / SAMPLE_RATE
        assert error_s < 0.25 * one_sample_s, (
            f"error {error_s * 1e6:.1f} us is not sub-sample "
            f"({one_sample_s * 1e6:.1f} us per sample)"
        )

    def test_sign_convention_delayed_lags_reference(self) -> None:
        signal = _noise_with_transients(seed=1)
        delayed = self._shift(signal, 80.0)
        assert gcc_phat(signal, delayed, SAMPLE_RATE, max_delay_ms=30.0) > 0
        assert gcc_phat(delayed, signal, SAMPLE_RATE, max_delay_ms=30.0) < 0

    def test_identical_signals_have_zero_delay(self) -> None:
        signal = _noise_with_transients(seed=2)
        # Not an exact-equality assertion: the correlation peak for identical
        # signals is symmetric about lag 0, so the parabolic sub-sample step
        # should return exactly 0.0 — but the FFT round-trip leaves the two
        # neighbours differing in the last few ulps, which interpolates to a
        # non-zero offset around 1e-25 s. A picosecond bound is ~13 orders of
        # magnitude tighter than anything audio timing can mean and still
        # catches a genuinely mis-centred peak.
        assert abs(gcc_phat(signal, signal, SAMPLE_RATE, max_delay_ms=10.0)) < 1e-12

    def test_interpolation_can_be_disabled(self) -> None:
        signal = _noise_with_transients(seed=0)
        delayed = self._shift(signal, 137.4)
        integral = gcc_phat(
            signal, delayed, SAMPLE_RATE, max_delay_ms=30.0, interpolate=False
        )
        assert abs(integral * SAMPLE_RATE - round(integral * SAMPLE_RATE)) < 1e-9

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            gcc_phat(np.zeros(0), np.zeros(10), SAMPLE_RATE)

    @pytest.mark.parametrize("kwargs", [{"sample_rate_hz": 0.0}, {"max_delay_ms": 0.0}])
    def test_argument_validation(self, kwargs: dict[str, float]) -> None:
        call = {"sample_rate_hz": SAMPLE_RATE, **kwargs}
        rate = call.pop("sample_rate_hz")
        with pytest.raises(ValueError):
            gcc_phat(np.zeros(100), np.zeros(100), rate, **call)  # type: ignore[arg-type]


class TestRefinedClapAlignment:
    def test_refined_onsets_recover_an_offset_far_better_than_coarse(self) -> None:
        """End-to-end precision: the number a rig claiming 5 ms cares about.

        Both trains come from *different* recordings here (independent
        noise), so the coarse detector's frame bias does **not** cancel:
        this is the case where refinement changes the answer rather than
        merely the digits.
        """
        onsets_s = (0.7, 1.9, 3.3, 4.4)
        offset_ns = 250 * MS
        audio_a = _noise_with_transients(seed=0, onsets_s=onsets_s)
        shifted_s = tuple(t + 0.0013 for t in onsets_s)  # 1.3 ms physical skew
        audio_b = _noise_with_transients(seed=7, onsets_s=shifted_s)

        def offset_error(refine: bool) -> float:
            a = detect_audio_onsets(audio_a, SAMPLE_RATE, refine=refine)
            b = [
                t + offset_ns
                for t in detect_audio_onsets(audio_b, SAMPLE_RATE, refine=refine)
            ]
            result = align_clap_events(a, b, max_offset_ms=500.0)
            true_offset = offset_ns + round(0.0013 * S)
            return abs(result.offset_ns - true_offset) / 1e6

        refined_error = offset_error(refine=True)
        assert refined_error <= 1.0, f"refined offset off by {refined_error:.3f} ms"
        assert refined_error < offset_error(refine=False)


class TestClapAlignment:
    def test_detected_onsets_align_to_a_known_offset(self) -> None:
        """The full path: detect on two signals, fit the mapping between them.

        The same detector runs on both sides, so its frame-start bias is
        common-mode and cancels out of the offset — which is exactly the
        property the module docstring claims.
        """
        onsets_s = (0.7, 1.9, 3.3, 4.4)
        signal = _noise_with_transients(seed=0, onsets_s=onsets_s)
        audio_a = detect_audio_onsets(signal, SAMPLE_RATE)
        # Same claps, observed by a clock running 250 ms behind.
        offset_ns = 250 * MS
        audio_b = [t + offset_ns for t in audio_a]

        result = align_clap_events(audio_a, audio_b, max_offset_ms=500.0)
        assert result.offset_ns == offset_ns
        assert len(result.matched) == len(onsets_s)
        assert result.residual_p95_ns == 0

    def test_single_clap_gives_an_offset_with_no_drift_claim(self) -> None:
        result = align_clap_events([1 * S], [1 * S + 33 * MS], max_offset_ms=100.0)
        assert result.offset_ns == 33 * MS
        assert result.drift_ppb == 0
        assert result.fit.mapping.variance_ns > 0  # the "unmeasured" marker

    def test_two_claps_refuse_drift_and_say_so(self) -> None:
        """Two claps determine a line exactly, which is the problem (A7).

        This used to assert that two claps at either end of a 60 s
        recording recover 100 ppm. They do — when the data is noiseless.
        Add any detection jitter and the same two points still fit a line
        exactly, residuals are still identically zero, and every
        residual-derived quality signal still reports perfection. The
        estimator cannot tell the two cases apart from two points, so it
        now declines the drift and names the reason rather than returning
        a slope it cannot stand behind. Three claps is the documented
        minimum; the physical protocol asks for 3-5.
        """
        a = [0, 60 * S]
        drift_ppb = 100_000  # 100 ppm
        b = [t + 20 * MS + round(t * drift_ppb / 1e9) for t in a]
        result = align_clap_events(a, b, max_offset_ms=100.0)
        assert result.drift_ppb == 0
        assert any("drift_needs_more_pairs" in p for p in result.problems)

    def test_four_claps_do_measure_drift(self) -> None:
        """The capability itself is intact above the minimum pair count."""
        a = [0, 20 * S, 40 * S, 60 * S]
        drift_ppb = 100_000
        b = [t + 20 * MS + round(t * drift_ppb / 1e9) for t in a]
        result = align_clap_events(a, b, max_offset_ms=100.0)
        assert result.offset_ns == 20 * MS
        assert result.drift_ppb == drift_ppb
