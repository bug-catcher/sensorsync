"""Clap / transient alignment: onset detection plus the event-train fit.

The clap test is the oldest trick in multi-camera capture and it is still
the right one: make a sharp, unambiguous event that every sensor can see,
then measure when each of them says it happened. This module supplies the
two halves — a numpy-only onset detector for audio, and a thin wrapper
that hands the detected times to
:func:`~embodied_sync.calibrate.events.match_event_trains`.

Why a hand-rolled detector rather than librosa: the dependency budget.
Detecting a clap is a half-wave-rectified log-energy difference against a
robust threshold — twenty lines of numpy. Detecting the onset of a
legato cello note is research, and this library does not need it. The
detector is deliberately tuned for *transients*: sharp attack, high
energy contrast, well separated in time.

Visual event times are an **input** here, not something this module
detects. "The gripper contacted the object in frame 412" is domain CV
that belongs in the caller's pipeline; converting frame 412 into a clock
mapping is metrology and belongs here. That boundary is the same one the
whole `calibrate/` subpackage draws.

Reported onset times and their bias
-----------------------------------
An onset is reported at the **start of the analysis frame in which the
energy rose**. The true attack lies somewhere inside that frame, so the
report is early by up to ``frame_ms``, systematically. When both trains
come from this same detector the bias is common-mode and cancels out of
the offset; when one train comes from elsewhere (video contact
detection, an external trigger log) it does **not** cancel, and it sets
a floor on the achievable accuracy. Shrink ``frame_ms``/``hop_ms`` if
that floor matters — at the cost of noise immunity.

Two-stage detection: candidates, then precision (A4)
----------------------------------------------------
The frame-based detector above is good at *deciding whether* a
transient happened and bad at saying *when*. Its resolution is the hop
(2.5 ms by default) and its bias is up to a frame (10 ms), and neither
improves by looking harder at one onset — averaging N of them only
shrinks the random part as ``10/√N`` while leaving the frame bias
exactly where it was. For a rig claiming 5 ms end-to-end that is not a
measurement, it is a rounding error with ambitions.

So detection is split in two, the standard arrangement for any
transient estimator:

1. **Candidate generation** — :func:`detect_audio_onsets`, unchanged.
   Robust, gain-invariant, conservative about false positives, coarse.
2. **Refinement** — :func:`refine_onsets`, which revisits the *raw
   waveform* in a short window around each candidate and locates the
   attack by correlating instantaneous power against a step kernel,
   then interpolating the correlation peak parabolically for sub-sample
   resolution. Typical improvement on planted transients is one to two
   orders of magnitude, from ~10 ms to well under 1 ms.

Refinement never invents onsets: it only moves ones the coarse stage
already accepted, so the false-positive behaviour is exactly the coarse
detector's. ``detect_audio_onsets(..., refine=True)`` runs both.

:func:`gcc_phat` is the other half of the toolkit, for the case where
the same physical event was recorded *twice* — two microphones, a
camera's audio track and a bench recorder. There the delay between the
two waveforms is directly estimable to sub-sample precision without
either signal needing an absolute onset time at all, and PHAT
weighting (whiten the cross-spectrum, keep only phase) is what makes
that robust to the two channels having wildly different frequency
responses. Prefer it whenever both recordings exist; use
:func:`refine_onsets` when one side is an event log rather than a
waveform.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from embodied_sync.calibrate.events import EventTrainAlignment, match_event_trains
from embodied_sync.time.clock_domain import ClockDomain

__all__ = [
    "align_clap_events",
    "detect_audio_onsets",
    "gcc_phat",
    "refine_onsets",
]

#: Robust-scale constant (1.4826·MAD ≈ σ for Gaussian data).
_MAD_TO_SIGMA = 1.4826

#: Frames needed before a threshold means anything.
_MIN_FRAMES = 4

#: Default half-width of the refinement search around a coarse onset. The
#: coarse report can be early by a frame and late by a hop, so ±15 ms
#: comfortably brackets the default 10 ms frame without letting a
#: neighbouring transient into the window.
_DEFAULT_SEARCH_MS = 15.0

#: Default half-width of the step kernel. Long enough to average away the
#: sample-to-sample fluctuation of instantaneous power (which is
#: chi-squared with one degree of freedom, i.e. extremely noisy), short
#: enough that the correlation peak stays sharp.
_DEFAULT_EDGE_MS = 1.0


def _to_mono(waveform: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(waveform, dtype=np.float64)
    if array.ndim == 2:
        # (samples, channels) — average, the standard downmix.
        return array.mean(axis=1)
    if array.ndim != 1:
        raise ValueError(
            f"waveform must be 1-D or 2-D (samples, channels), got shape "
            f"{array.shape}"
        )
    return array


def _parabolic_offset(curve: NDArray[np.float64], peak: int) -> float:
    """Sub-sample offset of a discrete maximum, by parabola through 3 points.

    Fits ``y = a·x² + b·x + c`` through ``curve[peak-1:peak+2]`` and
    returns the vertex position relative to ``peak``, in samples. This
    is the standard sub-sample peak estimator: a correlation peak is
    locally quadratic, and three samples are exactly enough to place its
    vertex between them.

    Returns ``0.0`` at the array edges (no neighbours to fit) and
    whenever the three points do not actually describe a maximum — a
    flat or upward-curving triple would put the vertex outside the
    bracket, which is not a refinement but a jump to somewhere the
    evidence does not point. The result is clamped to ``±0.5`` for the
    same reason.
    """
    if peak <= 0 or peak >= curve.size - 1:
        return 0.0
    y0 = float(curve[peak - 1])
    y1 = float(curve[peak])
    y2 = float(curve[peak + 1])
    denominator = y0 - 2.0 * y1 + y2
    if denominator >= 0.0:
        return 0.0
    offset = 0.5 * (y0 - y2) / denominator
    return max(-0.5, min(0.5, offset))


def gcc_phat(
    reference: Sequence[float] | NDArray[np.float64],
    delayed: Sequence[float] | NDArray[np.float64],
    sample_rate_hz: float,
    *,
    max_delay_ms: float | None = None,
    interpolate: bool = True,
) -> float:
    """Sub-sample delay of ``delayed`` relative to ``reference``, in seconds.

    Generalised cross-correlation with phase transform: take the
    cross-spectrum, divide out its magnitude so only phase survives,
    transform back. A positive result means ``delayed`` lags
    ``reference`` by that many seconds.

    Whitening is what makes this the standard choice rather than plain
    cross-correlation. Two recordings of one clap differ in level, in
    microphone response, and in room colouration; plain correlation
    weights the loudest shared frequencies most, so its peak drifts with
    whichever band the two channels happen to agree on. PHAT gives every
    frequency equal say, which turns a broad, spectrum-dependent hump
    into a sharp spike at the true delay. The price is that PHAT is
    fragile on narrowband or very low-SNR signals, where the phase of
    near-empty bins is noise amplified to unit weight — claps, which are
    broadband and loud, are close to the ideal case for it.

    ``max_delay_ms`` restricts the search to physically plausible lags,
    which is worth setting: an unrestricted search over a periodic-ish
    signal will happily find a taller spike one period away.
    ``interpolate`` adds the parabolic sub-sample step (see
    :func:`_parabolic_offset`); turn it off to get the integer-sample
    answer.

    numpy's FFT only — no scipy (D-0038).
    """
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")
    if max_delay_ms is not None and max_delay_ms <= 0:
        raise ValueError(f"max_delay_ms must be > 0 or None, got {max_delay_ms}")
    a = _to_mono(reference)
    b = _to_mono(delayed)
    if a.size == 0 or b.size == 0:
        raise ValueError("gcc_phat needs non-empty reference and delayed signals")
    # Pad to a power of two at least as long as the linear correlation, so
    # the circular correlation the FFT computes cannot wrap a real peak
    # around into a spurious lag.
    length = 1 << (a.size + b.size - 1).bit_length()
    spectrum_a = np.fft.rfft(a, length)
    spectrum_b = np.fft.rfft(b, length)
    cross = spectrum_b * np.conj(spectrum_a)
    magnitude = np.abs(cross)
    if float(magnitude.max()) <= 0.0:
        return 0.0
    # Whiten. The floor keeps empty bins from dividing by ~0 and being
    # promoted to unit weight alongside the bins that carry the signal.
    floor = np.finfo(np.float64).eps * float(magnitude.max())
    correlation = np.fft.irfft(cross / np.maximum(magnitude, floor), length)

    limit = length // 2
    if max_delay_ms is not None:
        limit = min(limit, max(1, int(round(max_delay_ms * 1e-3 * sample_rate_hz))))
    # Re-centre so index `limit` is zero lag and negative lags precede it.
    centred = np.concatenate([correlation[-limit:], correlation[: limit + 1]])
    peak = int(np.argmax(centred))
    offset = _parabolic_offset(centred, peak) if interpolate else 0.0
    return ((peak - limit) + offset) / sample_rate_hz


def refine_onsets(
    waveform: Sequence[float] | NDArray[np.float64],
    sample_rate_hz: float,
    onsets_ns: Sequence[int],
    *,
    start_time_ns: int = 0,
    search_ms: float = _DEFAULT_SEARCH_MS,
    edge_ms: float = _DEFAULT_EDGE_MS,
) -> list[int]:
    """Re-time coarse onsets against the raw waveform, to sub-sample precision.

    Each time in ``onsets_ns`` is treated as a *candidate* — a claim that
    a transient happened near here, not a claim about when. Around each
    one this function takes a ``±search_ms`` window of the raw samples
    and finds the attack properly:

    1. instantaneous power ``x²`` at full sample resolution (no frames,
       no hop — the quantisation those impose is the thing being fixed);
    2. correlation against a step kernel of half-width ``edge_ms``
       (``−1`` before, ``+1`` after, i.e. "how much more power is there
       just after this instant than just before it"). This is the
       matched filter for an onset, and its peak is the attack;
    3. parabolic interpolation of that peak for a sub-sample position.

    Returns refined times in the same clock as ``onsets_ns``, sorted,
    integer ns. The list is always the same length as the input: this
    stage **never adds or removes onsets**, so the coarse detector's
    false-positive behaviour is preserved exactly — refine a run of pure
    noise and you get the empty list you started with.

    What it does and does not fix. The frame bias and hop quantisation
    of :func:`detect_audio_onsets` are removed, because the answer no
    longer depends on frame boundaries at all. What remains is the
    genuine ambiguity in "when did this clap start" for a transient with
    a finite rise time, plus noise: at usable SNR that is tens of
    microseconds, well inside the sub-millisecond claim. It cannot
    rescue a candidate that was never near a real transient — garbage in
    is garbage relocated to sub-sample precision.

    A candidate whose window falls outside the waveform, or is too short
    to hold the kernel, is returned unchanged rather than dropped: the
    coarse estimate is still the best available answer, and silently
    losing an onset would corrupt the event-train matching downstream.
    """
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")
    if search_ms <= 0:
        raise ValueError(f"search_ms must be > 0, got {search_ms}")
    if edge_ms <= 0:
        raise ValueError(f"edge_ms must be > 0, got {edge_ms}")
    if not onsets_ns:
        return []

    signal = _to_mono(waveform)
    power = signal * signal
    search = max(1, round(search_ms * sample_rate_hz / 1000.0))
    edge = max(1, round(edge_ms * sample_rate_hz / 1000.0))
    kernel = np.concatenate(
        [np.full(edge, -1.0 / edge), np.full(edge, 1.0 / edge)]
    )

    refined: list[int] = []
    for onset_ns in onsets_ns:
        centre = int(round((onset_ns - start_time_ns) / 1e9 * sample_rate_hz))
        low = max(0, centre - search)
        high = min(power.size, centre + search + 1)
        window = power[low:high]
        if window.size < 2 * edge + 3:
            refined.append(int(onset_ns))
            continue
        correlation = np.correlate(window, kernel, mode="valid")
        peak = int(np.argmax(correlation))
        if float(correlation[peak]) <= 0.0:
            # No power step anywhere in the window: nothing to refine
            # against, so the coarse time stands.
            refined.append(int(onset_ns))
            continue
        position = low + peak + edge + _parabolic_offset(correlation, peak)
        refined.append(start_time_ns + round(position / sample_rate_hz * 1e9))
    return sorted(refined)


def detect_audio_onsets(
    waveform: Sequence[float] | NDArray[np.float64],
    sample_rate_hz: float,
    *,
    start_time_ns: int = 0,
    frame_ms: float = 10.0,
    hop_ms: float = 2.5,
    threshold: float = 6.0,
    min_separation_ms: float = 50.0,
    refine: bool = False,
    search_ms: float = _DEFAULT_SEARCH_MS,
    edge_ms: float = _DEFAULT_EDGE_MS,
) -> list[int]:
    """Detect transient onsets, returning integer-ns times.

    :param waveform: mono ``(n,)`` or multi-channel ``(n, channels)``
        samples. Amplitude scale is irrelevant — the detector works on
        log-energy differences, so it is gain-invariant.
    :param sample_rate_hz: sample rate of ``waveform``.
    :param start_time_ns: clock time of ``waveform[0]``. Returned onsets
        are absolute times in that clock, which is what
        :func:`align_clap_events` needs.
    :param frame_ms: analysis frame length. Longer means smoother energy
        and later-biased onsets.
    :param hop_ms: frame step, and therefore the time quantum of the
        result.
    :param threshold: detection threshold in robust sigmas above the
        median of the onset-strength curve. 6 is conservative — it wants
        a clap, not a footstep.
    :param min_separation_ms: minimum gap between reported onsets; the
        stronger of two close candidates wins.
    :param refine: run :func:`refine_onsets` on the detections, moving
        each one to the sub-sample position of the actual attack. This
        is a pure re-timing pass — the *set* of onsets is unchanged, so
        it cannot introduce a false positive — and it removes both the
        hop quantisation and the frame bias described below. Off by
        default so the function's historical output is unchanged; turn
        it on whenever the times matter more than the count.
    :param search_ms: refinement search half-width (see
        :func:`refine_onsets`). Ignored unless ``refine`` is set.
    :param edge_ms: refinement step-kernel half-width. Ignored unless
        ``refine`` is set.

    Returns onset times sorted ascending. An empty list means nothing
    crossed the threshold, which is a legitimate answer and not an error.

    The strength curve is the frame-to-frame difference of log-energy:
    it responds to *relative* energy increases, so a clap over quiet
    room tone and a clap over louder room tone score alike.

    The threshold is ``median + threshold·1.4826·MAD`` of that curve,
    computed on the **signed** differences before any rectification.
    That ordering matters: rectifying first makes half the samples
    exactly zero, which collapses the median and the MAD toward zero
    and turns a 6σ threshold into a noise detector. On the signed
    curve the background is symmetric and well scaled — stationary room
    tone peaks near 4σ while a clap lands beyond 50σ, so the default
    separates them by a wide margin. Median/MAD rather than mean/σ
    because the events being detected are precisely the outliers that
    would inflate a mean.
    """
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")
    if frame_ms <= 0 or hop_ms <= 0:
        raise ValueError(
            f"frame_ms and hop_ms must be > 0, got {frame_ms} and {hop_ms}"
        )
    signal = _to_mono(waveform)
    frame_len = max(1, round(frame_ms * sample_rate_hz / 1000.0))
    hop_len = max(1, round(hop_ms * sample_rate_hz / 1000.0))
    if signal.size < frame_len + hop_len * _MIN_FRAMES:
        return []

    squared = signal * signal
    frames = np.lib.stride_tricks.sliding_window_view(squared, frame_len)[::hop_len]
    energy = frames.mean(axis=1)
    if energy.size < _MIN_FRAMES:
        return []

    # Floor relative to the loudest frame keeps the log gain-invariant and
    # stops digital silence from producing an infinite "rise".
    floor = max(float(energy.max()) * 1e-10, np.finfo(np.float64).tiny)
    log_energy = np.log10(np.maximum(energy, floor))
    strength = np.diff(log_energy)

    centre = float(np.median(strength))
    mad = float(np.median(np.abs(strength - centre)))
    sigma = _MAD_TO_SIGMA * mad
    if sigma <= 0.0:
        # Degenerate curve (a noiseless signal, or digital silence between
        # bursts): there is no background scale to threshold against, so
        # fall back to a fraction of the peak rise.
        peak = float(strength.max())
        if peak <= 0.0:
            return []
        cutoff = peak * 0.5
    else:
        cutoff = centre + threshold * sigma

    candidates = [
        index
        for index in range(strength.size)
        if strength[index] > cutoff
        and (index == 0 or strength[index] >= strength[index - 1])
        and (index == strength.size - 1 or strength[index] >= strength[index + 1])
    ]
    if not candidates:
        return []

    min_separation_frames = max(
        1, int(round(min_separation_ms * sample_rate_hz / 1000.0 / hop_len))
    )
    chosen: list[int] = []
    for index in sorted(candidates, key=lambda k: (-float(strength[k]), k)):
        if all(abs(index - kept) >= min_separation_frames for kept in chosen):
            chosen.append(index)

    ns_per_hop = hop_len / sample_rate_hz * 1e9
    # strength[k] compares frame k+1 against frame k, so the rise is in
    # frame k+1, which starts at sample (k+1)*hop_len.
    coarse = sorted(start_time_ns + round((k + 1) * ns_per_hop) for k in chosen)
    if not refine:
        return coarse
    return refine_onsets(
        signal,
        sample_rate_hz,
        coarse,
        start_time_ns=start_time_ns,
        search_ms=search_ms,
        edge_ms=edge_ms,
    )


def align_clap_events(
    audio_onsets_ns: Sequence[int],
    visual_events_ns: Sequence[int],
    *,
    max_offset_ms: float = 1000.0,
    max_drift_ppm: float = 500.0,
    match_tolerance_ms: float | None = None,
    source_domain: ClockDomain | str | None = None,
    target_domain: ClockDomain | str | None = None,
) -> EventTrainAlignment:
    """Fit the audio-clock → visual-clock mapping from clap events.

    A thin wrapper over
    :func:`~embodied_sync.calibrate.events.match_event_trains`: the
    algorithm is identical, the name exists because "align my claps" is
    the question people actually arrive with. ``audio_onsets_ns`` is the
    source and ``visual_events_ns`` the target, so the resulting mapping
    restates audio times in the video clock.

    With a single clap in each train the result is an offset with
    :data:`~embodied_sync.calibrate.estimator.SINGLE_PAIR_VARIANCE_NS`
    — usable, and explicitly not a drift measurement. Clap at the start
    *and* the end of a recording to get one.
    """
    return match_event_trains(
        audio_onsets_ns,
        visual_events_ns,
        max_offset_ms=max_offset_ms,
        max_drift_ppm=max_drift_ppm,
        match_tolerance_ms=match_tolerance_ms,
        source_domain=source_domain,
        target_domain=target_domain,
    )
