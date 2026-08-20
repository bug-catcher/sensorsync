"""Alignment engine — offline NN + ZoH + linear interpolation (Milestone 3).

Aligns a run to a fixed-rate ``target_time_ns`` grid. Three policies are
supported in v0: nearest-neighbor, zero-order hold (ZoH), and linear
interpolation. Every aligned frame carries per-stream metadata (source
time, skew, method, missing flag, confidence) and the returned
:class:`AlignmentReport` summarises missing samples and, when ground
truth is supplied, cross-checks against corruption drops.

Design contract
---------------
Frame grid: aligned to the world-time origin (period multiples of ``0``),
then clipped to the intersection of streams' acquisition-time windows —
``start = max(s[0].acquisition_time_ns for s in run)`` and
``end   = min(s[-1].acquisition_time_ns for s in run)``. This is the
"only align frames every stream could plausibly cover" rule; a sparse
event stream that starts late narrows the window rather than being
resampled.

Per-stream tolerance: half the median inter-sample acquisition-time
interval when >= 2 samples exist, else half the target period.

Skew convention: ``skew_ns = source_time_ns - target_time_ns``. Positive
skew means the source is in the future relative to the target; negative
means the source is stale. For ZoH, skew is always ``<= 0``. For
``linear_interp`` successful interpolation the metadata reports
``skew_ns = 0`` (the value is synthesized *at* the target time).

Methods (v0):

- ``"nearest_neighbor"``: pick the sample with the smallest
  ``|acquisition_time_ns - target|``. Missing if ``|skew| > tolerance``.
  Confidence is ``1.0 - |skew| / tolerance``.
- ``"zoh"`` (zero-order hold): pick the most recent sample whose
  ``acquisition_time_ns <= target`` (the "hold last value" semantic).
  Missing if no such sample exists (target precedes every sample) or
  the sample is older than tolerance. Confidence is
  ``1.0 - (target - source) / tolerance``.
- ``"linear_interp"`` (D-0025): linearly interpolate the numeric
  payloads of the two nearest samples straddling the target. Skips
  streams whose payloads are not numeric vectors — those streams fall
  back to ZoH for the entire run with a one-shot warning; per-frame
  edge cases (no bracketing pair, or a bracket anchor beyond tolerance)
  also fall back to ZoH within a linear_interp stream. Interpolated
  samples are synthesized at ``target_time_ns`` and carry the
  ``interpolated`` quality flag — see
  :data:`~embodied_sync.core.sample.QUALITY_INTERPOLATED` — to
  distinguish them from fallback picks. ``metadata.method`` stays
  ``"linear_interp"`` for every frame of a requested-``linear_interp``
  stream (interp *or* fallback); the quality flag is the honest signal
  for "was this frame actually interpolated?".

Window aggregation and online ring buffers are future NEXT_TASKS.

Clock domains: this slice assumes a single shared domain (the synth
harness uses ``"host_mono"`` everywhere). Cross-domain lowering of
confidence is deferred until ``embodied_sync.time`` lands.
"""

from __future__ import annotations

import bisect
import statistics
import warnings
from typing import Literal, Mapping, Union

import numpy as np
from numpy.typing import NDArray

from embodied_sync.core.episode import (
    AlignedFrame,
    AlignedRun,
    AlignedSampleMetadata,
    AlignmentReport,
)
from embodied_sync.core.policy import AlignmentPolicy
from embodied_sync.core.sample import QUALITY_INTERPOLATED, Sample
from embodied_sync.time.alignment import cross_domain_confidence_factor
from embodied_sync.time.clock_domain import LatencyEstimate, translate_ns

NEAREST_NEIGHBOR = "nearest_neighbor"
ZERO_ORDER_HOLD = "zoh"
LINEAR_INTERPOLATION = "linear_interp"
Method = Literal["nearest_neighbor", "zoh", "linear_interp"]
_KNOWN_METHODS: tuple[Method, ...] = ("nearest_neighbor", "zoh", "linear_interp")

#: Accepted shapes for the ``method`` argument to :func:`align_run`.
#: A single string picks the same policy for every stream (the pre-
#: Milestone-3 shape); a mapping picks per stream and may mix
#: ``Method`` string values with :class:`AlignmentPolicy` values so
#: callers can also override tolerance.
MethodArg = Union[Method, Mapping[str, "Method | AlignmentPolicy"]]

# Re-exported for callers that historically imported the data types from
# ``embodied_sync.align.engine``. The canonical home is
# ``embodied_sync.core.episode`` (D-0024).
__all__ = [
    "AlignedFrame",
    "AlignedRun",
    "AlignedSampleMetadata",
    "AlignmentReport",
    "LINEAR_INTERPOLATION",
    "Method",
    "MethodArg",
    "NEAREST_NEIGHBOR",
    "ZERO_ORDER_HOLD",
    "align_run",
    "aggregate_window",
]


def _median_skew_ns_by_stream(
    frames: list[AlignedFrame],
    stream_names: list[str],
) -> dict[str, int | None]:
    """Signed median of per-frame ``skew_ns`` for each stream.

    Ignores frames flagged missing (skew is meaningless there) and
    frames whose ``skew_ns`` is ``None``. Returns ``None`` for a stream
    whose every frame is missing. Same computation the sync-quality
    report's ``median_skew_ns`` column uses; lifted onto
    :class:`AlignmentReport` so downstream tools reading a loaded
    :class:`AlignedRun` can read the signed direction without
    recomputing.
    """
    result: dict[str, int | None] = {}
    for name in stream_names:
        skews: list[int] = []
        for frame in frames:
            md = frame.metadata.get(name)
            if md is None or md.missing or md.skew_ns is None:
                continue
            skews.append(md.skew_ns)
        result[name] = int(statistics.median(skews)) if skews else None
    return result


def _median_interval_ns(samples: list[Sample]) -> int:
    """Median inter-sample acquisition-time interval (0 for < 2 samples)."""
    if len(samples) < 2:
        return 0
    diffs = sorted(
        samples[i + 1].acquisition_time_ns - samples[i].acquisition_time_ns
        for i in range(len(samples) - 1)
    )
    return diffs[len(diffs) // 2]


def _nearest_index(acq_times: list[int], target_ns: int) -> int:
    """Index of the sample with acquisition time closest to ``target_ns``."""
    idx = bisect.bisect_left(acq_times, target_ns)
    if idx == 0:
        return 0
    if idx == len(acq_times):
        return len(acq_times) - 1
    left_skew = target_ns - acq_times[idx - 1]
    right_skew = acq_times[idx] - target_ns
    return idx if right_skew < left_skew else idx - 1


def _zoh_index(acq_times: list[int], target_ns: int) -> int | None:
    """Index of the most recent sample with ``acquisition_time_ns <= target_ns``.

    Returns ``None`` if every sample is strictly newer than ``target_ns``
    (ZoH is undefined before the first sample).
    """
    idx = bisect.bisect_right(acq_times, target_ns)
    if idx == 0:
        return None
    return idx - 1


def _numeric_payload(payload: object) -> list[float] | None:
    """Return ``payload`` coerced to a list of floats if interpolable, else ``None``.

    Scalars become one-element lists so a stream of scalar readings
    (e.g. audio RMS) interpolates as easily as a vector. ``bool`` is
    explicitly rejected — it is an ``int`` subclass in Python but a
    categorical value in every domain we care about.
    """
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


def _stream_supports_interpolation(samples: list[Sample]) -> bool:
    """True iff every sample's payload is a numeric vector of the same length.

    Non-numeric or heterogeneous-shape streams cannot be safely
    interpolated in v0 (D-0025). We check every sample, not just the
    first, so a mid-stream payload shift can't silently break
    interpolation later.
    """
    if not samples:
        return False
    first = _numeric_payload(samples[0].payload)
    if first is None:
        return False
    length = len(first)
    for s in samples[1:]:
        v = _numeric_payload(s.payload)
        if v is None or len(v) != length:
            return False
    return True


def _payload_matrix(samples: list[Sample]) -> NDArray[np.float64]:
    """Numeric payloads as a 2-D float matrix for a known-interpolable stream."""
    return np.asarray([_numeric_payload(s.payload) for s in samples], dtype=float)


def _interpolate(
    samples: list[Sample],
    acq_times: list[int],
    payloads: NDArray[np.float64],
    target_ns: int,
    tolerance: int,
    interp_max_gap: int,
) -> tuple[Sample, float] | None:
    """Interpolate a synthetic sample at ``target_ns`` between bracketing anchors.

    Two gates:

    - ``bracket exists``: a sample strictly before and one at-or-after
      ``target_ns`` (target inside ``[first_acq, last_acq]``).
    - ``max(left_gap, right_gap) <= interp_max_gap``: the bracket is at
      most one median inter-sample interval wide, so we're not
      interpolating across a big drop or missing window.

    Returns ``(interpolated_sample, confidence)`` on success, ``None``
    otherwise (caller falls back to ZoH).

    Confidence peaks at ``1.0`` when the target sits on one of the
    anchors (``min_gap == 0``) and decays to ``0.0`` at the halfway
    point (``min_gap == tolerance``), mirroring the NN convention that
    "distance to the nearest anchor" drives trust.
    """
    idx_right = bisect.bisect_right(acq_times, target_ns)
    if idx_right == 0 or idx_right == len(acq_times):
        return None
    idx_left = idx_right - 1
    left = samples[idx_left]
    right = samples[idx_right]
    left_gap = target_ns - left.acquisition_time_ns
    right_gap = right.acquisition_time_ns - target_ns
    if tolerance <= 0 or interp_max_gap <= 0:
        return None
    if max(left_gap, right_gap) > interp_max_gap:
        return None
    a = payloads[idx_left]
    b = payloads[idx_right]
    span = left_gap + right_gap
    t = 0.0 if span == 0 else left_gap / span
    payload = (a + (b - a) * t).tolist()
    flags = left.quality_flags | frozenset({QUALITY_INTERPOLATED})
    interp_sample = Sample(
        stream_name=left.stream_name,
        modality=left.modality,
        sequence_id=left.sequence_id,
        acquisition_time_ns=target_ns,
        receive_time_ns=target_ns,
        source_clock_domain=left.source_clock_domain,
        payload=payload,
        quality_flags=flags,
    )
    min_gap = min(left_gap, right_gap)
    confidence = max(0.0, 1.0 - min_gap / tolerance)
    return interp_sample, confidence


def _resolve_methods(
    method: MethodArg,
    stream_names: list[str],
) -> tuple[dict[str, Method], dict[str, int | None]]:
    """Normalise ``method`` into per-stream (method, tolerance-override).

    Returns ``(methods, tolerance_overrides)``. ``tolerance_overrides[name]``
    is ``None`` when the caller wants the engine's derived default.
    """
    methods: dict[str, Method] = {}
    tolerance_overrides: dict[str, int | None] = {name: None for name in stream_names}
    if isinstance(method, str):
        if method not in _KNOWN_METHODS:
            raise ValueError(
                f"unknown alignment method {method!r}; "
                f"known methods: {list(_KNOWN_METHODS)}"
            )
        for name in stream_names:
            methods[name] = method
        return methods, tolerance_overrides

    unknown = set(method) - set(stream_names)
    if unknown:
        raise ValueError(
            f"per-stream method mapping references unknown streams: "
            f"{sorted(unknown)}; known streams: {stream_names}"
        )
    for name in stream_names:
        entry = method.get(name, NEAREST_NEIGHBOR)
        if isinstance(entry, AlignmentPolicy):
            picked = entry.method
            if picked not in _KNOWN_METHODS:
                raise ValueError(
                    f"stream {name!r}: unknown method {picked!r}; "
                    f"known methods: {list(_KNOWN_METHODS)}"
                )
            methods[name] = picked
            tolerance_overrides[name] = entry.tolerance_ns
        else:
            if entry not in _KNOWN_METHODS:
                raise ValueError(
                    f"stream {name!r}: unknown method {entry!r}; "
                    f"known methods: {list(_KNOWN_METHODS)}"
                )
            methods[name] = entry
    return methods, tolerance_overrides


def align_run(
    run: dict[str, list[Sample]],
    *,
    target_rate_hz: float,
    method: MethodArg = "nearest_neighbor",
    ground_truth: dict[str, tuple[Sample, ...]] | None = None,
    clock_map: Mapping[str, LatencyEstimate] | None = None,
) -> AlignedRun:
    """Align ``run`` to a ``target_rate_hz`` grid.

    ``method`` selects the per-sample policy: ``"nearest_neighbor"`` picks
    the closest sample by ``|skew|`` (D-0020); ``"zoh"`` picks the most
    recent sample whose ``acquisition_time_ns <= target`` (D-0022);
    ``"linear_interp"`` interpolates numeric payloads between the two
    bracketing samples, falling back to ZoH per-stream (non-numeric
    payloads, with a warning) or per-frame (target outside the
    bracketing window or an anchor beyond tolerance) as needed
    (D-0025).

    A dict mapping stream name to method selects per stream — either
    ``str`` values (``"zoh"`` etc.) or :class:`AlignmentPolicy` values
    (also carries a tolerance override). Streams missing from the dict
    default to ``"nearest_neighbor"``.

    ``clock_map`` provides per-stream
    :class:`~embodied_sync.time.LatencyEstimate`s. When present, the
    aligner translates that stream's acquisition timestamps into the
    target domain before scoring (``translate_ns``) and multiplies the
    per-frame confidence by
    :func:`~embodied_sync.time.cross_domain_confidence_factor` — a
    high-variance mapping visibly weakens trust without hiding the
    frame. Streams without a mapping are treated as already in the
    target domain (identity translation, factor ``1.0``).

    Returns an :class:`AlignedRun` whose ``frames`` cover the intersection
    of the streams' acquisition-time windows on the world-time grid, and
    whose ``report`` summarises per-stream missing counts (plus a ground
    truth cross-check when ``ground_truth`` is provided).
    """
    if target_rate_hz <= 0.0:
        raise ValueError(f"target_rate_hz must be > 0, got {target_rate_hz!r}")

    period_ns = round(1e9 / target_rate_hz)
    if period_ns <= 0:
        raise ValueError(
            f"target_rate_hz={target_rate_hz!r} rounds to a non-positive period"
        )

    stream_names = list(run.keys())
    per_stream_methods, tolerance_overrides = _resolve_methods(method, stream_names)
    missing_count: dict[str, int] = {name: 0 for name in stream_names}

    clock_map = dict(clock_map or {})
    unknown_clock_streams = set(clock_map) - set(stream_names)
    if unknown_clock_streams:
        raise ValueError(
            f"clock_map references unknown streams: "
            f"{sorted(unknown_clock_streams)}; known streams: {stream_names}"
        )
    translated: dict[str, list[int]] = {}
    for name in stream_names:
        samples = run[name]
        mapping = clock_map.get(name)
        if mapping is None:
            translated[name] = [s.acquisition_time_ns for s in samples]
        else:
            translated[name] = [translate_ns(s.acquisition_time_ns, mapping) for s in samples]

    empty_median: dict[str, int | None] = {name: None for name in stream_names}
    non_empty = {name: samples for name, samples in run.items() if samples}
    if not non_empty:
        return AlignedRun(
            frames=[],
            report=AlignmentReport(
                missing_count=missing_count,
                median_skew_ns=empty_median,
            ),
        )

    window_start = max(translated[name][0] for name in non_empty)
    window_end = min(translated[name][-1] for name in non_empty)
    if window_end < window_start:
        return AlignedRun(
            frames=[],
            report=AlignmentReport(
                missing_count=missing_count,
                median_skew_ns=empty_median,
            ),
        )

    # First frame >= window_start on the world-time grid (multiples of period_ns from 0).
    first_frame = -(-window_start // period_ns) * period_ns
    if first_frame > window_end:
        return AlignedRun(
            frames=[],
            report=AlignmentReport(
                missing_count=missing_count,
                median_skew_ns=empty_median,
            ),
        )
    frame_targets = list(range(first_frame, window_end + 1, period_ns))

    per_stream: dict[str, tuple[list[Sample], list[int], int, int]] = {}
    for name in stream_names:
        samples = run[name]
        acq_times = translated[name]
        if samples:
            if len(acq_times) < 2:
                interval_ns = 0
            else:
                diffs = sorted(
                    acq_times[i + 1] - acq_times[i] for i in range(len(acq_times) - 1)
                )
                interval_ns = diffs[len(diffs) // 2]
            default_tolerance = (
                interval_ns // 2 if interval_ns > 0 else period_ns // 2
            )
            # Interp gate: bracket must be at most one median interval wide,
            # so we don't interpolate across drops or missing windows.
            interp_max_gap = interval_ns if interval_ns > 0 else period_ns
        else:
            default_tolerance = period_ns // 2
            interp_max_gap = period_ns
        override = tolerance_overrides.get(name)
        tolerance = override if override is not None else default_tolerance
        per_stream[name] = (samples, acq_times, tolerance, interp_max_gap)

    # For linear_interp: decide once per stream whether payloads are
    # numeric. Streams that can't be interpolated fall back to ZoH for
    # the whole run with a one-shot warning (D-0025).
    interp_streams: set[str] = set()
    payload_matrices: dict[str, NDArray[np.float64]] = {}
    for name in stream_names:
        if per_stream_methods[name] != LINEAR_INTERPOLATION:
            continue
        samples = run[name]
        if not samples:
            continue
        if _stream_supports_interpolation(samples):
            interp_streams.add(name)
            payload_matrices[name] = _payload_matrix(samples)
        else:
            warnings.warn(
                f"linear_interp: stream {name!r} has non-numeric payloads; "
                "falling back to zoh for this stream",
                stacklevel=2,
            )

    def _missing_meta(stream_method: str) -> AlignedSampleMetadata:
        return AlignedSampleMetadata(
            source_time_ns=None,
            skew_ns=None,
            method=stream_method,
            missing=True,
            confidence=0.0,
        )

    def _domain_factor(name: str, tolerance: int) -> float:
        mapping = clock_map.get(name)
        if mapping is None:
            return 1.0
        return cross_domain_confidence_factor(mapping, tolerance)

    frames: list[AlignedFrame] = []
    for target in frame_targets:
        samples_at_frame: dict[str, Sample | None] = {}
        metadata: dict[str, AlignedSampleMetadata] = {}
        for name in stream_names:
            samples, acq_times, tolerance, interp_max_gap = per_stream[name]
            stream_method = per_stream_methods[name]
            if not samples:
                samples_at_frame[name] = None
                metadata[name] = _missing_meta(stream_method)
                missing_count[name] += 1
                continue

            # Try linear interpolation first when applicable.
            if stream_method == LINEAR_INTERPOLATION and name in interp_streams:
                interp = _interpolate(
                    samples,
                    acq_times,
                    payload_matrices[name],
                    target,
                    tolerance,
                    interp_max_gap,
                )
                if interp is not None:
                    interp_sample, confidence = interp
                    samples_at_frame[name] = interp_sample
                    metadata[name] = AlignedSampleMetadata(
                        source_time_ns=target,
                        skew_ns=0,
                        method=LINEAR_INTERPOLATION,
                        missing=False,
                        confidence=confidence * _domain_factor(name, tolerance),
                    )
                    continue
                # else: fall through to ZoH for this frame.

            # NN, ZoH, or linear_interp per-frame/per-stream fallback.
            use_nn = stream_method == NEAREST_NEIGHBOR
            if use_nn:
                idx: int | None = _nearest_index(acq_times, target)
            else:
                # Explicit "zoh" or fallback from "linear_interp".
                idx = _zoh_index(acq_times, target)

            if idx is None:
                samples_at_frame[name] = None
                metadata[name] = _missing_meta(stream_method)
                missing_count[name] += 1
                continue

            sample = samples[idx]
            translated_acq = acq_times[idx]
            skew = translated_acq - target
            skew_for_tolerance = abs(skew) if use_nn else -skew
            is_missing = tolerance == 0 or skew_for_tolerance > tolerance
            if is_missing:
                samples_at_frame[name] = None
                metadata[name] = _missing_meta(stream_method)
                missing_count[name] += 1
            else:
                base_confidence = max(0.0, 1.0 - skew_for_tolerance / tolerance)
                confidence = base_confidence * _domain_factor(name, tolerance)
                samples_at_frame[name] = sample
                metadata[name] = AlignedSampleMetadata(
                    source_time_ns=translated_acq,
                    skew_ns=skew,
                    method=stream_method,
                    missing=False,
                    confidence=confidence,
                )
        frames.append(
            AlignedFrame(target_time_ns=target, samples=samples_at_frame, metadata=metadata)
        )

    ground_truth_missing_count: dict[str, int] = {}
    if ground_truth:
        first_target = frame_targets[0]
        last_target = frame_targets[-1]
        for name, dropped in ground_truth.items():
            entry = per_stream.get(name, ([], [], period_ns // 2, period_ns))
            tolerance_for_check = entry[2]
            lo = first_target - tolerance_for_check
            hi = last_target + tolerance_for_check
            mapping = clock_map.get(name)
            if mapping is None:
                ground_truth_missing_count[name] = sum(
                    1 for s in dropped if lo <= s.acquisition_time_ns <= hi
                )
            else:
                ground_truth_missing_count[name] = sum(
                    1
                    for s in dropped
                    if lo <= translate_ns(s.acquisition_time_ns, mapping) <= hi
                )

    return AlignedRun(
        frames=frames,
        report=AlignmentReport(
            missing_count=missing_count,
            ground_truth_missing_count=ground_truth_missing_count,
            median_skew_ns=_median_skew_ns_by_stream(frames, stream_names),
        ),
    )


def aggregate_window(
    samples: list[Sample],
    *,
    target_ns: int,
    window_ns: int,
    reducer: Literal["mean", "median", "last"] = "mean",
) -> tuple[list[float] | None, int]:
    """Aggregate a numeric stream over a look-back window ending at ``target_ns``.

    A companion to the per-sample aligners for streams where "the value
    right now" is better summarised by "the average value over the last
    ``window_ns``". Used, for example, to smooth a jittery force reading
    into a single scalar per policy tick.

    ``samples`` must be a single stream's samples sorted by
    ``acquisition_time_ns`` — usually the value ``run[stream_name]``
    from a saved run. Eligibility: ``target_ns - window_ns <=
    acquisition_time_ns <= target_ns``. The returned value is the
    reducer applied element-wise across the eligible samples' numeric
    payloads (they must all share the same numeric-vector shape); the
    second element is the count of eligible samples.

    Non-numeric or heterogeneous-shape samples are skipped. If no
    eligible sample has a numeric payload, returns ``(None, 0)``.

    Reducers:

    - ``"mean"`` — element-wise arithmetic mean.
    - ``"median"`` — element-wise median (odd-count exact, even-count
      average of the two middles, mirroring :func:`statistics.median`).
    - ``"last"`` — the numeric payload of the newest eligible sample.
    """
    if window_ns <= 0:
        raise ValueError(f"window_ns must be > 0, got {window_ns}")
    if reducer not in ("mean", "median", "last"):
        raise ValueError(
            f"unknown reducer {reducer!r}; known: 'mean', 'median', 'last'"
        )
    low = target_ns - window_ns
    eligible_payloads: list[list[float]] = []
    for sample in samples:
        acq = sample.acquisition_time_ns
        if acq < low:
            continue
        if acq > target_ns:
            break
        payload = _numeric_payload(sample.payload)
        if payload is None:
            continue
        if eligible_payloads and len(payload) != len(eligible_payloads[0]):
            continue
        eligible_payloads.append(payload)
    if not eligible_payloads:
        return None, 0
    dim = len(eligible_payloads[0])
    if reducer == "last":
        result = list(eligible_payloads[-1])
    elif reducer == "mean":
        n = len(eligible_payloads)
        result = [sum(row[k] for row in eligible_payloads) / n for k in range(dim)]
    else:  # median
        result = [
            statistics.median(row[k] for row in eligible_payloads) for k in range(dim)
        ]
    return result, len(eligible_payloads)
