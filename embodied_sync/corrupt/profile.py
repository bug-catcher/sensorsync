"""Corruption profile schema v0: YAML loading and strict validation.

Schema (D-0009)::

    format_version: 0        # required, must be 0
    seed: 1234                # required int; drives all corruption randomness
    corruptions:              # required list, applied in order
      - stream: cam_wrist
        kind: fixed_latency
        offset_ms: 45.0       # added to receive_time_ns
      - stream: cam_front
        kind: jitter
        distribution: gaussian
        std_ms: 8.0
        clip_ms: 30.0         # optional
      - stream: cam_front
        kind: dropped_frames
        probability: 0.02     # per-sample drop probability in [0, 1]
      - stream: robot_state
        kind: clock_drift
        drift_ppm: 100.0      # receive_time_ns drift per elapsed acquisition time
      - stream: cam_front
        kind: burst_stall
        count: 3              # number of stall events; positive int
        stall_ms: 80.0        # each stall duration in ms; > 0
      - stream: cam_front
        kind: duplicate_samples
        probability: 0.01     # per-sample duplication probability in [0, 1]
      - stream: cam_front
        kind: non_monotonic
        count: 2              # number of adjacent-pair receive-time swaps; positive int
      - stream: robot_state
        kind: missing_interval
        start_ms: 100.0       # >= 0; offset from stream's first acquisition_time_ns
        duration_ms: 40.0     # > 0; length of the removed contiguous window

Profiles are human-authored, so times are milliseconds and drift is ppm
(floats allowed). Parsing converts those to integer nanoseconds / ppb
immediately; everything downstream stays on the integer-ns contract (D-0002).
Unknown keys and unknown kinds are hard errors — fail loudly over silent
assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, NoReturn

import yaml

PROFILE_FORMAT_VERSION = 0

_MS_PER_NS = 1_000_000
_PPB_PER_PPM = 1_000


class ProfileError(ValueError):
    """A corruption profile failed schema validation."""


@dataclass(frozen=True, slots=True)
class FixedLatencyCorruption:
    """Constant transport delay: ``receive_time_ns += offset_ns``."""

    kind: ClassVar[str] = "fixed_latency"
    stream: str
    offset_ns: int


@dataclass(frozen=True, slots=True)
class JitterCorruption:
    """Random gaussian noise on ``receive_time_ns``."""

    kind: ClassVar[str] = "jitter"
    stream: str
    distribution: str
    std_ns: int
    clip_ns: int | None


@dataclass(frozen=True, slots=True)
class DroppedFramesCorruption:
    """Independent per-sample drops with fixed probability."""

    kind: ClassVar[str] = "dropped_frames"
    stream: str
    probability: float


@dataclass(frozen=True, slots=True)
class ClockDriftCorruption:
    """Linear receive-time drift relative to the stream's first sample."""

    kind: ClassVar[str] = "clock_drift"
    stream: str
    drift_ppb: int


@dataclass(frozen=True, slots=True)
class BurstStallCorruption:
    """Contiguous receive-time stalls clustering delivery at burst release."""

    kind: ClassVar[str] = "burst_stall"
    stream: str
    count: int
    stall_ns: int


@dataclass(frozen=True, slots=True)
class DuplicateSamplesCorruption:
    """Independent per-sample duplication with fixed probability."""

    kind: ClassVar[str] = "duplicate_samples"
    stream: str
    probability: float


@dataclass(frozen=True, slots=True)
class NonMonotonicCorruption:
    """Adjacent-pair receive-time swaps producing out-of-order delivery."""

    kind: ClassVar[str] = "non_monotonic"
    stream: str
    count: int


@dataclass(frozen=True, slots=True)
class MissingIntervalCorruption:
    """Removes a contiguous acquisition-time window from the target stream."""

    kind: ClassVar[str] = "missing_interval"
    stream: str
    start_ns: int
    duration_ns: int


Corruption = (
    FixedLatencyCorruption
    | JitterCorruption
    | DroppedFramesCorruption
    | ClockDriftCorruption
    | BurstStallCorruption
    | DuplicateSamplesCorruption
    | NonMonotonicCorruption
    | MissingIntervalCorruption
)

_KNOWN_KINDS = (
    "fixed_latency",
    "jitter",
    "dropped_frames",
    "clock_drift",
    "burst_stall",
    "duplicate_samples",
    "non_monotonic",
    "missing_interval",
)


@dataclass(frozen=True, slots=True)
class CorruptionProfile:
    """A validated corruption profile. ``corruptions`` apply in order."""

    seed: int
    corruptions: tuple[Corruption, ...]


def _fail(context: str, message: str) -> NoReturn:
    raise ProfileError(f"{context}: {message}")


def _as_real(value: object) -> float | None:
    """Return a float for int/float inputs (bools rejected), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _ms_to_ns(value_ms: float) -> int:
    return round(value_ms * _MS_PER_NS)


def _ppm_to_ppb(value_ppm: float) -> int:
    return round(value_ppm * _PPB_PER_PPM)


def _require_ms(entry: dict[str, Any], key: str, context: str) -> int:
    value = _as_real(entry[key])
    if value is None:
        _fail(context, f"{key!r} must be a number (milliseconds), got {entry[key]!r}")
    return _ms_to_ns(value)


def _check_keys(entry: dict[str, Any], required: set[str], optional: set[str], context: str) -> None:
    keys = set(entry)
    base = {"stream", "kind"}
    missing = (base | required) - keys
    if missing:
        _fail(context, f"missing required key(s): {sorted(missing)}")
    unknown = keys - base - required - optional
    if unknown:
        _fail(context, f"unknown key(s): {sorted(unknown)}")


def _parse_corruption(entry: object, index: int) -> Corruption:
    context = f"corruptions[{index}]"
    if not isinstance(entry, dict):
        _fail(context, f"must be a mapping, got {type(entry).__name__}")
    stream = entry.get("stream")
    if not isinstance(stream, str) or not stream:
        _fail(context, f"'stream' must be a non-empty string, got {stream!r}")
    kind = entry.get("kind")
    if kind not in _KNOWN_KINDS:
        _fail(context, f"unknown corruption kind {kind!r}; known kinds: {list(_KNOWN_KINDS)}")

    if kind == "fixed_latency":
        _check_keys(entry, {"offset_ms"}, set(), context)
        return FixedLatencyCorruption(stream=stream, offset_ns=_require_ms(entry, "offset_ms", context))

    if kind == "jitter":
        _check_keys(entry, {"distribution", "std_ms"}, {"clip_ms"}, context)
        distribution = entry["distribution"]
        if distribution != "gaussian":
            _fail(context, f"unsupported jitter distribution {distribution!r} (v0 supports 'gaussian')")
        std_ns = _require_ms(entry, "std_ms", context)
        if std_ns <= 0:
            _fail(context, "'std_ms' must be > 0")
        clip_ns = _require_ms(entry, "clip_ms", context) if "clip_ms" in entry else None
        if clip_ns is not None and clip_ns <= 0:
            _fail(context, "'clip_ms' must be > 0")
        return JitterCorruption(stream=stream, distribution=distribution, std_ns=std_ns, clip_ns=clip_ns)

    if kind == "dropped_frames":
        _check_keys(entry, {"probability"}, set(), context)
        probability = _as_real(entry["probability"])
        if probability is None or not 0.0 <= probability <= 1.0:
            _fail(
                context,
                f"'probability' must be a number in [0, 1], got {entry['probability']!r}",
            )
        return DroppedFramesCorruption(stream=stream, probability=probability)

    if kind == "clock_drift":
        _check_keys(entry, {"drift_ppm"}, set(), context)
        drift_ppm = _as_real(entry["drift_ppm"])
        if drift_ppm is None or drift_ppm == 0.0:
            _fail(context, f"'drift_ppm' must be a non-zero number, got {entry['drift_ppm']!r}")
        return ClockDriftCorruption(stream=stream, drift_ppb=_ppm_to_ppb(drift_ppm))

    if kind == "burst_stall":
        _check_keys(entry, {"count", "stall_ms"}, set(), context)
        count_raw = entry["count"]
        if not isinstance(count_raw, int) or isinstance(count_raw, bool) or count_raw <= 0:
            _fail(context, f"'count' must be a positive int, got {count_raw!r}")
        stall_ns = _require_ms(entry, "stall_ms", context)
        if stall_ns <= 0:
            _fail(context, "'stall_ms' must be > 0")
        return BurstStallCorruption(stream=stream, count=count_raw, stall_ns=stall_ns)

    if kind == "duplicate_samples":
        _check_keys(entry, {"probability"}, set(), context)
        probability = _as_real(entry["probability"])
        if probability is None or not 0.0 <= probability <= 1.0:
            _fail(
                context,
                f"'probability' must be a number in [0, 1], got {entry['probability']!r}",
            )
        return DuplicateSamplesCorruption(stream=stream, probability=probability)

    if kind == "non_monotonic":
        _check_keys(entry, {"count"}, set(), context)
        count_raw = entry["count"]
        if not isinstance(count_raw, int) or isinstance(count_raw, bool) or count_raw <= 0:
            _fail(context, f"'count' must be a positive int, got {count_raw!r}")
        return NonMonotonicCorruption(stream=stream, count=count_raw)

    _check_keys(entry, {"start_ms", "duration_ms"}, set(), context)
    start_val = _as_real(entry["start_ms"])
    if start_val is None or start_val < 0.0:
        _fail(context, f"'start_ms' must be a number >= 0, got {entry['start_ms']!r}")
    duration_val = _as_real(entry["duration_ms"])
    if duration_val is None or duration_val <= 0.0:
        _fail(
            context,
            f"'duration_ms' must be a number > 0, got {entry['duration_ms']!r}",
        )
    return MissingIntervalCorruption(
        stream=stream,
        start_ns=_ms_to_ns(start_val),
        duration_ns=_ms_to_ns(duration_val),
    )


def parse_profile(data: object) -> CorruptionProfile:
    """Validate a decoded YAML document and return a :class:`CorruptionProfile`."""
    if not isinstance(data, dict):
        _fail("profile", f"top level must be a mapping, got {type(data).__name__}")
    unknown = set(data) - {"format_version", "seed", "corruptions"}
    if unknown:
        _fail("profile", f"unknown top-level key(s): {sorted(unknown)}")
    version = data.get("format_version")
    if version != PROFILE_FORMAT_VERSION:
        _fail(
            "profile",
            f"unsupported format_version {version!r} "
            f"(this build reads version {PROFILE_FORMAT_VERSION})",
        )
    seed = data.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        _fail("profile", f"'seed' must be an int, got {seed!r}")
    corruptions = data.get("corruptions")
    if not isinstance(corruptions, list):
        _fail("profile", f"'corruptions' must be a list, got {type(corruptions).__name__}")
    parsed = tuple(_parse_corruption(entry, i) for i, entry in enumerate(corruptions))
    return CorruptionProfile(seed=seed, corruptions=parsed)


def load_profile(path: str | Path) -> CorruptionProfile:
    """Load and validate a corruption profile YAML file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data: object = yaml.safe_load(f)
    return parse_profile(data)
