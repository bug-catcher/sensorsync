"""Deterministic synthetic streams — the "truth harness" for Milestone 1.

Design (see ARCHITECTURE.md, DECISIONS.md D-0004/D-0006). This module is
currently the *designed API*: signatures and contracts are final enough to be
under test, but the generator bodies raise :class:`NotImplementedError` until
the next session implements them (TDD red).

Determinism contract
--------------------
- Same ``(duration_s, seed, start_time_ns)`` → identical output, always.
- All randomness flows through ``numpy.random.Generator(PCG64(child_seed))``
  where child seeds are derived per-stream from
  ``numpy.random.SeedSequence(seed).spawn(len(streams))`` in spec order.
- Clean streams are perfectly regular:
  ``acquisition_time_ns = start_time_ns + round(i * 1e9 / rate_hz)``.
- ``receive_time_ns = acquisition_time_ns + transport_latency_ns`` with a
  *fixed* per-stream constant (D-0006) so that corruption profiles are the
  only source of timing noise.
- ``sequence_id`` is contiguous from 0 per stream.
- Every synthetic sample carries the ``synthetic`` quality flag.
- The ``events`` stream is irregular: exponential inter-arrival times drawn
  from the stream's own child generator (still deterministic per seed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from numpy.random import PCG64, Generator, SeedSequence

from embodied_sync.core.sample import QUALITY_SYNTHETIC, Modality, Sample

NS_PER_S = 1_000_000_000


@dataclass(frozen=True, slots=True)
class SyntheticStreamSpec:
    """Static description of one synthetic stream.

    ``rate_hz`` is None for irregular streams (event markers). ``payload_dim``
    is the length of the deterministic numeric payload vector, or None when
    the payload is scheme-specific (cameras, events).
    """

    name: str
    modality: Modality
    rate_hz: float | None
    transport_latency_ns: int
    payload_dim: int | None = None
    clock_domain: str = "host_mono"


#: The default synthetic rig (Milestone 1). Order matters: child seeds are
#: assigned in this order, so reordering breaks determinism (bump run format
#: version if this tuple ever changes).
DEFAULT_SPECS: tuple[SyntheticStreamSpec, ...] = (
    SyntheticStreamSpec("cam_front", Modality.CAMERA, 30.0, 12_000_000),
    SyntheticStreamSpec("cam_wrist", Modality.CAMERA, 30.0, 15_000_000),
    SyntheticStreamSpec("robot_state", Modality.ROBOT_STATE, 250.0, 1_000_000, payload_dim=7),
    SyntheticStreamSpec("tactile", Modality.TACTILE, 60.0, 2_000_000, payload_dim=16),
    SyntheticStreamSpec("audio", Modality.AUDIO, 50.0, 20_000_000),
    SyntheticStreamSpec("actions", Modality.ACTION, 10.0, 500_000, payload_dim=7),
    SyntheticStreamSpec("events", Modality.EVENT, None, 100_000),
)


_SYNTH_FLAGS = frozenset({QUALITY_SYNTHETIC})

#: Mean inter-arrival time of irregular event markers (seconds).
EVENT_MEAN_INTERVAL_S = 0.3

_EVENT_MARKERS = ("contact", "release", "waypoint")


def _name_offset(name: str) -> float:
    """Deterministic per-stream phase offset. Never use ``hash()`` here — it
    varies with PYTHONHASHSEED and would break cross-process determinism."""
    return 0.01 * sum(ord(c) for c in name)


def _sample_count(rate_hz: float, duration_s: float) -> int:
    """Number of regular samples: sample i exists iff ``i / rate_hz < duration_s``."""
    n = math.floor(duration_s * rate_hz)
    if n / rate_hz < duration_s:
        n += 1
    return n


def _camera_payload(spec: SyntheticStreamSpec, i: int) -> dict[str, Any]:
    """Small deterministic stand-in for an image: index + signature vector.

    Full images are out of scope for the truth harness; alignment only needs
    identity-checkable payloads (payload_ref will carry real frames later).
    """
    off = _name_offset(spec.name)
    return {
        "frame_index": i,
        "signature": [round(math.sin(0.37 * (i + k) + off), 6) for k in range(4)],
    }


def _vector_payload(spec: SyntheticStreamSpec, t_s: float) -> list[float]:
    """Smooth deterministic trajectory: sums of per-dimension sinusoids."""
    dim = spec.payload_dim if spec.payload_dim is not None else 1
    off = _name_offset(spec.name)
    return [
        round(0.5 * math.sin(2.0 * math.pi * (0.1 + 0.05 * j) * t_s + off + j), 9)
        for j in range(dim)
    ]


def _audio_payload(spec: SyntheticStreamSpec, acquisition_time_ns: int, t_s: float) -> dict[str, Any]:
    """RMS energy per window with explicit window bounds (D-0007)."""
    assert spec.rate_hz is not None
    window_ns = round(NS_PER_S / spec.rate_hz)
    rms = round(0.05 + 0.04 * abs(math.sin(2.0 * math.pi * 0.8 * t_s + _name_offset(spec.name))), 9)
    return {
        "window_start_ns": acquisition_time_ns,
        "window_end_ns": acquisition_time_ns + window_ns,
        "rms": rms,
    }


def _regular_payload(spec: SyntheticStreamSpec, i: int, acquisition_time_ns: int) -> Any:
    assert spec.rate_hz is not None
    t_s = i / spec.rate_hz
    if spec.modality is Modality.CAMERA:
        return _camera_payload(spec, i)
    if spec.modality is Modality.AUDIO:
        return _audio_payload(spec, acquisition_time_ns, t_s)
    return _vector_payload(spec, t_s)


def _generate_events(
    spec: SyntheticStreamSpec,
    rng: Generator,
    duration_s: float,
    start_time_ns: int,
) -> list[Sample]:
    """Irregular markers: exponential inter-arrivals in [start, start + duration)."""
    end_ns = start_time_ns + int(round(duration_s * NS_PER_S))
    samples: list[Sample] = []
    t_ns = start_time_ns
    while True:
        dt_ns = max(1, int(round(float(rng.exponential(EVENT_MEAN_INTERVAL_S)) * NS_PER_S)))
        t_ns += dt_ns
        if t_ns >= end_ns:
            break
        marker = _EVENT_MARKERS[int(rng.integers(len(_EVENT_MARKERS)))]
        samples.append(
            Sample(
                stream_name=spec.name,
                modality=spec.modality,
                sequence_id=len(samples),
                acquisition_time_ns=t_ns,
                receive_time_ns=t_ns + spec.transport_latency_ns,
                source_clock_domain=spec.clock_domain,
                payload={"marker": marker, "event_index": len(samples)},
                quality_flags=_SYNTH_FLAGS,
            )
        )
    return samples


def generate_stream(
    spec: SyntheticStreamSpec,
    *,
    duration_s: float,
    child_seed: int,
    start_time_ns: int = 0,
) -> list[Sample]:
    """Generate one clean, deterministic stream according to ``spec``.

    Returns samples ordered by ``acquisition_time_ns`` with contiguous
    ``sequence_id`` starting at 0. Regular streams contain exactly the samples
    with ``i / rate_hz < duration_s``. Irregular (event) streams contain every
    marker whose time falls in ``[start, start + duration)``.

    Regular streams use no randomness at all (D-0006): timing is a perfect
    grid, payloads are pure functions of ``(spec, i)``. Only event streams
    consume ``child_seed``.
    """
    rng = Generator(PCG64(child_seed))
    if spec.rate_hz is None:
        return _generate_events(spec, rng, duration_s, start_time_ns)

    samples: list[Sample] = []
    for i in range(_sample_count(spec.rate_hz, duration_s)):
        acquisition_time_ns = start_time_ns + round(i * NS_PER_S / spec.rate_hz)
        samples.append(
            Sample(
                stream_name=spec.name,
                modality=spec.modality,
                sequence_id=i,
                acquisition_time_ns=acquisition_time_ns,
                receive_time_ns=acquisition_time_ns + spec.transport_latency_ns,
                source_clock_domain=spec.clock_domain,
                payload=_regular_payload(spec, i, acquisition_time_ns),
                quality_flags=_SYNTH_FLAGS,
            )
        )
    return samples


def generate_synthetic_run(
    *,
    duration_s: float = 10.0,
    seed: int = 0,
    start_time_ns: int = 0,
    specs: tuple[SyntheticStreamSpec, ...] = DEFAULT_SPECS,
) -> dict[str, list[Sample]]:
    """Generate the full default rig as ``{stream_name: [Sample, ...]}``.

    Byte-identical output for identical arguments: per-stream child seeds are
    derived from ``SeedSequence(seed).spawn(len(specs))`` in ``specs`` order,
    so the result is independent of which other streams exist only if the
    spec tuple itself is unchanged (see ``DEFAULT_SPECS`` note).
    """
    children = SeedSequence(seed).spawn(len(specs))
    run: dict[str, list[Sample]] = {}
    for spec, child in zip(specs, children):
        child_seed = int(child.generate_state(1)[0])
        run[spec.name] = generate_stream(
            spec,
            duration_s=duration_s,
            child_seed=child_seed,
            start_time_ns=start_time_ns,
        )
    return run
