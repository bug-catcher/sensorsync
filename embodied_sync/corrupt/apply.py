"""Apply corruption profiles to runs.

Implemented kinds: ``fixed_latency``, ``jitter``, ``dropped_frames``,
``clock_drift``, ``burst_stall``, ``duplicate_samples``, ``non_monotonic``,
``missing_interval``.

Determinism (D-0010): all corruption randomness derives from the profile
seed; corruption entry *i* uses ``SeedSequence(profile.seed).spawn(n)[i]``
(mirroring the synthetic generator), so a profile applied to the same run
always produces the same output, and entries never share RNG state.

Corruptions never mutate their input: `Sample` is frozen and the run dict /
sample lists are copied. Quality flags are set only where a corruption is
*observable* to a real system: drops (both per-sample-random and
interval-shaped) leave ``gap_before`` on the first surviving sample after a
removed block (sequence-id gaps are observable), duplicates carry
``duplicate`` on the extra copy (sequence-id repetition is observable),
non-monotonic swaps flag the sample of every observed downward step in
``receive_time_ns``; latency, jitter, smooth drift, and burst stalls set no
flags. The unobservable ground truth — exactly which samples were removed
— is returned in :class:`CorruptionResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from numpy.random import PCG64, Generator, SeedSequence

from embodied_sync.core.sample import (
    QUALITY_DUPLICATE,
    QUALITY_GAP_BEFORE,
    QUALITY_NON_MONOTONIC,
    Sample,
)
from embodied_sync.corrupt.profile import (
    BurstStallCorruption,
    ClockDriftCorruption,
    CorruptionProfile,
    DroppedFramesCorruption,
    DuplicateSamplesCorruption,
    FixedLatencyCorruption,
    JitterCorruption,
    MissingIntervalCorruption,
    NonMonotonicCorruption,
)

_PPB_DENOMINATOR = 1_000_000_000


@dataclass(frozen=True, slots=True)
class CorruptionResult:
    """A corrupted run plus the ground truth a real system could not observe.

    ``dropped`` maps stream name to the exact samples removed by
    ``dropped_frames`` corruptions (in original order, accumulated across
    corruption entries), so reports can be validated against known removals.
    """

    run: dict[str, list[Sample]]
    dropped: dict[str, tuple[Sample, ...]] = field(default_factory=dict)


def apply_fixed_latency(samples: list[Sample], offset_ns: int) -> list[Sample]:
    """Add a constant transport delay: ``receive_time_ns += offset_ns``.

    ``acquisition_time_ns`` is untouched (the sensor still observed the world
    at the same instant; only delivery is delayed).
    """
    return [
        replace(sample, receive_time_ns=sample.receive_time_ns + offset_ns)
        for sample in samples
    ]


def apply_jitter(
    samples: list[Sample],
    *,
    std_ns: int,
    clip_ns: int | None,
    rng: Generator,
) -> list[Sample]:
    """Add gaussian noise to ``receive_time_ns`` (acquisition untouched).

    Offsets are ``round(standard_normal() * std_ns)``, then clipped to
    ``[-clip_ns, clip_ns]`` when a clip is set. Offsets may be negative;
    jitter can even push ``receive_time_ns`` before ``acquisition_time_ns``,
    which downstream must treat as a timing anomaly rather than something
    this module hides. Sample order is preserved — out-of-order receive
    times are exactly the corruption alignment has to cope with.
    """
    noise = rng.standard_normal(len(samples))
    jittered: list[Sample] = []
    for sample, gauss in zip(samples, noise):
        offset_ns = round(float(gauss) * std_ns)
        if clip_ns is not None:
            offset_ns = max(-clip_ns, min(clip_ns, offset_ns))
        jittered.append(replace(sample, receive_time_ns=sample.receive_time_ns + offset_ns))
    return jittered


def apply_dropped_frames(
    samples: list[Sample],
    *,
    probability: float,
    rng: Generator,
) -> tuple[list[Sample], tuple[Sample, ...]]:
    """Remove samples independently with ``probability`` each.

    Returns ``(survivors, removed)``. Survivors keep their original
    ``sequence_id`` (the gap is the observable symptom) and the first
    survivor after each removed block gains the ``gap_before`` flag.
    """
    drop_mask = rng.random(len(samples)) < probability
    survivors: list[Sample] = []
    removed: list[Sample] = []
    pending_gap = False
    for sample, drop in zip(samples, drop_mask):
        if drop:
            removed.append(sample)
            pending_gap = True
            continue
        if pending_gap:
            sample = replace(sample, quality_flags=sample.quality_flags | {QUALITY_GAP_BEFORE})
            pending_gap = False
        survivors.append(sample)
    return survivors, tuple(removed)


def apply_clock_drift(samples: list[Sample], *, drift_ppb: int) -> list[Sample]:
    """Apply linear receive-time drift anchored at the first sample.

    Offset is ``round((acquisition_time_ns - anchor_ns) * drift_ppb / 1e9)``.
    Positive drift makes later samples arrive increasingly late; negative
    drift makes them arrive increasingly early. Acquisition timestamps and
    sample order are preserved.
    """
    if not samples:
        return []
    anchor_ns = samples[0].acquisition_time_ns
    drifted: list[Sample] = []
    for sample in samples:
        elapsed_ns = sample.acquisition_time_ns - anchor_ns
        offset_ns = round(elapsed_ns * drift_ppb / _PPB_DENOMINATOR)
        drifted.append(replace(sample, receive_time_ns=sample.receive_time_ns + offset_ns))
    return drifted


def apply_burst_stall(
    samples: list[Sample],
    *,
    count: int,
    stall_ns: int,
    rng: Generator,
) -> list[Sample]:
    """Simulate ``count`` contiguous receive-time stalls of duration ``stall_ns``.

    Release times are drawn deterministically-uniform from
    ``[first_receive + stall_ns, last_receive]`` and processed in ascending
    order. Each stall bumps any sample whose current ``receive_time_ns`` sits
    in ``[release_ns - stall_ns, release_ns)`` to ``release_ns`` — the
    clustered-flush semantic. Acquisition timestamps, sample order, and
    quality flags are preserved; overall ``receive_time_ns`` remains
    monotonic non-decreasing so long as the input is. Sequential composition
    lets overlapping stalls cascade naturally.

    Bursts silently no-op when the input is empty or the stream's
    receive-time window is narrower than a single stall.
    """
    if not samples or count == 0:
        return list(samples)
    first_recv = samples[0].receive_time_ns
    last_recv = samples[-1].receive_time_ns
    if last_recv - first_recv < stall_ns:
        return list(samples)
    release_lo = first_recv + stall_ns
    release_hi = last_recv
    release_times = sorted(
        int(x) for x in rng.integers(release_lo, release_hi + 1, size=count)
    )
    receive_times = [sample.receive_time_ns for sample in samples]
    for release_ns in release_times:
        window_lo = release_ns - stall_ns
        for i, recv in enumerate(receive_times):
            if window_lo <= recv < release_ns:
                receive_times[i] = release_ns
    return [
        replace(sample, receive_time_ns=recv)
        for sample, recv in zip(samples, receive_times)
    ]


def apply_duplicate_samples(
    samples: list[Sample],
    *,
    probability: float,
    rng: Generator,
) -> list[Sample]:
    """Duplicate samples independently with ``probability`` each.

    Each duplicate is inserted immediately after its original and carries the
    original's ``sequence_id``, ``acquisition_time_ns``, ``receive_time_ns``,
    and ``payload`` — a byte-copy retransmission scenario. Sequence-id
    repetition is observable to any recorder tracking sequence ids, so the
    extra copy gains the ``duplicate`` quality flag; the original is left
    unchanged. The originals' order and count in the output preserve the input
    contract for downstream corruptions applied after this one.
    """
    dup_mask = rng.random(len(samples)) < probability
    duplicated: list[Sample] = []
    for sample, duplicate in zip(samples, dup_mask):
        duplicated.append(sample)
        if duplicate:
            duplicated.append(
                replace(sample, quality_flags=sample.quality_flags | {QUALITY_DUPLICATE})
            )
    return duplicated


def apply_missing_interval(
    samples: list[Sample],
    *,
    start_ns: int,
    duration_ns: int,
) -> tuple[list[Sample], tuple[Sample, ...]]:
    """Remove a contiguous acquisition-time window from the stream.

    The window starts at ``samples[0].acquisition_time_ns + start_ns`` and
    has length ``duration_ns`` (half-open ``[start, start + duration)``).
    Every sample whose ``acquisition_time_ns`` sits in the window is
    removed. Survivors keep their original ``sequence_id`` — the gap *is*
    the observable symptom — and the first survivor after each removed
    block gains the ``gap_before`` flag (identical treatment to
    :func:`apply_dropped_frames`).

    Returns ``(survivors, removed)``; ``removed`` preserves the input
    order so ground truth reports can be validated against exact removals.
    Silent no-op when the stream is empty; a window that doesn't intersect
    any sample yields ``(list(samples), ())`` naturally.
    """
    if not samples:
        return [], ()
    anchor_ns = samples[0].acquisition_time_ns
    window_start = anchor_ns + start_ns
    window_end = window_start + duration_ns
    survivors: list[Sample] = []
    removed: list[Sample] = []
    pending_gap = False
    for sample in samples:
        if window_start <= sample.acquisition_time_ns < window_end:
            removed.append(sample)
            pending_gap = True
            continue
        if pending_gap:
            sample = replace(sample, quality_flags=sample.quality_flags | {QUALITY_GAP_BEFORE})
            pending_gap = False
        survivors.append(sample)
    return survivors, tuple(removed)


def apply_non_monotonic(
    samples: list[Sample],
    *,
    count: int,
    rng: Generator,
) -> list[Sample]:
    """Swap ``count`` adjacent receive-time pairs to produce out-of-order delivery.

    Positions are drawn without replacement from ``[0, len(samples) - 1)``
    using the per-entry child RNG and processed in ascending order; each
    position ``i`` swaps ``receive_time_ns`` between samples ``i`` and
    ``i + 1`` in place. Acquisition timestamps, sequence ids, order in the
    list, and payloads are preserved.

    After all swaps, the flag pass scans the output: every sample ``i > 0``
    with ``receive_time_ns[i] < receive_time_ns[i - 1]`` gains
    ``non_monotonic`` — the observation any recorder tracking receive-time
    monotonicity would make. Non-overlapping swap positions produce exactly
    ``count`` flagged samples; overlapping positions cascade, which may
    yield fewer flagged samples than ``count`` (the observation, not the
    intent, is what gets marked).

    Silent no-op when the stream cannot support the request: fewer than 2
    samples, ``count == 0``, or ``count > len(samples) - 1``.
    """
    n = len(samples)
    if n < 2 or count == 0 or count > n - 1:
        return list(samples)
    positions = sorted(int(x) for x in rng.choice(n - 1, size=count, replace=False))
    receives = [sample.receive_time_ns for sample in samples]
    for i in positions:
        receives[i], receives[i + 1] = receives[i + 1], receives[i]
    swapped: list[Sample] = []
    prev_receive: int | None = None
    for sample, receive in zip(samples, receives):
        flags = sample.quality_flags
        if prev_receive is not None and receive < prev_receive:
            flags = flags | {QUALITY_NON_MONOTONIC}
        swapped.append(replace(sample, receive_time_ns=receive, quality_flags=flags))
        prev_receive = receive
    return swapped


def apply_profile(
    run: dict[str, list[Sample]],
    profile: CorruptionProfile,
) -> CorruptionResult:
    """Apply ``profile`` to a copy of ``run``; corruptions apply in order.

    Fails loudly: a corruption targeting a stream absent from the run raises
    ``KeyError`` (a typo in a profile must not silently do nothing).
    """
    corrupted = {name: list(samples) for name, samples in run.items()}
    dropped: dict[str, list[Sample]] = {}
    children = SeedSequence(profile.seed).spawn(len(profile.corruptions))
    for corruption, child in zip(profile.corruptions, children):
        if corruption.stream not in corrupted:
            raise KeyError(
                f"corruption targets stream {corruption.stream!r}, which is not in the run "
                f"(streams: {sorted(corrupted)})"
            )
        rng = Generator(PCG64(int(child.generate_state(1)[0])))
        samples = corrupted[corruption.stream]
        if isinstance(corruption, FixedLatencyCorruption):
            corrupted[corruption.stream] = apply_fixed_latency(samples, corruption.offset_ns)
        elif isinstance(corruption, JitterCorruption):
            corrupted[corruption.stream] = apply_jitter(
                samples, std_ns=corruption.std_ns, clip_ns=corruption.clip_ns, rng=rng
            )
        elif isinstance(corruption, DroppedFramesCorruption):
            survivors, removed = apply_dropped_frames(
                samples, probability=corruption.probability, rng=rng
            )
            corrupted[corruption.stream] = survivors
            dropped.setdefault(corruption.stream, []).extend(removed)
        elif isinstance(corruption, ClockDriftCorruption):
            corrupted[corruption.stream] = apply_clock_drift(
                samples, drift_ppb=corruption.drift_ppb
            )
        elif isinstance(corruption, BurstStallCorruption):
            corrupted[corruption.stream] = apply_burst_stall(
                samples, count=corruption.count, stall_ns=corruption.stall_ns, rng=rng
            )
        elif isinstance(corruption, DuplicateSamplesCorruption):
            corrupted[corruption.stream] = apply_duplicate_samples(
                samples, probability=corruption.probability, rng=rng
            )
        elif isinstance(corruption, NonMonotonicCorruption):
            corrupted[corruption.stream] = apply_non_monotonic(
                samples, count=corruption.count, rng=rng
            )
        elif isinstance(corruption, MissingIntervalCorruption):
            survivors, removed = apply_missing_interval(
                samples,
                start_ns=corruption.start_ns,
                duration_ns=corruption.duration_ns,
            )
            corrupted[corruption.stream] = survivors
            dropped.setdefault(corruption.stream, []).extend(removed)
        else:  # future kinds must fail loudly, not silently no-op
            raise NotImplementedError(f"corruption kind {corruption.kind!r} is not implemented")
    return CorruptionResult(
        run=corrupted,
        dropped={name: tuple(samples) for name, samples in dropped.items()},
    )
