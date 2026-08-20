"""Online-replay stress case: offline-style tolerance meets base transport
latency (NEXT_TASKS #5).

Companion to ``tests/test_align_online_replay.py``. That file deliberately
uses a *generous* per-stream tolerance (``10 * median_interval``) so
tolerance-driven missing frames don't dominate the median-skew comparison.
This file runs the same replay with the **offline-style** tolerance
(``interval // 2``, the same rule
:func:`~embodied_sync.align.align_run` derives per stream) and exercises
what the generous test avoids: the interaction between staleness gating
and base transport latency.

Why this is a stress case, not a baseline. Offline ZoH at a target grid
sees every sample regardless of ``receive_time_ns``, so an aligned-grid
target that coincides with a sample's ``acquisition_time_ns`` picks that
sample with zero skew. Online ZoH at ``deadline_ns == 0`` cannot see a
sample until ``receive_time_ns <= target_ns`` — so at any target the
freshest eligible sample is at least one ``transport_latency_ns`` old,
and may be up to one full source period older if the target falls just
before the next arrival. On the default 250 Hz robot_state stream with
1 ms base latency, this alone puts online staleness at ~4 ms even when
offline reports zero missing — and offline-style tolerance is 2 ms.

The test therefore pins two invariants:

1. **Monotonicity.** For every stream,
   ``online_missing_rate >= offline_missing_rate``. Staleness gating
   never *reduces* missing counts.
2. **Non-trivial stress.** At least one stream must show
   ``online_missing_rate`` substantially higher than
   ``offline_missing_rate`` on today's rig — otherwise the test is
   vacuous and offline-style tolerance would be a safe online default,
   contradicting `docs/concepts/online_vs_offline_alignment.md`.

Additionally, a "guaranteed incompatible" bucket is asserted at ~100%
missing for streams whose *minimum* profile-shifted online staleness
already exceeds tolerance (``cam_wrist`` at 60 ms shift under 16 ms
tolerance today). This documents the failure mode from
`online_vs_offline_alignment.md` in an executable form.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from embodied_sync.align import MultiStreamAligner, StreamRingBuffer, align_run
from embodied_sync.core.episode import AlignedFrame, AlignedRun
from embodied_sync.core.sample import Sample
from embodied_sync.corrupt import apply_profile, load_profile
from embodied_sync.streams.synthetic import DEFAULT_SPECS, generate_synthetic_run

REPO_ROOT = Path(__file__).parent.parent
CORRUPT_PROFILE = REPO_ROOT / "configs" / "corrupt_camera_jitter.yaml"

TARGET_RATE_HZ = 10.0
DURATION_S = 2.0
SEED = 0

# Per-stream ``[min, max]`` extra positive shift that
# ``configs/corrupt_camera_jitter.yaml`` can add to ``receive_time_ns``.
# ``fixed_latency`` contributes a constant shift (min == max). ``jitter``
# is Gaussian with a clip: the min is 0 (or negative, absorbed as 0 for
# the staleness bound), the max is the clip.
PROFILE_RECEIVE_SHIFT_RANGE_NS: dict[str, tuple[int, int]] = {
    "cam_front": (0, 30_000_000),  # jitter clip_ms=30.0 (stochastic)
    "cam_wrist": (45_000_000, 45_000_000),  # fixed_latency offset_ms=45.0
}

# A stream is "meaningfully stressed" if online missing exceeds offline
# missing by at least this fraction of frames. Ensures the stress case is
# doing observable work vs. the generous-tolerance baseline.
STRESS_DELTA = 0.5

# Minimum online missing_rate required from a "guaranteed incompatible"
# stream. Slack below 1.0 covers the very last few grid targets on a
# 2 s replay where the buffer may not yet be primed.
GUARANTEED_INCOMPATIBLE_FLOOR = 0.95


def _base_transport_latency_ns(name: str) -> int:
    """Look up the synth harness's baseline transport latency for ``name``.

    Derived from :data:`DEFAULT_SPECS` — not hardcoded — so a change to
    the default rig moves the compatibility split automatically.
    """
    for spec in DEFAULT_SPECS:
        if spec.name == name:
            return spec.transport_latency_ns
    raise KeyError(name)


def _median_interval_ns(samples: list[Sample]) -> int:
    if len(samples) < 2:
        return 0
    diffs = sorted(
        samples[i + 1].acquisition_time_ns - samples[i].acquisition_time_ns
        for i in range(len(samples) - 1)
    )
    return diffs[len(diffs) // 2]


def _regular_streams(run: dict[str, list[Sample]]) -> dict[str, list[Sample]]:
    return {name: samples for name, samples in run.items() if name != "events"}


def _missing_rate_online(frames: list[AlignedFrame], name: str) -> float:
    if not frames:
        return 0.0
    missing = sum(1 for f in frames if f.metadata[name].missing)
    return missing / len(frames)


def _missing_rate_offline(aligned: AlignedRun, name: str) -> float:
    if not aligned.frames:
        return 0.0
    return aligned.report.missing_count.get(name, 0) / len(aligned.frames)


def _min_online_shift_ns(name: str) -> int:
    """Minimum ``receive_time - acquisition_time`` this stream can produce.

    The freshest online-eligible sample at target ``T`` has
    ``receive_time <= T``, i.e. ``acquisition_time <= T - shift``. So the
    online staleness is *at least* ``min_online_shift`` regardless of
    grid alignment or jitter draw.
    """
    base = _base_transport_latency_ns(name)
    lo, _ = PROFILE_RECEIVE_SHIFT_RANGE_NS.get(name, (0, 0))
    return base + lo


@pytest.fixture(scope="module")
def corrupted_run() -> dict[str, list[Sample]]:
    clean = generate_synthetic_run(duration_s=DURATION_S, seed=SEED)
    profile = load_profile(CORRUPT_PROFILE)
    return apply_profile(clean, profile).run


@pytest.fixture(scope="module")
def offline_zoh(corrupted_run: dict[str, list[Sample]]) -> AlignedRun:
    return align_run(
        _regular_streams(corrupted_run),
        target_rate_hz=TARGET_RATE_HZ,
        method="zoh",
    )


@pytest.fixture(scope="module")
def online_tight_replay(
    corrupted_run: dict[str, list[Sample]], offline_zoh: AlignedRun
) -> tuple[list[AlignedFrame], dict[str, int]]:
    """Replay with ``tolerance_ns = interval // 2`` (offline-style).

    Returns ``(online_frames, per_stream_tolerance_ns)``. The tolerance
    dict is what the assertions in this file quote when explaining a
    failure — the exact number that decided each stream's category.
    """
    regular = _regular_streams(corrupted_run)
    tolerances: dict[str, int] = {}
    buffers: dict[str, StreamRingBuffer] = {}
    for name, samples in regular.items():
        if not samples:
            continue
        interval = _median_interval_ns(samples)
        tolerance = max(interval // 2, 1)
        tolerances[name] = tolerance
        buffers[name] = StreamRingBuffer(
            capacity=len(samples), tolerance_ns=tolerance
        )
    aligner = MultiStreamAligner(buffers)

    all_samples: list[Sample] = []
    for name in buffers:
        all_samples.extend(regular[name])
    all_samples.sort(
        key=lambda s: (s.receive_time_ns, s.stream_name, s.sequence_id)
    )

    online_frames: list[AlignedFrame] = []
    idx = 0
    for offline_frame in offline_zoh.frames:
        target = offline_frame.target_time_ns
        while idx < len(all_samples) and all_samples[idx].receive_time_ns <= target:
            aligner.push(all_samples[idx])
            idx += 1
        online_frames.append(aligner.get_latest_policy_frame(now_ns=target))
    return online_frames, tolerances


class TestOnlineReplayTightToleranceStress:
    def test_replay_produces_frames_at_every_offline_target(
        self,
        online_tight_replay: tuple[list[AlignedFrame], dict[str, int]],
        offline_zoh: AlignedRun,
    ) -> None:
        """Sanity: one online frame per offline target, at the same target ns."""
        frames, _ = online_tight_replay
        assert len(frames) == len(offline_zoh.frames) > 0
        for online_frame, offline_frame in zip(frames, offline_zoh.frames):
            assert online_frame.target_time_ns == offline_frame.target_time_ns

    def test_online_missing_rate_is_monotone_in_staleness(
        self,
        online_tight_replay: tuple[list[AlignedFrame], dict[str, int]],
        offline_zoh: AlignedRun,
    ) -> None:
        """For every stream, ``online_missing_rate >= offline_missing_rate``.

        Online eligibility is a strict subset of offline eligibility, so
        staleness gating can only *add* missing frames, never remove
        them. This invariant holds independently of tolerance choice —
        it just becomes the tightest statement we can make when
        tolerance is set tight enough to bite.
        """
        frames, tolerances = online_tight_replay
        for name in tolerances:
            online_missing = _missing_rate_online(frames, name)
            offline_missing = _missing_rate_offline(offline_zoh, name)
            assert online_missing >= offline_missing - 1e-9, (
                f"stream {name!r}: online missing_rate {online_missing:.3f} "
                f"is *below* offline {offline_missing:.3f}. Staleness "
                f"gating should be monotone; a bug in eligibility likely."
            )

    def test_guaranteed_incompatible_streams_are_almost_all_missing(
        self,
        online_tight_replay: tuple[list[AlignedFrame], dict[str, int]],
        offline_zoh: AlignedRun,
    ) -> None:
        """Streams whose min online shift already exceeds tolerance must
        report near-100% missing.

        ``min_online_shift`` is a lower bound on staleness that holds
        regardless of grid alignment or jitter draw. When it already
        exceeds tolerance, no eligible sample can ever be inside the
        tolerance window at any target — the tight-tolerance failure
        mode from `docs/concepts/online_vs_offline_alignment.md`.
        """
        frames, tolerances = online_tight_replay
        incompatible = [
            name
            for name, tol in tolerances.items()
            if _min_online_shift_ns(name) > tol
        ]
        assert incompatible, (
            "No stream in the default rig is guaranteed incompatible "
            "under offline-style tolerance — this test cannot exercise "
            "the failure mode it exists to document. Check "
            "PROFILE_RECEIVE_SHIFT_RANGE_NS and the synth defaults."
        )
        for name in incompatible:
            online_missing = _missing_rate_online(frames, name)
            offline_missing = _missing_rate_offline(offline_zoh, name)
            assert online_missing >= GUARANTEED_INCOMPATIBLE_FLOOR, (
                f"stream {name!r} is guaranteed incompatible "
                f"(min shift {_min_online_shift_ns(name)} > tolerance "
                f"{tolerances[name]}) but online missing_rate "
                f"{online_missing:.3f} < floor "
                f"{GUARANTEED_INCOMPATIBLE_FLOOR:.2f}; a stale pick "
                f"is being admitted (offline was {offline_missing:.3f})"
            )

    def test_stress_is_non_vacuous(
        self,
        online_tight_replay: tuple[list[AlignedFrame], dict[str, int]],
        offline_zoh: AlignedRun,
    ) -> None:
        """At least one stream must be meaningfully worse online than offline.

        Guards against a future refactor that makes offline-style
        tolerance safe online (which would be a silent, load-bearing
        change to the concept doc's causality argument). The stress
        case exists to document a real, observable gap.
        """
        frames, tolerances = online_tight_replay
        stressed: list[tuple[str, float, float]] = []
        for name in tolerances:
            online_missing = _missing_rate_online(frames, name)
            offline_missing = _missing_rate_offline(offline_zoh, name)
            if online_missing - offline_missing >= STRESS_DELTA:
                stressed.append((name, offline_missing, online_missing))
        assert stressed, (
            f"no stream shows online missing_rate at least {STRESS_DELTA:.2f} "
            f"above offline on the default rig + profile; the stress case "
            f"is vacuous. Either the harness collapsed the receive/acquisition "
            f"latency budget, or the online engine started tolerating stale "
            f"picks. See docs/concepts/online_vs_offline_alignment.md."
        )
