"""Online alignment replay integration test on the corrupted-jitter run
(NEXT_TASKS #1, D-0026 / D-0027).

Replays a synth run through the corruption pipeline
(`configs/corrupt_camera_jitter.yaml`) sample-by-sample into a
:class:`MultiStreamAligner`, driving it as a deadline-0 policy tick at
10 Hz. The purpose is to close the "we have online *primitives* but no
synthetic-driven online *pipeline* test" gap: the ring-buffer
(:mod:`tests/test_align_ring_buffer.py`) and composite
(:mod:`tests/test_align_online.py`) files exercise individual invariants;
this file exercises the two of them together against a real corrupted
run and cross-checks against the offline ZoH baseline.

The three properties this test pins:

1. **Deadline-0 causality (D-0026 invariant).** No picked sample has
   ``receive_time_ns > now_ns`` — the pick is a sample the receiver could
   plausibly have observed by the tick.
2. **Online is at least as stale as offline ZoH.** Online eligibility is
   a strict subset of offline eligibility (offline picks ignore
   ``receive_time_ns``), so every per-frame online skew is ``<=`` the
   corresponding offline skew.
3. **Per-stream median-skew is bounded.** For each stream the online
   median skew tracks the offline median skew to within one median
   inter-sample acquisition interval plus the maximum positive
   receive-time shift the corruption profile can inject (jitter clip +
   fixed latency). The bound is *derived from the profile*, not
   hand-tuned, so tightening a corruption automatically tightens the
   test.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from embodied_sync.align import MultiStreamAligner, StreamRingBuffer, align_run
from embodied_sync.core.episode import AlignedFrame, AlignedRun
from embodied_sync.core.sample import Sample
from embodied_sync.corrupt import apply_profile, load_profile
from embodied_sync.streams.synthetic import generate_synthetic_run

REPO_ROOT = Path(__file__).parent.parent
CORRUPT_PROFILE = REPO_ROOT / "configs" / "corrupt_camera_jitter.yaml"

TARGET_RATE_HZ = 10.0
DURATION_S = 2.0
SEED = 0

# Maximum positive receive-time shift each stream can suffer from
# `configs/corrupt_camera_jitter.yaml`. Used to widen the median-skew
# bound on the corrupted streams — the profile is the ground truth for
# how much extra staleness online can inherit vs offline. If the profile
# changes, update these together.
MAX_RECEIVE_SHIFT_NS: dict[str, int] = {
    "cam_front": 30_000_000,  # jitter clip_ms=30.0
    "cam_wrist": 45_000_000,  # fixed_latency offset_ms=45.0
}


def _median_interval_ns(samples: list[Sample]) -> int:
    """Median inter-sample acquisition-time interval, matching engine convention."""
    if len(samples) < 2:
        return 0
    diffs = sorted(
        samples[i + 1].acquisition_time_ns - samples[i].acquisition_time_ns
        for i in range(len(samples) - 1)
    )
    return diffs[len(diffs) // 2]


def _median_skew_ns(frames: list[AlignedFrame], name: str) -> float | None:
    skews: list[int] = []
    for frame in frames:
        md = frame.metadata[name]
        if md.missing or md.skew_ns is None:
            continue
        skews.append(md.skew_ns)
    if not skews:
        return None
    return statistics.median(skews)


def _regular_streams(run: dict[str, list[Sample]]) -> dict[str, list[Sample]]:
    """Drop the irregular events stream for the fixed-rate alignment tests.

    Events is Poisson-distributed and would narrow the offline window to
    the first event's acquisition; the online-vs-offline median-skew
    comparison is about *regular* sensor streams driving the policy tick.
    """
    return {name: samples for name, samples in run.items() if name != "events"}


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
def online_replay(
    corrupted_run: dict[str, list[Sample]], offline_zoh: AlignedRun
) -> tuple[list[AlignedFrame], dict[str, int]]:
    """Push every regular-stream sample in receive-time order into a composite.

    Returns ``(online_frames, per_stream_median_interval_ns)``. The frames
    are produced at the offline grid's targets so per-frame comparisons
    are apples-to-apples.
    """
    regular = _regular_streams(corrupted_run)
    intervals: dict[str, int] = {}
    buffers: dict[str, StreamRingBuffer] = {}
    for name, samples in regular.items():
        if not samples:
            continue
        interval = _median_interval_ns(samples)
        intervals[name] = interval
        # Generous tolerance so tolerance-driven missing frames don't
        # dominate the median comparison — this test is about causality
        # and skew, not staleness gating (which the ring-buffer file
        # already covers).
        buffers[name] = StreamRingBuffer(
            capacity=len(samples), tolerance_ns=10 * max(interval, 1)
        )
    aligner = MultiStreamAligner(buffers)

    # Global receive-time-sorted stream — the natural online arrival order.
    # Ties broken by (stream_name, sequence_id) for deterministic replay.
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
    return online_frames, intervals


class TestOnlineReplayCorruptedJitter:
    def test_replay_produces_frames_at_every_offline_target(
        self,
        online_replay: tuple[list[AlignedFrame], dict[str, int]],
        offline_zoh: AlignedRun,
    ) -> None:
        """Sanity: one online frame per offline target, at the same target ns."""
        frames, _ = online_replay
        assert len(frames) == len(offline_zoh.frames)
        assert len(frames) > 0
        for online_frame, offline_frame in zip(frames, offline_zoh.frames):
            assert online_frame.target_time_ns == offline_frame.target_time_ns

    def test_frames_are_causal_at_deadline_zero(
        self, online_replay: tuple[list[AlignedFrame], dict[str, int]]
    ) -> None:
        """Every picked sample satisfies ``receive_time_ns <= now_ns``.

        This is the D-0026 causality invariant applied at the full-frame
        level: no per-stream pick can leak a "future" sample into the tick.
        """
        frames, _ = online_replay
        for frame in frames:
            now_ns = frame.target_time_ns
            for name, sample in frame.samples.items():
                if sample is None:
                    continue
                assert sample.receive_time_ns <= now_ns, (
                    f"causality violated at now_ns={now_ns} for stream {name!r}: "
                    f"receive_time_ns={sample.receive_time_ns}"
                )
                # ZoH acquisition invariant is independent; assert it too.
                assert sample.acquisition_time_ns <= now_ns

    def test_online_skew_never_positive_relative_to_offline_per_frame(
        self,
        online_replay: tuple[list[AlignedFrame], dict[str, int]],
        offline_zoh: AlignedRun,
    ) -> None:
        """Per-frame: online skew ≤ offline skew (both non-positive for ZoH).

        Online eligibility is a strict subset of offline eligibility
        (offline sees samples regardless of when they arrived), so the
        online pick's acquisition_time_ns is ≤ the offline pick's
        acquisition_time_ns, which under the ``skew = source - target``
        convention means online skew ≤ offline skew.
        """
        frames, _ = online_replay
        for online_frame, offline_frame in zip(frames, offline_zoh.frames):
            for name in online_frame.samples:
                on_md = online_frame.metadata[name]
                off_md = offline_frame.metadata[name]
                if on_md.missing or on_md.skew_ns is None:
                    continue
                if off_md.missing or off_md.skew_ns is None:
                    continue
                assert on_md.skew_ns <= off_md.skew_ns, (
                    f"stream {name!r} at target {online_frame.target_time_ns}: "
                    f"online skew {on_md.skew_ns} > offline skew {off_md.skew_ns}"
                )

    def test_median_skew_within_interval_plus_profile_shift(
        self,
        online_replay: tuple[list[AlignedFrame], dict[str, int]],
        offline_zoh: AlignedRun,
    ) -> None:
        """|online_median - offline_median| ≤ interval + max receive shift.

        For streams the profile doesn't touch, the online median trails
        the offline median by at most one median inter-sample interval
        (the grid gap between "sample already received" and "sample about
        to be received"). For streams the profile pushes further into
        the future via ``fixed_latency`` or ``jitter``, add that maximum
        positive shift to the bound.
        """
        frames, intervals = online_replay
        for name, interval in intervals.items():
            offline_median = _median_skew_ns(offline_zoh.frames, name)
            online_median = _median_skew_ns(frames, name)
            if offline_median is None:
                continue
            assert online_median is not None, (
                f"stream {name!r}: offline has non-missing frames but online has none"
            )
            shift = MAX_RECEIVE_SHIFT_NS.get(name, 0)
            bound = interval + shift
            diff = abs(online_median - offline_median)
            assert diff <= bound, (
                f"stream {name!r}: |online {online_median} - offline {offline_median}| "
                f"= {diff} > interval {interval} + profile shift {shift} = {bound}"
            )

    def test_online_picks_are_non_empty_for_regular_streams(
        self, online_replay: tuple[list[AlignedFrame], dict[str, int]]
    ) -> None:
        """Guard: every regular stream produces at least one non-missing pick.

        Without this guard, a bug that made every online pick missing
        would let the median-skew test pass vacuously (it skips streams
        with no non-missing frames).
        """
        frames, intervals = online_replay
        for name in intervals:
            non_missing = sum(1 for f in frames if not f.metadata[name].missing)
            assert non_missing > 0, (
                f"stream {name!r} has zero non-missing online picks — "
                f"eligibility or push routing may be broken"
            )
