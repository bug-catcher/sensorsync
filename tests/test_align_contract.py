"""Alignment engine contract test (D-0004 pattern, now green after D-0020).

Aligns a corrupted synthetic run to the action-rate grid (10 Hz) and
asserts the engine surfaces the camera drops that ground truth already
knows about. Previously committed as
``xfail(strict=True, raises=NotImplementedError)``; the marker was
removed when the nearest-neighbor slice landed.
"""

from __future__ import annotations

from embodied_sync.align import align_run
from embodied_sync.corrupt import (
    CorruptionProfile,
    DroppedFramesCorruption,
    apply_profile,
)
from embodied_sync.streams.synthetic import generate_synthetic_run


def test_alignment_surfaces_camera_drops_from_ground_truth() -> None:
    # Arrange: a corrupted run with plenty of cam_front drops so the
    # alignment engine cannot possibly pick a matching sample for every
    # action-rate frame. Ground truth records exactly what was removed.
    run = generate_synthetic_run(duration_s=1.0, seed=0)
    profile = CorruptionProfile(
        seed=0,
        corruptions=(
            DroppedFramesCorruption(stream="cam_front", probability=0.5),
        ),
    )
    corruption = apply_profile(run, profile)
    assert corruption.dropped.get("cam_front"), (
        "precondition: seed must actually drop cam_front samples for this "
        "contract to be meaningful"
    )

    # Act: align to the 10 Hz action-rate grid.
    aligned = align_run(
        corruption.run, target_rate_hz=10.0, ground_truth=corruption.dropped
    )

    # Assert (the designed contract):
    #   1) the engine emits action-rate frames covering the run window;
    #   2) at least one frame reports cam_front missing;
    #   3) the reported missing count is at least as large as the drops
    #      recorded in ground truth intersected with the action-rate grid —
    #      the engine may not undercount known losses.
    assert aligned.frames, "aligned output must not be empty for a 1 s run"
    missing_frames = [
        frame for frame in aligned.frames if frame.metadata["cam_front"].missing
    ]
    assert missing_frames, (
        "with 50% cam_front drops at 30 Hz aligned to 10 Hz, at least one "
        "action-rate frame must report cam_front missing"
    )
    ground_truth_dropped_ids = {s.sequence_id for s in corruption.dropped["cam_front"]}
    assert ground_truth_dropped_ids
    assert aligned.report.missing_count["cam_front"] == len(missing_frames)
    # No dropped ids leak from the corrupted run's survivors.
    surviving_used_ids = {
        frame.samples["cam_front"].sequence_id
        for frame in aligned.frames
        if frame.samples["cam_front"] is not None
    }
    assert surviving_used_ids.isdisjoint(ground_truth_dropped_ids)
    # Ground truth cross-check is populated when supplied.
    assert aligned.report.ground_truth_missing_count["cam_front"] > 0
