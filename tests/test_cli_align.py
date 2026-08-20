"""CLI tests: `embsync align` writes an on-disk aligned episode.

Covers the full pipeline: synth → save_run → corrupt (with ground truth
sidecar) → align → save_episode → load_episode round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodied_sync.align import align_run
from embodied_sync.cli.main import main
from embodied_sync.core import AlignmentPolicy
from embodied_sync.corrupt import (
    CorruptionProfile,
    DroppedFramesCorruption,
    apply_profile,
)
from embodied_sync.datasets.io import (
    load_episode,
    load_run,
    save_corruption_ground_truth,
    save_run,
)
from embodied_sync.streams.synthetic import generate_synthetic_run


def _save_clean(tmp_path: Path) -> Path:
    clean_dir = tmp_path / "clean"
    save_run(generate_synthetic_run(duration_s=1.0, seed=0), clean_dir)
    return clean_dir


def _save_corrupted_with_gt(tmp_path: Path) -> Path:
    clean = generate_synthetic_run(duration_s=1.0, seed=0)
    profile = CorruptionProfile(
        seed=0,
        corruptions=(DroppedFramesCorruption(stream="cam_front", probability=0.5),),
    )
    result = apply_profile(clean, profile)
    corr_dir = tmp_path / "bad"
    save_run(result.run, corr_dir)
    save_corruption_ground_truth(result.dropped, corr_dir)
    return corr_dir


def test_align_writes_loadable_episode_and_reports_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_dir = _save_clean(tmp_path)
    out = tmp_path / "aligned"
    assert (
        main(["align", str(clean_dir), "--out", str(out), "--target-rate-hz", "10.0"])
        == 0
    )
    aligned = load_episode(out)
    assert aligned.frames, "1 s run at 10 Hz must produce frames"
    # Load direct in-memory reference and compare fully.
    expected = align_run(load_run(clean_dir), target_rate_hz=10.0)
    assert aligned == expected

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["type"] == "aligned_episode"
    assert manifest["target_rate_hz"] == 10.0
    assert manifest["target_period_ns"] == 100_000_000
    assert manifest["frame_count"] == len(aligned.frames)
    assert manifest["source_run"] == str(clean_dir)
    stdout = capsys.readouterr().out
    assert "aligned frames" in stdout
    assert "missing per stream" in stdout


def test_align_with_ground_truth_populates_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corr_dir = _save_corrupted_with_gt(tmp_path)
    out = tmp_path / "aligned"
    assert (
        main(
            [
                "align",
                str(corr_dir),
                "--out",
                str(out),
                "--target-rate-hz",
                "10.0",
                "--check-ground-truth",
            ]
        )
        == 0
    )
    aligned = load_episode(out)
    assert aligned.report.ground_truth_missing_count["cam_front"] > 0


def test_align_check_ground_truth_missing_sidecar_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A clean run has no ground-truth sidecar; --check-ground-truth must
    # emit a stderr warning and still succeed (empty cross-check).
    clean_dir = _save_clean(tmp_path)
    out = tmp_path / "aligned"
    assert (
        main(
            [
                "align",
                str(clean_dir),
                "--out",
                str(out),
                "--target-rate-hz",
                "10.0",
                "--check-ground-truth",
            ]
        )
        == 0
    )
    err = capsys.readouterr().err
    assert "no corruption_ground_truth.json" in err or "corruption_ground_truth" in err
    aligned = load_episode(out)
    assert aligned.report.ground_truth_missing_count == {}


def test_align_refuses_non_empty_out_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_dir = _save_clean(tmp_path)
    out = tmp_path / "aligned"
    out.mkdir()
    (out / "existing.txt").write_text("x", encoding="utf-8")
    assert (
        main(["align", str(clean_dir), "--out", str(out), "--target-rate-hz", "10.0"])
        == 1
    )
    err = capsys.readouterr().err
    assert "non-empty" in err


def test_align_unknown_run_dir_fails_loudly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "aligned"
    assert (
        main(
            [
                "align",
                str(tmp_path / "does_not_exist"),
                "--out",
                str(out),
                "--target-rate-hz",
                "10.0",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "not a run directory" in err or "does_not_exist" in err


def test_align_negative_rate_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_dir = _save_clean(tmp_path)
    out = tmp_path / "aligned"
    assert (
        main(["align", str(clean_dir), "--out", str(out), "--target-rate-hz", "-1.0"])
        == 1
    )
    err = capsys.readouterr().err
    assert "target_rate_hz" in err


def test_align_method_zoh_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_dir = _save_clean(tmp_path)
    out = tmp_path / "aligned"
    assert (
        main(
            [
                "align",
                str(clean_dir),
                "--out",
                str(out),
                "--target-rate-hz",
                "10.0",
                "--method",
                "zoh",
            ]
        )
        == 0
    )
    aligned = load_episode(out)
    for frame in aligned.frames:
        for md in frame.metadata.values():
            assert md.method == "zoh"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["method"] == "zoh"


def test_align_method_linear_interp_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI accepts ``--method linear_interp`` and echoes it into the manifest."""
    clean_dir = _save_clean(tmp_path)
    out = tmp_path / "aligned"
    # Non-numeric streams in the synth rig (cameras, audio, events) fall
    # back to ZoH with a one-shot warning per stream (D-0025); filter
    # them here so the test output stays clean.
    with pytest.warns(UserWarning, match="linear_interp"):
        assert (
            main(
                [
                    "align",
                    str(clean_dir),
                    "--out",
                    str(out),
                    "--target-rate-hz",
                    "20.0",
                    "--method",
                    "linear_interp",
                ]
            )
            == 0
        )
    aligned = load_episode(out)
    for frame in aligned.frames:
        for md in frame.metadata.values():
            assert md.method == "linear_interp"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["method"] == "linear_interp"


def test_align_alignment_policy_json_echoes_manifest(tmp_path: Path) -> None:
    clean_dir = _save_clean(tmp_path)
    out = tmp_path / "aligned"
    raw_policy = json.dumps(
        {
            "cam_front": "zoh",
            "robot_state": {
                "method": "linear_interp",
                "tolerance_ns": 5_000_000,
            },
        }
    )
    assert (
        main(
            [
                "align",
                str(clean_dir),
                "--out",
                str(out),
                "--target-rate-hz",
                "10.0",
                "--alignment-policy",
                raw_policy,
            ]
        )
        == 0
    )

    expected_policy = {
        "cam_front": "zoh",
        "robot_state": {
            "method": "linear_interp",
            "tolerance_ns": 5_000_000,
        },
    }
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 0
    assert manifest["method"] == "nearest_neighbor"
    assert manifest["alignment_policy"] == expected_policy

    loaded = load_episode(out)
    assert loaded.report.alignment_policy == expected_policy
    expected = align_run(
        load_run(clean_dir),
        target_rate_hz=10.0,
        method={
            "cam_front": "zoh",
            "robot_state": AlignmentPolicy(
                method="linear_interp",
                tolerance_ns=5_000_000,
            ),
        },
    )
    assert loaded == expected


def test_align_method_default_is_nearest_neighbor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_dir = _save_clean(tmp_path)
    out = tmp_path / "aligned"
    assert (
        main(["align", str(clean_dir), "--out", str(out), "--target-rate-hz", "10.0"])
        == 0
    )
    aligned = load_episode(out)
    for frame in aligned.frames:
        for md in frame.metadata.values():
            assert md.method == "nearest_neighbor"


def test_align_unknown_method_rejected_by_argparse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_dir = _save_clean(tmp_path)
    out = tmp_path / "aligned"
    with pytest.raises(SystemExit):
        main(
            [
                "align",
                str(clean_dir),
                "--out",
                str(out),
                "--target-rate-hz",
                "10.0",
                "--method",
                "linear",
            ]
        )
    err = capsys.readouterr().err
    assert "invalid choice" in err or "linear" in err


def test_align_single_method_applies_to_every_stream(
    tmp_path: Path,
) -> None:
    """`--method` today is a single string applied to every stream.

    Milestone 3 still lists per-stream method selection as future
    work; a per-stream selector would be a comma-separated key=value
    list (e.g. ``--method cam_front=zoh,robot_state=linear_interp,
    default=nearest_neighbor``). This test pins the current
    single-string behavior on the multi-stream synth run so the next
    session that extends the argparse surface has to consciously
    touch both the CLI layer and the ``_KNOWN_METHODS`` layer in
    ``embodied_sync/align/engine.py`` (D-0022) rather than
    accidentally regressing the current shape.
    """
    clean_dir = _save_clean(tmp_path)
    out = tmp_path / "aligned"
    assert (
        main(
            [
                "align",
                str(clean_dir),
                "--out",
                str(out),
                "--target-rate-hz",
                "10.0",
                "--method",
                "zoh",
            ]
        )
        == 0
    )
    aligned = load_episode(out)
    # Every frame, every stream: the same requested method string.
    # A per-stream selector would break this pin, which is intended
    # — that PR must consciously extend the CLI *and* engine layers.
    stream_names = set(aligned.frames[0].samples.keys())
    assert len(stream_names) > 1, "test requires the multi-stream synth rig"
    for frame in aligned.frames:
        for name in stream_names:
            assert frame.metadata[name].method == "zoh"


@pytest.mark.parametrize(
    "rate_str,expected_period_ns",
    [
        ("10.0", 100_000_000),
        ("60.0", 16_666_667),
        # NTSC-shaped rate — irrational at float precision. Pinned via
        # SESSION_STATE.md "Open questions". The manifest's
        # ``target_rate_hz`` echoes what argparse parsed; the
        # authoritative integer is ``target_period_ns``.
        (repr(60.0 / 1.001), round(1e9 / (60.0 / 1.001))),
    ],
)
def test_align_target_rate_hz_echoes_verbatim_into_manifest(
    tmp_path: Path, rate_str: str, expected_period_ns: int
) -> None:
    """``--target-rate-hz`` round-trips through the manifest as the
    float argparse parsed, and ``target_period_ns`` is the integer
    the engine actually used (D-0021).

    The two fields answer different questions: ``target_rate_hz`` is
    a diagnostic echo of the user's request (may lose float precision
    on irrational rates like NTSC's 60/1.001); ``target_period_ns`` is
    authoritative. A downstream tool comparing manifests must use the
    integer.
    """
    clean_dir = _save_clean(tmp_path)
    out = tmp_path / "aligned"
    assert (
        main(
            [
                "align",
                str(clean_dir),
                "--out",
                str(out),
                "--target-rate-hz",
                rate_str,
            ]
        )
        == 0
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    # The echoed float is exactly what argparse's `type=float` produced
    # from the CLI string, verbatim in the manifest.
    assert manifest["target_rate_hz"] == float(rate_str)
    # And the authoritative integer is `round(1e9 / rate)`, computed
    # from the same float.
    assert manifest["target_period_ns"] == expected_period_ns
