"""CLI tests: `embsync synth` writes a loadable, reproducible run (NEXT_TASKS #1).

End-to-end over the public entry point: parse args -> generate -> save ->
load back -> compare against a direct generator call (exact equality,
including every timestamp nanosecond).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodied_sync.cli.main import main
from embodied_sync.datasets.io import load_run
from embodied_sync.streams.synthetic import generate_synthetic_run


def test_synth_writes_loadable_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "run"
    assert main(["synth", "--out", str(out), "--seed", "7", "--duration-s", "1.0"]) == 0

    loaded = load_run(out)
    expected = generate_synthetic_run(duration_s=1.0, seed=7)
    assert loaded == expected

    stdout = capsys.readouterr().out
    total = sum(len(samples) for samples in expected.values())
    assert str(total) in stdout


def test_synth_manifest_records_generator_inputs(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert main(["synth", "--out", str(out), "--seed", "3", "--duration-s", "0.5"]) == 0

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic"] == {"seed": 3, "duration_s": 0.5, "start_time_ns": 0}
    assert manifest["format_version"] == 0
    assert set(manifest["streams"]) == set(generate_synthetic_run(duration_s=0.5, seed=3))


def test_synth_refuses_non_empty_out_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "existing.txt").write_text("x", encoding="utf-8")

    assert main(["synth", "--out", str(out), "--duration-s", "0.1"]) == 1
    err = capsys.readouterr().err
    assert "non-empty" in err


def test_synth_cli_is_byte_deterministic_across_runs(tmp_path: Path) -> None:
    """Two CLI invocations at the same seed and duration write identical bytes.

    Pins the Milestone 1 acceptance criterion "tests verify deterministic
    output" for `embsync synth` end-to-end — not just at the generator API
    (already covered by ``test_synth_writes_loadable_run``) but through
    the CLI's manifest-writing path so any nondeterminism introduced by
    save layout is caught here.
    """
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    args = ["--seed", "5", "--duration-s", "0.75"]
    assert main(["synth", "--out", str(out_a), *args]) == 0
    assert main(["synth", "--out", str(out_b), *args]) == 0

    a_files = sorted(p.relative_to(out_a) for p in out_a.rglob("*.json*"))
    b_files = sorted(p.relative_to(out_b) for p in out_b.rglob("*.json*"))
    assert a_files == b_files
    for rel in a_files:
        assert (out_a / rel).read_bytes() == (out_b / rel).read_bytes(), (
            f"nondeterministic bytes in {rel}: CLI runs at the same "
            f"seed/duration must be byte-identical"
        )


def test_synth_start_time_ns_flag_shifts_acquisitions_and_echoes_manifest(
    tmp_path: Path,
) -> None:
    """`--start-time-ns N` offsets every regular stream and lands in the manifest.

    Compares two CLI runs at the same (seed, duration): one at the default
    ``start_time_ns = 0`` and one at a large positive value. Every regular
    stream's first sample's ``acquisition_time_ns`` must shift by exactly
    the requested offset; the manifest's ``synthetic.start_time_ns`` must
    echo the value verbatim. Guards NEXT_TASKS #1: today's CLI hardcodes
    ``start_time_ns=0`` in the manifest even though the synth harness
    already supports non-zero offsets.
    """
    zero_dir = tmp_path / "zero"
    shifted_dir = tmp_path / "shifted"
    offset_ns = 1_500_000_000_000  # 1500 s — large enough to be obvious.

    args = ["--seed", "11", "--duration-s", "0.5"]
    assert main(["synth", "--out", str(zero_dir), *args]) == 0
    assert main(
        [
            "synth",
            "--out",
            str(shifted_dir),
            "--start-time-ns",
            str(offset_ns),
            *args,
        ]
    ) == 0

    zero_manifest = json.loads((zero_dir / "manifest.json").read_text(encoding="utf-8"))
    shifted_manifest = json.loads(
        (shifted_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert zero_manifest["synthetic"]["start_time_ns"] == 0
    assert shifted_manifest["synthetic"]["start_time_ns"] == offset_ns

    zero_run = load_run(zero_dir)
    shifted_run = load_run(shifted_dir)
    assert set(zero_run) == set(shifted_run)
    # Regular streams shift acquisition times by exactly ``offset_ns``.
    for name, zero_samples in zero_run.items():
        if not zero_samples:
            continue
        shifted_samples = shifted_run[name]
        assert len(shifted_samples) == len(zero_samples)
        assert (
            shifted_samples[0].acquisition_time_ns
            == zero_samples[0].acquisition_time_ns + offset_ns
        )
        # Receive-time shifts by the same offset too (clean streams add a
        # per-stream constant to acquisition — offsetting acquisition
        # offsets receive by exactly the same amount, D-0006).
        assert (
            shifted_samples[0].receive_time_ns
            == zero_samples[0].receive_time_ns + offset_ns
        )


def test_synth_cli_reproduces_committed_fixture(tmp_path: Path) -> None:
    """CLI at (seed=0, duration=1.0) reproduces ``data/fixtures/synth_mini/``.

    This is the "CLI honours the same reproducibility contract the
    committed fixture pins" check — the fixture is the byte-identity
    ground truth (see ``test_fixture_synth_mini.py``) and this test
    routes through the CLI to make sure the CLI is what fed it.
    """
    fixture_dir = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "synth_mini"
    regen_dir = tmp_path / "synth_mini"
    assert main(
        ["synth", "--out", str(regen_dir), "--seed", "0", "--duration-s", "1.0"]
    ) == 0
    fixture_files = sorted(p.relative_to(fixture_dir) for p in fixture_dir.rglob("*.json*"))
    regen_files = sorted(p.relative_to(regen_dir) for p in regen_dir.rglob("*.json*"))
    assert fixture_files == regen_files
    for rel in fixture_files:
        assert (fixture_dir / rel).read_bytes() == (regen_dir / rel).read_bytes(), (
            f"CLI at seed=0/duration=1.0 no longer reproduces the "
            f"committed fixture bytes for {rel} — regenerate per "
            f"data/fixtures/README.md if this drift was intentional"
        )
