"""Committed-fixture guard: `data/fixtures/synth_mini` (NEXT_TASKS #2).

The fixture is a 1-second clean run committed to git. These tests pin the
on-disk format: if the generator or the run format changes, regeneration no
longer matches the committed bytes and the diff must be made deliberately
(regeneration command in `data/fixtures/README.md`).
"""

from __future__ import annotations

from pathlib import Path

from embodied_sync.cli.main import main
from embodied_sync.datasets.io import load_run
from embodied_sync.streams.synthetic import generate_synthetic_run

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "synth_mini"

#: Exact regeneration command (see data/fixtures/README.md).
REGEN_ARGS = ["synth", "--out", str(FIXTURE_DIR), "--seed", "0", "--duration-s", "1.0"]


def test_fixture_loads_and_matches_generator() -> None:
    loaded = load_run(FIXTURE_DIR)
    assert loaded == generate_synthetic_run(duration_s=1.0, seed=0)


def test_fixture_is_byte_identical_to_regeneration(tmp_path: Path) -> None:
    regen_dir = tmp_path / "synth_mini"
    regen_args = ["synth", "--out", str(regen_dir), "--seed", "0", "--duration-s", "1.0"]
    assert main(regen_args) == 0

    fixture_files = sorted(p.relative_to(FIXTURE_DIR) for p in FIXTURE_DIR.rglob("*.json*"))
    regen_files = sorted(p.relative_to(regen_dir) for p in regen_dir.rglob("*.json*"))
    assert fixture_files == regen_files
    for rel in fixture_files:
        assert (FIXTURE_DIR / rel).read_bytes() == (regen_dir / rel).read_bytes(), (
            f"format drift in {rel}: committed fixture differs from regeneration "
            f"(see data/fixtures/README.md)"
        )
