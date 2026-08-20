# Fixtures

Tiny generated or hand-authored fixtures that are safe to commit (kilobytes,
text-preferred formats). Regenerable fixtures must document the exact command
or function that produced them.

External datasets never go here — they belong under `data/external/`
(git-ignored, manually provided; see `TESTING_STRATEGY.md`).

## `synth_mini/`

A 1-second clean synthetic run (run format v0, 434 samples, 7 streams).
Regenerate with:

```bash
rm -rf data/fixtures/synth_mini
embsync synth --out data/fixtures/synth_mini --seed 0 --duration-s 1.0
```

`tests/test_fixture_synth_mini.py` asserts the committed files are
byte-identical to regeneration, so any change to the generator or the run
format shows up as a fixture diff (bump `format_version` and regenerate
deliberately).

## `synth_mini_aligned/`

The nearest-neighbor-aligned episode of `synth_mini/` at 10 Hz (D-0021
aligned-episode format v0, 6 frames × 7 streams). Regenerate with:

```bash
rm -rf data/fixtures/synth_mini_aligned
embsync align data/fixtures/synth_mini \
    --out data/fixtures/synth_mini_aligned \
    --target-rate-hz 10.0
```

(Run from the repo root — the source-run path is relative so the
manifest's `source_run` field records the same string that
regeneration would produce.)

`tests/test_fixture_synth_mini_aligned.py` asserts three things:
(a) the committed fixture round-trips against a fresh `align_run` on
the source, (b) the committed bytes match CLI regeneration, and (c)
the session-13 `median_skew_ns` field on `AlignmentReport` is
populated and echoed into the manifest. A change to the alignment
engine (D-0020), the episode format (D-0021), or `AlignmentReport`'s
shape fails the byte-identity test and forces a deliberate
regeneration.

## `synth_mini_corrupted/`

`synth_mini/` after the committed camera-jitter profile — the same
profile `docs/user/quickstart.md` walks through. Regenerate with:

```bash
rm -rf data/fixtures/synth_mini_corrupted
embsync corrupt data/fixtures/synth_mini \
    --profile configs/corrupt_camera_jitter.yaml \
    --out data/fixtures/synth_mini_corrupted
```

(Run from the repo root — both paths are relative so the manifest's
`profile_path` matches the regenerated string.)

`tests/test_fixture_synth_mini_corrupted.py` asserts three things:
(a) the committed run + ground-truth sidecar match `apply_profile` on
the clean source, (b) the committed bytes match CLI regeneration,
and (c) the ground-truth sidecar contains drops on `cam_front` only.
This pins the corruption-application layer (D-0009 / D-0010) as a
stable byte-level contract.

## `lsl_mini_replay/`

A tiny two-stream LSL/XDF replay JSON fixture (Milestone 7). Hand-
authored, not generated: 31 samples on a 100 Hz `eeg` stream and 4
samples on an irregular `marker` stream, each with its own
`xdf_clock_offset_ns` metadata. Loaded through
`embodied_sync.adapters.lsl.load_lsl_replay`.
`tests/test_fixture_lsl_mini_replay.py` pins two-stream loading, the
per-stream clock-offset propagation, and end-to-end 10 Hz alignment.
`tests/test_lsl_replay_jitter.py` pipes it through
`configs/corrupt_camera_jitter.yaml` — as an LSL-shaped profile —
to prove the corruption layer and adapter agree on the payload
contract.

## `surg_sync_mini/`

A tiny hand-authored SurgSync-shaped run fixture used to test shape
normalization and integration plumbing without committing a real
SurgSync dataset. `tests/test_corrupt_surg_sync_shape.py` and
`tests/test_surg_sync_integration.py` load it directly from
`data/fixtures/`, while native real-dataset tests continue to use
`EMBODIED_SYNC_EXTERNAL_DATA_ROOT/surg_sync/`.
