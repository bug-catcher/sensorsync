# Automatic dataset import

Automatic import handles unfamiliar *layouts* using reusable storage and clock
primitives. It does not ask a language model to generate or execute Python.
The output is a reviewable JSON plan consumed by deterministic code.

## Workflow

Inspect without writing a run:

```bash
embsync inspect-dataset /data/recording --out profile.json
```

Generate ranked interpretations:

```bash
embsync infer-import /data/recording --out inference.json
```

The inference document contains the observed profile, every candidate plan,
its evidence and warnings, and `selected`. A selection is made only when the
best candidate clears the default confidence (`0.75`) and lead over the
runner-up (`0.12`). Override a known native rate during inference when media
metadata does not carry one:

```bash
embsync infer-import /data/recording --rate-hz 30 --out inference.json
```

Execute the reviewed result:

```bash
embsync import-auto /data/recording --plan inference.json --out runs/recording
```

`import-auto` can infer and execute in one call when evidence is decisive:

```bash
embsync import-auto /data/recording --out runs/recording
```

If evidence is ambiguous, the command stops. `--accept-ambiguous` is an
explicit override for exploratory work; its use is printed and the chosen plan
is persisted in the run manifest.

## What is inferred

Known signatures select existing native adapters for LeRobot v3, SurgSync v1,
MCAP, XDF, canonical embodied-sync runs, and the UMI JSON contract.

The generic `indexed_episode` executor currently covers repeated episode
directories containing:

- one JSON array of state rows per episode;
- camera arrays grouped by episode in HDF5, or episode-local video files;
- a fixed media rate or a numeric timestamp field;
- camera/state joins by row or frame index.

For these datasets, inference compares a fixed row clock against a timestamp
field clock. It checks HDF5/video frame counts, media rate, monotonicity, and
episode duration agreement. Equal row counts alone prove correspondence, not
which clock is semantically correct, so clock evidence is reported separately.

## Plan safety

An import plan contains an executor name and JSON parameters. Executors are
registered in the package; arbitrary modules, expressions, and generated code
cannot be named. Relative paths are constrained to the dataset root. Optional
format dependencies are imported only after their executor is selected.

Plans are reusable for another dataset with the same layout. Applying a plan
to a changed layout still validates required files, row shape, timestamp type,
clock strategy, and path boundaries during execution.

## Current boundary

Inference is structural. It can decisively identify contracts such as “frame
`i` equals state row `i` at 10 Hz,” but it cannot infer every semantic fact
from filenames and timestamps. Content-based motion/kinematic correlation is a
future evidence provider. Until then, low-confidence and close-scoring plans
remain intentionally human-gated.
