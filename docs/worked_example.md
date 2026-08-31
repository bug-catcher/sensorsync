# Worked example: a redistributable failure

The committed `synth_mini` fixture is deterministic and redistributable. The
paired `synth_mini_corrupted` fixture removes camera samples and records the
removals in `corruption_ground_truth.json`.

```bash
embsync align data/fixtures/synth_mini_corrupted \
  --out /tmp/synth-aligned --target-rate-hz 10 --check-ground-truth
embsync report /tmp/synth-aligned --out /tmp/synth-report.html \
  --json-summary /tmp/synth-report.json
```

Acceptable: the clean fixture has zero missing frames within tolerance.
Unacceptable: the corrupted fixture has non-zero `cam_front` missing rate and
ground-truth drops. The report exposes a failure that ordinary trigger
monitoring does not surface at the policy-observation level.
