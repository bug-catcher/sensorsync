# Interpreting sync-quality reports

`missing_rate` means that no sample was available within the chosen
alignment tolerance; it does not prove that a source packet was lost. Use
the optional corruption ground truth or source metadata to establish loss.

Read the fields together:

- `median_skew_ns` shows signed bias. Negative is older/stale; positive means
  a picker reached forward in time.
- `median_abs_skew_ns` shows magnitude without direction.
- `median_confidence` is relative to the tolerance window, not to zero skew.
- `ground_truth_missing_count` appears only when `--check-ground-truth` loads
  a corruption sidecar.

An acceptable report has thresholds defined in advance, no unexpected
missing observations, and skew appropriate to the workload. A clean report
still does not prove physical simultaneity; use the LED-and-beep capture
protocol for that independent check.
