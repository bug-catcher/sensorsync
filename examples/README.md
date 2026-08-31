# Examples

Runnable examples and notebooks. Every example runs offline with only base
dependencies unless its name clearly indicates an optional integration.

## Notebooks

- [`notebooks/hardware_ground_truth_capture_protocol.ipynb`](notebooks/hardware_ground_truth_capture_protocol.ipynb) —
  field worksheet for LED-plus-beep capture, three clock/capture setups,
  physical-vs-timestamp interpretation, and acceptance evidence.

- [`notebooks/sync_quality_demo.ipynb`](notebooks/sync_quality_demo.ipynb) —
  the full validation loop in-memory: synthesize a deterministic seven-stream
  rig, inject the kitchen-sink corruption profile, align offline at 12 Hz,
  cross-check detected missing frames against corruption ground truth, and
  replay the same data through the online (causal) aligner to expose
  delivery latency that offline alignment cannot see. Needs `matplotlib`
  in addition to the base install; runs end-to-end in ~10 s. Committed
  with outputs so it reads on GitHub without executing.

Notebook-generated files land in `notebooks/output/` (git-ignored).
