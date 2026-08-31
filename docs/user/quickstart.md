# Synchronize robot sensor streams in Python

Install the public package, generate a deterministic run, align it, and write
a report:

```bash
python -m pip install embodied-sync
embsync synth --out runs/clean --seed 0 --duration-s 10
embsync align runs/clean --out episodes/clean --target-rate-hz 10
embsync report episodes/clean --out reports/clean.html --json-summary reports/clean.json
```

The report provides synchronization assurance: missing observations, skew,
and confidence under the selected policy. Physical simultaneity requires an
independent event such as the LED-and-beep protocol in the capture notebook.
