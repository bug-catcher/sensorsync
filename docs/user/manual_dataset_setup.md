# Manual dataset setup

External datasets are intentionally supplied by the user. Put local copies
under `data/external/<format>/`, install the matching optional extra, and set:

```bash
export EMBODIED_SYNC_EXTERNAL_DATA_ROOT=/path/to/data/external
pytest -q -m external_data
```

The public release does not download gated data, accept dataset agreements, or
redistribute third-party recordings. Use the committed fixtures for a smoke
test, then profile an unfamiliar layout before importing it.
