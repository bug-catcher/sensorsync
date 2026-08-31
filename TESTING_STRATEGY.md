# Testing strategy

Core tests run with the base install and never download external data.
Optional adapter tests skip when dependencies are absent; external-data tests
skip when a user has not supplied a local dataset. Synthetic tests cover
jitter, drift, drops, stalls, duplicates, and non-monotonic delivery.

```bash
python -m pip install -e '.[dev]'
pytest -q
```

Committed fixtures contain no third-party recordings and are deterministic.
