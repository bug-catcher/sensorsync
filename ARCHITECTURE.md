# Architecture

The package is organized around a canonical sample model: `core/` stores
timestamps and reports, `time/` maps clock domains, `align/` implements
offline and online pickers, `session/` handles live bundles, `calibrate/` fits
event-based mappings, and `reports/` emits diagnostics. Adapters and
exporters are optional integrations; the base install remains NumPy plus
PyYAML.

The public boundary validates timestamped observations and alignment choices;
it cannot infer physical simultaneity without independent capture evidence.
