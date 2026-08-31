# Choosing an alignment policy

Use `nearest_neighbor` for post-hoc reports, `zoh` for a causal online policy,
and `linear_interp` for numeric streams when interpolation is appropriate.
For a policy that will run online, train with the same causal rule used at
deployment. Do not interpolate across known missing intervals.

| Workload | Starting policy | Reason |
|---|---|---|
| Post-hoc report | `nearest_neighbor` | symmetric skew is useful for diagnosis |
| Online policy | `zoh` | never reads a future sample |
| Numeric resampling | `linear_interp` | smooth values between valid brackets |
| Events or images | `zoh` | interpolation is not meaningful |

See [online versus offline alignment](../concepts/online_vs_offline_alignment.md)
and [latency, jitter, and drift](../concepts/latency_vs_jitter_vs_drift.md)
before setting tolerances.
