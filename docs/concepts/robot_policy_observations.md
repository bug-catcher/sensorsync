# Robot policy observations

A policy running online consumes an *observation* at each policy tick.
This doc names the clocks and rates that the observation has to
straddle, explains why deadline-zero ZoH is the safe default, and shows
how to read the per-stream `median_skew_ns` recorded in an aligned
episode's `manifest.json` (D-0026, session 9) so you can catch a
training/deployment mismatch before it silently degrades the policy.

Prerequisites:
[`online_vs_offline_alignment.md`](./online_vs_offline_alignment.md)
pins the causality and bounded-memory invariants that apply here;
[`../user/choosing_alignment_policy.md`](../user/choosing_alignment_policy.md)
compares the three offline policies.

## Two clocks, not one

A policy tick and a sensor tick are two different clocks that only look
alike because both are measured in nanoseconds:

- **Policy tick**: the control loop's own clock. Fires at a fixed rate
  (10 Hz, 30 Hz, sometimes 100 Hz). Nothing sensor-side drives it. Each
  tick names a `target_time_ns` and demands an observation *now*.
- **Sensor tick**: the arrival clock of each stream. Robot state at
  250 Hz, cameras at 30 Hz, tactile at 60 Hz, event streams irregular.
  No two streams tick together, and none of them coincide with the
  policy tick except by coincidence.

The observation is the answer to: *for each stream, what sample should
I hand the policy at this policy tick?* That is the question
`MultiStreamAligner.get_latest_policy_frame(now_ns=...)` and
`StreamRingBuffer.get_latest_policy_observation(now_ns=...)` answer.

## Why deadline-zero ZoH is the default

Three properties any policy tick needs — and only ZoH at deadline zero
gives all three.

**Causality.** At `now_ns = T`, samples with
`receive_time_ns > T` have not physically arrived. Using them
requires either the future (impossible online) or a deadline > 0 (the
policy has to wait, burning latency budget). Deadline-zero ZoH picks
the newest sample whose `acquisition_time_ns <= T` *and*
`receive_time_ns <= T`. Nearest-neighbor on acquisition time is not
causal — it will reach for the newest future sample as soon as one
arrives. Linear interpolation between a past and a future sample is
worse: it hides the causality violation inside a synthesised value.

**Bounded latency.** A policy that sometimes waits for a slower stream
misses its tick. Deadline zero says "I will not wait; if the freshest
available sample is stale, mark it missing and let the policy
downstream deal with it". `StreamRingBuffer` reports missing when a
pick is stale beyond `tolerance_ns`; the composite frame surfaces this
per stream, so the policy can decide whether to hold, retry, or fall
back.

**Bounded memory.** `StreamRingBuffer` uses a `deque(maxlen=capacity)`
— O(1) push, O(1) FIFO eviction. Every stream keeps at most
`capacity` samples, regardless of how long the loop has been running.
An offline `align_run` cannot deploy in a control loop: it holds the
whole run in memory and looks ahead.

The deadline-zero ZoH answer is:

- The freshest sample from each stream with
  `acquisition_time_ns <= T` and `receive_time_ns <= T`;
- Or *missing* per stream, with the candidate reported in metadata so
  the caller can log staleness.

Concretely::

    aligner = MultiStreamAligner({
        "robot_state": StreamRingBuffer(capacity=512, tolerance_ns=2_000_000),
        "cam_front":   StreamRingBuffer(capacity=64,  tolerance_ns=33_000_000),
    })
    for sample in producer_thread():
        aligner.push(sample)
    frame = aligner.get_latest_policy_frame(now_ns=clock())
    action = policy(frame.samples)

Nothing in the library reads a wall clock. The caller injects
`now_ns` so tests get bit-identical output.

This is the engine, worth understanding once. In practice, reach for
`embodied_sync.session.SyncSession` (D-0037) instead of assembling
`MultiStreamAligner` and `StreamRingBuffer` by hand — it wraps exactly
this pattern (`push`/`attach` per stream, `get(reference=now_ns)` for a
`SyncBundle` with `bundle.ok` already computed) plus the parts a raw
policy loop needs and this snippet does not show: `SyncViolation`s for
every degraded pick instead of a metadata flag you have to remember to
check, cross-device clock mapping via `embodied_sync.calibrate`, and
recording the loop to the same run-directory format this doc's
`median_skew_ns` discussion reads back from. See the README quickstart
and `embodied_sync/session/session.py`'s module docstring.

## Reading `median_skew_ns` in the manifest

An offline aligned episode's `manifest.json` carries a per-stream
`median_skew_ns` field (session 9's median-skew round-trip). It is a
signed integer nanosecond count: the median value of
`sample.acquisition_time_ns - target_time_ns` across non-missing
frames of that stream, or `null` if every frame was missing. Same
formula as `SyncQualityReport.median_skew_ns`; the manifest is just a
cheaper place to read it from.

The signature to look for:

- **Near zero** (say ≤ half the fastest stream's inter-sample
  interval): frames are on average close to the target. Fine for any
  downstream use.
- **Negative for every stream**: the picker preferred samples in the
  *past* of the target. Standard for ZoH — offline ZoH is
  bit-identical to online ZoH at deadline zero for a clean stream, so
  matching signs is the goal.
- **Positive for every stream**: the picker preferred samples in the
  *future* of the target. Only nearest-neighbor and linear
  interpolation can do that; deploying a policy trained on such an
  episode online (where the future is unreachable) is the failure
  case walked through in
  [`online_vs_offline_alignment.md`](./online_vs_offline_alignment.md).
- **Mixed signs across streams**: a stream is being picked from the
  past while another is picked from the future. The policy sees a
  systematically inconsistent snapshot; retrain with a single policy
  (usually ZoH) across all streams.
- **Median deployment skew ≈ -median training skew**: the "4 ms
  distribution shift" bug fingerprint from the concept doc. Training
  used nearest-neighbor and got symmetric skew around zero;
  deployment uses ZoH and gets negative skew. The mismatch is
  detectable from two manifest reads — one from a training episode,
  one from an episode aligned from the deployment stream.

Because `median_skew_ns` is a manifest field and not a report field, a
downstream tool can spot the mismatch with three lines of JSON parsing
— no HTML scrape needed.

## Rules of thumb

- If you plan to deploy the policy online, align training data with
  ZoH offline too, at the same `target_rate_hz` the deployment tick
  will use.
- Set `tolerance_ns` on each ring buffer to *at most* one nominal
  inter-sample interval of that stream (half is the offline default;
  jitter margin can push it larger). A stale-beyond-tolerance pick
  should mark missing, not silently degrade the policy input.
- Set `capacity` big enough that a burst-stall (D-0015) worth of
  samples still fits without eviction. A 250 Hz stream with a 40 ms
  worst-case stall needs at least 10 slots plus safety; 128 or 256 is
  fine.
- Check that every stream's manifest `median_skew_ns` sits on the same
  side of zero before comparing training and deployment. Silent bugs
  hide in mismatched signs.
- Log `frame.metadata[stream].missing` per stream in the control loop;
  a missing pick is a diagnostic, not an error, but it *is* a signal
  that either `tolerance_ns` is too tight or the stream has actually
  stalled.

## Related material

- Concept: [`online_vs_offline_alignment.md`](./online_vs_offline_alignment.md)
  — causality, bounded memory, the training/deployment mismatch trap.
- Concept: [`latency_vs_jitter_vs_drift.md`](./latency_vs_jitter_vs_drift.md)
  — the three timing-error shapes that pick your `tolerance_ns`.
- User: [`../user/choosing_alignment_policy.md`](../user/choosing_alignment_policy.md)
  — decision table across the three offline policies.
- Decisions: D-0026 (online ring buffer, ZoH-only) and D-0027
  (multi-stream online composite).
