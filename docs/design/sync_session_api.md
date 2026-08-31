# Live session API

For the public, task-oriented explanation and example, see
[Message-filters-style synchronization without ROS](../user/sync_session_api.md).

The live API accepts SDK callbacks and polled samples, then returns causal
bundles with per-stream quality metadata. It records the same run format used
by offline alignment. It is an assurance boundary, not a hardware clock
synchronizer or a hard-real-time controller.
