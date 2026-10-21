# Changelog

## Unreleased

Additive and opt-in: without the new settings, runs, logs and the public API behave as in 0.1.0.

- Strict timing (`core/timing.py`): `eval(strict_timing=...)`, `--strict-timing` and
  `[rig] strict_timing` hold each step to the embodiment's rate and record overruns, missed
  ticks, lateness and the effective rate per trial; `max_consecutive_overruns` aborts.
- `AsyncController` (`--async-inference`): background inference with prefetch, stale-action
  dropping and hold-on-underrun; refuses `observe()` policies.
- Live commands (`core/command_stream.py`): `CommandStream` axes with ranges and deadman
  timeouts plus events, fed by a Linux gamepad, UDP, `robot run --commands/--commands-fd`
  and serve's new `command` RPC; injected as `extra["commands"]` and
  `extra["command_events"]`, recorded per step and in the action log; `end_episode` ends a trial.
- Robot presets (`robot/preset`): URDF/MJCF-backed robot TOMLs with joint selection, limit
  narrowing, per-step limits, home, parts, end effectors, cameras, sensors, state, commands
  and controllers; the `preset` embodiment with user drivers or a kinematic simulator;
  `[rig] preset`; examples in `robots/` including a Unitree G1 template.
- Locomotion: the `base_vel` control mode, base/IMU canonical state keys,
  `AccelerationLimitApprover`, a rate-mode watchdog and fall detection in presets, the
  agent's `drive` tool, and the `teleop` policy (`[vla] policy = "teleop"`).
- The orchestrator (`robot/orchestrator`, policy `orchestrator`, `[agent] policy =
  "orchestrator"`): extension tools, packaged policies and skills, run as background jobs
  whose events wake the model, with `wait`, `stop_job`, `steer_job` and `jobs`.
- `reference_state_keys` prefers the state field named after the control mode, so a
  `joint_pos` embodiment may also report a same-shaped `joint_vel`.
- `abstract robot holdcheck|health|pose` refuse rigs that are not `yam_arms`.

## 0.1.0 — unreleased

## 0.1.0 — unreleased

Initial standalone release candidate, derived in part from Robocurve inspect-robots and
inspect-robots-yam; see THIRD_PARTY_NOTICES.md for bases and attribution.

- Preserve fixed policy semantics and explicit callable contracts during binding.
- Align gateway camera names with YAM and collect stateful history at control cadence.
- Validate gateway sessions, dimensions, numeric limits, URLs and response semantics.
- Scope cloud credentials to their origin, reject literal secret arguments and redact setup.
- Retain raw data when export completion cannot be verified; recheck file hashes at deletion.
- Route CLI termination through teardown and bound inference/arm-initialization waits.
- Share Python/CLI guardrails, seed and refresh measured references, and preflight native failsafes.
- Preserve terminal metadata through callable conversion and temporal ensembling.
- Distinguish requested/approved actions, final commands and transitions without execution.
- Bundle third-party notices and typing metadata; separate the immutable native-driver group.
- Exercise minimal installs, built distributions and both supported Python versions in CI.

Before 1.0, public API changes may occur between minor versions. Patch releases should
preserve documented interfaces except when a correction is required to refuse unsafe or
invalid input; record those changes here. Underscore-prefixed modules are internal.
