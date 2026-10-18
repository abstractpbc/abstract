# Adapter guide

[examples/fake_arm.py](../examples/fake_arm.py) is a complete, runnable one-joint adapter.
It performs no I/O and exercises the physical default approval chain with an in-memory
driver. Run `uv run python examples/fake_arm.py`; CI runs it in the core environment.

## Contracts

An embodiment declares `EmbodimentInfo` with an action `Box`, semantics, observation
fields and control rate. `reset(scene, seed=...)` returns the first observation;
`step(action)` returns `StepResult`; `close()` releases owned resources. Construction
must be motion-free. Implement close for partial initialization as well as successful runs.

`joint_embodiment_info` builds a joint-position contract with radians for joints. Gripper
units, polarity, coordinate frame, rotation representation and dimension ordering must
be declared consistently by the policy and robot. The helper does not add a driver,
watchdog, collision model or emergency stop. Describe those separately for a real adapter.

Observations contain RGB uint8 camera arrays and named numeric state arrays matching the
declared spaces. Supply monotonic state and image publication timestamps in one documented
clock domain. Lazy extras may hold callable depth accessors; their lifetime remains the
adapter's responsibility. Numeric buffers are snapshotted when retained in trial records.

Absolute control needs a measured `joint_pos` or `eef_state` reference. An adapter can
implement `action_reference(observation)` to read fresher state after inference. The
default physical chain reanchors per dispatch so an actuator that fails to follow cannot
accumulate unrestricted commanded targets. `check_health()` may be called on the owning
thread during inference waits. These hooks must be short and must not compete with the
inference worker for an actuator lock.

`StepResult.action_executed` is false if the supplied action was refused or never sent.
Use `command` for the final driver command in explicitly documented units and
`command_time` for its dispatch time. For example, YAM Cartesian requests produce joint
commands, and its thermal-trip row has no executed action. Export omits no-command rows.
Adapters that cannot observe execution should document that `action_executed` means dispatch,
not independently measured completion.

## Policies and registration

`CallableVLAPolicy(fn)` accepts an `Action`, `ActionChunk`, numeric action vector or action
matrix. An `Action` retains its metadata. Explicit action spaces, control modes and
dimensions are authoritative; leave them unspecified only when the callable actually
adapts to the embodiment. A policy must never drive hardware itself. It can continue
running after timeout on an abandoned daemon worker, whose result is never dispatched.

To expose an adapter to TOML, register it with `@embodiment("name")` in an imported module.
To construct a registered object, use `resolve("embodiment", "name", **arguments)` from
`abstract_harness.robot.core.registry`. Registration decorators do not construct instances.
Factories must accept only documented arguments and must not log secret values.

Controllers select or transform predictions; approvers decide which targets may run.
Use `DefaultController` for buffered action chunks, `SmoothingController` for low-pass
filtering and `EnsemblingController` for compatible overlapping predictions. A stateful
gateway continuation requires its full fresh control window; an incompatible replan
interval is refused. A change of rate, units or frame requires an explicit transformation.

## Resource ownership and errors

Python `eval(..., embodiment=object)` leaves the object open; always close it in `finally`,
as the example does. A registry-name embodiment is owned by eval and closed there. CLI
execution owns its constructed components. Low-level rollout does not close the robot.

Raise `PolicyError` for an invalid prediction or model failure, `EmbodimentFault` for
device failure and `SafetyAbort` for a refused unsafe transition. The last two halt
evaluation regardless of the policy error tolerance. Keep original partial-trial evidence.
Report unknown physical state as unknown, never as a successful park.

Before adding real hardware, test nonfinite/incorrect dimensions, first-command limits,
stalled tracking, delayed inference, stop metadata, partial reset, repeated close, camera
loss and teardown faults with fakes. Then perform the operator-run acceptance matrix in
[CONTRIBUTING.md](../CONTRIBUTING.md). See [SAFETY.md](SAFETY.md) for the limits of these hooks.
