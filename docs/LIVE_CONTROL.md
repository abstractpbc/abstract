# Live control: strict timing, async inference and live commands

Three opt-in additions to the rollout loop. Without them, the loop behaves as described in
[SAFETY.md](SAFETY.md): it steps as soon as an action is ready, and the policy sees only
the embodiment's observations.

## Strict timing

By default the rollout applies no wall-clock pacing. Each step runs as soon as the policy,
controller and approver produce an action, and an embodiment that needs a fixed rate
paces itself (YAM and presets do). With strict timing, the loop holds every step to a
fixed tick rate. The default rate is the embodiment's declared `control_hz`, which is a
preset's `policy_hz`.

```sh
uv run abstract eval --strict-timing ...
uv run abstract robot run --config rigs/my-robot.toml --strict-timing "..."
```

```toml
[rig]
strict_timing = true
max_consecutive_overruns = 10   # optional: halt (and park) after 10 late steps in a row
```

```python
from abstract_harness import StrictTiming, eval

eval(task, policy, robot, strict_timing=True)
eval(task, policy, robot, strict_timing=StrictTiming(hz=50, max_consecutive_overruns=10))
```

Step `k` of a trial is dispatched no earlier than `start + k / hz`. A step that misses its
deadline by more than a tenth of a period is an overrun. The schedule never bursts to
catch up: a late step restarts the schedule at that step, so the next step still waits
a full period, and the skipped ticks are counted. Each trial's report lands in its metadata under `timing`:
`hz`, `ticks`, `overruns`, `missed_ticks`, `max_late_ms`, `mean_overrun_late_ms` and
`effective_hz`.

Strict timing paces dispatch. It does not make inference faster, so a slow policy
overruns every tick unless inference overlaps execution. A policy that declares
`requires_paced_loop = True` (the orchestrator) gets strict timing automatically on an
embodiment that does not pace itself.

## Async inference

`AsyncController` (`--async-inference`, `[rig] async_inference = true`) runs inference on
a worker thread while the buffered actions keep playing:

- The first step of a trial waits for one inference.
- When `prefetch` or fewer actions remain (half the last chunk by default), the next
  inference starts from the current observation.
- No observation is sent to the policy twice.
- A chunk computed from step `q` that arrives at step `t` drops its first `t - q` actions,
  which were for steps that already ran. The rest replace the buffer. If the whole chunk
  is stale, its last action runs one or more steps late, marked `meta["late"]`. This
  happens with a one-action reactive policy, or when latency exceeds the chunk.
- If the buffer runs dry, the robot holds. An absolute target repeats its last value.
  Displacements and velocities send zeros. Hold actions carry `meta["hold"]`;
  `underrun="wait"` blocks instead.
- A stop request is never lost. A buffered one is not replaced by a later prediction,
  and one on a dropped stale action is carried by the next action.
- Inference running longer than `timeout_s` (120 s) raises `SafetyAbort`.

The controller's counts (inferences, holds, late and stale actions) land in the trial
metadata under `controller`. Approvals, operator messages and command events count as
delivered at the step whose observation went to the policy, so none are lost while
inference runs. Only `act()` runs on the worker. Policies whose `observe()` or
`transcript_delta()` hooks would run concurrently with it are refused: stateful world
models that must see every control observation in order, and the LLM agent. At a
trial's end, the rollout waits up to `join_timeout_s` (2 s) for an inference still in
flight before resetting or reading the policy; one that does not return marks the policy
abandoned. Use async inference on a paced loop. On an unpaced
simulator, the loop spins through hold steps while inference runs.

## Live commands

A `CommandStream` carries live inputs into a running policy:

- **Axes** are continuous values such as `vx`, `yaw_rate` or `speed`. Each has a range,
  a resting default and a deadman timeout.
- **Events** are discrete presses such as `go` or `stop`.

Every step, the rollout puts the newest axis values in `observation.extra["commands"]`.
It puts the events since the previous `act()` in `observation.extra["command_events"]`,
each as `{"t", "name", "source"}`. Values are recorded with each step and in the action
log's `commands` column, and events in the trial's transcript.

```python
def locomotion_policy(observation):
    commands = observation.extra.get("commands", {})
    return controller(observation.state, commands.get("vx", 0.0), commands.get("yaw_rate", 0.0))
```

An axis that stops updating returns to its default: a dropped gamepad or network link
stops a velocity command instead of repeating it forever. A stream with declared axes
refuses other names and clamps values to their ranges. A stream without declared axes
accepts any name, drops a value that goes 0.5 s without an update, and joins observations only once it has carried a command. So `serve`'s command channel changes nothing until a client uses it. Each trial starts
at rest with no queued events. The `end_episode` event ends the trial as the console's
Esc does, including while inference is pending on a physical run. Commands are policy
inputs: they reach the robot only through the policy's actions, which still pass the
approver chain. A button is not an emergency stop.

Sources:

| Source | How |
| --- | --- |
| A preset's controllers | `[[robot.controllers]]` ([PRESETS.md](PRESETS.md)): a Linux gamepad (`/dev/input/js*`, no extra dependency), with deadzones, scales, a deadman button and button actions, or a UDP bridge. |
| UDP | `--commands udp:127.0.0.1:9870` on `abstract eval` and `abstract robot run`. One JSON datagram per message: `{"values": {"vx": 0.3}, "events": ["go"]}`, or `{"vx": 0.3}`. Loopback only unless the source is built with `allow_remote=True`. |
| HTTP | `abstract robot serve`'s `command` RPC ([RPC.md](RPC.md)), forwarded to the run through `robot run --commands-fd`. |
| Python | `stream.set("vx", 0.3)`, `stream.update({...})`, `stream.press("go")`, or a `CommandSource` with `start(stream)` and `close()`; pass `eval(..., commands=stream)`. |

A gamepad reports only changes, so the joystick source republishes held stick values
(50 Hz by default) to keep them inside their axes' timeouts. A pad that disconnects stops
republishing, and its axes return to rest. So does a wireless pad that goes silent with
its device still open, after `max_silence_s` (1 s) without any event.

## Teleop

The `teleop` policy maps each action dimension to the live command of the same name.
`map = "yaw_rate:turn"` renames; `scale` scales. A dimension without a command sends zero.
Teleop drives displacement and velocity modes only, where zero is rest. It refuses
absolute targets: there, a command timing out to its default would be a place to move
to.
With a `base_vel` preset and a gamepad, this is joystick driving with the bounds clamp,
the acceleration limit, the watchdog and fall detection in force.
[rigs/mobile-base-teleop.toml](../rigs/mobile-base-teleop.toml) does this in the simulator.
In a robot TOML it is `[vla] policy = "teleop"`.
