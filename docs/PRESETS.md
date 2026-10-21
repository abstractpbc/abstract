# Robot presets

A robot preset describes a robot in one TOML file: its model file (URDF or MJCF), the
joints a policy drives and their limits, the policy and servo rates, named end effectors,
cameras and sensor frames, the state its driver reports, a mobile base, live command
inputs, and the gamepads or bridges that feed them. The `preset` embodiment runs any
preset through a driver, either yours or the built-in kinematic simulator. A new robot
needs a preset and a driver, not a new embodiment class.

Examples: [robots/mobile-manipulator](../robots/mobile-manipulator) (a toy URDF with an
arm preset and a base preset) and [robots/unitree-g1](../robots/unitree-g1) (a G1 template
with custom end effectors, IMU state, gamepad velocity commands and a driver skeleton).

## Use a preset

Point a rig at it, or pass it to the embodiment directly:

```toml
# rigs/my-robot.toml
[vla]
policy = "teleop"                      # or a VLA server, or [agent]

[rig]
preset = "../robots/my-robot/preset.toml"   # relative to this file
strict_timing = true                   # hold steps to the preset's policy_hz

[rig.embodiment]
# Overrides for this rig: driver, pace, auto_start, or any [robot.driver_args] key.
driver = "sim"
```

```sh
uv run abstract robot check --config rigs/my-robot.toml      # resolves and audits; no motion
uv run abstract eval --embodiment preset -E preset=robots/my-robot/preset.toml \
    -E driver=sim --policy teleop --instruction "dry run" --strict-timing --no-prompt
```

`check` prints what the preset resolved to: model, action layout and rates, parts, end
effectors, cameras, state, commands and controllers, followed by the conformance audit.
Every reference is validated before any driver exists. A misspelled joint or frame is
refused with the closest names from the model.

## The file

Keys of `[robot]` come before its sub-tables, as TOML requires.

```toml
[robot]
name = "g1"                    # the embodiment name in logs: lower case, digits, - and _
model = "g1_29dof.urdf"        # URDF or MJCF, relative to the preset (optional with [[robot.joint]])
policy_hz = 50                 # the policy step rate: EmbodimentInfo.control_hz
control_hz = 500               # the driver's servo rate, passed to the driver (default policy_hz)
mode = "joint_pos"             # joint_pos | joint_delta | joint_vel | base_vel
driver = "g1_driver.py:G1Driver"   # "sim", a registered driver, "module:Class" or "file.py:Class"
pace = true                    # step() waits out the policy period (default true)
auto_start = false             # physical drivers: skip the readiness gates (a notice prints)
command_timeout_s = 0.25       # rate modes: the watchdog's timeout (at least two policy periods)
command_events = ["go", "stop"]
description = "Free text appended to the operating notes policies read."

[robot.driver_args]            # keyword arguments for the driver's constructor
network_interface = "eth0"
```

**Joints.** `[robot.joints]` selects and orders the action's joints. Its entries are joint
names or globs:

| Key | Meaning |
| --- | --- |
| `select` | Joints in action order. Default: every movable joint, in the MJCF's actuator order when it has actuators, else in document order. |
| `exclude` | Joints to drop from the selection. |
| `limits` | `glob = [low, high]` in the mode's units. Narrows the model's limits, and supplies them where the model has none (a continuous joint). A preset can never widen a limit. |
| `max_step` | `glob = value`: the largest change per step. For `joint_pos` it becomes `ActionSemantics.max_step` (the delta limiter's budget). For `joint_delta` it is the per-step bound itself and is required for every joint. |
| `max_accel` | `joint_vel` only: `glob = value`, the largest change of velocity per second. It becomes the acceleration limit, as for a base. |
| `home`, `home_secs` | The reset pose and the ramp's duration (`joint_pos` only). Joints left out stay where they are. |

`[[robot.joint]]` declares joints the model lacks, such as a custom end effector's
actuator: `name`, `type` (`revolute`, `prismatic` or `continuous`), `lower`, `upper` and
`velocity`. With such declarations, a preset needs no model at all.

For `joint_vel`, limits come from each joint's velocity limit. In `joint_delta` and
`joint_vel` modes, a command that would carry a joint past the model's position limits is
shortened to stop at the limit. It is never reversed, so a joint already past a limit can
only move back in. The check reads the joint just before each command. A delta is taken
relative to the measured position, so a driver must apply it there. A velocity is
assumed to persist until the watchdog would stop it (`command_timeout_s`), so joints
slow down near their limits. MJCF hinge ranges are read
in degrees unless `<compiler angle="radian">`. Joint `<default>` classes, `<include>`
files, every `<worldbody>` and `<actuator>` section, and `<frame>` groups are followed. An
actuator that drives a tendon or a site, not a joint, makes the default selection refuse
until `select` lists the joints.

**Parts.** `[robot.parts]` maps a part name to joints (`left_arm = ["left_shoulder_*",
...]`). Parts become the action's `dim_parts`: episode labels report per part, and
orchestrator tools can claim `part:left_arm`. Joints in no part belong to `body`.

**End effectors, cameras, sensors.**

```toml
[[robot.end_effectors]]
name = "left_gripper"
frame = "left_wrist_yaw_link"    # a URDF link, or an MJCF body, site or camera
offset = [0.12, 0.0, 0.0]        # metres in that frame
joints = ["left_ee_joint"]       # selected joints that actuate it
kind = "gripper"                 # gripper | hand | suction | tool | custom
gripper = "continuous"           # or "binary"
description = "jaw gap in metres, 0 closed, 0.05 open"

[[robot.cameras]]
name = "head_cam"
frame = "d435_link"
width = 640
height = 480

[[robot.sensors]]
name = "pelvis_imu"
frame = "imu_in_pelvis"
kind = "imu"                     # imu | force_torque | contact | lidar | depth | other
```

All of this reaches policies that read text, such as the agent and the orchestrator,
through the embodiment's operating notes.

**State.** `[robot.state] fields` lists what the driver reports each step. The known
fields are `joint_pos`, `joint_vel` and `joint_effort` (one value per selected joint),
plus `base_lin_vel`, `base_ang_vel`, `projected_gravity` (3 each), `base_quat` (wxyz) and
`base_pose` (planar x, y, yaw). `custom = { foot_contact = 4 }` adds others with their
lengths. `joint_pos` control requires `joint_pos`: it is the measured reference that
every target is limited against. The default is `joint_pos` for joint modes, and base
velocities and gravity for `base_vel`.

## Locomotion

`mode = "base_vel"` makes the action a base velocity in the base frame, from
`[robot.base]`:

```toml
[robot.base]
command = ["vx", "yaw_rate"]                  # the action's dimensions (default vx, vy, yaw_rate)
limits = { vx = [-0.5, 1.0], yaw_rate = [-1.5, 1.5] }   # required; must contain zero
max_accel = { vx = 1.0, yaw_rate = 3.0 }      # per second: the acceleration limit
command_timeout_s = 0.25                      # the watchdog
max_tilt_deg = 30                             # fall detection
```

- The **acceleration limit** is an `AccelerationLimitApprover` in the guardrail chain,
  shown as `accel-limit`. It caps each step's change at `max_accel / policy_hz`, starting
  from rest. Stopping ramps too. `joint_vel` presets get the same limit from
  `[robot.joints] max_accel`.
- The **watchdog** stops the driver when no step arrives within `command_timeout_s`
  (here or in `[robot]`; at least two policy periods). Examples are slow synchronous
  inference or a stalled loop. The next step resumes motion, and the acceleration limit
  ramps it up again from rest. A `stop()` that fails is retried, and the next step or
  health check faults the episode. It runs for `base_vel` and `joint_vel`.
- **Fall detection** works in any mode. Past `max_tilt_deg` from upright, measured from
  `projected_gravity` or `base_quat`, the driver stops and the episode ends with
  `fall_detected`. A reset refuses to start from a robot already past it.

A legged robot that walks with a learned controller usually keeps `mode = "joint_pos"`: the
locomotion policy outputs joint targets and reads velocity commands from the live
command stream (`[[robot.commands]]`, [LIVE_CONTROL.md](LIVE_CONTROL.md)). The G1 template
works this way, with `[robot.base] max_tilt_deg` for fall detection. `base_vel` is for
robots whose vendor controller takes velocities directly. One action cannot yet mix
joint targets and a base velocity.

## Commands and controllers

`[[robot.commands]]` declares the live inputs a policy reads from
`observation.extra["commands"]`: `name`, `low`, `high`, `default`, and either `timeout_s`
(back to `default` after that long without an update, 0.5 s by default) or `hold = true`.
`command_events` names the discrete events. `end_episode` is always accepted.

`[[robot.controllers]]` feeds them:

```toml
[[robot.controllers]]
name = "pad"
kind = "joystick"            # Linux joystick API, /dev/input/js*
device = "/dev/input/js0"
layout = "xbox"              # names for buttons and axes; "raw" for indices only
deadman = "lb"               # sticks count only while held
# max_silence_s = 1.0        # opt-in: rest after 1 s with no pad event (a dead wireless link)

[robot.controllers.axes]     # command = stick
vx = { axis = "ly", scale = -1.0, deadzone = 0.05 }
yaw_rate = { axis = "rx", scale = -1.5 }

[robot.controllers.buttons]  # button = event, or a table that sets commands
a = "go"
b = "stop"
start = "end_episode"
y = { set = { speed = 1.5 } }

[[robot.controllers]]
name = "bridge"
kind = "udp"                 # JSON datagrams from another program
port = 9870                  # loopback; host and allow_remote for a trusted network
```

The `xbox` layout follows the Linux xpad driver. Its buttons are `a b x y lb rb back
start guide ls rs`, and its axes are `lx ly lt rx ry rt dx dy`. For another pad, name its
inputs in a table:

```toml
[robot.controllers.layout]
buttons = { cross = 0, circle = 1, l1 = 4 }
axes = { left_x = 0, left_y = 1 }
```

Any other `kind` is a command source of your own. Give it as a registered name, as
`"module:Class"`, or as `"file.py:Class"` relative to the preset. Examples are a vendor's
wireless remote, a ROS topic, or a gamepad API on another platform.
`factory(**args)` builds it from its `[robot.controllers.args]` table. It must have
`start(stream)` and `close()` (`CommandSource` in `core/command_stream.py`). It writes
with `stream.update(...)` and `stream.press(...)`:

```toml
[[robot.controllers]]
name = "remote"
kind = "g1_remote.py:WirelessRemote"
[robot.controllers.args]
network_interface = "eth0"
```

## Drivers

A driver is the only code that talks to the robot or its SDK. The preset embodiment
constructs it as `factory(description, **driver_args)` at the first reset. The
`RobotDescription` holds joint names in action order (`description.joint_names`),
limits, rates, end effectors and cameras. The constructor must not open hardware.

```python
class MyDriver:
    def __init__(self, description, *, port="/dev/ttyUSB0"): ...
    def connect(self): ...  # open the robot; once, at the first reset
    def read(self) -> dict: ...  # every [robot.state] field, joint fields in action order
    def command(self, values): ...  # one approved action, in the preset's mode and order
    def stop(self): ...  # rest now; idempotent, callable from the watchdog thread
    def close(self): ...

    # optional: images() -> {camera: HxWxC uint8}, health() (raise to fault), simulated = False
```

The embodiment owns everything else. It runs the readiness gates, ramps to the home pose
no faster than the per-step limits, and paces steps to `policy_hz`. A joint left past a
narrowed limit walks back in at the step limit; a joint without a home value does not
move. It validates every
`read()`: a missing field, a wrong shape or a non-finite value is an `EmbodimentFault`.
It also runs the watchdog and fall detection, and calls `stop()` and `close()` on every
exit path.

Run a servo loop at `control_hz` inside your driver if the robot needs one. Python's step
loop runs at the policy rate. The harness checks every action before `command()` receives
it. Firmware limits, torque limits and the hardware emergency stop remain the robot's
responsibility.

On a physical driver, `reset()` asks the operator twice through the run's console: once
before the home ramp (once per connection) and once before the episode starts.
Without an attended terminal, a physical preset refuses to reset, before connecting
anything, unless it sets `auto_start = true`, which prints a notice instead. A driver whose
`connect()` fails is closed. Keep someone at the hardware stop.

A plugin package can register its driver with `@registry.driver("my_robot")`. A preset
then names it `driver = "my_robot"` ([ADAPTERS.md](ADAPTERS.md)). A robot whose parts sit
on separate buses still has one driver, and that driver composes the parts. For
example, its `command()` splits the action by `description.parts`.

`driver = "sim"` is the built-in `SimDriver`, a kinematic stand-in with no physics.
Joint targets track with an optional lag (`tau_s`), deltas and velocities integrate, a
base velocity moves a planar pose, and cameras render gray frames. Use it to dry-run a
preset's limits, rates, commands and controllers. `-E driver=sim` dry-runs a physical
preset without touching its hardware.
