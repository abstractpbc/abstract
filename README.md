# abstract-harness

A Python runtime for evaluating robot policies and supervising their execution. Connect a
policy, an embodiment, a controller and an approval chain; keep a structured record of
what the policy requested and what the robot did.

The package includes a bimanual I2RT YAM adapter, a deterministic CubePick test environment,
HTTP action-policy clients, Python callable adapters, and a tool-calling agent. The core,
agent and YAM components derive from Robocurve's inspect-robots projects; their immutable
upstream bases and licenses are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

This is an early, supervised robotics runtime. Its software guardrails require a measured
robot configuration and independent hardware protection. Read [the safety contract](docs/SAFETY.md)
before connecting an actuator. Parking is an attempted lifecycle operation and can fail.

## Start without hardware

Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/) are required for source development.
From this directory:

```sh
uv sync --locked
uv run abstract eval --no-prompt
uv run pytest
```

The default evaluation uses CubePick and a scripted policy. It connects no robot, downloads
no model weights and calls no paid service.

```python
from abstract_harness import Scene, Task, eval, success_at_end

logs = eval(
    Task(
        name="reach",
        scenes=[Scene(id="s0", instruction="reach the cube", init_seed=0)],
        scorer=success_at_end(),
        max_steps=80,
    ),
    policy="scripted",
    embodiment="cubepick",
    log_dir="runs/reach",
)
print(logs[0].results.metrics)
```

## Installation and support

| Environment | Source-checkout installation | Supported contract |
| --- | --- | --- |
| Core, Python 3.12/3.13 | `uv sync --locked` | Protocols, evaluation, controllers, agents with mocked transports, CubePick, HTTP clients |
| Generic robot runtime | `uv sync --locked --extra robot` | Rerun recording, LeRobot v3 writer, OpenCV and MuJoCo collision queries |
| YAM, Linux and SocketCAN | `uv sync --locked --extra yam --group yam-driver` | Bimanual YAM adapter and RealSense; requires operator acceptance on the configured rig |

The YAM native driver is a separate dependency group pinned to an immutable Git revision.
The `yam` wheel extra supplies the runtime libraries and Linux RealSense SDK; it does not
install a bare `i2rt` from a package index. Use the source-checkout recipe above for a rig.
The guided missing-driver error also provides a separately pinned installation command.
Driver revisions require new hardware acceptance evidence.

A built core wheel can be installed independently. CI builds it, checks its bundled
licenses and assets, and exercises it in a fresh environment outside the checkout.
CubePick is a small deterministic test environment. It does not validate real-world
robot dynamics. Other physical robots are extension points, with no supplied Franka or UR driver.

## Policies and controllers

Every policy implements `act(observation) -> ActionChunk`. Use `CallableVLAPolicy` to wrap
an in-process model, or the registered `policy` factory to select a callable, a `/v1`
gateway or a local `/act` server. `policy` at the package top level is a registration
decorator; `resolve("policy", name, **arguments)` constructs a registered policy.

Fixed model semantics stay fixed. Joint positions, joint deltas and Cartesian targets are
not interchangeable. Compatibility checks reject conflicting control modes, frames,
gripper conventions and declared joint order; transforming values requires an explicit adapter.
A callable adopts the robot's spaces only where the caller has left its contract unspecified.

`DefaultController` executes buffered chunks; `SmoothingController` filters them;
`EnsemblingController` combines overlapping predictions. Terminal action metadata survives
these paths. The controller's timing contract and the adapter's own pacing must match.
`ActionChunk.control_hz` is advisory; it does not resample actions automatically.

The gateway presets include stateless action models and stateful DreamZero-Astra/Picasso
sessions. For the latter, continuation samples control observations at offsets
`(-23, -16, -8, 0)` from a 24-step window, per camera. The controller observes every step,
including buffered steps. Premature replanning is refused rather than padded with repeated
inference frames. A direct caller must supply intermediate observations through `observe()`.
Session IDs, step numbers, dimensions, rates and frame counts are checked before accepting a reply.
Model availability and any model-weight licenses belong to the serving provider.

## Configure a robot

The examples in [rigs/](rigs/) declare policy, embodiment, recording formats and optional
storage. All examples are checked against the actual factories without opening hardware.
Use a local copy with the rig's real interfaces, camera identities, limits and measured geometry.

For a prepared YAM rig, the operator runs:

```sh
uv run abstract robot check --config rigs/yam-abstract.toml
uv run abstract robot run --config rigs/yam-abstract.toml "fold the towel"
```

`run`, `holdcheck`, `health` and pose capture/goto can energize or move the arms. `check`
reads readiness information without enabling motors. Immediately before native YAM bring-up,
the runtime also claims the configured devices and refuses unarmed loss-of-comms motor timeouts.
Keep an operator at the hardware stop throughout a run.

Reference secrets by environment-variable name, for example `api_key_env`, rather than
putting key values in TOML or command-line overrides. Literal credential arguments are
rejected; persisted setup snapshots are redacted defensively. Gateway clients automatically
use `ABSTRACT_API_KEY` only for `https://api.abstract.inc`. A custom endpoint uses its own
explicit credential reference. Remote `/v1` gateways require HTTPS; loopback HTTP is allowed.

The separately distributed Abstract account CLI provides `abstract login`. This Python
package also installs an `abstract` executable: keep their executable paths separate if both
are installed. Setting the required environment variable avoids relying on command lookup.
The agent's local budget meter is an estimate, not a provider-enforced invoice cap; agent
runs require `--allow-paid`. Gateway inference may also incur provider charges.

## Records and export

Python `eval()` writes JSON and action logs. Raw frames require `store_frames=True`.
The robot command adds recording and runs the formats chosen in `[data]` after the episode.
Rerun and LeRobot require the robot runtime; they are not created by every core evaluation.

Action rows distinguish requested action, approved action, execution outcome and an
adapter-supplied command. YAM supplies the final joint command and command time; observation
records include state and camera publication times. LeRobot omits transitions where no
action executed, and uses final joint commands when exporting joint-position control. Its timestamps use the nominal frame rate; use forensic logs for actual
elapsed time, inference stalls and settling. `Task.max_seconds` resolves to a step budget.

Upload verifies transported bytes. Local raw frames are deleted only when selected exports
have completion evidence whose hashes still match the actual files. Failed conversions,
missing receipts and degraded recordings retain raw frames locally. A local `.uploaded.json`
receipt prevents automatic repeated uploads; repair the export and explicitly upload the
run again when ready. `runs.jsonl` keeps the upload result and retention reason.

```sh
uv run abstract robot show <run_dir>
uv run abstract robot view <run_dir>
uv run abstract robot video <run_dir>
```

## Extending and contributing

See [ADAPTERS.md](docs/ADAPTERS.md) for a complete fake adapter, lifecycle ownership and units;
[ROBOT.md](docs/ROBOT.md) for the YAM runbook and storage; [RPC.md](docs/RPC.md) for the
operator server; and [AGENTS.md](AGENTS.md) for coding-agent architecture and invariants.
[CONTRIBUTING.md](CONTRIBUTING.md) describes validation and release gates.

Version 0.1 is pre-stable: public API changes are recorded in [CHANGELOG.md](CHANGELOG.md).
Do not depend on underscore-prefixed internals. Report sensitive defects privately as
described in [SECURITY.md](SECURITY.md).

## License

Abstract-authored code is Apache-2.0. The distribution also contains MIT-licensed derived
code and model assets. See [LICENSE](LICENSE), [NOTICE](NOTICE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); all ship in the wheel and source archive.
