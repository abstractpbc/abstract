# The orchestrator: policies and skills as LLM tools

The agent policy drives the robot through motion tools (`move_joints`, `move_to`, ...),
one motion per model call. The orchestrator gives the model higher-level tools instead,
in the style of the Pi coding agent's extensions:

- Each tool is a trained policy packaged as a tool (`pour_water`) or a skill with its
  own logic.
- Calling a tool starts a **job** and returns its id at once.
- The job runs on the robot's control loop while the model keeps thinking, and streams
  events back (`"cup grabbed"`, `"pouring begins"`, `"cup is put down"`).
- The model reacts as the events arrive: it waits, stops or redirects the job, or starts
  another.

## Extensions

An extension is a module or file with a `register(harness)` function. One module can
register any number of tools:

```python
# skills/kitchen.py
import numpy as np
from abstract_harness.robot.orchestrator import ExtensionAPI, JobContext


def cup_events(observation, ctx: JobContext) -> None:
    """A monitor: runs after every step of the job, on the control loop."""
    gripper = float(observation.state["joint_pos"][-1])
    if gripper < 0.01 and not ctx.state.get("grabbed"):
        ctx.state["grabbed"] = True
        ctx.emit("cup grabbed")


class Wave:
    """A skill: its own execution loop, one step per control tick."""

    def __init__(self, ctx: JobContext, times: int = 2) -> None:
        self.ctx, self.left = ctx, times * 20

    def step(self, observation):
        self.left -= 1
        if self.left == 0:
            self.ctx.finish("waved")
        return np.array([0.3 * np.sin(self.left / 3.0)])  # values for its dims

    def steer(self, times: int) -> str:  # optional
        self.left = times * 20
        return f"waving {times} more times"


def register(harness: ExtensionAPI) -> None:
    harness.register_policy_tool(
        name="pour_water",
        description="Pour from the cup in the right gripper into the glass in front.",
        policy=load_pour_policy,  # a Policy, or a factory called once at startup
        instruction="pour the water into the glass",
        monitors=[cup_events],
        controls=["part:right_arm", "part:right_ee"],
        max_seconds=60,
    )
    harness.register_tool(
        name="wave",
        description="Wave the left wrist.",
        parameters={"type": "object", "properties": {"times": {"type": "integer"}}},
        start=Wave,  # (ctx, **arguments) -> a behavior
        controls=["left_wrist_roll_joint"],
    )
```

Name the extensions in `extensions`, separated by commas: a file (`./skills/kitchen.py`),
a module (`my_pkg.kitchen`), `module:function` for another registration function, or the
name of an installed package's `abstract_harness.extensions` entry point. Extensions
load when the orchestrator is constructed, before anything moves. Each policy tool's
policy is built and bound once, when the orchestrator binds to the robot, and checked
for compatibility with the dimensions it drives. A mismatch is refused at startup.

**Behaviors.** A tool's `start(ctx, **arguments)` returns a behavior. The orchestrator
calls `behavior.step(observation)` once per control tick on the rollout thread. It
returns this tick's values: an `Action`, an array or a list, either for every action
dimension (the job's are taken) or for exactly the job's dimensions. It can also return
an `ActionChunk`, played one action per tick, or `None` to hold. Optional methods are
`steer(**arguments) -> str` to redirect a running job and `stop()`, which runs once when
the job ends for any reason.

**The context.** `ctx` is the job's link to the orchestrator:

| Member | What it does |
| --- | --- |
| `emit(text, wake=True)` | Sends progress to the model. `wake=False` waits for the model's next turn, for high-rate telemetry. |
| `finish(summary)`, `fail(reason)` | End the job. |
| `job_id`, `tool`, `arguments` | The job's identity and call. |
| `labels`, `dims` | The dimensions the job drives. |
| `embodiment` | The robot's `EmbodimentInfo`. |
| `elapsed_s`, `steps` | How long the job has run. |
| `commands` | Job-local live commands. |
| `state` | Scratch space shared with monitors. |

**Policy tools.** Each job resets the policy with its instruction (the call's
`instruction` argument, or the tool's default) and steps it through a controller.
`controller=AsyncController` overlaps inference with execution. When the policy
requests a stop, the job finishes; the trial does not end. Strings in an action's
`meta["events"]` are reported as progress. `commands=["vx", "yaw_rate"]` makes those
values call arguments and `steer_job` targets, delivered to the policy as live commands.
This packages a command-conditioned locomotion policy as a `walk` tool that the model
steers.

**Controls.** `controls` lists label globs or `part:<name>` entries, and defaults to every
dimension. Jobs that drive the same dimensions cannot run at once, so the model must stop
one before starting the other, in the same turn if it likes. Dimensions no job drives
hold. For absolute modes, a job's dimensions hold the robot's measured position from
when the job ended, not its last target, so a stopped job stops the robot. For
displacements and velocities they hold zero. A policy tool's next job waits until its
previous job's background inference has returned.
`max_jobs` caps how many jobs run at once.

## What the model sees

The model gets the extension tools, the control tools `wait`, `stop_job`, `steer_job` and
`jobs`, the agent's motion tool as a built-in job (`builtin_motion=false` drops it), and
`done` and `give_up`. Every message carries:

- the new events since the last call (`[event] pour_water#1: cup grabbed`);
- the running jobs, with their ages and dimensions;
- the current observation: state, approvals, and camera frames unless
  `images="on_demand"`, which adds `take_pic`.

The model is called again when one of these happens:

- an event that wakes arrives: progress, a job's end, operator feedback or a button
  press;
- a `wait` with `seconds` runs out;
- `heartbeat_s` passes while jobs run with no new events (15 s by default; 0 disables
  it).

Starting, stopping and steering wait for the next event; status calls continue at once.
A `done` or `give_up` stops every job and ends the trial.

## Threads and safety

- **Two threads.** The model runs on its own thread. Jobs run on the rollout thread
  inside `act()`, which is called once per control step and returns immediately. The
  model thread never calls a behavior or touches the robot: it queues starts, stops and
  steers, and `act()` applies them at the next step. A behavior therefore only ever runs
  on one thread.
- **Guardrails.** Every composed action passes the rollout's approver chain like any
  policy's.
- **First turn.** The first step of a trial waits for the model's first decision, up to
  `first_turn_timeout_s` (300 s), so the robot holds until then.
- **Errors.** A model or transport error ends the trial as a policy error, and the robot
  parks.
- **Idle rollout.** The model thread stops calling the model when `act()` has not been
  called for `idle_timeout_s` (10 s), so a rollout that ended unexpectedly spends nothing
  more. A long-running step does not count as idle, and if the trial resumes, the next
  `act()` restarts the thread.
- **Tool errors.** A tool call that raises becomes an error result for the model; steer
  arguments are checked against the tool's schema.
- **Pacing.** The orchestrator needs a loop that keeps ticking while the model thinks. It
  declares `requires_paced_loop`, so it gets strict timing automatically on an
  embodiment that does not pace itself ([LIVE_CONTROL.md](LIVE_CONTROL.md)).
- **Records.** Each trial's metadata keeps the jobs' timeline under `jobs`: tool,
  arguments, status, summary, and start and end times.

## Running it

```sh
# Python or the eval CLI, any provider the agent supports:
uv run abstract eval --policy orchestrator --instruction "find and reach the cube" \
    -P model=anthropic/claude-sonnet-5-5 \
    -P extensions=examples/extensions/cubepick_tools.py --no-prompt

# A robot config: the spend meter, the budget and --allow-paid apply as for [agent].
uv run abstract robot run --config rigs/cubepick-orchestrator.toml --allow-paid "find the cube"
```

```toml
[agent]
policy = "orchestrator"
model = "gpt-6.1-sol"
# ... pricing, budget and limits as for the agent ...

[agent.args]
extensions = "../skills/kitchen.py"   # files are relative to this TOML
heartbeat_s = 10.0
max_jobs = 2
```

`abstract robot check` rehearses an orchestrator config like an agent config: it runs on
CubePick against a stand-in API, with no spend and no motion.
