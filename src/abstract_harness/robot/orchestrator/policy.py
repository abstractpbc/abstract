"""OrchestratorPolicy: an LLM that runs tools on the robot instead of moving it joint by joint.

The model's tools come from extensions (``extensions.py``): trained policies packaged as
tools (``pour_water``) and skills with their own logic. Calling a tool starts a *job*
and returns its id at once; the job runs on the control loop, every tick, while the
model keeps thinking. Jobs report progress as events (``"cup grabbed"``,
``"pouring begins"``) that wake the model, which can then ``wait``, ``stop_job``,
``steer_job`` (redirect a running job) or start another, much like background tasks.
Every message to the model carries the new events, the running jobs and the current
observation (camera frames unless ``images="on_demand"``). The agent's own motion tool
(``move_joints``, ``move_to``, ``move_by`` or ``drive``) stays available as a built-in
tool, so the model can still nudge the robot directly.

Threads. The model runs on a background thread; jobs run on the rollout thread inside
``act()``, which is called once per control step and returns at once. The model thread
never calls a behavior or the robot: it queues starts, stops and steers that ``act()``
applies at the next step, so a behavior only ever runs on one thread. Jobs driving the
same action dimensions cannot run at once; dimensions no job drives hold (the last
target for absolute modes, zero for displacements and velocities). Every composed
action passes the rollout's approver chain.

Lifecycle. The first ``act()`` of a trial starts the model and waits for its first turn,
so the robot holds until the model has decided something. The trial ends when the model
calls ``done`` or ``give_up`` (all jobs are stopped first), when its call budget runs
out, or by the rollout's own limits; a model or transport error ends it as a policy
error. The model thread stops at trial end and also stops calling the model when
``act()`` has not been called for ``idle_timeout_s`` (a rollout that ended without
``on_trial_end``). The orchestrator needs a loop that keeps ticking while the model
thinks: it declares ``requires_paced_loop``, and ``eval()`` gives it strict timing on
an embodiment that does not pace itself.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from abstract_harness.robot.agent._anthropic import AnthropicClient
from abstract_harness.robot.agent._responses import ResponsesClient
from abstract_harness.robot.agent.policy import (
    AgentPolicyConfig,
    LLMAgentPolicy,
    _evicted_view,
    _image_parts,
    _observation_content,
)
from abstract_harness.robot.core.approver import default_step_limits
from abstract_harness.robot.core.compat import check_compatibility
from abstract_harness.robot.core.embodiment import EmbodimentInfo
from abstract_harness.robot.core.errors import ConfigError
from abstract_harness.robot.core.policy import PolicyInfo
from abstract_harness.robot.core.scene import Scene
from abstract_harness.robot.core.spaces import ABSOLUTE_CONTROL_MODES, reference_state_keys
from abstract_harness.robot.core.types import Action, ActionChunk, Observation
from abstract_harness.robot.orchestrator.extensions import (
    ExtensionAPI,
    ToolSpec,
    load_extensions,
    validate_arguments,
)
from abstract_harness.robot.orchestrator.jobs import (
    Job,
    JobContext,
    JobEvent,
    MotionBehavior,
    PolicyBehavior,
    resolve_controls,
    sub_embodiment,
)

if TYPE_CHECKING:
    from abstract_harness.robot.core.rollout import TrialRecord

_SYSTEM_TEMPLATE = """You supervise a real robot embodiment named {name!r}. You do not \
drive it joint by joint: you run tools, trained policies and skills that act on the robot \
by themselves, and you watch what they report. Calling a tool starts a job and returns its \
id at once; the job keeps running while you think. Its progress arrives as [event] lines \
in later messages, and every message also lists the running jobs and shows the current \
observation. Call wait to let jobs progress until the next event, stop_job to stop one, \
steer_job to redirect one (for example a new instruction or speed), and jobs for their \
status. Jobs that drive the same parts of the robot cannot run at once: stop one before \
starting the other, in the same turn if you like. Safety approvers clamp out-of-bounds \
and too-fast actions below you. Operator feedback lines are trusted guidance from the \
human supervising the robot. When the goal is achieved call done; if it cannot be \
achieved call give_up. You have a budget of {budget} LLM calls for the whole trial, so \
prefer wait over checking in repeatedly."""

_CONTROL_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": (
                "Let running jobs progress. You are called again at the next event, or "
                "after seconds (at most {max_wait}) if given."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "number", "description": "How long to wait at most."},
                    "note": {"type": "string", "description": "What you are waiting for."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_job",
            "description": "Stop a running job. Its dimensions hold where they are.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "steer_job",
            "description": (
                "Redirect a running job without stopping it, e.g. a new instruction for a "
                "policy or new command values. Its reply arrives as an event."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "arguments": {"type": "object", "description": "What to change."},
                },
                "required": ["job_id", "arguments"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jobs",
            "description": "List running jobs and the ones that ended this trial.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
_MAX_CONSECUTIVE_FAILURES = 3
# Observation extras the orchestrator reports as events instead of state lines.
_EVENT_EXTRAS = frozenset({"operator_messages", "command_events"})


@dataclass(frozen=True)
class OrchestratorPolicyConfig(AgentPolicyConfig):
    """The agent's recorded configuration plus the orchestrator's own."""

    extensions: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    builtin_motion: bool = True
    heartbeat_s: float = 15.0
    max_wait_s: float = 30.0
    max_jobs: int = 4


def _number(value: object, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not low < value <= high:
        raise ConfigError(f"{name} must be a number in ({low:g}, {high:g}], got {value!r}")
    return float(value)


class OrchestratorPolicy(LLMAgentPolicy):
    """An LLM that runs extension tools as background jobs (see the module docstring).

    Takes every ``LLMAgentPolicy`` argument, plus ``extensions`` (modules or files,
    comma-separated), ``builtin_motion`` (keep the agent's motion tool, default true),
    ``heartbeat_s`` (call the model with a status update after this long without one
    while jobs run; 0 disables), ``max_wait_s``, ``max_jobs`` (concurrent jobs),
    ``idle_timeout_s``, ``first_turn_timeout_s`` (how long the first step waits for the
    model's first decision) and ``api`` (a pre-built ``ExtensionAPI``, for Python callers).
    """

    accepts_operator_messages: bool = True
    requires_paced_loop: bool = True
    config: AgentPolicyConfig

    def __init__(
        self,
        *,
        extensions: str | list[str] | None = None,
        api: ExtensionAPI | None = None,
        builtin_motion: bool = True,
        heartbeat_s: float = 15.0,
        max_wait_s: float = 30.0,
        max_jobs: int = 4,
        idle_timeout_s: float = 10.0,
        first_turn_timeout_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        **agent_kwargs: Any,
    ) -> None:
        if agent_kwargs.get("pre_check") is not None:
            raise ConfigError("the orchestrator takes no pre_check; give tools their own checks")
        super().__init__(**agent_kwargs)
        if self._images not in ("always", "on_demand"):  # pragma: no cover - parent validates
            raise ConfigError("images must be 'always' or 'on_demand'")
        if isinstance(extensions, str):
            named = tuple(item.strip() for item in extensions.split(",") if item.strip())
        else:
            named = tuple(extensions or ())
        self._api = api if api is not None else load_extensions(named)
        if not isinstance(builtin_motion, bool):
            raise ConfigError("builtin_motion must be true or false")
        self._builtin_motion = builtin_motion
        self._heartbeat_s = (
            0.0 if heartbeat_s == 0 else _number(heartbeat_s, "heartbeat_s", 0.0, 3600.0)
        )
        self._max_wait_s = _number(max_wait_s, "max_wait_s", 0.0, 3600.0)
        if isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < 1:
            raise ConfigError("max_jobs must be an integer >= 1")
        self._max_jobs = max_jobs
        self._idle_timeout_s = _number(idle_timeout_s, "idle_timeout_s", 0.0, 3600.0)
        self._first_turn_timeout_s = _number(
            first_turn_timeout_s, "first_turn_timeout_s", 0.0, 3600.0
        )
        self._clock = clock
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._tool_dims: dict[str, tuple[int, ...]] = {}
        self._schemas: list[dict[str, Any]] = []
        self._absolute = False
        self._state_key: str | None = None
        self._labels: tuple[str, ...] = ()
        self._generation = 0
        # Each policy tool's policy, built once and bound to the dims it drives.
        self._instances: dict[str, Any] = {}
        self._reset_trial_state()
        self.config = OrchestratorPolicyConfig(
            **asdict(self.config),
            extensions=named,
            tools=tuple(self._api.tools),
            builtin_motion=builtin_motion,
            heartbeat_s=self._heartbeat_s,
            max_wait_s=self._max_wait_s,
            max_jobs=max_jobs,
        )

    # -- binding ------------------------------------------------------------------

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Bind the motion toolset, every policy tool's policy, and the tools' dims."""
        super().bind(embodiment_info)
        self.info = PolicyInfo(
            name="orchestrator",
            action_space=embodiment_info.action_space,
            observation_space=embodiment_info.observation_space,
            control_hz=embodiment_info.control_hz,
        )
        self._embodiment_info = embodiment_info
        space = embodiment_info.action_space
        semantics = space.semantics
        assert semantics is not None  # the toolset refuses spaces without semantics
        self._labels = space.labels
        self._absolute = semantics.control_mode in ABSOLUTE_CONTROL_MODES
        self._step_limit = default_step_limits(space)
        if self._absolute:
            keys = reference_state_keys(
                embodiment_info.observation_space.state, space.dim, semantics.control_mode
            )
            self._state_key = keys[0]
        for spec in self._api.tools.values():
            dims = resolve_controls(spec.controls, embodiment_info)
            self._tool_dims[spec.name] = dims
            if spec.policy is None:
                continue
            policy = self._instances.get(spec.name)
            if policy is None:
                # Built once, before any robot moves, and reused by every job of the tool.
                source = spec.policy
                is_instance = not isinstance(source, type) and hasattr(source, "act")
                policy = source if is_instance else source()
                self._instances[spec.name] = policy
            sub_info = sub_embodiment(embodiment_info, dims)
            bind = getattr(policy, "bind", None)
            if callable(bind):
                bind(sub_info)
            report = check_compatibility(policy, cast(Any, SimpleNamespace(info=sub_info)))
            if not report.ok:
                problems = "; ".join(f"{i.code}: {i.message}" for i in report.errors)
                raise ConfigError(
                    f"policy tool {spec.name!r} cannot drive {sub_info.name!r}: {problems}"
                )
        toolset = self._toolset
        assert toolset is not None
        agent_tools = {schema["function"]["name"]: schema for schema in toolset.schemas()}
        schemas = [spec.schema() for spec in self._api.tools.values()]
        for control in _CONTROL_TOOLS:
            control = json.loads(json.dumps(control))
            function = control["function"]
            function["description"] = function["description"].format(
                max_wait=f"{self._max_wait_s:g}"
            )
            schemas.append(control)
        if self._builtin_motion:
            schemas.append(agent_tools[toolset.move_tool])
        if "take_pic" in agent_tools:
            schemas.append(agent_tools["take_pic"])
        schemas += [agent_tools["done"], agent_tools["give_up"]]
        self._schemas = schemas

    @property
    def tools(self) -> list[str]:
        """The model-facing tool names, in schema order."""
        return [schema["function"]["name"] for schema in self._schemas]

    # -- trial lifecycle ------------------------------------------------------------

    def _reset_trial_state(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._claims: dict[int, str] = {}
        self._ops: deque[tuple[Any, ...]] = deque()
        self._inbox: list[JobEvent] = []
        self._history: list[dict[str, Any]] = []
        self._counter = 0
        self._finish: dict[str, Any] | None = None
        self._error: BaseException | None = None
        self._brain: threading.Thread | None = None
        self._first_turn = threading.Event()
        self._latest: Observation | None = None
        self._last_act = self._clock()
        self._in_act = False
        self._policy_jobs: dict[str, PolicyBehavior] = {}
        self._hold: npt.NDArray[np.float64] | None = None
        self._started_at = self._clock()

    def reset(self, scene: Scene) -> None:
        """Stop any previous trial's model thread and jobs, then start a conversation."""
        self._shutdown()
        if isinstance(self._client, ResponsesClient):
            self._client._reset_cache_tracking()
        with self._lock:
            self._generation += 1
            self._reset_trial_state()
            prompt = self._with_context(
                _SYSTEM_TEMPLATE.format(name=self._embodiment_name, budget=self._max_llm_calls),
                None
                if self._max_steps is None
                else (
                    f"The trial ends after {self._max_steps} control steps "
                    f"(about {self._max_steps / (self.info.control_hz or 10.0):.0f}s)."
                ),
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Goal: {scene.instruction}"},
            ]
            self._messages = messages
            self._delta_cursor = 0
            self._calls_used = 0
            self._usage_totals.clear()
            self._hindsight = None
        self._echo(f"[orchestrator] goal: {scene.instruction}")

    def on_trial_end(self, record: TrialRecord, log_dir: str, run_id: str) -> None:
        """Stop the model thread and every job, and record the jobs' timeline."""
        self._shutdown()
        with self._lock:
            history = [dict(entry) for entry in self._history]
        if history:
            record.metadata["jobs"] = history
        super().on_trial_end(record, log_dir, run_id)

    def _shutdown(self) -> None:
        with self._lock:
            self._generation += 1
            brain, self._brain = self._brain, None
            self._wake.notify_all()
        if brain is not None:
            brain.join(timeout=1.0)
        self._stop_all("trial ended")

    def transcript(self) -> list[dict[str, Any]] | None:
        with self._lock:  # the model thread appends to the conversation
            return super().transcript()

    def transcript_delta(self) -> list[dict[str, Any]] | None:
        with self._lock:
            return super().transcript_delta()

    # -- the control loop (rollout thread) -----------------------------------------------

    def act(self, observation: Observation) -> ActionChunk:
        """Apply queued job changes, step every job, and compose this tick's action."""
        if self._toolset is None:
            raise RuntimeError("OrchestratorPolicy.act() before bind()")
        snapshot = replace(
            observation,
            images={key: np.array(value, copy=True) for key, value in observation.images.items()},
            state={key: np.array(value, copy=True) for key, value in observation.state.items()},
            extra={k: v for k, v in observation.extra.items() if k not in _EVENT_EXTRAS},
        )
        with self._lock:
            self._last_act = self._clock()
            self._in_act = True
            self._latest = snapshot
            self._intake(observation)
            if self._hold is None:
                self._hold = self._initial_hold(observation)
            generation = self._generation
            first_turn = self._first_turn
            if self._brain is None and self._finish is None and self._error is None:
                # The first start of the trial, or a restart after the thread found the
                # rollout idle (a job's step that took longer than idle_timeout_s).
                self._brain = threading.Thread(
                    target=self._think,
                    args=(generation, first_turn),
                    name="orchestrator-model",
                    daemon=True,
                )
                self._brain.start()
        try:
            return self._act(observation, generation, first_turn)
        finally:
            with self._lock:
                self._in_act = False
                self._last_act = self._clock()

    def _act(
        self, observation: Observation, generation: int, first_turn: threading.Event
    ) -> ActionChunk:
        # The robot holds until the model's first decision.
        deadline = self._clock() + self._first_turn_timeout_s
        while not first_turn.wait(0.05):
            if self._clock() > deadline:
                raise RuntimeError(
                    f"the model made no first decision in {self._first_turn_timeout_s:g}s"
                )
        if generation != self._generation:
            raise RuntimeError("the trial ended while the model was deciding")
        if self._error is not None:
            self._stop_all("the model failed")
            error = self._error
            raise RuntimeError(
                f"orchestrator model failed: {type(error).__name__}: {error}"
            ) from error
        self._apply_ops()
        if self._finish is not None:
            self._stop_all(str(self._finish.get("stop_reason", "done")))
            assert self._hold is not None
            data = self._hold.copy()  # zeros unless absolute
            return ActionChunk(
                actions=[Action(data=data, meta=dict(self._finish))],
                control_hz=self.info.control_hz,
            )
        data = self._compose(observation)
        return ActionChunk(actions=[Action(data=data)], control_hz=self.info.control_hz)

    def _initial_hold(self, observation: Observation) -> npt.NDArray[np.float64]:
        if self._absolute and self._state_key is not None:
            return np.asarray(observation.state[self._state_key], dtype=np.float64).copy()
        return np.zeros(len(self._labels))

    def _intake(self, observation: Observation) -> None:
        """Operator feedback and button events wake the model (called under the lock)."""
        for message in observation.extra.get("operator_messages") or ():
            if isinstance(message, Mapping) and isinstance(message.get("text"), str):
                self._post(JobEvent("operator", f"operator feedback: {message['text']}"))
        for event in observation.extra.get("command_events") or ():
            if isinstance(event, Mapping) and isinstance(event.get("name"), str):
                self._post(JobEvent("operator", f"operator command: {event['name']}"))

    def _post(self, event: JobEvent) -> None:
        with self._lock:
            self._inbox.append(event)
            self._wake.notify_all()

    def _apply_ops(self) -> None:
        with self._lock:
            ops = list(self._ops)
            self._ops.clear()
        for op in ops:
            kind = op[0]
            if kind == "start":
                self._start(*op[1:])
            elif kind == "stop":
                job = self._jobs.get(op[1])
                if job is not None:
                    self._end(job, "stopped", op[2] or "stopped by the model")
            elif kind == "steer":
                job = self._jobs.get(op[1])
                if job is None:
                    self._post(JobEvent(op[1], "steer ignored: the job already ended"))
                    continue
                try:
                    reply = job.steer(op[2])
                except Exception as exc:
                    self._post(JobEvent(job.id, f"steer refused: {exc}"))
                else:
                    self._post(JobEvent(job.id, reply, wake=False))

    def _start(
        self,
        job_id: str,
        tool: str,
        arguments: dict[str, Any],
        dims: tuple[int, ...],
        factory: Callable[[JobContext], Any],
        max_seconds: float | None,
        monitors: tuple[Any, ...],
    ) -> None:
        context = JobContext(
            job_id=job_id,
            tool=tool,
            arguments=arguments,
            embodiment=self._embodiment_info,
            dims=dims,
            post=self._post,
            clock=self._clock,
        )
        try:
            previous = self._policy_jobs.get(tool)
            if previous is not None and previous.busy():
                raise RuntimeError(f"the previous {tool} inference is still running")
            behavior = factory(context)
            job = Job(behavior, context, max_seconds=max_seconds, monitors=monitors)
            if isinstance(behavior, PolicyBehavior):
                self._policy_jobs[tool] = behavior
        except Exception as exc:
            with self._lock:
                self._release(job_id)
                self._history.append(self._entry(job_id, tool, arguments, "failed", str(exc)))
            self._post(JobEvent(job_id, f"failed to start: {type(exc).__name__}: {exc}"))
            return
        with self._lock:
            self._jobs[job_id] = job
            self._history.append(self._entry(job_id, tool, arguments, "running", ""))
        self._post(JobEvent(job_id, f"started; drives {', '.join(context.labels)}", wake=False))

    def _entry(
        self, job_id: str, tool: str, arguments: dict[str, Any], status: str, text: str
    ) -> dict[str, Any]:
        return {
            "job": job_id,
            "tool": tool,
            "arguments": arguments,
            "status": status,
            "summary": text,
            "started_s": round(self._clock() - self._started_at, 3),
            "ended_s": None if status == "running" else round(self._clock() - self._started_at, 3),
        }

    def _compose(self, observation: Observation) -> npt.NDArray[np.float64]:
        assert self._hold is not None
        dim = len(self._labels)
        data = self._hold.copy()  # zeros unless absolute
        ended: list[tuple[Job, str, str]] = []
        for job in list(self._jobs.values()):
            try:
                values = job.step(observation, dim)
            except Exception as exc:
                ended.append((job, "failed", f"{type(exc).__name__}: {exc}"))
                continue
            if values is not None:
                if not np.all(np.isfinite(values)):
                    ended.append((job, "failed", "returned a non-finite action"))
                    continue
                data[list(job.context.dims)] = values
            if job.ended is not None:
                ended.append((job, job.ended[0], job.ended[1]))
        if self._absolute:
            self._hold = data.copy()
        for job, status, text in ended:
            self._end(job, status, text)
        return data

    def _end(self, job: Job, status: str, text: str) -> None:
        with self._lock:
            if self._jobs.pop(job.id, None) is None:
                return
            self._release(job.id)
            latest = self._latest
            if (
                status != "finished"
                and self._absolute
                and self._hold is not None
                and latest is not None
            ):
                # A stopped, failed or timed-out job's last target may be far ahead of the
                # robot, and the approvers would keep walking toward it. Hold at most one
                # step limit from where the robot is: a moving joint stops, a target it
                # already reached (a gripper squeezing an object) stays. A job that
                # finished keeps its final target.
                assert self._state_key is not None
                measured = np.asarray(latest.state[self._state_key], dtype=np.float64)
                dims = list(job.context.dims)
                reach = self._step_limit[dims]
                self._hold[dims] = np.clip(
                    self._hold[dims], measured[dims] - reach, measured[dims] + reach
                )
            for entry in reversed(self._history):
                if entry["job"] == job.id:
                    entry.update(
                        status=status,
                        summary=text,
                        ended_s=round(self._clock() - self._started_at, 3),
                    )
                    break
        try:
            job.stop()
        except Exception as exc:
            text = f"{text} (its stop() raised {type(exc).__name__}: {exc})"
        self._post(JobEvent(job.id, f"{status}: {text}" if text else status))

    def _release(self, job_id: str) -> None:
        for index in [i for i, owner in self._claims.items() if owner == job_id]:
            del self._claims[index]

    def _stop_all(self, reason: str) -> None:
        for job in list(self._jobs.values()):
            self._end(job, "stopped", reason)

    # -- the model (its own thread) ------------------------------------------------------

    def _current(self, generation: int) -> bool:
        return generation == self._generation and self._finish is None and self._error is None

    def _think(self, generation: int, first_turn: threading.Event) -> None:
        failures = 0
        resume_now = True
        waiting_until: float | None = None
        last_turn = self._clock()
        try:
            while True:
                with self._wake:
                    while True:
                        if not self._current(generation):
                            return
                        now = self._clock()
                        if not self._in_act and now - self._last_act > self._idle_timeout_s:
                            # The rollout stopped calling act(): spend nothing more. The
                            # next act() starts a new thread if the trial goes on.
                            if self._brain is threading.current_thread():
                                self._brain = None
                            return
                        if resume_now or any(event.wake for event in self._inbox):
                            break
                        if waiting_until is not None and now >= waiting_until:
                            break
                        if (
                            self._jobs
                            and self._heartbeat_s
                            and now - last_turn >= self._heartbeat_s
                        ):
                            break
                        self._wake.wait(0.05)
                    events, self._inbox = self._inbox, []
                    observation = self._latest
                resume_now, waiting_until = False, None
                last_turn = self._clock()
                outcome = self._turn(generation, first_turn, events, observation)
                if outcome is None:
                    return
                if outcome == "nudged":
                    failures += 1
                    if failures >= _MAX_CONSECUTIVE_FAILURES:
                        raise RuntimeError(
                            f"LLM produced no tool call in {failures} consecutive turns"
                        )
                    resume_now = True
                    continue
                failures = 0
                kind, seconds = outcome if isinstance(outcome, tuple) else (outcome, None)
                if kind == "continue":
                    resume_now = True
                elif kind == "wait":
                    limit = seconds if seconds is not None else self._max_wait_s
                    waiting_until = self._clock() + limit
                elif kind == "jobs" and not self._jobs and not self._ops:
                    # Nothing left to wait for: the model has to decide again.
                    waiting_until = self._clock() + self._max_wait_s
        except BaseException as exc:
            with self._lock:
                if generation == self._generation:
                    self._error = exc
        finally:
            first_turn.set()

    def _turn(
        self,
        generation: int,
        first_turn: threading.Event,
        events: list[JobEvent],
        observation: Observation | None,
    ) -> str | tuple[str, float | None] | None:
        """One model call and its tool calls. Returns what the thread does next:
        "continue", ("wait", seconds), "jobs" (wait for job events), "nudged", or None
        when the trial is over."""
        if self._calls_used >= self._max_llm_calls:
            with self._lock:
                if generation == self._generation:
                    self._finish = {
                        "request_stop": True,
                        "stop_reason": "give_up",
                        "stop_detail": "LLM call budget exhausted",
                    }
            first_turn.set()
            return None
        content = self._message(events, observation)
        with self._lock:
            if not self._current(generation):
                return None
            self._messages.append({"role": "user", "content": content})
            outgoing = list(self._messages)
        if self._image_horizon is not None:
            outgoing = _evicted_view(
                outgoing,
                self._image_horizon,
                mark_anchor=isinstance(self._client, AnthropicClient | ResponsesClient),
            )
        message = self._client.complete(
            outgoing, self._schemas, temperature=self._temperature, reasoning_effort=self._effort
        )
        with self._lock:
            if not self._current(generation):
                return None
            self._calls_used += 1
            if message.usage is not None:
                for key, value in message.usage.items():
                    self._usage_totals[key] = self._usage_totals.get(key, 0) + value
            self._messages.append(message.raw())
        if message.content:
            self._echo(f"[orchestrator] << {message.content}")
        if message.refusal and not message.tool_calls:
            raise RuntimeError(f"LLM refused: {message.refusal}")
        if not message.tool_calls:
            with self._lock:
                self._messages.append(
                    {
                        "role": "user",
                        "content": "Respond with tool calls: start a tool, wait, or call done.",
                    }
                )
            first_turn.set()
            return "nudged"
        next_step: str | tuple[str, float | None] = "continue"
        frames: tuple[str, ...] = ()
        ended = False
        for call in message.tool_calls:
            if ended:
                result = "ignored: the trial ends with an earlier call"
            else:
                try:
                    result, effect = self._call(call, observation, generation)
                except Exception as exc:  # a tool's bug is the model's error, not the thread's
                    result, effect = f"{call.name} failed: {type(exc).__name__}: {exc}", None
                self._echo(f"[orchestrator] {call.name}: {result}")
                if effect == "end":
                    ended = True
                elif effect == "frames":
                    frames = tuple(observation.images) if observation is not None else ()
                elif isinstance(effect, tuple) and effect[0] == "wait":
                    next_step = effect
                elif effect == "jobs" and next_step == "continue":
                    next_step = "jobs"
            with self._lock:
                self._messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        if frames and observation is not None:
            with self._lock:
                self._messages.append({"role": "user", "content": _image_parts(observation)})
        first_turn.set()
        if ended:
            return None
        return next_step

    def _message(
        self, events: list[JobEvent], observation: Observation | None
    ) -> list[dict[str, Any]]:
        elapsed = self._clock() - self._started_at
        lines = [f"[harness] t={elapsed:.1f}s"]
        for event in events:
            lines.append(f"[event] {event.job}: {event.text}")
        with self._lock:
            running = list(self._jobs.values())
            pending = [op[1] for op in self._ops if op[0] == "start"]
        if running or pending:
            lines.append(
                "Running jobs: "
                + "; ".join(
                    [_describe(job) for job in running]
                    + [f"{job_id} (starting)" for job_id in pending]
                )
            )
        else:
            lines.append("No jobs running.")
        if observation is None:
            return [{"type": "text", "text": "\n".join(lines)}]
        reveal = () if self._images == "on_demand" else None
        content = _observation_content(
            observation, self._state_labels, max_steps=self._max_steps, reveal=reveal
        )
        content[0] = {"type": "text", "text": "\n".join(lines) + "\n" + content[0]["text"]}
        return content

    def _call(self, call: Any, observation: Observation | None, generation: int) -> tuple[str, Any]:
        """Run one tool call on the model thread: a result for the model and its effect."""
        try:
            arguments = json.loads(call.arguments) if call.arguments else {}
        except (TypeError, ValueError):
            return f"arguments for {call.name} are not valid JSON", None
        if not isinstance(arguments, dict):
            return f"arguments for {call.name} must be a JSON object", None
        toolset = self._toolset
        assert toolset is not None
        name = call.name
        if name in self._api.tools:
            return self._start_tool(self._api.tools[name], arguments)
        if name == "wait":
            seconds = arguments.get("seconds")
            if seconds is not None:
                try:
                    seconds = _number(seconds, "seconds", 0.0, self._max_wait_s)
                except ConfigError as exc:
                    return str(exc), None
            return ("waiting" + (f" up to {seconds:g}s" if seconds else " for the next event")), (
                "wait",
                seconds,
            )
        if name == "stop_job":
            job_id = str(arguments.get("job_id", ""))
            with self._lock:
                known = job_id in self._jobs or any(
                    op[0] == "start" and op[1] == job_id for op in self._ops
                )
                if not known:
                    return f"no running job {job_id!r}; running: {self._running_ids()}", None
                self._release(job_id)
                self._ops.append(("stop", job_id, str(arguments.get("reason") or "")))
            return f"stopping {job_id}", "jobs"
        if name == "steer_job":
            job_id = str(arguments.get("job_id", ""))
            changes = arguments.get("arguments")
            if not isinstance(changes, dict) or not changes:
                return "arguments must be a non-empty object of what to change", None
            with self._lock:
                job = self._jobs.get(job_id)
                # A job started earlier in this turn is still queued; queued operations
                # apply in order, so its steer follows its start.
                pending = [op[2] for op in self._ops if op[0] == "start" and op[1] == job_id]
                tool = job.context.tool if job is not None else pending[0] if pending else None
                if tool is None:
                    return f"no running job {job_id!r}; running: {self._running_ids()}", None
                spec = self._api.tools.get(tool)
                if spec is not None:
                    problem = validate_arguments(spec.parameters, changes, partial=True)
                    if problem is not None:
                        return f"steer_job {job_id}: {problem}", None
                self._ops.append(("steer", job_id, changes))
            return f"steering {job_id}; its reply arrives as an event", "jobs"
        if name == "jobs":
            return self._status(), None
        if name == "take_pic":
            return "frames attached below", "frames"
        if name in ("done", "give_up"):
            if observation is None:
                return "no observation yet", None
            result = toolset.execute(call, observation)
            if result.chunk is None:
                return result.error or "invalid call", None
            meta = dict(result.chunk.actions[0].meta)
            with self._lock:
                if generation != self._generation:
                    return "ignored: the trial already ended", "end"
                self._hindsight = meta.get("stop_hindsight")
                self._finish = meta
            return result.note, "end"
        if name == toolset.move_tool and self._builtin_motion:
            return self._start_motion(call, arguments, observation)
        return f"unknown tool {name!r}; available: {', '.join(self.tools)}", None

    def _running_ids(self) -> str:
        ids = list(self._jobs) + [op[1] for op in self._ops if op[0] == "start"]
        return ", ".join(ids) if ids else "none"

    def _status(self) -> str:
        with self._lock:
            lines = [
                f"{job.id}: running, {job.context.elapsed_s:.1f}s, drives "
                + ", ".join(job.context.labels)
                for job in self._jobs.values()
            ]
            lines += [
                f"{entry['job']}: {entry['status']} {entry['summary']}".rstrip()
                for entry in self._history
                if entry["status"] != "running"
            ]
        return "\n".join(lines) if lines else "no jobs yet"

    def _claim(self, dims: tuple[int, ...]) -> str | None:
        """Why ``dims`` cannot be claimed now, or ``None`` (called under the lock)."""
        busy = sorted({self._claims[i] for i in dims if i in self._claims})
        if busy:
            labels = [self._labels[i] for i in dims if i in self._claims]
            return (
                f"{', '.join(busy)} already drives {', '.join(labels)}; stop it first "
                "(stop_job, in this same turn if you like) or wait for it to finish"
            )
        running = len(set(self._claims.values()))
        if running >= self._max_jobs:
            return f"{running} jobs are running, the most at once; stop one first"
        return None

    def _enqueue(
        self,
        tool: str,
        arguments: dict[str, Any],
        dims: tuple[int, ...],
        factory: Callable[[JobContext], Any],
        max_seconds: float | None,
        monitors: tuple[Any, ...] = (),
    ) -> tuple[str, Any]:
        with self._lock:
            problem = self._claim(dims)
            if problem is not None:
                return problem, None
            self._counter += 1
            job_id = f"{tool}#{self._counter}"
            for index in dims:
                self._claims[index] = job_id
            self._ops.append(
                ("start", job_id, tool, arguments, dims, factory, max_seconds, monitors)
            )
        labels = ", ".join(self._labels[i] for i in dims)
        return (
            f"started {job_id} (drives {labels}); it begins at the next control step and "
            "reports events as it runs",
            "jobs",
        )

    def _start_tool(self, spec: ToolSpec, arguments: dict[str, Any]) -> tuple[str, Any]:
        problem = validate_arguments(spec.parameters, arguments)
        if problem is not None:
            return f"{spec.name}: {problem}", None
        dims = self._tool_dims[spec.name]
        previous = self._policy_jobs.get(spec.name)
        if previous is not None and previous.busy():
            return f"the previous {spec.name} inference is still running; try again shortly", None
        factory: Callable[[JobContext], Any]
        if spec.policy is not None:
            policy = self._instances[spec.name]

            def policy_job(context: JobContext) -> PolicyBehavior:
                return PolicyBehavior(
                    policy,
                    context,
                    instruction=str(arguments.get("instruction") or spec.instruction or ""),
                    controller=spec.controller() if spec.controller is not None else None,
                    commands=spec.commands,
                )

            factory = policy_job
        else:
            start = spec.start
            assert start is not None

            def behavior_job(context: JobContext) -> Any:
                return start(context, **arguments)

            factory = behavior_job
        return self._enqueue(spec.name, arguments, dims, factory, spec.max_seconds, spec.monitors)

    def _start_motion(
        self, call: Any, arguments: dict[str, Any], observation: Observation | None
    ) -> tuple[str, Any]:
        if observation is None:
            return "no observation yet", None
        toolset = self._toolset
        assert toolset is not None
        result = toolset.execute(call, observation)
        if result.error is not None or result.chunk is None:
            return result.error or "invalid motion", None
        values_key = toolset.values_key
        index = {label: i for i, label in enumerate(self._labels)}
        dims = tuple(sorted(index[str(label)] for label in arguments[values_key]))
        chunk, target = result.chunk, result.target

        def residual(after: Observation) -> str | None:
            if target is None:
                return None
            measured = toolset.residual(target, after)
            if measured is None:
                return None
            label, magnitude = measured
            return f"motion complete; largest remaining offset {magnitude:.4g} on {label}"

        def factory(context: JobContext) -> MotionBehavior:
            return MotionBehavior(chunk, context, residual)

        text, effect = self._enqueue(call.name, arguments, dims, factory, None)
        if effect is not None:
            text = f"{text}; {result.note}"
        return text, effect


def _describe(job: Job) -> str:
    labels = ", ".join(job.context.labels)
    return f"{job.id} ({job.context.elapsed_s:.1f}s, drives {labels})"


def orchestrator_policy(**kwargs: Any) -> OrchestratorPolicy:
    """Factory the registry calls (policy ``orchestrator``); the CLI's ``-P key=value``
    and ``[agent.args]`` arrive here."""
    return OrchestratorPolicy(**kwargs)


__all__ = ["OrchestratorPolicy", "OrchestratorPolicyConfig", "orchestrator_policy"]
