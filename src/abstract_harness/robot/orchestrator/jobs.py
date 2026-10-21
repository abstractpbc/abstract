"""Jobs: what a tool call starts, and how the control loop steps it.

A tool's factory returns a ``Behavior``. While its job runs, the orchestrator calls
``behavior.step(observation)`` once per control tick, on the rollout thread, and the
result drives the dimensions the job controls. A behavior never touches the robot: its
actions are composed with the other running jobs' and pass the rollout's approver
chain like any policy's. It reports through its ``JobContext``: ``emit`` sends a
progress line to the model (``"cup grabbed"``), ``finish`` and ``fail`` end the job.

``step`` returns one of:

* an ``Action``, an array or a sequence of floats: this tick's values, either for every
  action dimension (the job's are taken) or for exactly the job's dimensions;
* an ``ActionChunk``: played one action per tick, ``step`` is called again when it runs out;
* ``None``: hold this tick.

An action whose ``meta`` requests a stop (``request_stop``, as policies signal ``done``)
finishes the job; the trial ends only when the model calls ``done`` or ``give_up``.
Optional methods: ``steer(**arguments) -> str | None`` redirects a running job (its reply
is shown to the model), and ``stop()`` runs once when the job ends for any reason.
"""

from __future__ import annotations

import fnmatch
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from abstract_harness.robot.core.controller import DefaultController
from abstract_harness.robot.core.embodiment import EmbodimentInfo
from abstract_harness.robot.core.errors import ConfigError
from abstract_harness.robot.core.scene import Scene
from abstract_harness.robot.core.spaces import ActionSemantics, Box
from abstract_harness.robot.core.types import Action, ActionChunk, Observation

if TYPE_CHECKING:
    from abstract_harness.robot.core.controller import Controller
    from abstract_harness.robot.core.policy import Policy


@runtime_checkable
class Behavior(Protocol):
    """One running job's logic (see the module docstring)."""

    def step(self, observation: Observation) -> Any:
        """This tick's action for the job's dimensions, a chunk, or ``None`` to hold."""
        ...


@dataclass
class JobEvent:
    """One line for the model: a job's progress, its end, or a reply to a steer."""

    job: str
    text: str
    wake: bool = True
    kind: str = "progress"  # progress | finished | failed | stopped | steered | started
    time: float = field(default_factory=time.monotonic)


class JobContext:
    """What a behavior gets from the orchestrator: its identity, the robot, and the
    channel back to the model."""

    def __init__(
        self,
        *,
        job_id: str,
        tool: str,
        arguments: Mapping[str, Any],
        embodiment: EmbodimentInfo,
        dims: tuple[int, ...],
        post: Callable[[JobEvent], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.job_id = job_id
        self.tool = tool
        self.arguments = dict(arguments)
        self.embodiment = embodiment
        self.dims = dims
        labels = (
            embodiment.action_space.semantics.dim_labels
            if embodiment.action_space.semantics is not None
            else None
        )
        all_labels = labels or tuple(str(i) for i in range(embodiment.action_space.dim))
        self.labels = tuple(all_labels[i] for i in dims)
        #: Job-local live commands, merged over the stream's in ``extra["commands"]``.
        self.commands: dict[str, float] = {}
        #: Scratch space for the behavior and its monitors.
        self.state: dict[str, Any] = {}
        self.steps = 0
        self._post = post
        self._clock = clock
        self._started = clock()
        self.outcome: tuple[str, str] | None = None  # (finished | failed, text)

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self._started

    def emit(self, text: str, *, wake: bool = True) -> None:
        """Report progress to the model; ``wake=False`` waits for its next turn."""
        self._post(JobEvent(self.job_id, str(text)[:2000], wake=wake))

    def finish(self, summary: str = "") -> None:
        """End the job successfully after this tick."""
        if self.outcome is None:
            self.outcome = ("finished", str(summary)[:2000])

    def fail(self, reason: str) -> None:
        """End the job as failed after this tick."""
        if self.outcome is None:
            self.outcome = ("failed", str(reason)[:2000])


def resolve_controls(patterns: Sequence[str] | None, info: EmbodimentInfo) -> tuple[int, ...]:
    """Action dimensions from label globs or ``part:<name>`` entries (``None``: all)."""
    dim = info.action_space.dim
    if not patterns:
        return tuple(range(dim))
    semantics = info.action_space.semantics
    labels = (semantics.dim_labels if semantics is not None else None) or tuple(
        str(i) for i in range(dim)
    )
    parts = semantics.dim_parts if semantics is not None else None
    chosen: list[int] = []
    for pattern in patterns:
        if pattern.startswith("part:"):
            name = pattern.removeprefix("part:")
            found = [i for i, part in enumerate(parts or ()) if part == name]
        else:
            found = [i for i, label in enumerate(labels) if fnmatch.fnmatchcase(label, pattern)]
        if not found:
            raise ConfigError(
                f"controls entry {pattern!r} matches no action dimension of {info.name!r}"
            )
        chosen += [i for i in found if i not in chosen]
    return tuple(sorted(chosen))


def sub_embodiment(info: EmbodimentInfo, dims: tuple[int, ...]) -> EmbodimentInfo:
    """The embodiment as a policy that drives only ``dims`` sees it."""
    space = info.action_space
    if dims == tuple(range(space.dim)):
        return info
    index = list(dims)
    semantics = space.semantics
    sub_semantics = None
    if semantics is not None:

        def pick(values: tuple[Any, ...] | None) -> tuple[Any, ...] | None:
            return tuple(values[i] for i in index) if values is not None else None

        sub_semantics = ActionSemantics(
            control_mode=semantics.control_mode,
            rotation_repr=semantics.rotation_repr,
            gripper=semantics.gripper,
            frame=semantics.frame,
            dim_labels=pick(semantics.dim_labels),
            max_step=pick(semantics.max_step),
            dim_parts=pick(semantics.dim_parts),
        )
    box = Box(
        shape=(len(index),),
        low=None if space.low is None else np.asarray(space.low).reshape(-1)[index],
        high=None if space.high is None else np.asarray(space.high).reshape(-1)[index],
        semantics=sub_semantics,
    )
    return replace(info, action_space=box)


class Job:
    """A running tool call: its behavior, context, dimensions and chunk buffer."""

    def __init__(
        self,
        behavior: Any,
        context: JobContext,
        *,
        max_seconds: float | None = None,
        monitors: Sequence[Callable[[Observation, JobContext], None]] = (),
    ) -> None:
        if not callable(getattr(behavior, "step", None)):
            raise ConfigError(
                f"tool {context.tool!r} returned {type(behavior).__name__}, which has no step()"
            )
        self.behavior = behavior
        self.context = context
        self.max_seconds = max_seconds
        self.monitors = tuple(monitors)
        self._buffer: deque[Action] = deque()
        self.ended: tuple[str, str] | None = None

    @property
    def id(self) -> str:
        return self.context.job_id

    def step(self, observation: Observation, dim: int) -> npt.NDArray[np.float64] | None:
        """This tick's values for the job's dims (``None``: hold); sets ``ended`` when the
        job finishes, fails or times out."""
        context = self.context
        if self.max_seconds is not None and context.elapsed_s > self.max_seconds:
            context.fail(f"timed out after {self.max_seconds:g}s")
            self.ended = context.outcome
            return None
        if not self._buffer:
            out = self.behavior.step(observation)
            if isinstance(out, ActionChunk):
                self._buffer.extend(out.actions)
            elif out is not None:
                self._buffer.append(
                    out if isinstance(out, Action) else Action(data=np.asarray(out))
                )
        values: npt.NDArray[np.float64] | None = None
        if self._buffer:
            action = self._buffer.popleft()
            values = self._values(action, dim)
            if action.meta.get("request_stop"):
                detail = action.meta.get("stop_detail") or action.meta.get("stop_reason") or "done"
                context.finish(str(detail))
            events = action.meta.get("events")
            if isinstance(events, list | tuple):
                for text in events:
                    context.emit(str(text))
        for monitor in self.monitors:
            monitor(observation, context)
        context.steps += 1
        if context.outcome is not None:
            self.ended = context.outcome
        return values

    def _values(self, action: Action, dim: int) -> npt.NDArray[np.float64]:
        data = np.asarray(action.data, dtype=np.float64).reshape(-1)
        dims = self.context.dims
        if data.size == len(dims):
            return data
        if data.size == dim:
            return data[list(dims)]
        raise ValueError(
            f"returned {data.size} values; expected {len(dims)} (its dimensions) or {dim}"
        )

    def steer(self, arguments: Mapping[str, Any]) -> str:
        steer = getattr(self.behavior, "steer", None)
        if not callable(steer):
            raise ValueError(f"{self.context.tool} cannot be steered; stop it and start another")
        reply = steer(**arguments)
        return str(reply) if reply else "steered"

    def stop(self) -> None:
        stop = getattr(self.behavior, "stop", None)
        if callable(stop):
            stop()


class PolicyBehavior:
    """A trained policy packaged as a tool: each job resets it with its instruction and
    steps it through a controller. ``steer(instruction=...)`` changes the instruction;
    other steered values become the job's live commands (for command-conditioned
    policies such as a locomotion controller). The policy's ``request_stop`` finishes
    the job; action ``meta["events"]`` strings are reported as progress."""

    def __init__(
        self,
        policy: Policy,
        context: JobContext,
        *,
        instruction: str | None,
        controller: Controller | None = None,
        commands: Sequence[str] = (),
    ) -> None:
        self.policy = policy
        self.context = context
        self.instruction = instruction
        self.commands = tuple(commands)
        self.controller = controller or DefaultController(policy.config.replan_interval)
        self._store: dict[str, Any] = {}
        self._t = 0
        for name in self.commands:
            if name in context.arguments:
                context.commands[name] = float(context.arguments[name])
        policy.reset(Scene(id=context.job_id, instruction=instruction))

    def step(self, observation: Observation) -> Action:
        extra = dict(observation.extra)
        if self.context.commands:
            extra["commands"] = {**dict(extra.get("commands") or {}), **self.context.commands}
        local = replace(
            observation,
            instruction=self.instruction or observation.instruction,
            extra={**extra, "env_step": self._t},
        )
        action = self.controller.next_action(self.policy, local, self._t, self._store)
        self._t += 1
        return action

    def steer(self, **arguments: Any) -> str:
        changed = []
        instruction = arguments.pop("instruction", None)
        if instruction is not None:
            self.instruction = str(instruction)
            changed.append(f"instruction {self.instruction!r}")
        for name, value in arguments.items():
            if self.commands and name not in self.commands:
                raise ValueError(f"unknown command {name!r}; this tool takes {list(self.commands)}")
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be a number")
            self.context.commands[name] = float(value)
            changed.append(f"{name}={float(value):g}")
        if not changed:
            raise ValueError("nothing to steer: pass instruction or command values")
        return "now " + ", ".join(changed)

    def stop(self) -> None:
        return None


class MotionBehavior:
    """The built-in motion tool (``move_joints``, ``move_to``, ``move_by``, ``drive``)
    as a job: plays the speed-limited chunk the agent toolset synthesized, then
    finishes with the measured residual when one exists."""

    def __init__(
        self, chunk: ActionChunk, context: JobContext, residual: Callable[[Observation], str | None]
    ):
        self._actions = deque(chunk.actions)
        self._context = context
        self._residual = residual

    def step(self, observation: Observation) -> Action | None:
        if not self._actions:
            self._context.finish(self._residual(observation) or "motion complete")
            return None
        action = self._actions.popleft()
        return Action(data=action.data)

    def stop(self) -> None:
        self._actions.clear()
