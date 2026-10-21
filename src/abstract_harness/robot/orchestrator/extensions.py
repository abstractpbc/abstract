"""Extensions: modules that add tools to the orchestrator, as Pi's coding agent does.

An extension is a Python module (or file) with a ``register(harness)`` function; one
module may register any number of tools::

    # kitchen.py
    from abstract_harness.robot.orchestrator import ExtensionAPI

    def register(harness: ExtensionAPI) -> None:
        harness.register_policy_tool(
            name="pour_water",
            description="Pour from the cup in the right gripper into the glass.",
            policy=load_pour_policy,          # a Policy, or a factory returning one
            instruction="pour the water into the glass",
            monitors=[cup_events],            # (observation, ctx) -> None, every tick
            controls=["part:right_arm"],
            max_seconds=60,
        )
        harness.register_tool(
            name="wave",
            description="Wave the left arm.",
            parameters={"type": "object", "properties": {"times": {"type": "integer"}}},
            start=Wave,                       # (ctx, **arguments) -> Behavior
            controls=["left_*"],
        )

Extensions load when the orchestrator is constructed, before any robot moves, and their
tools appear in the model's context. Name them in ``extensions`` (comma-separated):
a module path (``my_pkg.kitchen``), a file (``./skills/kitchen.py``), ``module:function``
for another registration function, or a name registered with ``@registry.extension``
(how an installed plugin package provides one; see ``core/registry.py``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from abstract_harness.robot.core import registry
from abstract_harness.robot.core.errors import ConfigError

if TYPE_CHECKING:
    from abstract_harness.robot.core.controller import Controller
    from abstract_harness.robot.core.policy import Policy
    from abstract_harness.robot.core.types import Observation
    from abstract_harness.robot.orchestrator.jobs import JobContext

TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
#: Names the orchestrator itself uses for its control and built-in motion tools.
RESERVED = frozenset(
    {"wait", "stop_job", "steer_job", "jobs", "done", "give_up", "take_pic"}
    | {"move_joints", "move_to", "move_by", "drive"}
)
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


@dataclass(frozen=True)
class ToolSpec:
    """One registered tool: what the model sees and how a job starts. A behavior tool
    has ``start``; a policy tool has ``policy`` (a ``Policy`` or a factory for one)."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    start: Callable[..., Any] | None = None
    controls: tuple[str, ...] | None = None
    max_seconds: float | None = None
    monitors: tuple[Callable[[Observation, JobContext], None], ...] = ()
    extension: str = ""
    policy: Any = None
    instruction: str | None = None
    commands: tuple[str, ...] = ()
    controller: Callable[[], Controller] | None = None

    def schema(self) -> dict[str, Any]:
        """The model-facing function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description
                + " Runs as a background job: the call returns a job id at once, and the "
                "job's progress arrives as events while it runs.",
                "parameters": dict(self.parameters),
            },
        }


def validate_arguments(
    schema: Mapping[str, Any], arguments: Any, *, partial: bool = False
) -> str | None:
    """A small JSON Schema check of a call's arguments: an object, required keys present
    (unless ``partial``, for a steer that changes some of them), declared property types
    and enums respected, no extra keys when ``additionalProperties`` is false. Returns
    the problem, or ``None``."""
    if not isinstance(arguments, dict):
        return "arguments must be a JSON object"
    properties = schema.get("properties") or {}
    for name in () if partial else schema.get("required") or ():
        if name not in arguments:
            return f"missing required argument {name!r}"
    if schema.get("additionalProperties") is False:
        extra = sorted(set(arguments) - set(properties))
        if extra:
            return f"unknown arguments {extra}; valid: {sorted(properties)}"
    for name, value in arguments.items():
        spec = properties.get(name)
        if not isinstance(spec, Mapping):
            continue
        kind = spec.get("type")
        allowed = _JSON_TYPES.get(kind) if isinstance(kind, str) else None
        if allowed is not None:
            if isinstance(value, bool) and kind in ("number", "integer"):
                return f"{name} must be a {kind}"
            if not isinstance(value, allowed):
                return f"{name} must be a {kind}"
        if "enum" in spec and value not in spec["enum"]:
            return f"{name} must be one of {list(spec['enum'])}"
    return None


class ExtensionAPI:
    """What an extension's ``register(harness)`` receives."""

    def __init__(self) -> None:
        self.tools: dict[str, ToolSpec] = {}
        self._current = ""

    def register_tool(
        self,
        *,
        name: str,
        description: str,
        start: Callable[..., Any],
        parameters: Mapping[str, Any] | None = None,
        controls: Sequence[str] | None = None,
        max_seconds: float | None = None,
        monitors: Sequence[Callable[[Observation, JobContext], None]] = (),
    ) -> None:
        """Add a tool whose job runs ``start(ctx, **arguments)``, a ``Behavior``.

        ``parameters`` is the JSON Schema of the arguments (default: none).
        ``controls`` names the action dimensions it drives, as label globs or
        ``part:<name>`` (default: all); jobs that drive the same dimensions cannot run
        at once. ``max_seconds`` fails a job that runs longer. ``monitors`` run after
        every step with ``(observation, ctx)``, to emit progress or end the job.
        """
        self._add(
            ToolSpec(
                name=name,
                description=description,
                parameters=parameters or {"type": "object", "properties": {}},
                start=start,
                controls=tuple(controls) if controls else None,
                max_seconds=max_seconds,
                monitors=tuple(monitors),
                extension=self._current,
            )
        )

    def register_policy_tool(
        self,
        *,
        name: str,
        description: str,
        policy: Policy | Callable[[], Policy],
        instruction: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        commands: Sequence[str] = (),
        controls: Sequence[str] | None = None,
        max_seconds: float | None = None,
        monitors: Sequence[Callable[[Observation, JobContext], None]] = (),
        controller: Callable[[], Controller] | None = None,
    ) -> None:
        """Package a trained policy as a tool.

        Each job resets ``policy`` (built once, from a factory if one is given, when the
        orchestrator binds to the robot) with ``instruction`` or the call's own
        ``instruction`` argument, then steps it every tick through ``controller()``
        (default: ``DefaultController``; ``AsyncController`` overlaps inference).
        ``commands`` are live-command names the call (and ``steer_job``) may set, for
        command-conditioned policies: a locomotion policy's ``vx`` and ``yaw_rate``.
        The policy's ``request_stop`` finishes the job, and action ``meta["events"]``
        strings are reported as progress; ``monitors`` add events of your own.
        """
        if parameters is None:
            properties: dict[str, Any] = {
                "instruction": {
                    "type": "string",
                    "description": "Language instruction for the policy"
                    + (f" (default: {instruction!r})." if instruction else "."),
                }
            }
            for command in commands:
                properties[command] = {"type": "number", "description": f"live command {command}"}
            parameters = {"type": "object", "properties": properties}
        self._add(
            ToolSpec(
                name=name,
                description=description,
                parameters=parameters,
                controls=tuple(controls) if controls else None,
                max_seconds=max_seconds,
                monitors=tuple(monitors),
                extension=self._current,
                policy=policy,
                instruction=instruction,
                commands=tuple(commands),
                controller=controller,
            )
        )

    def _add(self, spec: ToolSpec) -> None:
        if not TOOL_NAME.match(spec.name):
            raise ConfigError(f"tool name {spec.name!r} must match {TOOL_NAME.pattern}")
        if spec.name in RESERVED:
            raise ConfigError(f"tool name {spec.name!r} is reserved by the orchestrator")
        if spec.name in self.tools:
            raise ConfigError(
                f"tool {spec.name!r} is registered twice ({self.tools[spec.name].extension} "
                f"and {spec.extension})"
            )
        if (spec.start is None) == (spec.policy is None):
            raise ConfigError(f"tool {spec.name!r}: give start or policy, not both")
        if spec.start is not None and not callable(spec.start):
            raise ConfigError(f"tool {spec.name!r}: start must be callable")
        if not isinstance(spec.parameters, Mapping) or spec.parameters.get("type") != "object":
            raise ConfigError(f"tool {spec.name!r}: parameters must be an object JSON Schema")
        if spec.max_seconds is not None and not spec.max_seconds > 0:
            raise ConfigError(f"tool {spec.name!r}: max_seconds must be positive")
        if not spec.description.strip():
            raise ConfigError(f"tool {spec.name!r} needs a description the model can read")
        self.tools[spec.name] = spec


def _register_function(spec: str) -> Callable[[ExtensionAPI], Any]:
    function = registry.registered("extension").get(spec)
    if function is None:
        try:
            function = registry.load_object(spec, default="register")
        except ConfigError as exc:
            raise ConfigError(f"extension {exc}") from exc
    if not callable(function):
        raise ConfigError(f"extension {spec!r} has no register(harness) function")
    return cast(Callable[[ExtensionAPI], Any], function)


def load_extensions(specs: str | Sequence[str] | None) -> ExtensionAPI:
    """Import each extension and let it register its tools."""
    api = ExtensionAPI()
    if specs is None:
        return api
    items = [item.strip() for item in specs.split(",")] if isinstance(specs, str) else list(specs)
    for item in items:
        if not item:
            continue
        api._current = item
        try:
            function = _register_function(item)
            function(api)
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(
                f"extension {item!r} failed to load: {type(exc).__name__}: {exc}"
            ) from exc
    api._current = ""
    return api
