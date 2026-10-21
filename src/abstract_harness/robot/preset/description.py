"""A robot preset resolved against its model: the spaces, limits and inputs a run uses.

``load_preset(path)`` reads the preset TOML, reads its model file, and resolves every
cross-reference: joint selection and order, limits (the model's, narrowed by the
preset), per-step limits, the home pose, parts, end-effector and camera frames, the
state the driver reports, the base command, live command axes and controller bindings.
Everything is checked before any driver is constructed, so ``abstract robot check``
reports a wrong joint name or frame without touching hardware.
"""

from __future__ import annotations

import fnmatch
import hashlib
import math
import tomllib
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError

from abstract_harness.robot.core.command_stream import (
    END_EPISODE,
    LAYOUTS,
    AxisBinding,
    AxisSpec,
    ButtonBinding,
    CommandSource,
    CommandStream,
    JoystickSource,
    UdpCommandSource,
)
from abstract_harness.robot.core.embodiment import RESETTABLE, SELF_PACED, EmbodimentInfo
from abstract_harness.robot.core.errors import ConfigError
from abstract_harness.robot.core.spaces import (
    CANONICAL_STATE_UNITS,
    ActionSemantics,
    Box,
    CameraSpec,
    ControlMode,
    GripperKind,
    ObservationSpace,
    StateField,
    StateSpec,
)
from abstract_harness.robot.preset.model import ModelError, ModelJoint, RobotModel, load_model
from abstract_harness.robot.preset.schema import (
    KNOWN_STATE,
    ButtonActionDecl,
    CameraDecl,
    ControllerDecl,
    EndEffector,
    PresetFile,
    RobotSection,
    SensorDecl,
)

_DEFAULT_STATE = {"base_vel": ("base_lin_vel", "base_ang_vel", "projected_gravity")}
_REST_PART = "body"


class PresetError(ConfigError):
    """A preset that cannot be resolved; the message names the field and the fix."""


def _suggest(name: str, choices: list[str] | set[str] | frozenset[str]) -> str:
    close = get_close_matches(name, sorted(choices), n=3)
    return f" (did you mean {', '.join(repr(c) for c in close)}?)" if close else ""


@dataclass(frozen=True)
class ResolvedEndEffector:
    """An end effector with its actuating joints resolved to action dimensions."""

    name: str
    frame: str
    offset: tuple[float, float, float]
    joints: tuple[str, ...]
    kind: str
    gripper: str
    description: str


@dataclass(frozen=True, eq=False)
class RobotDescription:
    """Everything a run needs from a preset, resolved and validated. Drivers receive it."""

    name: str
    path: Path
    section: RobotSection
    model: RobotModel | None
    mode: ControlMode
    policy_hz: float
    control_hz: float
    labels: tuple[str, ...]
    low: npt.NDArray[np.float64]
    high: npt.NDArray[np.float64]
    max_step: tuple[float | None, ...] | None
    home: tuple[float | None, ...] | None
    parts: tuple[str, ...] | None
    joints: tuple[ModelJoint, ...]
    end_effectors: tuple[ResolvedEndEffector, ...]
    cameras: tuple[CameraDecl, ...]
    sensors: tuple[SensorDecl, ...]
    state_fields: tuple[StateField, ...]
    digest: str
    driver_args: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.labels)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """The joints the action drives, in action order (empty for ``base_vel``)."""
        return tuple(joint.name for joint in self.joints)

    def state_shape(self, key: str) -> tuple[int, ...]:
        for state_field in self.state_fields:
            if state_field.key == key:
                return state_field.shape
        raise KeyError(key)

    def embodiment_info(self, *, simulated: bool, pace: bool) -> EmbodimentInfo:
        gripper_kinds = {ee.gripper for ee in self.end_effectors if ee.joints}
        gripper: GripperKind = (
            "binary" if "binary" in gripper_kinds else "continuous" if gripper_kinds else "none"
        )
        semantics = ActionSemantics(
            control_mode=self.mode,
            gripper=gripper,
            dim_labels=self.labels,
            max_step=self.max_step,
            dim_parts=self.parts,
        )
        capabilities = {RESETTABLE}
        if pace:
            capabilities.add(SELF_PACED)
        return EmbodimentInfo(
            name=self.name,
            action_space=Box(
                shape=(self.dim,), low=self.low.copy(), high=self.high.copy(), semantics=semantics
            ),
            observation_space=ObservationSpace(
                cameras=tuple(
                    CameraSpec(c.name, height=c.height, width=c.width, channels=c.channels)
                    for c in self.cameras
                ),
                state=StateSpec(self.state_fields),
            ),
            control_hz=self.policy_hz,
            is_simulated=simulated,
            capabilities=frozenset(capabilities),
            docs=self.docs(),
            environment_id=f"preset:{self.name}",
            environment_revision=self.digest[:16],
        )

    def docs(self) -> str:
        """Operating notes for policies that read text (the agent's system prompt)."""
        section = self.section
        lines = [f"Robot preset {self.name!r}"]
        if self.model is not None:
            lines[0] += f", model {self.model.path.name} ({self.model.format.upper()})"
        lines[0] += (
            f"; policy rate {self.policy_hz:g} Hz, driver servo rate {self.control_hz:g} Hz."
        )
        unit = {"joint_pos": "", "joint_delta": " per step", "joint_vel": "/s"}.get(self.mode, "")
        if self.mode == "base_vel":
            lines.append(
                "The action is the base velocity in the base frame: "
                + ", ".join(
                    f"{label} [{lo:.3g}, {hi:.3g}]"
                    for label, lo, hi in zip(self.labels, self.low, self.high, strict=True)
                )
                + " (m/s for linear, rad/s for yaw). Zero is a stop."
            )
        else:
            by_part: dict[str, list[str]] = {}
            for index, label in enumerate(self.labels):
                part = self.parts[index] if self.parts else "joints"
                joint = self.joints[index]
                by_part.setdefault(part, []).append(
                    f"{label} [{self.low[index]:.3g}, {self.high[index]:.3g}] {joint.unit}{unit}"
                )
            lines.append(f"Action ({self.mode}), by part:")
            lines.extend(f"- {part}: {', '.join(items)}" for part, items in by_part.items())
        for ee in self.end_effectors:
            offset = ", ".join(f"{value:g}" for value in ee.offset)
            text = f"End effector {ee.name} ({ee.kind}) at frame {ee.frame}, offset [{offset}] m"
            if ee.joints:
                text += f", actuated by {', '.join(ee.joints)} ({ee.gripper})"
            if ee.description:
                text += f": {ee.description}"
            lines.append(text + ".")
        for camera in self.cameras:
            text = f"Camera {camera.name} ({camera.width}x{camera.height})"
            if camera.frame:
                text += f" at frame {camera.frame}"
            if camera.description:
                text += f": {camera.description}"
            lines.append(text + ".")
        for sensor in self.sensors:
            text = f"Sensor {sensor.name} ({sensor.kind})"
            if sensor.frame:
                text += f" at frame {sensor.frame}"
            if sensor.description:
                text += f": {sensor.description}"
            lines.append(text + ".")
        if section.commands:
            lines.append(
                "Live commands (observation.extra['commands']): "
                + "; ".join(
                    f"{axis.name} in [{axis.low:g}, {axis.high:g}]"
                    + (f" ({axis.description})" if axis.description else "")
                    for axis in section.commands
                )
                + "."
            )
        if section.description.strip():
            lines.append(section.description.strip())
        return "\n".join(lines)

    def command_stream(self) -> CommandStream | None:
        """The preset's live command axes, events and controllers, or ``None``."""
        section = self.section
        if not section.commands and not section.controllers and not section.command_events:
            return None
        axes = [
            AxisSpec(
                axis.name,
                low=float(axis.low),
                high=float(axis.high),
                default=float(axis.default),
                timeout_s=None if axis.hold else axis.timeout_s,
                description=axis.description,
            )
            for axis in section.commands
        ]
        events = section.command_events or None
        sources = [_controller_source(decl, axes) for decl in section.controllers]
        return CommandStream(axes, events=events, sources=sources)

    def summary(self) -> str:
        """A few lines for the doctor: what the preset resolved to."""
        model = (
            f"{self.model.path.name} ({self.model.format}, {len(self.model.joints)} joints)"
            if self.model is not None
            else "none (joints declared in the preset)"
        )
        lines = [
            f"preset: {self.path}",
            f"  model: {model}",
            f"  action: {self.mode}, {self.dim} dims at {self.policy_hz:g} Hz "
            f"(driver servo {self.control_hz:g} Hz), driver {self.section.driver}",
        ]
        if self.parts:
            counts: dict[str, int] = {}
            for part in self.parts:
                counts[part] = counts.get(part, 0) + 1
            lines.append("  parts: " + ", ".join(f"{p} ({n})" for p, n in counts.items()))
        if self.end_effectors:
            lines.append(
                "  end effectors: "
                + ", ".join(f"{ee.name} @ {ee.frame}" for ee in self.end_effectors)
            )
        if self.cameras:
            lines.append("  cameras: " + ", ".join(c.name for c in self.cameras))
        lines.append("  state: " + ", ".join(f.key for f in self.state_fields))
        section = self.section
        if section.commands or section.command_events:
            # Described from the declarations: building the stream would open its sources.
            axes = [f"{a.name} [{a.low:g}, {a.high:g}]" for a in section.commands]
            events = (
                [f"events {', '.join(section.command_events)}"] if section.command_events else []
            )
            lines.append("  commands: " + "; ".join(axes + events))
        if section.controllers:
            lines.append(
                "  controllers: " + ", ".join(f"{c.name} ({c.kind})" for c in section.controllers)
            )
        return "\n".join(lines)


def _button_index(key: str | int, layout: str, controller: str) -> int:
    if isinstance(key, int):
        return key
    if key.isdigit():
        return int(key)
    names = LAYOUTS.get(layout, {}).get("buttons", {})
    if key not in names:
        raise PresetError(
            f"controller {controller}: unknown button {key!r} for layout {layout!r}"
            f"{_suggest(key, set(names))}"
        )
    return names[key]


def _axis_index(key: str | int, layout: str, controller: str) -> int:
    if isinstance(key, int):
        return key
    names = LAYOUTS.get(layout, {}).get("axes", {})
    if key not in names:
        raise PresetError(
            f"controller {controller}: unknown axis {key!r} for layout {layout!r}"
            f"{_suggest(key, set(names))}"
        )
    return names[key]


def _controller_source(decl: ControllerDecl, axes: list[AxisSpec]) -> CommandSource:
    if decl.kind == "udp":
        return UdpCommandSource(decl.host, decl.port, allow_remote=decl.allow_remote)
    bindings = [
        AxisBinding(
            command,
            axis=_axis_index(binding.axis, decl.layout, decl.name),
            scale=float(binding.scale),
            deadzone=binding.deadzone,
        )
        for command, binding in decl.axes.items()
    ]
    buttons = []
    for key, action in decl.buttons.items():
        index = _button_index(key, decl.layout, decl.name)
        if isinstance(action, ButtonActionDecl):
            buttons.append(
                ButtonBinding(
                    index, event=action.event, set={k: float(v) for k, v in action.set.items()}
                )
            )
        else:
            buttons.append(ButtonBinding(index, event=action))
    deadman = (
        _button_index(decl.deadman, decl.layout, decl.name) if decl.deadman is not None else None
    )
    return JoystickSource(
        decl.device,
        axes=bindings,
        buttons=buttons,
        deadman=deadman,
        refresh_hz=decl.refresh_hz,
        source=decl.name,
    )


def _matches(pattern: str, names: list[str]) -> list[str]:
    return [name for name in names if fnmatch.fnmatchcase(name, pattern)]


def _select_joints(section: RobotSection, model: RobotModel | None) -> list[ModelJoint]:
    declared = [
        ModelJoint(
            name=decl.name,
            type=decl.type,
            lower=None if decl.lower is None else float(decl.lower),
            upper=None if decl.upper is None else float(decl.upper),
            velocity=decl.velocity,
        )
        for decl in section.joint
    ]
    pool: list[ModelJoint] = list(model.movable) if model is not None else []
    if model is not None:
        clash = sorted({j.name for j in declared} & {j.name for j in model.joints})
        if clash:
            raise PresetError(f"[[robot.joint]] {clash} are already joints of the model")
    pool += declared
    by_name = {joint.name: joint for joint in pool}
    names = list(by_name)
    if section.joints.select is None:
        chosen = names
    else:
        chosen = []
        for pattern in section.joints.select:
            found = _matches(pattern, names)
            if not found:
                every = [j.name for j in model.joints] if model is not None else names
                fixed = [n for n in every if fnmatch.fnmatchcase(n, pattern) and n not in by_name]
                why = (
                    f" (a fixed or unsupported joint: {fixed})"
                    if fixed
                    else _suggest(pattern, names)
                )
                raise PresetError(
                    f"[robot.joints] select {pattern!r} matches no movable joint{why}"
                )
            chosen += [name for name in found if name not in chosen]
    for pattern in section.joints.exclude:
        if not _matches(pattern, names):
            raise PresetError(
                f"[robot.joints] exclude {pattern!r} matches no movable joint"
                f"{_suggest(pattern, names)}"
            )
        chosen = [name for name in chosen if not fnmatch.fnmatchcase(name, pattern)]
    if not chosen:
        raise PresetError("[robot.joints] selects no joints")
    return [by_name[name] for name in chosen]


def _per_joint(table: dict[str, Any], names: list[str], what: str) -> dict[str, Any]:
    """Glob-keyed values per joint; later patterns win, every pattern must match."""
    out: dict[str, Any] = {}
    for pattern, value in table.items():
        found = _matches(pattern, names)
        if not found:
            raise PresetError(
                f"[robot.joints] {what} {pattern!r} matches no selected joint"
                f"{_suggest(pattern, names)}"
            )
        for name in found:
            out[name] = value
    return out


def _frame(name: str, model: RobotModel | None, what: str) -> str:
    if model is not None and name not in model.frames:
        raise PresetError(
            f"{what}: frame {name!r} is not a link, body, site or camera of "
            f"{model.path.name}{_suggest(name, model.frames)}"
        )
    return name


def _joint_limits(
    section: RobotSection, joints: list[ModelJoint]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    names = [joint.name for joint in joints]
    overrides = _per_joint(section.joints.limits, names, "limits")
    steps = _per_joint(section.joints.max_step, names, "max_step")
    low, high = np.zeros(len(joints)), np.zeros(len(joints))
    for index, joint in enumerate(joints):
        if section.mode == "joint_delta":
            if joint.name not in steps:
                raise PresetError(
                    f"joint_delta needs [robot.joints] max_step for every joint; "
                    f"{joint.name} has none"
                )
            low[index], high[index] = -steps[joint.name], steps[joint.name]
            continue
        if section.mode == "joint_vel":
            base = (-joint.velocity, joint.velocity) if joint.velocity is not None else None
            kind = "velocity"
        else:
            base = (
                (joint.lower, joint.upper)
                if joint.lower is not None and joint.upper is not None
                else None
            )
            kind = "position"
        override = overrides.get(joint.name)
        if override is not None:
            lo, hi = float(override[0]), float(override[1])
            if base is not None and (lo < base[0] - 1e-12 or hi > base[1] + 1e-12):
                raise PresetError(
                    f"[robot.joints] limits for {joint.name} [{lo:g}, {hi:g}] exceed the "
                    f"model's {kind} limits [{base[0]:g}, {base[1]:g}]; a preset can only "
                    "narrow them"
                )
            base = (lo, hi)
        if base is None:
            raise PresetError(
                f"joint {joint.name} ({joint.type}) has no {kind} limits in the model; set "
                f"them in [robot.joints] limits"
            )
        low[index], high[index] = base
    return low, high


def _base_limits(section: RobotSection) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert section.base is not None
    missing = [name for name in section.base.command if name not in section.base.limits]
    if missing:
        raise PresetError(f'mode = "base_vel" needs [robot.base] limits for {missing}')
    low = np.array([float(section.base.limits[name][0]) for name in section.base.command])
    high = np.array([float(section.base.limits[name][1]) for name in section.base.command])
    if bool(np.any(low > 0)) or bool(np.any(high < 0)):
        raise PresetError("[robot.base] limits must contain zero (a stop) for every command")
    return low, high


def _state_fields(section: RobotSection, joints: list[ModelJoint]) -> tuple[StateField, ...]:
    keys = section.state.fields
    if keys is None:
        keys = list(_DEFAULT_STATE.get(section.mode, ("joint_pos",)))
    if section.mode == "joint_pos" and "joint_pos" not in keys:
        raise PresetError(
            "joint_pos control needs joint_pos in [robot.state] fields: it is the measured "
            "reference every target is limited against"
        )
    fields = []
    for key in keys:
        length = KNOWN_STATE[key]
        if length is None:
            if not joints:
                raise PresetError(f"[robot.state] {key} needs joints; base_vel presets have none")
            length = len(joints)
        fields.append(StateField(key, (length,), CANONICAL_STATE_UNITS.get(key, "")))
    fields += [StateField(key, (length,), "") for key, length in section.state.custom.items()]
    if section.base is not None and section.base.max_tilt_deg is not None:
        if not {"projected_gravity", "base_quat"} & {f.key for f in fields}:
            raise PresetError(
                "[robot.base] max_tilt_deg needs projected_gravity or base_quat in "
                "[robot.state] fields"
            )
    return tuple(fields)


def _check_commands(section: RobotSection) -> None:
    declared = {axis.name for axis in section.commands}
    events = set(section.command_events) | {END_EPISODE}
    for axis in section.commands:
        if axis.low > axis.high or not axis.low <= axis.default <= axis.high:
            raise PresetError(f"command {axis.name}: need low <= default <= high")
    for controller in section.controllers:
        for command in controller.axes:
            if declared and command not in declared:
                raise PresetError(
                    f"controller {controller.name}: axis binding {command!r} is not a "
                    f"[[robot.commands]] name{_suggest(command, declared)}"
                )
        for key, action in controller.buttons.items():
            event = action.event if isinstance(action, ButtonActionDecl) else action
            if event is not None and section.command_events and event not in events:
                raise PresetError(
                    f"controller {controller.name}: button {key} sends {event!r}, which is "
                    f"not in command_events{_suggest(event, events)}"
                )
            if isinstance(action, ButtonActionDecl):
                for command in action.set:
                    if declared and command not in declared:
                        raise PresetError(
                            f"controller {controller.name}: button {key} sets {command!r}, "
                            "which is not a [[robot.commands]] name"
                        )
        if controller.kind == "joystick":
            for binding in controller.axes.values():
                _axis_index(binding.axis, controller.layout, controller.name)
            for key in controller.buttons:
                _button_index(key, controller.layout, controller.name)
            if controller.deadman is not None:
                _button_index(controller.deadman, controller.layout, controller.name)


def load_preset(path: Path | str) -> RobotDescription:
    """Read, validate and resolve a robot preset. Raises ``PresetError`` (a
    ``ConfigError``) naming the field at fault."""
    path = Path(path).expanduser().resolve()
    try:
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PresetError(f"robot preset {path}: {exc}") from None
    try:
        section = PresetFile.model_validate(data).robot
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise PresetError(f"robot preset {path}: {details}") from None
    model: RobotModel | None = None
    if section.model is not None:
        model_path = section.model if section.model.is_absolute() else path.parent / section.model
        try:
            model = load_model(model_path.expanduser())
        except ModelError as exc:
            raise PresetError(f"robot preset {path}: {exc}") from None
    joints: list[ModelJoint] = []
    if section.mode == "base_vel":
        assert section.base is not None  # the schema requires it for base_vel
        labels = tuple(section.base.command)
        low, high = _base_limits(section)
        max_step: tuple[float | None, ...] | None = None
        home = None
        parts: tuple[str, ...] | None = ("base",) * len(labels)
        if section.joints.select or section.joints.limits or section.joints.home:
            raise PresetError('mode = "base_vel" drives the base only; drop [robot.joints]')
    else:
        joints = _select_joints(section, model)
        names = [joint.name for joint in joints]
        labels = tuple(names)
        low, high = _joint_limits(section, joints)
        steps = _per_joint(section.joints.max_step, names, "max_step")
        max_step = None
        if section.mode == "joint_pos" and steps:
            max_step = tuple(steps.get(name) for name in names)
            for name, step in steps.items():
                index = names.index(name)
                if high[index] == low[index]:
                    raise PresetError(f"joint {name} has zero range; drop its max_step")
                if step >= high[index] - low[index]:
                    raise PresetError(
                        f"max_step for {name} ({step:g}) is not smaller than its range"
                    )
        elif section.mode == "joint_vel" and steps:
            raise PresetError("joint_vel limits velocities, not steps; drop max_step")
        home_values = _per_joint(section.joints.home, names, "home")
        home = None
        if home_values:
            if section.mode != "joint_pos":
                raise PresetError('[robot.joints] home needs mode = "joint_pos"')
            home = tuple(
                float(home_values[name]) if name in home_values else None for name in names
            )
            for index, value in enumerate(home):
                if value is not None and not low[index] <= value <= high[index]:
                    raise PresetError(
                        f"home for {names[index]} ({value:g}) is outside its limits "
                        f"[{low[index]:g}, {high[index]:g}]"
                    )
        parts = None
        if section.parts:
            assigned: dict[str, str] = {}
            for part, patterns in section.parts.items():
                for pattern in patterns:
                    found = _matches(pattern, names)
                    if not found:
                        raise PresetError(
                            f"[robot.parts] {part}: {pattern!r} matches no selected joint"
                            f"{_suggest(pattern, names)}"
                        )
                    for name in found:
                        assigned.setdefault(name, part)
            parts = tuple(assigned.get(name, _REST_PART) for name in names)
    end_effectors = []
    for ee in section.end_effectors:
        _frame(ee.frame, model, f"end effector {ee.name}")
        actuators: list[str] = []
        for pattern in ee.joints:
            found = _matches(pattern, list(labels)) if section.mode != "base_vel" else []
            if not found:
                raise PresetError(
                    f"end effector {ee.name}: joint {pattern!r} is not a selected joint"
                    f"{_suggest(pattern, list(labels))}"
                )
            actuators += [name for name in found if name not in actuators]
        end_effectors.append(_resolved(ee, tuple(actuators)))
    for camera in section.cameras:
        if camera.frame is not None:
            _frame(camera.frame, model, f"camera {camera.name}")
    for sensor in section.sensors:
        if sensor.frame is not None:
            _frame(sensor.frame, model, f"sensor {sensor.name}")
    state_fields = _state_fields(section, joints)
    _check_commands(section)
    digest = hashlib.sha256(raw + (model.digest.encode() if model is not None else b"")).hexdigest()
    return RobotDescription(
        name=section.name,
        path=path,
        section=section,
        model=model,
        mode=section.mode,
        policy_hz=float(section.policy_hz),
        control_hz=float(section.control_hz or section.policy_hz),
        labels=labels,
        low=low,
        high=high,
        max_step=max_step,
        home=home,
        parts=parts,
        joints=tuple(joints),
        end_effectors=tuple(end_effectors),
        cameras=tuple(section.cameras),
        sensors=tuple(section.sensors),
        state_fields=state_fields,
        digest=digest,
        driver_args=dict(section.driver_args),
    )


def _resolved(ee: EndEffector, joints: tuple[str, ...]) -> ResolvedEndEffector:
    offset = (float(ee.offset[0]), float(ee.offset[1]), float(ee.offset[2]))
    if not all(math.isfinite(value) for value in offset):
        raise PresetError(f"end effector {ee.name}: offset must be finite")
    return ResolvedEndEffector(
        name=ee.name,
        frame=ee.frame,
        offset=offset,
        joints=joints,
        kind=ee.kind,
        gripper=ee.gripper,
        description=ee.description,
    )
