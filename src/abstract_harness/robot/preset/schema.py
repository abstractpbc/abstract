"""The robot preset TOML, validated field by field (``docs/PRESETS.md`` is the reference).

Cross-field rules that need the model file (joint names, frames, limits) are checked when
the preset is resolved (``description.py``); this module checks shapes and values only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from abstract_harness.models import Record

# Explicit TypeAlias rather than `type` statements: pydantic validates these annotations.
Name: TypeAlias = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")]  # noqa: UP040
RobotName: TypeAlias = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]  # noqa: UP040
Glob: TypeAlias = Annotated[str, Field(min_length=1, max_length=200)]  # noqa: UP040
Number: TypeAlias = StrictFloat | StrictInt  # noqa: UP040
Range: TypeAlias = tuple[Number, Number]  # noqa: UP040
Scalar: TypeAlias = StrictBool | StrictInt | StrictFloat | StrictStr  # noqa: UP040
Mode: TypeAlias = Literal["joint_pos", "joint_delta", "joint_vel", "base_vel"]  # noqa: UP040

#: State fields a driver can report, with their lengths (``None``: one per joint).
KNOWN_STATE: dict[str, int | None] = {
    "joint_pos": None,
    "joint_vel": None,
    "joint_effort": None,
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "base_quat": 4,
    "base_pose": 3,
}


def _range(value: Range, what: str) -> Range:
    low, high = value
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ValueError(f"{what}: expected [low, high] with low <= high, got {list(value)}")
    return value


class JointsSection(Record):
    """Which joints the action drives, in order, and their limits, steps and home."""

    # Joint names or globs, in action order; default every movable joint (an MJCF's
    # actuator order when it has actuators).
    select: list[Glob] | None = None
    exclude: list[Glob] = Field(default_factory=list)
    # Glob -> [low, high] in the mode's units; narrows the model's limits, and supplies
    # them where the model has none (a continuous joint).
    limits: dict[Glob, Range] = Field(default_factory=dict)
    # Glob -> the largest per-step change (joint_pos: rad or m per step; joint_delta:
    # the per-step bound itself).
    max_step: dict[Glob, Annotated[float, Field(gt=0)]] = Field(default_factory=dict)
    # joint_vel: glob -> the largest change of velocity per second (rad/s^2 or m/s^2).
    max_accel: dict[Glob, Annotated[float, Field(gt=0)]] = Field(default_factory=dict)
    # Glob -> the reset pose; joints left out stay where they are at reset.
    home: dict[Glob, Number] = Field(default_factory=dict)
    home_secs: Annotated[float, Field(gt=0, le=60)] = 3.0

    @model_validator(mode="after")
    def ranges(self) -> Self:
        for pattern, value in self.limits.items():
            _range(value, f"[robot.joints] limits {pattern!r}")
        return self


class JointDecl(Record):
    """A joint declared in the preset, for robots without a model file or with joints the
    model lacks (a custom end effector)."""

    name: Name
    type: Literal["revolute", "prismatic", "continuous"] = "revolute"
    lower: Number | None = None
    upper: Number | None = None
    velocity: Annotated[float, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def bounds(self) -> Self:
        if (self.lower is None) != (self.upper is None):
            raise ValueError(f"joint {self.name}: give both lower and upper, or neither")
        if self.lower is not None and self.upper is not None:
            _range((self.lower, self.upper), f"joint {self.name}")
        return self


class EndEffector(Record):
    """A named tool on the robot: where it is (a model frame and an offset) and which
    joints, if any, actuate it."""

    name: Name
    frame: Glob
    offset: tuple[Number, Number, Number] = (0.0, 0.0, 0.0)
    joints: list[Glob] = Field(default_factory=list)
    kind: Literal["gripper", "hand", "suction", "tool", "custom"] = "gripper"
    # How its actuating joints read: continuous (0 closed .. 1 open) or binary.
    gripper: Literal["continuous", "binary"] = "continuous"
    description: str = ""


class CameraDecl(Record):
    name: Name
    frame: Glob | None = None
    width: Annotated[int, Field(ge=1, le=8192)]
    height: Annotated[int, Field(ge=1, le=8192)]
    channels: Literal[1, 3, 4] = 3
    description: str = ""


class SensorDecl(Record):
    name: Name
    frame: Glob | None = None
    kind: Literal["imu", "force_torque", "contact", "lidar", "depth", "other"] = "other"
    description: str = ""


class StateSection(Record):
    """The proprioception the driver reports each step."""

    # Known fields (KNOWN_STATE); default joint_pos for joint modes and base velocities
    # and gravity for base_vel.
    fields: list[Name] | None = None
    # Other fields and their lengths, e.g. {foot_contact = 4}.
    custom: dict[Name, Annotated[int, Field(ge=1, le=4096)]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def known(self) -> Self:
        unknown = sorted(set(self.fields or ()) - set(KNOWN_STATE))
        if unknown:
            raise ValueError(
                f"[robot.state] fields {unknown} are not known fields "
                f"({sorted(KNOWN_STATE)}); declare them under custom with their lengths"
            )
        clash = sorted(set(self.custom) & set(KNOWN_STATE))
        if clash:
            raise ValueError(f"[robot.state] custom fields {clash} shadow known fields")
        return self


class BaseSection(Record):
    """A mobile base: its velocity command (``mode = "base_vel"``) and locomotion safety."""

    command: list[Name] = Field(default_factory=lambda: ["vx", "vy", "yaw_rate"])
    # Command -> [low, high] in m/s or rad/s; required for every command in base_vel mode.
    limits: dict[Name, Range] = Field(default_factory=dict)
    # Command -> the largest change per second (m/s^2 or rad/s^2): an acceleration limit.
    max_accel: dict[Name, Annotated[float, Field(gt=0)]] = Field(default_factory=dict)
    # The watchdog stops the base when no step arrives for this long.
    command_timeout_s: Annotated[float, Field(gt=0, le=10)] = 0.25
    # Ends the episode (and stops the driver) past this tilt from upright; needs
    # projected_gravity or base_quat in [robot.state].
    max_tilt_deg: Annotated[float, Field(gt=0, lt=180)] | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if len(set(self.command)) != len(self.command):
            raise ValueError("[robot.base] command names must be distinct")
        for name, value in self.limits.items():
            _range(value, f"[robot.base] limits {name!r}")
        for table in ("limits", "max_accel"):
            extra = sorted(set(getattr(self, table)) - set(self.command))
            if extra:
                raise ValueError(f"[robot.base] {table} names {extra} not in command")
        return self


class CommandAxisDecl(Record):
    """A continuous input a policy reads from ``observation.extra["commands"]``."""

    name: Name
    low: Number = -1.0
    high: Number = 1.0
    default: Number = 0.0
    # Back to default after this long without an update; hold = true keeps the last value.
    timeout_s: Annotated[float, Field(gt=0, le=60)] = 0.5
    hold: StrictBool = False
    description: str = ""


class AxisBindingDecl(Record):
    # A stick axis: its index, or its name in the controller's layout ("ly").
    axis: StrictInt | Name
    scale: Number = 1.0
    deadzone: Annotated[float, Field(ge=0, lt=1)] = 0.05


class ButtonActionDecl(Record):
    event: Name | None = None
    set: dict[Name, Number] = Field(default_factory=dict)

    @model_validator(mode="after")
    def something(self) -> Self:
        if self.event is None and not self.set:
            raise ValueError("a button action needs an event, a set table, or both")
        return self


class LayoutDecl(Record):
    """A gamepad's button and axis names, by index, for a pad the built-in layouts miss."""

    buttons: dict[Name, Annotated[int, Field(ge=0, le=1023)]] = Field(default_factory=dict)
    axes: dict[Name, Annotated[int, Field(ge=0, le=1023)]] = Field(default_factory=dict)


#: Controller kinds the preset schema knows the fields of; any other kind is a
#: registered command source, "module:attribute" or "file.py:attribute" taking ``args``.
BUILTIN_CONTROLLERS = ("joystick", "udp")
_JOYSTICK_FIELDS = {"device", "layout", "deadman", "refresh_hz", "max_silence_s", "axes", "buttons"}
_UDP_FIELDS = {"host", "port", "allow_remote"}


class ControllerDecl(Record):
    """An external input device and what each of its inputs does."""

    name: Name
    # "joystick" (Linux /dev/input/js*), "udp" (JSON datagrams), or another command
    # source: a registered name, "module:attribute" or "file.py:attribute" (relative to
    # the preset), constructed with ``args``.
    kind: Annotated[str, Field(min_length=1, max_length=300)]
    # joystick
    device: str = "/dev/input/js0"
    # "xbox", "raw" (indices only), or a table of button and axis names.
    layout: Literal["xbox", "raw"] | LayoutDecl = "xbox"
    deadman: StrictInt | Name | None = None
    refresh_hz: Annotated[float, Field(gt=0, le=1000)] = 50.0
    # Opt-in: after this long without any pad event, sticks publish rest (a wireless pad
    # gone silent). A stick held still sends no events either, so leave it off for
    # sustained full-deflection driving.
    max_silence_s: Annotated[float, Field(gt=0, le=60)] | None = None
    # Command name -> stick binding.
    axes: dict[Name, AxisBindingDecl] = Field(default_factory=dict)
    # Button (index or layout name) -> an event name, or {event, set}.
    buttons: dict[str, Name | ButtonActionDecl] = Field(default_factory=dict)
    # udp
    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 9870
    allow_remote: StrictBool = False
    # Other kinds: the source factory's keyword arguments.
    args: dict[Name, Scalar | list[Scalar]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def kind_fields(self) -> Self:
        allowed = {"joystick": _JOYSTICK_FIELDS, "udp": _UDP_FIELDS}.get(self.kind, {"args"})
        wrong = (_JOYSTICK_FIELDS | _UDP_FIELDS | {"args"}) - allowed
        if wrong & self.model_fields_set:
            raise ValueError(
                f"controller {self.name}: {sorted(wrong & self.model_fields_set)} are not "
                f"{self.kind} fields"
            )
        return self


class RobotSection(Record):
    name: RobotName
    # A URDF or MJCF file, relative to the preset file.
    model: Path | None = None
    # The rate the policy steps at: EmbodimentInfo.control_hz, enforced by strict timing.
    policy_hz: Annotated[float, Field(gt=0, le=2000)]
    # The driver's servo rate, passed to the driver (default: policy_hz).
    control_hz: Annotated[float, Field(gt=0, le=100_000)] | None = None
    mode: Mode = "joint_pos"
    # "sim" (the built-in kinematic simulator) or "module:attribute" / "file.py:attribute".
    driver: Annotated[str, Field(min_length=1, max_length=300)] = "sim"
    driver_args: dict[Name, Scalar | list[Scalar]] = Field(default_factory=dict)
    description: str = ""
    # The embodiment waits out the policy period inside step(), like YAM.
    pace: StrictBool = True
    # Physical drivers only: skip the operator's readiness gates (a notice still prints).
    auto_start: StrictBool = False
    # Rate modes (base_vel, joint_vel): the watchdog stops the driver when no step arrives
    # for this long (default [robot.base] command_timeout_s, else 0.25 s); at least two
    # policy periods. joint_vel also limits approach speed near position limits by it.
    command_timeout_s: Annotated[float, Field(gt=0, le=10)] | None = None
    command_events: list[Name] = Field(default_factory=list)
    joints: JointsSection = Field(default_factory=JointsSection)
    joint: list[JointDecl] = Field(default_factory=list)
    # Part name -> joint names or globs (the action's dim_parts).
    parts: dict[Name, list[Glob]] = Field(default_factory=dict)
    end_effectors: list[EndEffector] = Field(default_factory=list)
    cameras: list[CameraDecl] = Field(default_factory=list)
    sensors: list[SensorDecl] = Field(default_factory=list)
    state: StateSection = Field(default_factory=StateSection)
    base: BaseSection | None = None
    commands: list[CommandAxisDecl] = Field(default_factory=list)
    controllers: list[ControllerDecl] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        for table, items in (
            ("end_effectors", self.end_effectors),
            ("cameras", self.cameras),
            ("sensors", self.sensors),
            ("commands", self.commands),
            ("controllers", self.controllers),
            ("joint", self.joint),
        ):
            names = [item.name for item in items]
            if len(set(names)) != len(names):
                raise ValueError(f"[robot] {table} names must be distinct")
        if self.mode == "base_vel" and self.base is None:
            raise ValueError('mode = "base_vel" needs a [robot.base] section')
        if self.model is None and not self.joint and self.mode != "base_vel":
            raise ValueError("a preset needs a model file or [[robot.joint]] declarations")
        if self.control_hz is not None and self.control_hz < self.policy_hz:
            raise ValueError("control_hz (the driver's servo rate) must be >= policy_hz")
        timeout = self.command_timeout
        if self.mode in ("base_vel", "joint_vel") and timeout < 2.0 / self.policy_hz:
            raise ValueError(
                f"command_timeout_s {timeout:g} is shorter than two policy periods "
                f"({2.0 / self.policy_hz:g}s): the watchdog would stop the robot between steps"
            )
        return self

    @property
    def command_timeout(self) -> float:
        """The rate-mode watchdog's timeout, in seconds."""
        if self.command_timeout_s is not None:
            return self.command_timeout_s
        if self.base is not None:
            return self.base.command_timeout_s
        return 0.25


class PresetFile(Record):
    robot: RobotSection
