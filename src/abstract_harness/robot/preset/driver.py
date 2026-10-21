"""The driver contract a preset plugs in, and the built-in kinematic simulator.

A driver is the only code that talks to a preset robot's hardware (or vendor SDK). The
preset embodiment owns the lifecycle, pacing, the readiness gates, the watchdog, fall
detection and observation assembly; every action it hands the driver has already
passed the approver chain. A driver is constructed as
``factory(description, **driver_args)`` (``RobotDescription`` holds joint names in
action order, limits, rates, end effectors and cameras) and its constructor must not
open hardware: ``connect()`` does. The methods:

``connect()``
    Open the robot: ports, SDK sessions, enabling. Called once, at the first reset.
``read() -> Mapping[str, array]``
    The current state: every field the preset declares in ``[robot.state]``, joint fields
    in action order (``description.joint_names``).
``command(values)``
    Send one action, in the preset's mode and dimension order.
``stop()``
    Bring the robot to rest now (zero velocity, hold position, or damp, as the robot
    requires). Called by the watchdog, on a fall, on close and after errors; must be
    safe to call repeatedly and from another thread than ``command()``.
``close()``
    Release the hardware.

Optional: ``images() -> Mapping[str, ndarray]`` (one ``(H, W, C)`` uint8 frame per
declared camera), ``health()`` (raise to fault the episode), and a ``simulated``
attribute (``True`` marks the embodiment simulated: no operator gates, no physical
guardrail requirements).
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

from abstract_harness.robot.core.errors import ConfigError

if TYPE_CHECKING:
    from abstract_harness.robot.preset.description import RobotDescription


@runtime_checkable
class RobotDriver(Protocol):
    """The hardware side of a preset robot (see the module docstring)."""

    def connect(self) -> None:
        """Open the robot; called once, at the first reset."""
        ...

    def read(self) -> Mapping[str, Any]:
        """Every declared state field, joint fields in action order."""
        ...

    def command(self, values: npt.NDArray[np.float64]) -> None:
        """Send one approved action in the preset's mode and dimension order."""
        ...

    def stop(self) -> None:
        """Bring the robot to rest now; idempotent and thread-safe."""
        ...

    def close(self) -> None:
        """Release the hardware."""
        ...


def load_driver_factory(spec: str, base: Path) -> Callable[..., Any]:
    """``"sim"``, ``"package.module:Class"``, or ``"path/to/file.py:Class"`` (relative to
    the preset's directory) to the driver factory."""
    if spec == "sim":
        return SimDriver
    module_name, sep, attribute = spec.rpartition(":")
    if not sep or not module_name or not attribute:
        raise ConfigError(
            f'driver {spec!r}: expected "sim", "module:attribute" or "file.py:attribute"'
        )
    try:
        if module_name.endswith(".py"):
            file = Path(module_name)
            file = file if file.is_absolute() else base / file
            unique = f"_abstract_preset_driver_{abs(hash(str(file.resolve())))}"
            module_spec = importlib.util.spec_from_file_location(unique, file)
            if module_spec is None or module_spec.loader is None:
                raise ConfigError(f"driver {spec!r}: cannot load {file}")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[unique] = module
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"driver {spec!r}: import failed: {type(exc).__name__}: {exc}") from exc
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ConfigError(f"driver {spec!r}: {module_name} has no callable {attribute!r}")
    return cast(Callable[..., Any], factory)


# Base command labels the simulator understands, by axis.
_VX = ("vx", "linear_x", "forward")
_VY = ("vy", "linear_y", "lateral")
_WZ = ("yaw_rate", "wz", "angular_z", "turn")


class SimDriver:
    """A kinematic stand-in for a preset robot: no physics, no hardware.

    Joint targets are tracked with a first-order lag (``tau_s``; 0 tracks exactly),
    deltas and velocities are integrated at the policy rate, and a base velocity moves
    a planar pose. Cameras render flat gray frames. ``tilt(degrees)`` tips the simulated
    base for fall-detection tests. Use it to dry-run a preset: limits, rates, commands
    and controllers behave as on the robot, the dynamics do not.
    """

    simulated = True

    def __init__(self, description: RobotDescription, *, tau_s: float = 0.0, **_: Any) -> None:
        if isinstance(tau_s, bool) or not math.isfinite(tau_s) or tau_s < 0:
            raise ConfigError("sim driver: tau_s must be finite and >= 0")
        self.description = description
        self.dt = 1.0 / description.policy_hz
        self.alpha = 1.0 if tau_s == 0 else 1.0 - math.exp(-self.dt / tau_s)
        n = len(description.joints)
        start = np.clip(np.zeros(n), description.low, description.high) if n else np.zeros(0)
        if description.mode in ("joint_delta", "joint_vel") and n:
            start = np.zeros(n)
        self.position: npt.NDArray[np.float64] = start.astype(np.float64)
        self.velocity = np.zeros(n)
        self.target = self.position.copy()
        self.pose = np.zeros(3)  # x, y, yaw
        self.base_velocity = np.zeros(3)  # vx, vy, yaw_rate (base frame)
        self.tilt_rad = 0.0
        self.connected = False
        self.commands: list[npt.NDArray[np.float64]] = []
        self.stops = 0
        self.closed = False

    def connect(self) -> None:
        self.connected = True

    def tilt(self, degrees: float) -> None:
        """Tip the simulated base (about its x axis) to exercise fall detection."""
        self.tilt_rad = math.radians(degrees)

    def command(self, values: npt.NDArray[np.float64]) -> None:
        values = np.asarray(values, dtype=np.float64)
        self.commands.append(values.copy())
        mode = self.description.mode
        if mode == "base_vel":
            by_label = dict(zip(self.description.labels, values.tolist(), strict=True))
            vx = next((by_label[k] for k in _VX if k in by_label), 0.0)
            vy = next((by_label[k] for k in _VY if k in by_label), 0.0)
            wz = next((by_label[k] for k in _WZ if k in by_label), 0.0)
            self.base_velocity = np.array([vx, vy, wz])
            yaw = self.pose[2]
            self.pose = self.pose + self.dt * np.array(
                [
                    vx * math.cos(yaw) - vy * math.sin(yaw),
                    vx * math.sin(yaw) + vy * math.cos(yaw),
                    wz,
                ]
            )
            return
        previous = self.position.copy()
        if mode == "joint_pos":
            self.target = values.copy()
            self.position = self.position + self.alpha * (self.target - self.position)
        elif mode == "joint_delta":
            self.position = self.position + values
        else:  # joint_vel
            self.position = self.position + values * self.dt
        self.velocity = (self.position - previous) / self.dt

    def stop(self) -> None:
        self.stops += 1
        self.base_velocity = np.zeros(3)
        self.velocity = np.zeros_like(self.velocity)
        self.target = self.position.copy()

    def read(self) -> dict[str, npt.NDArray[np.float64]]:
        gravity = np.array([0.0, math.sin(self.tilt_rad), -math.cos(self.tilt_rad)])
        yaw = self.pose[2]
        quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        known = {
            "joint_pos": self.position.copy(),
            "joint_vel": self.velocity.copy(),
            "joint_effort": np.zeros_like(self.position),
            "base_lin_vel": np.array([self.base_velocity[0], self.base_velocity[1], 0.0]),
            "base_ang_vel": np.array([0.0, 0.0, self.base_velocity[2]]),
            "projected_gravity": gravity,
            "base_quat": quat,
            "base_pose": self.pose.copy(),
        }
        state = {}
        for state_field in self.description.state_fields:
            value = known.get(state_field.key)
            state[state_field.key] = (
                value if value is not None else np.zeros(state_field.shape, np.float64)
            )
        return state

    def images(self) -> dict[str, npt.NDArray[np.uint8]]:
        return {
            camera.name: np.full((camera.height, camera.width, camera.channels), 96, np.uint8)
            for camera in self.description.cameras
        }

    def close(self) -> None:
        self.closed = True
