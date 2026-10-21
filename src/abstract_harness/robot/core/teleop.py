"""``teleop``: drive any embodiment from live commands, through the same guardrails.

Each action dimension takes the live command of the same name
(``observation.extra["commands"]``, ``core/command_stream.py``) times ``scale``. A
dimension without a command holds: zero for displacements and velocities, the measured
position for absolute targets. With a robot preset whose gamepad maps sticks to
``vx``, ``vy`` and ``yaw_rate`` on a ``base_vel`` base, this is joystick driving with
the bounds clamp, the acceleration limit, the watchdog and fall detection in force.
``map`` renames: ``"yaw_rate:turn,vx:forward"`` reads the ``turn`` and ``forward``
commands. The policy never ends the trial; ``end_episode`` (a button) or the console does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from abstract_harness.robot.core.errors import ConfigError
from abstract_harness.robot.core.policy import PolicyBase, PolicyConfig, PolicyInfo
from abstract_harness.robot.core.spaces import ABSOLUTE_CONTROL_MODES, Box, reference_state_keys
from abstract_harness.robot.core.types import Action, ActionChunk, Observation

if TYPE_CHECKING:
    from abstract_harness.robot.core.embodiment import EmbodimentInfo


class TeleopPolicy(PolicyBase):
    """Live commands as actions (see the module docstring)."""

    def __init__(self, *, scale: float = 1.0, map: str | None = None) -> None:  # noqa: A002
        if isinstance(scale, bool) or not isinstance(scale, int | float) or not np.isfinite(scale):
            raise ConfigError("teleop scale must be a finite number")
        self.scale = float(scale)
        self.renames: dict[str, str] = {}
        for item in (map or "").split(","):
            if not item.strip():
                continue
            label, sep, command = item.strip().partition(":")
            if not sep or not label or not command:
                raise ConfigError(f"teleop map entry {item!r}: expected label:command")
            self.renames[label] = command
        self.info = PolicyInfo(name="teleop", action_space=Box(shape=(1,)))
        self.config = PolicyConfig()
        self._labels: tuple[str, ...] = ()
        self._absolute = False
        self._state_key: str | None = None

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        space = embodiment_info.action_space
        semantics = space.semantics
        if semantics is None or semantics.dim_labels is None:
            raise ConfigError("teleop needs an action space with semantics and dim_labels")
        self._labels = semantics.dim_labels
        unknown = sorted(set(self.renames) - set(self._labels))
        if unknown:
            raise ConfigError(f"teleop map names {unknown}, not action dimensions")
        self._absolute = semantics.control_mode in ABSOLUTE_CONTROL_MODES
        if self._absolute:
            keys = reference_state_keys(
                embodiment_info.observation_space.state, space.dim, semantics.control_mode
            )
            if len(keys) != 1:
                raise ConfigError("teleop on absolute targets needs one reference state field")
            self._state_key = keys[0]
        self.info = PolicyInfo(
            name="teleop",
            action_space=space,
            observation_space=embodiment_info.observation_space,
            control_hz=embodiment_info.control_hz,
        )

    def act(self, observation: Observation) -> ActionChunk:
        if not self._labels:
            raise ConfigError("teleop.act() before bind()")
        commands: Any = observation.extra.get("commands") or {}
        if self._absolute:
            assert self._state_key is not None
            data = np.asarray(observation.state[self._state_key], dtype=np.float64).copy()
        else:
            data = np.zeros(len(self._labels))
        for index, label in enumerate(self._labels):
            value = commands.get(self.renames.get(label, label))
            if isinstance(value, int | float) and not isinstance(value, bool):
                data[index] = float(value) * self.scale
        return ActionChunk(actions=[Action(data=data)], control_hz=self.info.control_hz)
