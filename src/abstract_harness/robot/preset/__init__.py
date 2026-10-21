"""Robot presets: describe a robot in TOML, point at its model file, plug in a driver.

A preset (``docs/PRESETS.md``) names the robot's model file (URDF or MJCF), the joints a
policy drives and their limits, the policy and servo rates, end effectors, cameras and
sensor frames, the state the driver reports, a mobile base, live command axes, and the
controllers (gamepads, UDP bridges) that feed them. ``load_preset`` resolves it against
the model; the ``preset`` embodiment (``PresetEmbodiment``) runs it through a driver: the
built-in kinematic simulator (``driver = "sim"``) or yours (``RobotDriver``).
"""

from __future__ import annotations

from abstract_harness.robot.preset.description import (
    PresetError,
    ResolvedEndEffector,
    RobotDescription,
    load_preset,
)
from abstract_harness.robot.preset.driver import RobotDriver, SimDriver, load_driver_factory
from abstract_harness.robot.preset.embodiment import FALL, PresetEmbodiment
from abstract_harness.robot.preset.model import ModelError, ModelJoint, RobotModel, load_model

__all__ = [
    "FALL",
    "ModelError",
    "ModelJoint",
    "PresetEmbodiment",
    "PresetError",
    "ResolvedEndEffector",
    "RobotDescription",
    "RobotDriver",
    "RobotModel",
    "SimDriver",
    "load_driver_factory",
    "load_model",
    "load_preset",
]
