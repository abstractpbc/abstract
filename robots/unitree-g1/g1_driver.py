"""A G1 driver skeleton for robots/unitree-g1/preset.toml: fill in your SDK calls.

The preset embodiment calls these methods; every action it passes to ``command`` has
already passed the guardrails (bounds clamp, per-step limit). See docs/PRESETS.md,
"Drivers", for the contract. Nothing here talks to a robot until you write it: until
then, dry-run the preset with ``-E driver=sim``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class G1Driver:
    def __init__(self, description: Any, *, network_interface: str = "eth0") -> None:
        # Hardware-free: keep the description and settings; open nothing here.
        self.description = description
        self.network_interface = network_interface
        self.joint_names = description.joint_names  # action order, as the preset selects
        self.servo_hz = description.control_hz

    def connect(self) -> None:
        # Open the SDK channel on self.network_interface, enter low-level mode, and start
        # a servo loop at self.servo_hz that tracks the latest target with your PD gains.
        raise NotImplementedError("fill in G1Driver.connect with your SDK, or use -E driver=sim")

    def read(self) -> dict[str, Any]:
        # Return every field in [robot.state], joint fields in self.joint_names order:
        # joint_pos, joint_vel (rad, rad/s), base_ang_vel (IMU gyro, rad/s, pelvis frame)
        # and projected_gravity (gravity in the pelvis frame, from the IMU quaternion).
        raise NotImplementedError

    def command(self, values: np.ndarray) -> None:
        # values: one absolute joint target per self.joint_names entry. Hand it to the
        # servo loop; map the *_ee_joint entries to your end effectors.
        raise NotImplementedError

    def stop(self) -> None:
        # Bring the robot to a safe rest now (for a humanoid: hold the current pose with
        # damping, or your SDK's damp mode). Must be safe to call repeatedly.
        raise NotImplementedError

    def close(self) -> None:
        # Leave low-level mode and release the channel.
        pass
