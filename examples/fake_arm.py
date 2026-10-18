"""Complete, motion-free adapter example: uv run python examples/fake_arm.py."""

from __future__ import annotations

import tempfile
import time

import numpy as np

from abstract_harness import (
    Action,
    EmbodimentBase,
    Observation,
    Scene,
    StepResult,
    Task,
    eval,
    joint_embodiment_info,
    success_at_end,
)
from abstract_harness.robot.vla import CallableVLAPolicy


class FakeArm(EmbodimentBase):
    """In-memory joint target adapter; no ports, cameras or actuators are opened."""

    def __init__(self) -> None:
        self.info = joint_embodiment_info(
            "fake-arm",
            low=[-1.0],
            high=[1.0],
            dim_labels=["joint_0"],
            control_hz=10.0,
        )
        # This intentionally exercises the physical approval profile using fake I/O.
        self.position = np.zeros(1)
        self.closed = False

    def _observe(self) -> Observation:
        return Observation(state={"joint_pos": self.position.copy()}, state_time=time.monotonic())

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self.position[:] = 0
        return self._observe()

    def step(self, action: Action) -> StepResult:
        self.position[:] = np.asarray(action.data)
        return StepResult(
            observation=self._observe(),
            command={"joint_pos": self.position.copy()},
            command_time=time.monotonic(),
        )

    def close(self) -> None:
        self.closed = True


def main() -> None:
    arm = FakeArm()
    try:
        with tempfile.TemporaryDirectory() as directory:
            logs = eval(
                Task(
                    "fake-reach", [Scene("s", "reach 0.3 radians")], success_at_end(), max_steps=3
                ),
                CallableVLAPolicy(lambda observation: np.array([0.3])),
                arm,
                log_dir=directory,
            )
            assert logs[0].status == "success"
            np.testing.assert_allclose(arm.position, [0.3])
            print("fake adapter: bounded targets and evaluation passed")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
