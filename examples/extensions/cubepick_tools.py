"""An example orchestrator extension for the CubePick test world (docs/ORCHESTRATOR.md).

Two tools: ``reach_cube`` packages a policy (the scripted CubePick oracle) as a tool,
with a monitor that reports progress; ``scan`` is a skill with its own execution loop
that sweeps the effector in a square and can be steered to another speed. Try it with
any model the agent supports::

    uv run abstract eval --policy orchestrator --instruction "find and reach the cube" \\
        -P model=anthropic/claude-sonnet-5-5 \\
        -P extensions=examples/extensions/cubepick_tools.py --no-prompt
"""

from __future__ import annotations

from typing import Any

import numpy as np

from abstract_harness.robot.core.mock import ScriptedPolicy
from abstract_harness.robot.orchestrator import ExtensionAPI, JobContext


def cube_progress(observation: Any, ctx: JobContext) -> None:
    """Report the approach to the cube at halfway and on arrival."""
    distance = float(np.linalg.norm(observation.state["cube_pos"] - observation.state["eef_pos"]))
    start = ctx.state.setdefault("start_distance", distance)
    if distance < start / 2 and not ctx.state.get("halfway"):
        ctx.state["halfway"] = True
        ctx.emit(f"halfway to the cube ({distance:.2f} left)")
    if distance < 0.06:
        ctx.finish("reached the cube")


class Scan:
    """Sweep a square of side ``side`` at ``speed`` per step, one event per corner."""

    def __init__(self, ctx: JobContext, side: float = 0.2, speed: float = 0.02) -> None:
        self.ctx = ctx
        self.side = side
        self.speed = speed
        self.corner = 0
        self.travelled = 0.0

    def step(self, observation: Any) -> np.ndarray:
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dx, dy = directions[self.corner % 4]
        self.travelled += self.speed
        if self.travelled >= self.side:
            self.travelled = 0.0
            self.corner += 1
            self.ctx.emit(
                f"corner {self.corner} at {observation.state['eef_pos'].round(2).tolist()}"
            )
            if self.corner == 4:
                self.ctx.finish("square complete")
        return np.array([dx, dy], dtype=np.float64) * self.speed

    def steer(self, speed: float | None = None, side: float | None = None) -> str:
        if speed is not None:
            self.speed = float(speed)
        if side is not None:
            self.side = float(side)
        return f"speed {self.speed:g}, side {self.side:g}"


def register(harness: ExtensionAPI) -> None:
    harness.register_policy_tool(
        name="reach_cube",
        description="Move the effector straight to the cube and stop there.",
        policy=lambda: ScriptedPolicy(chunk_size=1),
        instruction="reach the cube",
        monitors=[cube_progress],
        max_seconds=30,
    )
    harness.register_tool(
        name="scan",
        description="Sweep the effector around a square to look around; reports each corner.",
        parameters={
            "type": "object",
            "properties": {
                "side": {"type": "number", "description": "Square side, in world units."},
                "speed": {"type": "number", "description": "Distance per control step."},
            },
            "additionalProperties": False,
        },
        start=Scan,
        max_seconds=60,
    )
