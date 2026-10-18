"""Register the in-tree builtin components with the registry.

Imported lazily by ``abstract_harness.robot.core.registry`` on first
lookup, so importing ``abstract_harness.robot.core`` stays cheap and free of import cycles.
"""

from __future__ import annotations

from abstract_harness.robot.core.grader import operator_grader, vlm_grader
from abstract_harness.robot.core.logging import JsonLogSink, LiveLogSink, RerunSink
from abstract_harness.robot.core.mock import (
    CubePickEmbodiment,
    NoopPolicy,
    RandomPolicy,
    ScriptedPolicy,
)
from abstract_harness.robot.core.registry import embodiment, grader, policy, scorer, sink, task
from abstract_harness.robot.core.scene import Scene
from abstract_harness.robot.core.scorer import (
    episode_length,
    min_distance_to_goal,
    operator_scorer,
    reached_goal_state,
    success_at_end,
)
from abstract_harness.robot.core.task import Task
from abstract_harness.robot.yam.policy import MolmoAct2Policy, gr00t_policy

# Embodiments
embodiment("cubepick")(CubePickEmbodiment)

# Policies
policy("scripted")(ScriptedPolicy)
policy("random")(RandomPolicy)
policy("noop")(NoopPolicy)
policy("molmoact2")(MolmoAct2Policy)
policy("gr00t")(gr00t_policy)

# Scorers
scorer("success_at_end")(success_at_end)
scorer("episode_length")(episode_length)
scorer("min_distance_to_goal")(min_distance_to_goal)
scorer("reached_goal_state")(reached_goal_state)
scorer("operator")(operator_scorer)

# Graders
grader("operator")(operator_grader)
grader("vlm")(vlm_grader)

# Sinks
sink("json")(JsonLogSink)
sink("live-json")(LiveLogSink)
sink("rerun")(RerunSink)


@task("cubepick-reach")
def _cubepick_reach(num_scenes: int = 4, max_steps: int = 80) -> Task:
    """A simple reach benchmark over a handful of seeded cube layouts."""
    return Task(
        name="cubepick-reach",
        scenes=[
            Scene(id=f"scene-{i}", instruction="reach the cube", init_seed=i)
            for i in range(num_scenes)
        ],
        scorer=success_at_end(),
        max_steps=max_steps,
    )
