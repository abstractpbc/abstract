"""The robot harness core: evaluations of a policy on an embodiment.

An evaluation framework for VLA (vision-language-action) models. A benchmark is
defined once as a ``Task`` and run against any compatible
pairing of a ``Policy`` (the VLA) and an
``Embodiment`` (a real robot or simulator).

The public API is every name and public submodule exported here via ``__all__``.
Anything not listed (or prefixed with ``_``) is private and carries no stability
guarantee.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("abstract-harness")
except PackageNotFoundError:  # pragma: no cover - only hit in a non-installed tree
    __version__ = "0.0.0+unknown"

from abstract_harness.robot.core.console import (
    ConsolePoll,
    EndRequest,
    OperatorConsole,
    OperatorInput,
)
from abstract_harness.robot.core.embodiment import (
    Embodiment,
    EmbodimentBase,
    EmbodimentInfo,
)
from abstract_harness.robot.core.eval import eval, eval_set
from abstract_harness.robot.core.grader import Grader, operator_grader, vlm_grader
from abstract_harness.robot.core.log import (
    EvalLog,
    EvalResults,
    EvalSpec,
    EvalStats,
    SceneResult,
    read_eval_log,
)
from abstract_harness.robot.core.policy import Policy, PolicyBase, PolicyConfig, PolicyInfo
from abstract_harness.robot.core.registry import (
    embodiment,
    grader,
    operator_input,
    policy,
    registered,
    resolve,
    scorer,
    sink,
    task,
)
from abstract_harness.robot.core.rollout import TrialRecord
from abstract_harness.robot.core.scene import Scene, Target
from abstract_harness.robot.core.scorer import (
    Score,
    Scorer,
    episode_length,
    is_affirmative_verdict,
    min_distance_to_goal,
    operator_scorer,
    reached_goal_state,
    success_at_end,
)
from abstract_harness.robot.core.session import OperatorSession
from abstract_harness.robot.core.spaces import (
    ABSOLUTE_CONTROL_MODES,
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from abstract_harness.robot.core.task import Epochs, Task, TaskEnvelope
from abstract_harness.robot.core.taskgen import generate_scene
from abstract_harness.robot.core.types import (
    OPERATOR_END,
    Action,
    ActionChunk,
    Observation,
    StepResult,
)

# The public, stability-guaranteed API. Anything not listed here (or prefixed
# with ``_``) is private. Authoring a benchmark, policy, or embodiment should only
# need these names.
__all__ = [
    "ABSOLUTE_CONTROL_MODES",
    "OPERATOR_END",
    "Action",
    "ActionChunk",
    "ActionSemantics",
    "Box",
    "CameraSpec",
    "ConsolePoll",
    "Embodiment",
    "EmbodimentBase",
    "EmbodimentInfo",
    "EndRequest",
    "Epochs",
    "EvalLog",
    "EvalResults",
    "EvalSpec",
    "EvalStats",
    "Grader",
    "Observation",
    "ObservationSpace",
    "OperatorConsole",
    "OperatorInput",
    "OperatorSession",
    "Policy",
    "PolicyBase",
    "PolicyConfig",
    "PolicyInfo",
    "Scene",
    "SceneResult",
    "Score",
    "Scorer",
    "StateField",
    "StateSpec",
    "StepResult",
    "Target",
    "Task",
    "TaskEnvelope",
    "TrialRecord",
    "__version__",
    "embodiment",
    "episode_length",
    "eval",
    "eval_set",
    "generate_scene",
    "grader",
    "is_affirmative_verdict",
    "min_distance_to_goal",
    "operator_grader",
    "operator_input",
    "operator_scorer",
    "policy",
    "reached_goal_state",
    "read_eval_log",
    "registered",
    "resolve",
    "scorer",
    "sink",
    "success_at_end",
    "task",
    "vlm_grader",
]
