"""The orchestrator: an LLM that runs trained policies and skills as background jobs.

Extensions (``extensions.py``) register tools, each with its own execution loop
(``jobs.py``); the orchestrator policy (``policy.py``) gives them to the model, starts a
job per call, streams the jobs' events back to the model, and lets it stop or steer
them while the robot keeps moving. Registered as the policy ``orchestrator``; in a robot
config, ``[agent] policy = "orchestrator"`` with ``extensions`` in ``[agent.args]``.
"""

from __future__ import annotations

from abstract_harness.robot.orchestrator.extensions import (
    ExtensionAPI,
    ToolSpec,
    load_extensions,
    validate_arguments,
)
from abstract_harness.robot.orchestrator.jobs import (
    Behavior,
    Job,
    JobContext,
    JobEvent,
    MotionBehavior,
    PolicyBehavior,
    resolve_controls,
    sub_embodiment,
)
from abstract_harness.robot.orchestrator.policy import (
    OrchestratorPolicy,
    OrchestratorPolicyConfig,
    orchestrator_policy,
)

__all__ = [
    "Behavior",
    "ExtensionAPI",
    "Job",
    "JobContext",
    "JobEvent",
    "MotionBehavior",
    "OrchestratorPolicy",
    "OrchestratorPolicyConfig",
    "PolicyBehavior",
    "ToolSpec",
    "load_extensions",
    "orchestrator_policy",
    "resolve_controls",
    "sub_embodiment",
    "validate_arguments",
]
