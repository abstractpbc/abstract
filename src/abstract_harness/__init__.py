"""abstract-harness: run and evaluate robotics policies across sim and physical embodiments."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_CORE_EXPORTS = {
    "OPERATOR_END": "abstract_harness.robot.core",
    "Action": "abstract_harness.robot.core",
    "ActionChunk": "abstract_harness.robot.core",
    "ActionSemantics": "abstract_harness.robot.core",
    "Box": "abstract_harness.robot.core",
    "CameraSpec": "abstract_harness.robot.core",
    "Embodiment": "abstract_harness.robot.core",
    "EmbodimentBase": "abstract_harness.robot.core",
    "EmbodimentInfo": "abstract_harness.robot.core",
    "Epochs": "abstract_harness.robot.core",
    "EvalLog": "abstract_harness.robot.core",
    "Grader": "abstract_harness.robot.core",
    "Observation": "abstract_harness.robot.core",
    "ObservationSpace": "abstract_harness.robot.core",
    "Policy": "abstract_harness.robot.core",
    "PolicyBase": "abstract_harness.robot.core",
    "PolicyConfig": "abstract_harness.robot.core",
    "PolicyInfo": "abstract_harness.robot.core",
    "Scene": "abstract_harness.robot.core",
    "Score": "abstract_harness.robot.core",
    "Scorer": "abstract_harness.robot.core",
    "StepResult": "abstract_harness.robot.core",
    "Target": "abstract_harness.robot.core",
    "Task": "abstract_harness.robot.core",
    "TaskEnvelope": "abstract_harness.robot.core",
    "TrialRecord": "abstract_harness.robot.core",
    "embodiment": "abstract_harness.robot.core",
    "eval": "abstract_harness.robot.core",
    "eval_set": "abstract_harness.robot.core",
    "grader": "abstract_harness.robot.core",
    "operator_grader": "abstract_harness.robot.core",
    "operator_scorer": "abstract_harness.robot.core",
    "policy": "abstract_harness.robot.core",
    "read_eval_log": "abstract_harness.robot.core",
    "register": "abstract_harness.robot.core.registry",
    "registered": "abstract_harness.robot.core",
    "resolve": "abstract_harness.robot.core",
    "scorer": "abstract_harness.robot.core",
    "sink": "abstract_harness.robot.core",
    "success_at_end": "abstract_harness.robot.core",
    "task": "abstract_harness.robot.core",
    "vlm_grader": "abstract_harness.robot.core",
    "Controller": "abstract_harness.robot.core.controller",
    "DefaultController": "abstract_harness.robot.core.controller",
    "EnsemblingController": "abstract_harness.robot.core.controller",
    "SmoothingController": "abstract_harness.robot.core.controller",
    "StateField": "abstract_harness.robot.core.spaces",
    "StateSpec": "abstract_harness.robot.core.spaces",
    "assert_embodiment_conformant": "abstract_harness.robot.core.conformance",
    "check_embodiment": "abstract_harness.robot.core.conformance",
    "joint_embodiment_info": "abstract_harness.robot.core.embodiment",
    "rollout": "abstract_harness.robot.core.rollout",
    "AbstractGatewayPolicy": "abstract_harness.robot.vla",
    "CallablePolicy": "abstract_harness.robot.vla",
    "GatewayVLAPolicy": "abstract_harness.robot.vla",
    "WorldModelPolicy": "abstract_harness.robot.vla",
    "abstract_policy": "abstract_harness.robot.vla",
    "local_policy": "abstract_harness.robot.vla",
    "vla_policy": "abstract_harness.robot.vla",
    "world_model_policy": "abstract_harness.robot.vla",
}

if TYPE_CHECKING:
    from abstract_harness.robot.core import (
        OPERATOR_END as OPERATOR_END,
    )
    from abstract_harness.robot.core import (
        Action as Action,
    )
    from abstract_harness.robot.core import (
        ActionChunk as ActionChunk,
    )
    from abstract_harness.robot.core import (
        ActionSemantics as ActionSemantics,
    )
    from abstract_harness.robot.core import (
        Box as Box,
    )
    from abstract_harness.robot.core import (
        CameraSpec as CameraSpec,
    )
    from abstract_harness.robot.core import (
        Embodiment as Embodiment,
    )
    from abstract_harness.robot.core import (
        EmbodimentBase as EmbodimentBase,
    )
    from abstract_harness.robot.core import (
        EmbodimentInfo as EmbodimentInfo,
    )
    from abstract_harness.robot.core import (
        Epochs as Epochs,
    )
    from abstract_harness.robot.core import (
        EvalLog as EvalLog,
    )
    from abstract_harness.robot.core import (
        Grader as Grader,
    )
    from abstract_harness.robot.core import (
        Observation as Observation,
    )
    from abstract_harness.robot.core import (
        ObservationSpace as ObservationSpace,
    )
    from abstract_harness.robot.core import (
        Policy as Policy,
    )
    from abstract_harness.robot.core import (
        PolicyBase as PolicyBase,
    )
    from abstract_harness.robot.core import (
        PolicyConfig as PolicyConfig,
    )
    from abstract_harness.robot.core import (
        PolicyInfo as PolicyInfo,
    )
    from abstract_harness.robot.core import (
        Scene as Scene,
    )
    from abstract_harness.robot.core import (
        Score as Score,
    )
    from abstract_harness.robot.core import (
        Scorer as Scorer,
    )
    from abstract_harness.robot.core import (
        StepResult as StepResult,
    )
    from abstract_harness.robot.core import (
        Target as Target,
    )
    from abstract_harness.robot.core import (
        Task as Task,
    )
    from abstract_harness.robot.core import (
        TaskEnvelope as TaskEnvelope,
    )
    from abstract_harness.robot.core import (
        TrialRecord as TrialRecord,
    )
    from abstract_harness.robot.core import (
        embodiment as embodiment,
    )
    from abstract_harness.robot.core import (
        eval as eval,
    )
    from abstract_harness.robot.core import (
        eval_set as eval_set,
    )
    from abstract_harness.robot.core import (
        grader as grader,
    )
    from abstract_harness.robot.core import (
        operator_grader as operator_grader,
    )
    from abstract_harness.robot.core import (
        operator_scorer as operator_scorer,
    )
    from abstract_harness.robot.core import (
        policy as policy,
    )
    from abstract_harness.robot.core import (
        read_eval_log as read_eval_log,
    )
    from abstract_harness.robot.core import (
        registered as registered,
    )
    from abstract_harness.robot.core import (
        resolve as resolve,
    )
    from abstract_harness.robot.core import (
        scorer as scorer,
    )
    from abstract_harness.robot.core import (
        sink as sink,
    )
    from abstract_harness.robot.core import (
        success_at_end as success_at_end,
    )
    from abstract_harness.robot.core import (
        task as task,
    )
    from abstract_harness.robot.core import (
        vlm_grader as vlm_grader,
    )
    from abstract_harness.robot.core.conformance import (
        assert_embodiment_conformant as assert_embodiment_conformant,
    )
    from abstract_harness.robot.core.conformance import (
        check_embodiment as check_embodiment,
    )
    from abstract_harness.robot.core.controller import (
        Controller as Controller,
    )
    from abstract_harness.robot.core.controller import (
        DefaultController as DefaultController,
    )
    from abstract_harness.robot.core.controller import (
        EnsemblingController as EnsemblingController,
    )
    from abstract_harness.robot.core.controller import (
        SmoothingController as SmoothingController,
    )
    from abstract_harness.robot.core.embodiment import (
        joint_embodiment_info as joint_embodiment_info,
    )
    from abstract_harness.robot.core.registry import (
        register as register,
    )
    from abstract_harness.robot.core.rollout import (
        rollout as rollout,
    )
    from abstract_harness.robot.core.spaces import (
        StateField as StateField,
    )
    from abstract_harness.robot.core.spaces import (
        StateSpec as StateSpec,
    )
    from abstract_harness.robot.vla import (
        AbstractGatewayPolicy as AbstractGatewayPolicy,
    )
    from abstract_harness.robot.vla import (
        CallablePolicy as CallablePolicy,
    )
    from abstract_harness.robot.vla import (
        GatewayVLAPolicy as GatewayVLAPolicy,
    )
    from abstract_harness.robot.vla import (
        WorldModelPolicy as WorldModelPolicy,
    )
    from abstract_harness.robot.vla import (
        abstract_policy as abstract_policy,
    )
    from abstract_harness.robot.vla import (
        local_policy as local_policy,
    )
    from abstract_harness.robot.vla import (
        vla_policy as vla_policy,
    )
    from abstract_harness.robot.vla import (
        world_model_policy as world_model_policy,
    )


__all__ = ["__version__", *sorted(_CORE_EXPORTS)]


def __getattr__(name: str) -> Any:
    module_name = _CORE_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
