"""Extensions and jobs: registration, loading, argument checks, controls and the job
step contract."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from abstract_harness.robot.core.embodiment import EmbodimentInfo
from abstract_harness.robot.core.errors import ConfigError
from abstract_harness.robot.core.spaces import (
    ActionSemantics,
    Box,
    ObservationSpace,
    StateField,
    StateSpec,
)
from abstract_harness.robot.core.types import Action, ActionChunk, Observation
from abstract_harness.robot.orchestrator import (
    ExtensionAPI,
    Job,
    JobContext,
    JobEvent,
    PolicyBehavior,
    load_extensions,
    resolve_controls,
    sub_embodiment,
    validate_arguments,
)

EXAMPLE = Path(__file__).parents[2] / "examples" / "extensions" / "cubepick_tools.py"


def arm_info() -> EmbodimentInfo:
    return EmbodimentInfo(
        name="arm",
        action_space=Box(
            shape=(4,),
            low=-np.ones(4),
            high=np.ones(4),
            semantics=ActionSemantics(
                "joint_pos",
                dim_labels=("l_j1", "l_j2", "r_j1", "r_j2"),
                dim_parts=("left", "left", "right", "right"),
                max_step=(0.1, 0.1, None, None),
            ),
        ),
        observation_space=ObservationSpace(state=StateSpec((StateField("joint_pos", (4,)),))),
        control_hz=10.0,
    )


def test_the_example_extension_registers_two_tools() -> None:
    api = load_extensions(str(EXAMPLE))
    assert sorted(api.tools) == ["reach_cube", "scan"]
    scan = api.tools["scan"].schema()["function"]
    assert "background job" in scan["description"]
    assert scan["parameters"]["additionalProperties"] is False
    reach = api.tools["reach_cube"]
    assert reach.policy is not None and reach.start is None
    assert "instruction" in reach.parameters["properties"]


def test_extensions_load_from_modules_functions_and_registered_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = type(sys)("my_tools")

    def register(harness: ExtensionAPI) -> None:
        harness.register_tool(name="one", description="first", start=lambda ctx: None)

    def other(harness: ExtensionAPI) -> None:
        harness.register_tool(name="two", description="second", start=lambda ctx: None)

    module.register = register  # type: ignore[attr-defined]
    module.other = other  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "my_tools", module)
    assert sorted(load_extensions("my_tools, my_tools:other").tools) == ["one", "two"]

    # A registered extension (what an installed plugin does with @registry.extension).
    from abstract_harness.robot.core import registry

    registry.registered("extension")  # load the registry first, as a lookup does
    monkeypatch.setitem(registry._FACTORIES["extension"], "kitchen", other)
    assert list(load_extensions(["kitchen"]).tools) == ["two"]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("x = 1\n", "has no register"),
        (
            "def register(h):\n    raise RuntimeError('boom')\n",
            "failed to load: RuntimeError: boom",
        ),
        (
            "def register(h):\n    h.register_tool(name='wait', description='d', start=print)\n",
            "reserved",
        ),
        (
            "def register(h):\n"
            "    h.register_tool(name='Bad-Name', description='d', start=print)\n",
            "must match",
        ),
        (
            "def register(h):\n"
            "    h.register_tool(name='a', description='d', start=print)\n"
            "    h.register_tool(name='a', description='d', start=print)\n",
            "registered twice",
        ),
        (
            "def register(h):\n"
            "    h.register_tool(\n"
            "        name='a', description='d', start=print, parameters={'type': 'array'}\n"
            "    )\n",
            "object JSON Schema",
        ),
        (
            "def register(h):\n    h.register_tool(name='a', description=' ', start=print)\n",
            "description",
        ),
    ],
)
def test_extension_errors(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "bad_ext.py"
    path.write_text(body)
    with pytest.raises(ConfigError, match=message):
        load_extensions(str(path))


def test_a_missing_extension_file() -> None:
    with pytest.raises(ConfigError, match="no such file"):
        load_extensions("./nowhere/tools.py")


def test_validate_arguments() -> None:
    schema = {
        "type": "object",
        "properties": {
            "speed": {"type": "number"},
            "count": {"type": "integer"},
            "mode": {"type": "string", "enum": ["fast", "slow"]},
        },
        "required": ["speed"],
        "additionalProperties": False,
    }
    assert validate_arguments(schema, {"speed": 1, "count": 2, "mode": "fast"}) is None
    assert "missing required" in (validate_arguments(schema, {}) or "")
    assert "must be a number" in (validate_arguments(schema, {"speed": "x"}) or "")
    assert "must be a number" in (validate_arguments(schema, {"speed": True}) or "")
    assert "must be a integer" in (validate_arguments(schema, {"speed": 1, "count": 1.5}) or "")
    assert "one of" in (validate_arguments(schema, {"speed": 1, "mode": "medium"}) or "")
    assert "unknown arguments" in (validate_arguments(schema, {"speed": 1, "x": 2}) or "")
    assert "JSON object" in (validate_arguments(schema, [1]) or "")


def test_controls_resolve_globs_and_parts() -> None:
    info = arm_info()
    assert resolve_controls(None, info) == (0, 1, 2, 3)
    assert resolve_controls(["r_*"], info) == (2, 3)
    assert resolve_controls(["part:left", "r_j2"], info) == (0, 1, 3)
    with pytest.raises(ConfigError, match="matches no action dimension"):
        resolve_controls(["wheel"], info)
    sub = sub_embodiment(info, (2, 3))
    semantics = sub.action_space.semantics
    assert sub.action_space.shape == (2,) and semantics is not None
    assert semantics.dim_labels == ("r_j1", "r_j2") and semantics.max_step == (None, None)
    assert sub_embodiment(info, (0, 1, 2, 3)) is info


def context(dims: tuple[int, ...] = (2, 3)) -> tuple[JobContext, list[JobEvent]]:
    events: list[JobEvent] = []
    return (
        JobContext(
            job_id="t#1",
            tool="t",
            arguments={},
            embodiment=arm_info(),
            dims=dims,
            post=events.append,
        ),
        events,
    )


def test_jobs_take_their_dims_play_chunks_and_finish_on_request_stop() -> None:
    ctx, events = context()
    outputs = iter(
        [
            np.array([0.1, 0.2, 0.3, 0.4]),  # every dimension: the job's two are taken
            [0.5, 0.6],  # exactly the job's dimensions
            None,  # hold
            ActionChunk(
                actions=[
                    Action(data=np.array([0.7, 0.8]), meta={"events": ["halfway"]}),
                    Action(
                        data=np.array([0.9, 1.0]), meta={"request_stop": True, "stop_detail": "ok"}
                    ),
                ]
            ),
        ]
    )

    class Behavior:
        def step(self, observation: Observation) -> Any:
            return next(outputs)

    job = Job(Behavior(), ctx)
    observation = Observation()
    assert job.step(observation, 4).tolist() == [0.3, 0.4]
    assert job.step(observation, 4).tolist() == [0.5, 0.6]
    assert job.step(observation, 4) is None
    assert job.step(observation, 4).tolist() == [0.7, 0.8]
    assert [e.text for e in events] == ["halfway"]
    assert job.step(observation, 4).tolist() == [0.9, 1.0]
    assert job.ended == ("finished", "ok")


def test_jobs_refuse_wrong_sizes_time_out_and_run_monitors() -> None:
    ctx, events = context()

    class Wrong:
        def step(self, observation: Observation) -> Any:
            return [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match="returned 3 values; expected 2"):
        Job(Wrong(), ctx).step(Observation(), 4)

    clock = [0.0]
    timed_ctx = JobContext(
        job_id="slow#1",
        tool="slow",
        arguments={},
        embodiment=arm_info(),
        dims=(0,),
        post=events.append,
        clock=lambda: clock[0],
    )
    seen: list[int] = []

    def monitor(observation: Observation, ctx: JobContext) -> None:
        seen.append(ctx.steps)
        if ctx.steps == 1:
            ctx.emit("checked", wake=False)

    job = Job(Wrong(), timed_ctx, max_seconds=5, monitors=[monitor])
    job.behavior = type("Idle", (), {"step": lambda self, o: None})()
    job.step(Observation(), 4)
    job.step(Observation(), 4)
    assert seen == [0, 1] and events[-1].text == "checked" and not events[-1].wake
    clock[0] = 6.0
    job.step(Observation(), 4)
    assert job.ended == ("failed", "timed out after 5s")


def test_a_behavior_without_step_is_refused() -> None:
    ctx, _ = context()
    with pytest.raises(ConfigError, match="has no step"):
        Job(object(), ctx)


def test_policy_behaviors_steer_instructions_and_commands() -> None:
    seen: list[tuple[str | None, dict[str, float]]] = []

    class Walker:
        from abstract_harness.robot.core.policy import PolicyConfig, PolicyInfo

        info = PolicyInfo(name="walker", action_space=Box(shape=(2,)))
        config = PolicyConfig()

        def reset(self, scene: Any) -> None:
            seen.append((scene.instruction, {}))

        def act(self, observation: Observation) -> ActionChunk:
            seen.append((observation.instruction, dict(observation.extra.get("commands") or {})))
            return ActionChunk(actions=[Action(data=np.zeros(2))])

    ctx, _ = context()
    ctx.arguments = {"vx": 0.3}
    behavior = PolicyBehavior(Walker(), ctx, instruction="walk", commands=("vx", "yaw_rate"))
    behavior.step(Observation(extra={"commands": {"vx": 0.0, "speed": 1.0}}))
    assert seen[-1] == ("walk", {"vx": 0.3, "speed": 1.0})
    assert behavior.steer(instruction="walk to the door", yaw_rate=0.2) == (
        "now instruction 'walk to the door', yaw_rate=0.2"
    )
    behavior.step(Observation())
    assert seen[-1] == ("walk to the door", {"vx": 0.3, "yaw_rate": 0.2})
    with pytest.raises(ValueError, match="unknown command"):
        behavior.steer(height=1.0)
    with pytest.raises(ValueError, match="nothing to steer"):
        behavior.steer()
    assert seen[0] == ("walk", {})  # reset with the job's instruction
