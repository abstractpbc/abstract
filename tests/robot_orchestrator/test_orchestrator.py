"""The orchestrator end to end: a scripted model starts, steers and stops jobs while the
control loop keeps stepping them, on CubePick and through the robot CLI's rehearsal."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from abstract_harness.robot.core import eval
from abstract_harness.robot.core.approver import AutoApprover
from abstract_harness.robot.core.console import ConsolePoll
from abstract_harness.robot.core.controller import DefaultController
from abstract_harness.robot.core.errors import ConfigError
from abstract_harness.robot.core.logging.sink import NullSink
from abstract_harness.robot.core.mock import CubePickEmbodiment
from abstract_harness.robot.core.policy import PolicyConfig, PolicyInfo
from abstract_harness.robot.core.rollout import rollout
from abstract_harness.robot.core.scene import Scene
from abstract_harness.robot.core.scorer import success_at_end
from abstract_harness.robot.core.spaces import ActionSemantics, Box
from abstract_harness.robot.core.task import Task
from abstract_harness.robot.orchestrator import ExtensionAPI, OrchestratorPolicy, load_extensions

EXAMPLE = Path(__file__).parents[2] / "examples" / "extensions" / "cubepick_tools.py"
Calls = list[tuple[str, dict[str, Any]]]


class FakeModel:
    """A Chat Completions stand-in that answers with ``decide(history, n)``'s tool calls,
    where ``history`` is every user-side text so far and ``n`` counts calls."""

    def __init__(self, decide: Callable[[str, int], Calls | httpx.Response]) -> None:
        self.decide = decide
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        with self.lock:
            self.requests.append(body)
            n = len(self.requests)
        history = "\n".join(_texts(body["messages"]))
        calls = self.decide(history, n)
        if isinstance(calls, httpx.Response):
            return calls
        tool_calls = [
            {
                "id": f"call_{n}_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for i, (name, arguments) in enumerate(calls)
        ]
        message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        return httpx.Response(200, json={"choices": [{"message": message}]})


def _texts(messages: list[dict[str, Any]]) -> list[str]:
    out = []
    for message in messages:
        if message.get("role") not in ("user", "tool"):
            continue
        content = message.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out += [part["text"] for part in content if part.get("type") == "text"]
    return out


def orchestrator(model: FakeModel, **kwargs: Any) -> OrchestratorPolicy:
    return OrchestratorPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(model),
        env={},
        **kwargs,
    )


def fast_cube() -> CubePickEmbodiment:
    cube = CubePickEmbodiment()
    cube.info = replace(cube.info, control_hz=100.0)
    return cube


def task(max_steps: int = 400) -> Task:
    return Task(
        name="orchestrated",
        scenes=[Scene(id="s0", instruction="find and reach the cube", init_seed=0)],
        scorer=success_at_end(),
        max_steps=max_steps,
    )


DONE = ("done", {"summary": "finished", "hindsight": "none"})


def test_the_model_starts_steers_and_switches_jobs_while_the_robot_moves(tmp_path: Path) -> None:
    flags: dict[str, bool] = {}

    def decide(history: str, n: int) -> Calls:
        if n == 1:
            return [("scan", {"speed": 0.02})]
        if "[event] scan#1: corner 1" in history and not flags.get("steered"):
            flags["steered"] = True
            return [("steer_job", {"job_id": "scan#1", "arguments": {"speed": 0.04}})]
        if "[event] scan#1: corner 2" in history and not flags.get("switched"):
            flags["switched"] = True
            return [("stop_job", {"job_id": "scan#1"}), ("reach_cube", {})]
        return [("wait", {})]

    model = FakeModel(decide)
    policy = orchestrator(model, extensions=str(EXAMPLE))
    (log,) = eval(task(), policy, fast_cube(), log_dir=str(tmp_path))
    sample = log.samples[0]
    metadata = sample.trial_metadata[0]
    assert log.status == "success" and sample.termination_reasons == ("success",)
    jobs = {entry["job"]: entry for entry in metadata["jobs"]}
    assert jobs["scan#1"]["status"] == "stopped"
    assert jobs["scan#1"]["arguments"] == {"speed": 0.02}
    assert jobs["reach_cube#2"]["status"] in ("stopped", "finished")
    # The model ran on a paced loop it asked for: this CubePick does not pace itself.
    assert metadata["timing"]["auto"] is True and metadata["timing"]["hz"] == 100.0
    tools = [schema["function"]["name"] for schema in model.requests[0]["tools"]]
    assert tools == [
        "reach_cube",
        "scan",
        "wait",
        "stop_job",
        "steer_job",
        "jobs",
        "move_by",
        "done",
        "give_up",
    ]
    history = "\n".join(_texts(model.requests[-1]["messages"]))
    assert "started scan#1 (drives dx, dy)" in history
    assert "[event] scan#1: speed 0.04, side 0.2" in history  # the steer's reply
    assert "stopping scan#1" in history and "started reach_cube#2" in history
    transcript = sample.policy_transcripts[0]
    assert transcript[0]["role"] == "system" and "background" not in transcript[0]["content"][:20]
    assert log.eval.policy == "orchestrator"
    assert list(log.eval.policy_config["tools"]) == ["reach_cube", "scan"]


def test_done_ends_the_trial_at_once(tmp_path: Path) -> None:
    model = FakeModel(lambda history, n: [DONE])
    (log,) = eval(task(), orchestrator(model), fast_cube(), log_dir=str(tmp_path))
    assert log.samples[0].termination_reasons == ("done",)
    assert len(model.requests) == 1


def test_jobs_on_the_same_dims_conflict_until_one_is_stopped(tmp_path: Path) -> None:
    def decide(history: str, n: int) -> Calls:
        if n == 1:
            return [("scan", {}), ("scan", {}), ("stop_job", {"job_id": "scan#1"}), ("scan", {})]
        return [DONE]

    model = FakeModel(decide)
    (log,) = eval(
        task(), orchestrator(model, extensions=str(EXAMPLE)), fast_cube(), log_dir=str(tmp_path)
    )
    results = [m["content"] for m in model.requests[1]["messages"] if m.get("role") == "tool"]
    assert results[0].startswith("started scan#1")
    assert results[1].startswith("scan#1 already drives dx, dy; stop it first")
    assert results[2] == "stopping scan#1"
    assert results[3].startswith("started scan#2")
    jobs = {e["job"]: e["status"] for e in log.samples[0].trial_metadata[0]["jobs"]}
    assert jobs == {"scan#1": "stopped", "scan#2": "stopped"}


def test_the_built_in_motion_tool_runs_as_a_job(tmp_path: Path) -> None:
    def decide(history: str, n: int) -> Calls:
        if n == 1:
            return [("move_by", {"deltas": {"dx": 0.3}, "note": "the cube is to the right"})]
        if "move_by#1: finished" in history:
            return [DONE]
        return [("wait", {})]

    cube = fast_cube()
    model = FakeModel(decide)
    (log,) = eval(task(), orchestrator(model), cube, log_dir=str(tmp_path))
    (job,) = log.samples[0].trial_metadata[0]["jobs"]
    assert job["job"] == "move_by#1" and job["status"] == "finished"
    assert job["summary"] == "motion complete"
    assert cube._eef[0] == pytest.approx(0.4)
    assert log.samples[0].termination_reasons == ("done",)


def test_bad_arguments_and_unknown_jobs_are_correctable(tmp_path: Path) -> None:
    def decide(history: str, n: int) -> Calls:
        if n == 1:
            return [
                ("scan", {"speed": "fast"}),
                ("scan", {"radius": 1}),
                ("stop_job", {"job_id": "scan#9"}),
                ("steer_job", {"job_id": "scan#9", "arguments": {}}),
                ("wait", {"seconds": 99}),
                ("jobs", {}),
                ("teleport", {}),
            ]
        return [DONE]

    model = FakeModel(decide)
    eval(task(), orchestrator(model, extensions=str(EXAMPLE)), fast_cube(), log_dir=str(tmp_path))
    results = [m["content"] for m in model.requests[1]["messages"] if m.get("role") == "tool"]
    assert results[0] == "scan: speed must be a number"
    assert results[1].startswith("scan: unknown arguments ['radius']")
    assert results[2] == "no running job 'scan#9'; running: none"
    assert results[3] == "arguments must be a non-empty object of what to change"
    assert results[4].startswith("seconds must be a number in (0, 30]")
    assert results[5] == "no jobs yet"
    assert results[6].startswith("unknown tool 'teleport'")


def test_a_model_failure_is_a_policy_error(tmp_path: Path) -> None:
    model = FakeModel(lambda history, n: httpx.Response(400, json={"error": "bad request"}))
    (log,) = eval(task(), orchestrator(model), fast_cube(), log_dir=str(tmp_path))
    assert log.samples[0].status == "error"
    assert "orchestrator model failed" in (log.samples[0].error or "")


def test_the_call_budget_ends_in_give_up(tmp_path: Path) -> None:
    model = FakeModel(lambda history, n: [("wait", {"seconds": 0.01})])
    (log,) = eval(task(), orchestrator(model, max_llm_calls=3), fast_cube(), log_dir=str(tmp_path))
    assert len(model.requests) == 3
    assert log.samples[0].termination_reasons == ("give_up",)


def test_operator_feedback_wakes_the_model() -> None:
    def decide(history: str, n: int) -> Calls:
        if "operator feedback: try the left side" in history:
            return [DONE]
        return [("wait", {})]

    class Console:
        def __init__(self) -> None:
            self.polls = 0

        def begin_trial(self) -> None:
            return None

        def poll(self) -> ConsolePoll:
            self.polls += 1
            return (
                ConsolePoll(messages=("try the left side",)) if self.polls == 5 else ConsolePoll()
            )

    model = FakeModel(decide)
    policy = orchestrator(model)
    cube = fast_cube()
    policy.bind(cube.info)
    from abstract_harness.robot.core.timing import Pacer

    started = time.monotonic()
    record = rollout(
        policy,
        cube,
        Scene(id="s", instruction="reach", init_seed=0),
        max_steps=500,
        seed=0,
        epoch=0,
        controller=DefaultController(),
        approver=AutoApprover(),
        sink=NullSink(),
        operator_input=Console(),  # type: ignore[arg-type]
        pacer=Pacer(100),
    )
    assert record.termination_reason == "done"
    # Woken by the feedback, not by the 30-second wait.
    assert time.monotonic() - started < 10 and len(model.requests) == 2


def test_the_model_stops_when_the_rollout_stops_calling() -> None:
    model = FakeModel(lambda history, n: [("wait", {"seconds": 0.05})])
    policy = orchestrator(model, idle_timeout_s=0.3)
    cube = fast_cube()
    policy.bind(cube.info)
    policy.reset(Scene(id="s", instruction="reach"))
    policy.act(cube.reset(Scene(id="s", instruction="reach")))
    brain = policy._brain
    assert brain is not None
    brain.join(timeout=3)
    assert not brain.is_alive()
    calls = len(model.requests)
    time.sleep(0.2)
    assert len(model.requests) == calls  # nothing more is spent


def test_policy_tools_are_checked_against_the_robot_at_bind() -> None:
    class ThreeJoints:
        info = PolicyInfo(
            name="three",
            action_space=Box(shape=(3,), semantics=ActionSemantics("joint_pos")),
        )
        config = PolicyConfig()

        def reset(self, scene: Any) -> None:
            return None

        def act(self, observation: Any) -> Any:
            raise AssertionError("never called")

    api = ExtensionAPI()
    api.register_policy_tool(name="grasp", description="grasp", policy=ThreeJoints())
    policy = orchestrator(FakeModel(lambda h, n: [DONE]), api=api)
    with pytest.raises(ConfigError, match="policy tool 'grasp' cannot drive 'cubepick'"):
        policy.bind(CubePickEmbodiment().info)


def test_orchestrator_arguments_are_validated() -> None:
    model = FakeModel(lambda h, n: [DONE])
    for kwargs, message in (
        ({"heartbeat_s": -1}, "heartbeat_s"),
        ({"max_wait_s": 0}, "max_wait_s"),
        ({"max_jobs": 0}, "max_jobs"),
        ({"builtin_motion": "yes"}, "builtin_motion"),
        ({"pre_check": lambda w: None}, "pre_check"),
    ):
        with pytest.raises(ConfigError, match=message):
            orchestrator(model, **kwargs)


def test_without_built_in_motion_only_extension_and_control_tools_remain() -> None:
    policy = orchestrator(
        FakeModel(lambda h, n: [DONE]), api=load_extensions(str(EXAMPLE)), builtin_motion=False
    )
    policy.bind(CubePickEmbodiment().info)
    assert "move_by" not in policy.tools and "scan" in policy.tools


def test_the_registry_and_cli_build_an_orchestrator(tmp_path: Path) -> None:
    from abstract_harness.robot.core.registry import resolve

    policy = resolve(
        "policy",
        "orchestrator",
        model="test/model",
        base_url="http://llm.test/v1",
        env={},
        extensions=str(EXAMPLE),
        transport=httpx.MockTransport(FakeModel(lambda h, n: [DONE])),
    )
    assert isinstance(policy, OrchestratorPolicy)
    assert policy.config.extensions == (str(EXAMPLE),)


def test_a_robot_config_can_choose_the_orchestrator_and_rehearses_it(tmp_path: Path) -> None:
    from abstract_harness.robot import cli as robot_cli
    from abstract_harness.robot.config import RobotConfig

    config = tmp_path / "robot.toml"
    config.write_text(
        '[agent]\npolicy = "orchestrator"\nmodel = "gpt-test"\nbudget_usd = 1.0\n'
        "max_calls = 5\nmax_output_tokens = 1000\n"
        "[agent.pricing]\ninput_per_million = 2.0\ncached_input_per_million = 0.2\n"
        "output_per_million = 10.0\n"
        f'[agent.args]\nextensions = "{EXAMPLE}"\n'
        '[rig]\nlog_root = "logs"\n[rig.embodiment]\nleft_channel = "can1"\n'
        'right_channel = "can0"\n'
    )
    loaded = RobotConfig.load(config)
    assert loaded.policy_name == "orchestrator"
    assert loaded.setup()["policy"] == "orchestrator"
    # The free, motion-free rehearsal: the orchestrator on CubePick through the spend
    # meter, against a stand-in Responses API that answers done, then a refused budget.
    ok, detail = robot_cli.rehearse(loaded)
    assert ok, detail
    assert detail == (
        "done: exit 0, 1 call(s) through the meter; "
        "budget refusal: exit 1, 0 upstream call(s), 1 blocked"
    )


def test_holding_absolute_dims_keeps_the_last_target() -> None:
    from abstract_harness.robot.core.embodiment import joint_embodiment_info
    from abstract_harness.robot.core.types import Observation

    info = joint_embodiment_info("arm", low=[-1, -1], high=[1, 1], dim_labels=["a", "b"])

    def decide(history: str, n: int) -> Calls:
        if n == 1:
            return [("move_joints", {"targets": {"a": 0.05}, "note": "nudge a"})]
        return [("wait", {})]

    policy = orchestrator(FakeModel(decide))
    policy.bind(info)
    policy.reset(Scene(id="s", instruction="nudge"))
    state = np.array([0.0, 0.5])
    seen = []
    for _ in range(40):
        chunk = policy.act(Observation(state={"joint_pos": state.copy()}))
        state = np.asarray(chunk.actions[0].data, float)
        seen.append(state.copy())
        if any(e["status"] == "finished" for e in policy._history):
            break
    # Joint a moved to its target; b, which no job drives, held its measured position.
    assert seen[-1][0] == pytest.approx(0.05) and seen[-1][1] == pytest.approx(0.5)
    assert all(row[1] == pytest.approx(0.5) for row in seen)
    policy._shutdown()


def test_extension_files_resolve_against_the_robot_config(tmp_path: Path) -> None:
    from abstract_harness.robot.config import RobotConfig

    rig = Path(__file__).parents[2] / "rigs" / "cubepick-orchestrator.toml"
    config = RobotConfig.load(rig)
    assert config.policy_name == "orchestrator" and config.rig.type == "cubepick"
    assert config.agent is not None
    assert config.agent.args["extensions"] == str(EXAMPLE.resolve())
    mixed = tmp_path / "robot.toml"
    mixed.write_text(
        rig.read_text().replace(
            'extensions = "../examples/extensions/cubepick_tools.py"',
            'extensions = "my_pkg.tools, ./local.py:setup"',
        )
    )
    assert RobotConfig.load(mixed).agent.args["extensions"] == (  # type: ignore[union-attr]
        f"my_pkg.tools,{tmp_path / 'local.py'}:setup"
    )


def test_list_form_extensions_resolve_too(tmp_path: Path) -> None:
    from abstract_harness.robot.config import RobotConfig

    rig = Path(__file__).parents[2] / "rigs" / "cubepick-orchestrator.toml"
    listed = tmp_path / "robot.toml"
    listed.write_text(
        rig.read_text().replace(
            'extensions = "../examples/extensions/cubepick_tools.py"',
            'extensions = ["tools.py", "my_pkg.tools"]',
        )
    )
    agent = RobotConfig.load(listed).agent
    assert agent is not None
    assert agent.args["extensions"] == f"{tmp_path / 'tools.py'},my_pkg.tools"


def _arm() -> Any:
    from abstract_harness.robot.core.embodiment import joint_embodiment_info

    return joint_embodiment_info("arm", low=[-2, -2], high=[2, 2], dim_labels=["a", "b"])


def _drive_policy(policy: OrchestratorPolicy, ticks: int, until: Callable[[], bool]) -> list[Any]:
    """Act like a robot whose approvers allow 0.01 per step toward each target."""
    from abstract_harness.robot.core.types import Observation

    state = np.zeros(2)
    trace = []
    for _ in range(ticks):
        chunk = policy.act(Observation(state={"joint_pos": state.copy()}))
        target = np.asarray(chunk.actions[0].data, float)
        state = state + np.clip(target - state, -0.01, 0.01)
        trace.append((target.copy(), state.copy()))
        if until():
            break
        time.sleep(0.005)
    return trace


def test_a_stopped_job_holds_where_the_robot_is_not_at_its_last_target() -> None:
    class Far:
        def __init__(self, ctx: Any) -> None:
            self.ctx = ctx

        def step(self, observation: Any) -> list[float]:
            return [1.5]  # far away: the approvers walk toward it 0.01 per step

    api = ExtensionAPI()
    api.register_tool(name="reach", description="reach far", start=Far, controls=["a"])
    stopped = threading.Event()

    def decide(history: str, n: int) -> Calls:
        if n == 1:
            return [("reach", {})]
        if not stopped.is_set():
            stopped.set()
            return [("stop_job", {"job_id": "reach#1"})]
        return [("wait", {})]

    policy = orchestrator(FakeModel(decide), api=api, heartbeat_s=0.05)
    policy.bind(_arm())
    policy.reset(Scene(id="s", instruction="reach"))
    trace = _drive_policy(
        policy, 400, lambda: False if not stopped.is_set() else len(policy._jobs) == 0
    )
    for _ in range(20):
        trace += _drive_policy(policy, 1, lambda: False)
    policy._shutdown()
    after = [target for target, _ in trace[-15:]]
    # Once stopped, the target is the measured position, so the robot stops moving.
    assert all(np.allclose(target, after[0]) for target in after)
    assert after[0][0] < 1.0


def test_a_slow_job_step_does_not_end_the_model_thread(tmp_path: Path) -> None:
    class Slow:
        def __init__(self, ctx: Any) -> None:
            self.calls = 0

        def step(self, observation: Any) -> None:
            self.calls += 1
            if self.calls == 2:
                time.sleep(0.6)  # longer than idle_timeout_s: a cold-starting model

    api = ExtensionAPI()
    api.register_tool(name="warm_up", description="slow", start=Slow)

    def decide(history: str, n: int) -> Calls:
        return [("warm_up", {})] if n == 1 else [DONE]

    model = FakeModel(decide)
    policy = orchestrator(model, api=api, heartbeat_s=0.1, idle_timeout_s=0.2)
    (log,) = eval(task(), policy, fast_cube(), log_dir=str(tmp_path))
    assert log.samples[0].termination_reasons == ("done",)


def test_policy_classes_are_built_and_steers_are_checked(tmp_path: Path) -> None:
    from abstract_harness.robot.core.mock import ScriptedPolicy

    api = ExtensionAPI()
    api.register_policy_tool(name="reach", description="reach", policy=ScriptedPolicy)
    api.register_tool(
        name="scan",
        description="scan",
        start=lambda ctx, speed=0.01: type("S", (), {"step": lambda self, o: [0.0, 0.0]})(),
        parameters={
            "type": "object",
            "properties": {"speed": {"type": "number"}},
            "additionalProperties": False,
        },
    )

    def decide(history: str, n: int) -> Calls:
        if n == 1:
            return [
                ("scan", {}),
                ("steer_job", {"job_id": "scan#1", "arguments": {"radius": 1}}),
                (
                    "move_by",
                    {"deltas": {"dx": 0.1}, "targets": {"bogus": 1}, "note": "nudge right"},
                ),
            ]
        return [DONE]

    model = FakeModel(decide)
    policy = orchestrator(model, api=api, heartbeat_s=0.05)
    eval(task(), policy, fast_cube(), log_dir=str(tmp_path))
    assert isinstance(policy._instances["reach"], ScriptedPolicy)
    results = [m["content"] for m in model.requests[1]["messages"] if m.get("role") == "tool"]
    assert results[0].startswith("started scan#1")
    assert results[1].startswith("steer_job scan#1: unknown arguments ['radius']")
    assert results[2].startswith("scan#1 already drives dx;")  # parsed, not a KeyError


def test_a_finished_trial_ignores_its_old_model_thread() -> None:
    policy = orchestrator(FakeModel(lambda h, n: [DONE]))
    policy.bind(CubePickEmbodiment().info)
    policy.reset(Scene(id="s", instruction="x"))
    from abstract_harness.robot.agent._llm import ToolCall

    stale = policy._generation - 1
    observation = CubePickEmbodiment().reset(Scene(id="s", instruction="x"))
    call = ToolCall("c", "done", json.dumps({"summary": "s", "hindsight": "none"}))
    assert policy._call(call, observation, stale) == ("ignored: the trial already ended", "end")
    assert policy._finish is None


def test_the_first_decision_has_a_deadline() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        time.sleep(1.0)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hmm"}}]})

    policy = OrchestratorPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(slow),
        env={},
        first_turn_timeout_s=0.2,
    )
    cube = CubePickEmbodiment()
    policy.bind(cube.info)
    policy.reset(Scene(id="s", instruction="x"))
    with pytest.raises(RuntimeError, match="no first decision in 0.2s"):
        policy.act(cube.reset(Scene(id="s", instruction="x")))
    policy._shutdown()
