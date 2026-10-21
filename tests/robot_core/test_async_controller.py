"""AsyncController: inference overlaps execution, stale predictions are dropped, and the
robot holds (never guesses) when the buffer runs dry."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from abstract_harness.robot.core import eval
from abstract_harness.robot.core.controller import _INFER_KEY, AsyncController
from abstract_harness.robot.core.errors import ConfigError, PolicyError, SafetyAbort
from abstract_harness.robot.core.mock import CubePickEmbodiment, ScriptedPolicy
from abstract_harness.robot.core.policy import PolicyConfig, PolicyInfo
from abstract_harness.robot.core.scene import Scene
from abstract_harness.robot.core.scorer import success_at_end
from abstract_harness.robot.core.spaces import ActionSemantics, Box
from abstract_harness.robot.core.task import Task
from abstract_harness.robot.core.types import Action, ActionChunk, Observation


class StepPolicy:
    """Predicts, from the observation of step q, a chunk whose action i is ``[q + i]``:
    an executed action's value names the step it was predicted for."""

    def __init__(self, mode: str = "joint_pos", horizon: int = 6, delay_s: float = 0.0):
        self.info = PolicyInfo(
            name="steps",
            action_space=Box(shape=(1,), semantics=ActionSemantics(control_mode=mode)),
        )
        self.config = PolicyConfig(action_horizon=horizon)
        self.horizon = horizon
        self.delay_s = delay_s
        self.calls: list[int] = []
        self.gate: threading.Event | None = None

    def reset(self, scene: Scene) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        q = int(observation.extra["env_step"])
        self.calls.append(q)
        if self.gate is not None:
            self.gate.wait(5)
        if self.delay_s:
            time.sleep(self.delay_s)
        return ActionChunk(
            actions=[Action(data=np.array([q + i], float)) for i in range(self.horizon)]
        )


def _report(store: dict[str, Any]) -> dict[str, Any]:
    report = AsyncController().report(store)  # the trial's state lives in the store
    assert report is not None
    return report


def _obs(t: int) -> Observation:
    return Observation(state={"joint_pos": np.zeros(1)}, extra={"env_step": t})


def _drive(controller: AsyncController, policy: Any, steps: int, tick_s: float = 0.0) -> Any:
    store: dict[str, Any] = {}
    actions = []
    for t in range(steps):
        actions.append(controller.next_action(policy, _obs(t), t, store))
        if tick_s:
            time.sleep(tick_s)
    return actions, store


def test_first_step_waits_for_inference_then_plays_the_chunk() -> None:
    policy = StepPolicy(horizon=4)
    actions, store = _drive(AsyncController(prefetch=0), policy, 4)
    assert [float(a.data[0]) for a in actions] == [0, 1, 2, 3]
    assert policy.calls[0] == 0
    assert _report(store)["controller"] == "async"


def test_executed_actions_always_match_their_step_under_latency() -> None:
    # Inference takes ~3 ticks; prefetch starts it early. Every non-hold action must be
    # the prediction for the step it runs at: stale predictions are dropped, not played.
    policy = StepPolicy(horizon=8, delay_s=0.03)
    actions, store = _drive(AsyncController(prefetch=4), policy, 40, tick_s=0.01)
    for t, action in enumerate(actions):
        if action.meta.get("late"):
            assert float(action.data[0]) < t  # an older prediction, played late
        elif not action.meta.get("hold"):
            assert float(action.data[0]) == t
    report = _report(store)
    assert report["inferences"] >= 4
    assert report["stale_actions"] > 0  # latency compensation dropped elapsed steps
    assert len(store[_INFER_KEY]) == report["inferences"]


def test_underrun_holds_the_last_absolute_target() -> None:
    policy = StepPolicy(horizon=2)
    controller = AsyncController(prefetch=0)
    store: dict[str, Any] = {}
    first = controller.next_action(policy, _obs(0), 0, store)
    policy.gate = threading.Event()  # the next inference blocks until released
    second = controller.next_action(policy, _obs(1), 1, store)
    assert [float(first.data[0]), float(second.data[0])] == [0, 1]
    hold = controller.next_action(policy, _obs(2), 2, store)
    assert hold.meta == {"hold": True}
    np.testing.assert_array_equal(hold.data, [1.0])
    policy.gate.set()
    deadline = time.monotonic() + 5
    while True:
        action = controller.next_action(policy, _obs(3), 3, store)
        if not action.meta.get("hold") or time.monotonic() > deadline:
            break
    # The chunk from step 1 predicted steps 1 and 2: all stale at step 3, so its newest
    # action runs late rather than the robot holding until the next inference.
    assert float(action.data[0]) == 2 and action.meta["late"] is True
    assert _report(store)["holds"] >= 1 and _report(store)["late_actions"] == 1


@pytest.mark.parametrize("mode", ["joint_delta", "eef_delta_pos", "joint_vel", "base_vel"])
def test_underrun_sends_zeros_for_rate_and_displacement_modes(mode: str) -> None:
    policy = StepPolicy(mode=mode, horizon=1)
    controller = AsyncController(prefetch=0)
    store: dict[str, Any] = {}
    controller.next_action(policy, _obs(0), 0, store)
    policy.gate = threading.Event()
    hold = controller.next_action(policy, _obs(1), 1, store)
    assert hold.meta == {"hold": True}
    np.testing.assert_array_equal(hold.data, [0.0])
    policy.gate.set()


def test_wait_mode_blocks_instead_of_holding() -> None:
    policy = StepPolicy(horizon=1, delay_s=0.02)
    actions, store = _drive(AsyncController(prefetch=0, underrun="wait"), policy, 5)
    assert [float(a.data[0]) for a in actions] == [0, 1, 2, 3, 4]
    assert _report(store)["holds"] == 0


def test_no_semantics_means_wait() -> None:
    policy = StepPolicy(horizon=1, delay_s=0.01)
    policy.info = PolicyInfo(name="bare", action_space=Box(shape=(1,)))
    actions, store = _drive(AsyncController(prefetch=0), policy, 3)
    assert not any(a.meta.get("hold") for a in actions)


def test_policies_with_observe_hooks_are_refused() -> None:
    policy = StepPolicy()
    policy.observe = lambda observation: None  # type: ignore[attr-defined]
    with pytest.raises(ConfigError, match="observe"):
        AsyncController().next_action(policy, _obs(0), 0, {})


def test_inference_errors_surface_on_the_rollout_thread() -> None:
    policy = StepPolicy(horizon=1)

    def boom(observation: Observation) -> ActionChunk:
        raise RuntimeError("model crashed")

    policy.act = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="model crashed"):
        AsyncController().next_action(policy, _obs(0), 0, {})


def test_empty_chunks_are_policy_errors() -> None:
    policy = StepPolicy()
    policy.act = lambda observation: "nothing"  # type: ignore[assignment,method-assign]
    with pytest.raises(PolicyError, match="empty or invalid"):
        AsyncController().next_action(policy, _obs(0), 0, {})


def test_a_stuck_inference_aborts() -> None:
    policy = StepPolicy()
    policy.gate = threading.Event()
    with pytest.raises(SafetyAbort, match="async inference exceeded"):
        AsyncController(timeout_s=0.1).next_action(policy, _obs(0), 0, {})
    policy.gate.set()


def test_a_stop_request_is_never_replaced_by_a_later_prediction() -> None:
    class Stopper(StepPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            chunk = super().act(observation)
            actions = list(chunk.actions)
            actions[3] = Action(
                data=actions[3].data, meta={"request_stop": True, "stop_reason": "done"}
            )
            return ActionChunk(actions=actions)

    policy = Stopper(horizon=5)
    actions, _ = _drive(AsyncController(prefetch=4), policy, 4, tick_s=0.005)
    assert actions[3].meta.get("request_stop") is True
    assert policy.calls == [0]  # no inference was dispatched once the stop was buffered


@pytest.mark.parametrize("prefetch", [-1, 0.0, 1.5, True, "half"])
def test_prefetch_is_validated(prefetch: Any) -> None:
    with pytest.raises(ConfigError):
        AsyncController(prefetch=prefetch)


def test_other_arguments_are_validated() -> None:
    with pytest.raises(ConfigError):
        AsyncController(underrun="guess")  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        AsyncController(timeout_s=0)


def test_eval_runs_cubepick_with_async_inference_and_strict_timing(tmp_path: Any) -> None:
    from dataclasses import replace

    cube = CubePickEmbodiment()
    cube.info = replace(cube.info, control_hz=200.0)
    (log,) = eval(
        Task(
            name="async",
            scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
            scorer=success_at_end(),
            max_steps=60,
        ),
        ScriptedPolicy(chunk_size=8),
        cube,
        log_dir=str(tmp_path),
        controller=AsyncController(),
        strict_timing=True,
    )
    metadata = log.samples[0].trial_metadata[0]
    assert log.status == "success"
    assert metadata["controller"]["controller"] == "async"
    assert metadata["timing"]["ticks"] >= 1
    assert log.results.metrics["success_at_end"] == 1.0


def test_a_stop_on_a_stale_action_is_delivered() -> None:
    class StopFirst(StepPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            chunk = super().act(observation)
            if not self.calls[1:]:
                return chunk  # the first chunk is ordinary
            first = Action(data=chunk.actions[0].data, meta={"request_stop": True})
            return ActionChunk(actions=[first, *chunk.actions[1:]])

    policy = StopFirst(horizon=4, delay_s=0.03)
    actions, _ = _drive(AsyncController(prefetch=3), policy, 30, tick_s=0.01)
    assert any(a.meta.get("request_stop") for a in actions)


def test_a_wholly_stale_stop_is_delivered() -> None:
    class Done(StepPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            chunk = super().act(observation)
            if len(self.calls) == 1:
                return chunk
            return ActionChunk(
                actions=[Action(data=chunk.actions[0].data, meta={"request_stop": True})]
            )

    policy = Done(horizon=1, delay_s=0.03)
    actions, _ = _drive(AsyncController(prefetch=0), policy, 20, tick_s=0.01)
    assert any(a.meta.get("request_stop") for a in actions)


def test_a_one_action_policy_keeps_moving_and_no_observation_is_sent_twice() -> None:
    policy = StepPolicy(horizon=1, delay_s=0.025)
    actions, store = _drive(AsyncController(prefetch=0), policy, 40, tick_s=0.01)
    moving = [a for a in actions if not a.meta.get("hold")]
    assert len(moving) > len(actions) / 4  # it acts on most inferences instead of holding
    assert len(policy.calls) == len(set(policy.calls))  # every dispatched step is distinct


def test_agent_like_policies_are_refused() -> None:
    policy = StepPolicy()
    policy.transcript_delta = lambda: None  # type: ignore[attr-defined]
    with pytest.raises(ConfigError, match="transcript_delta"):
        AsyncController().next_action(policy, _obs(0), 0, {})


def test_end_trial_waits_for_a_slow_inference_to_finish() -> None:
    # A normal slow inference (a remote VLA's few seconds) is waited for, not abandoned.
    policy = StepPolicy(horizon=2)
    controller = AsyncController(prefetch=0)
    store: dict[str, Any] = {}
    controller.next_action(policy, _obs(0), 0, store)
    policy.delay_s = 0.5
    controller.next_action(policy, _obs(1), 1, store)  # dispatches the slow inference
    started = time.monotonic()
    assert controller.end_trial(store) is True
    assert time.monotonic() - started >= 0.3


def test_end_trial_waits_for_inference_and_reports_a_stuck_one() -> None:
    policy = StepPolicy(horizon=2)
    controller = AsyncController(prefetch=0, timeout_s=0.3)
    store: dict[str, Any] = {}
    controller.next_action(policy, _obs(0), 0, store)
    policy.gate = threading.Event()
    controller.next_action(policy, _obs(1), 1, store)  # dispatches the next inference
    assert controller.end_trial(store) is False  # still running at its deadline
    with pytest.raises(PolicyError, match="still running past its deadline"):
        controller.prepare_trial(policy)
    policy.gate.set()
    assert controller.end_trial(store) is True
    controller.prepare_trial(policy)


def test_the_rollout_never_resets_a_policy_under_a_stuck_inference() -> None:
    from abstract_harness.robot.core.approver import AutoApprover
    from abstract_harness.robot.core.logging.sink import NullSink
    from abstract_harness.robot.core.rollout import rollout

    class Cube(CubePickEmbodiment):
        pass

    release = threading.Event()

    class Hangs(ScriptedPolicy):
        resets = 0

        def reset(self, scene: Any) -> None:
            Hangs.resets += 1
            super().reset(scene)

        def act(self, observation: Observation) -> ActionChunk:
            if self.num_inferences >= 1:
                release.wait(5)
            return super().act(observation)

    policy = Hangs(chunk_size=2)
    record = rollout(
        policy,
        Cube(),
        Scene(id="s", instruction="reach", init_seed=0),
        max_steps=6,
        seed=0,
        epoch=0,
        controller=AsyncController(prefetch=1, timeout_s=0.3),
        approver=AutoApprover(),
        sink=NullSink(),
    )
    assert record.metadata["policy_abandoned"] is True
    assert record.policy_transcript is None
    release.set()


def test_events_reach_the_policy_once_under_async_inference() -> None:
    from abstract_harness.robot.core.approver import AutoApprover
    from abstract_harness.robot.core.command_stream import CommandStream
    from abstract_harness.robot.core.logging.sink import NullSink
    from abstract_harness.robot.core.rollout import rollout
    from abstract_harness.robot.core.timing import Pacer

    stream = CommandStream(events=["go"])
    seen: list[int] = []

    class Counting(ScriptedPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            seen.extend(event["t"] for event in observation.extra["command_events"])
            time.sleep(0.03)
            return super().act(observation)

    class Pressing(CubePickEmbodiment):
        def step(self, action: Any) -> Any:
            stream.press("go", source="test")
            return super().step(action)

    cube = Pressing(max_step=0.001)
    record = rollout(
        Counting(chunk_size=8, max_step=0.001),
        cube,
        Scene(id="s", instruction="reach", init_seed=0),
        max_steps=40,
        seed=0,
        epoch=0,
        controller=AsyncController(prefetch=4),
        approver=AutoApprover(),
        sink=NullSink(),
        pacer=Pacer(100),
        commands=stream,
    )
    pressed = [e.t for e in record.events if e.kind == "command"]
    # Every press is delivered exactly once, except those after the last dispatch.
    assert seen == sorted(set(seen)) and set(seen) <= set(pressed)
    assert len(seen) >= len(pressed) - 10


def test_a_slow_inference_at_trial_end_is_not_abandoned_and_never_overlaps_reset(
    tmp_path: Any,
) -> None:
    from abstract_harness.robot.core import eval
    from abstract_harness.robot.core.task import Epochs

    running = threading.Event()
    overlaps: list[str] = []

    class Slow(ScriptedPolicy):
        resets = 0
        ended = 0

        def reset(self, scene: Any) -> None:
            if running.is_set():
                overlaps.append("reset() while act() runs")
            Slow.resets += 1
            super().reset(scene)

        def on_trial_end(self, record: Any, log_dir: str, run_id: str) -> None:
            Slow.ended += 1

        def act(self, observation: Observation) -> ActionChunk:
            running.set()
            try:
                if self.num_inferences >= 1:
                    time.sleep(0.4)
                return super().act(observation)
            finally:
                running.clear()

    (log,) = eval(
        Task(
            name="slow",
            scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
            scorer=success_at_end(),
            max_steps=3,
            epochs=Epochs(2),
        ),
        Slow(chunk_size=4, max_step=0.001),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        controller=AsyncController(prefetch=3),
    )
    assert overlaps == []
    assert Slow.resets == 2 and Slow.ended == 2
    assert not any(meta.get("policy_abandoned") for meta in log.samples[0].trial_metadata)
