"""AsyncController: inference overlaps execution, stale predictions are dropped, and the
robot holds (never guesses) when the buffer runs dry."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from abstract_harness.robot.core import eval
from abstract_harness.robot.core.controller import _INFER_KEY, _REPORT_KEY, AsyncController
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
    assert store[_REPORT_KEY]["controller"] == "async"


def test_executed_actions_always_match_their_step_under_latency() -> None:
    # Inference takes ~3 ticks; prefetch starts it early. Every non-hold action must be
    # the prediction for the step it runs at: stale predictions are dropped, not played.
    policy = StepPolicy(horizon=8, delay_s=0.03)
    actions, store = _drive(AsyncController(prefetch=4), policy, 40, tick_s=0.01)
    for t, action in enumerate(actions):
        if not action.meta.get("hold"):
            assert float(action.data[0]) == t
    report = store[_REPORT_KEY]
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
    assert float(action.data[0]) == 3  # chunk from step 1 lands with its stale actions dropped
    assert store[_REPORT_KEY]["holds"] >= 1


@pytest.mark.parametrize("mode", ["joint_delta", "eef_delta_pos", "joint_vel"])
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
    assert store[_REPORT_KEY]["holds"] == 0


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
