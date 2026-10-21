"""Locomotion primitives: the base_vel control mode through guardrails, controllers, the
agent's drive tool and episode labels."""

from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np
import pytest

from abstract_harness.robot.agent._llm import ToolCall
from abstract_harness.robot.agent._tools import ToolsetError, build_toolset
from abstract_harness.robot.core.approver import (
    AccelerationLimitApprover,
    ChainApprover,
    ClampApprover,
    DeltaLimitApprover,
)
from abstract_harness.robot.core.conformance import check_embodiment
from abstract_harness.robot.core.controller import AsyncController, EnsemblingController
from abstract_harness.robot.core.embodiment import EmbodimentInfo
from abstract_harness.robot.core.errors import SafetyAbort
from abstract_harness.robot.core.policy import PolicyConfig, PolicyInfo
from abstract_harness.robot.core.spaces import (
    CANONICAL_STATE_UNITS,
    RATE_CONTROL_MODES,
    ActionSemantics,
    Box,
    ObservationSpace,
    StateField,
    StateSpec,
)
from abstract_harness.robot.core.types import Action, ActionChunk, Observation

LABELS = ("vx", "vy", "yaw_rate")


def base_box(**semantics: Any) -> Box:
    return Box(
        shape=(3,),
        low=np.array([-0.5, -0.3, -1.0]),
        high=np.array([0.8, 0.3, 1.0]),
        semantics=ActionSemantics("base_vel", dim_labels=LABELS, **semantics),
    )


def base_obs() -> ObservationSpace:
    return ObservationSpace(
        state=StateSpec(
            (
                StateField("base_lin_vel", (3,), "m/s"),
                StateField("base_ang_vel", (3,), "rad/s"),
                StateField("projected_gravity", (3,), "unit_vec"),
            )
        )
    )


def test_base_vel_is_a_rate_mode_with_canonical_locomotion_state() -> None:
    assert "base_vel" in RATE_CONTROL_MODES
    for key in ("base_lin_vel", "base_ang_vel", "projected_gravity", "base_quat", "base_pose"):
        assert key in CANONICAL_STATE_UNITS
    with pytest.raises(ValueError, match="max_step"):
        ActionSemantics("base_vel", max_step=(0.1, None, None))


def test_a_base_velocity_embodiment_is_conformant() -> None:
    info = EmbodimentInfo(
        name="base", action_space=base_box(), observation_space=base_obs(), control_hz=50.0
    )
    report = check_embodiment(info)
    assert report.ok, report.summary()


def test_the_default_guardrails_clamp_velocities_to_the_box() -> None:
    space = base_box()
    limiter = DeltaLimitApprover(space)
    chain = ChainApprover(ClampApprover(space), limiter)
    approved = chain.review(Action(data=np.array([2.0, -1.0, 0.5])), {})
    np.testing.assert_allclose(approved.data, [0.8, -0.3, 0.5])


def test_acceleration_limit_ramps_from_rest_each_trial() -> None:
    space = base_box()
    approver = AccelerationLimitApprover(space, max_change=[0.1, 0.1, 0.2])
    store: dict[str, Any] = {}
    approver.begin_trial(None, store)
    seen = [approver.review(Action(data=np.array([0.8, 0.0, -1.0])), store) for _ in range(3)]
    np.testing.assert_allclose(
        [a.data for a in seen], [[0.1, 0, -0.2], [0.2, 0, -0.4], [0.3, 0, -0.6]]
    )
    assert seen[0].meta["accel_clamped"] is True
    # Stopping ramps down too: the watchdog, not the approver, halts at once.
    stop = approver.review(Action(data=np.zeros(3)), store)
    np.testing.assert_allclose(stop.data, [0.2, 0, -0.4])
    fresh: dict[str, Any] = {}
    approver.begin_trial(np.ones(3), fresh)
    unchanged = Action(data=np.array([0.05, 0.0, 0.1]))
    assert approver.review(unchanged, fresh) is unchanged  # identity when within limits


def test_acceleration_limit_refuses_non_rate_spaces_and_nan() -> None:
    joints = Box(
        shape=(1,), low=-np.ones(1), high=np.ones(1), semantics=ActionSemantics("joint_pos")
    )
    with pytest.raises(ValueError, match="rate control modes"):
        AccelerationLimitApprover(joints, 0.1)
    with pytest.raises(ValueError, match="max_delta"):
        AccelerationLimitApprover(base_box(), 0.0)
    approver = AccelerationLimitApprover(base_box(), 0.1)
    with pytest.raises(SafetyAbort, match="NaN"):
        approver.review(Action(data=np.array([np.nan, 0, 0])), {})


def test_ensembling_averages_base_velocities() -> None:
    EnsemblingController(base_box())


def test_async_underrun_stops_a_base() -> None:
    class Base:
        info = PolicyInfo(name="walker", action_space=base_box())
        config = PolicyConfig()
        gate: threading.Event | None = None

        def reset(self, scene: Any) -> None:
            return None

        def act(self, observation: Observation) -> ActionChunk:
            if self.gate is not None:
                self.gate.wait(5)
            return ActionChunk(actions=[Action(data=np.array([0.5, 0.0, 0.2]))])

    policy = Base()
    controller = AsyncController(prefetch=0)
    store: dict[str, Any] = {}
    obs = Observation(extra={"env_step": 0})
    controller.next_action(policy, obs, 0, store)  # type: ignore[arg-type]
    policy.gate = threading.Event()
    hold = controller.next_action(policy, obs, 1, store)  # type: ignore[arg-type]
    np.testing.assert_array_equal(hold.data, np.zeros(3))
    policy.gate.set()


def _drive(toolset: Any, observation: Observation, **arguments: Any) -> Any:
    return toolset.execute(ToolCall("c1", "drive", json.dumps(arguments)), observation)


def test_the_agent_drives_a_base_for_a_duration_then_stops() -> None:
    toolset = build_toolset(base_box(), base_obs(), control_hz=10.0)
    (move, *_rest) = toolset.schemas()
    function = move["function"]
    assert function["name"] == "drive"
    assert function["parameters"]["required"] == ["velocities", "duration_s", "note"]
    result = _drive(
        toolset, Observation(), velocities={"vx": 0.4}, duration_s=1.5, note="corridor clear"
    )
    assert result.error is None
    data = np.array([a.data for a in result.chunk.actions])
    assert len(data) == 16  # 15 steps at 10 Hz, then rest
    np.testing.assert_allclose(data[:15], [[0.4, 0.0, 0.0]] * 15)
    np.testing.assert_array_equal(data[-1], np.zeros(3))
    assert result.chunk.actions[-1].meta["chunk_final"] is True
    assert "executing drive over 15 steps (1.5s)" == result.note


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"velocities": {"vx": 2.0}, "duration_s": 1}, "outside"),
        ({"velocities": {"vz": 0.1}, "duration_s": 1}, "unknown dimension"),
        ({"velocities": {"vx": 0.1}, "duration_s": 0}, "finite and > 0"),
        ({"velocities": {"vx": 0.1}, "duration_s": "long"}, "number of seconds"),
        ({"velocities": {"vx": 0.1}, "duration_s": 60}, "playout cap"),
        ({"velocities": {}, "duration_s": 1}, "non-empty"),
    ],
)
def test_drive_errors_are_correctable(arguments: dict[str, Any], message: str) -> None:
    toolset = build_toolset(base_box(), base_obs(), control_hz=10.0)
    result = _drive(toolset, Observation(), note="checking", **arguments)
    assert result.chunk is None and message in (result.error or "")


def test_drive_refuses_pre_checks_and_bounds_without_zero() -> None:
    with pytest.raises(ToolsetError, match="velocity control modes"):
        build_toolset(base_box(), base_obs(), control_hz=10.0, pre_check=lambda w: None)
    forward_only = Box(
        shape=(3,),
        low=np.array([0.1, -0.3, -1.0]),
        high=np.array([0.8, 0.3, 1.0]),
        semantics=ActionSemantics("base_vel", dim_labels=LABELS),
    )
    with pytest.raises(ToolsetError, match="velocity control bounds must contain zero"):
        build_toolset(forward_only, base_obs(), control_hz=10.0)


def test_joint_velocity_stays_unsupported_by_the_agent() -> None:
    joint_vel = Box(
        shape=(2,), low=-np.ones(2), high=np.ones(2), semantics=ActionSemantics("joint_vel")
    )
    with pytest.raises(ToolsetError, match="joint_vel"):
        build_toolset(joint_vel, base_obs(), control_hz=10.0)


def test_episode_motion_integrates_base_velocities(tmp_path: Any) -> None:
    from abstract_harness.robot import episode

    path = tmp_path / "actions.jsonl"
    rows = [{"kind": "header", "labels": list(LABELS)}]
    rows += [{"t": t, "action": [0.5, 0.0, 0.0]} for t in range(10)]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    spaces = {"action": {"control_mode": "base_vel", "labels": list(LABELS)}}
    labels, _ = episode.motion(path, spaces, 10.0)
    no_rate, _ = episode.motion(path, spaces)
    assert labels != no_rate
