"""Tests for `AbstractGatewayPolicy`, top-level `abstract_harness` exports, and `abstract eval`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import abstract_harness
from abstract_harness import (
    AbstractGatewayPolicy,
    CallablePolicy,
    Observation,
    Scene,
    Task,
    eval,
    local_policy,
    resolve,
    success_at_end,
)
from abstract_harness.cli import main as cli_main
from abstract_harness.robot.config import RobotConfig


def sample_local_vla(obs: Observation, instruction: str) -> np.ndarray:
    """Example in-process local model function for `target=test_gateway_policy:sample_local_vla`."""
    assert isinstance(instruction, str)
    pos = obs.state["eef_pos"]
    goal = obs.state["cube_pos"]
    delta = goal - pos
    dist = float(np.linalg.norm(delta))
    step = np.zeros(2, dtype=np.float64) if dist < 1e-9 else delta / dist * min(0.1, dist)
    return np.stack([step, step], axis=0)


def _obs(seed: int = 0) -> Observation:
    rng = np.random.default_rng(seed)
    images = {
        cam: rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
        for cam in ("top_cam", "left_cam", "right_cam")
    }
    state = np.array(
        [0.1, -0.2, 0.3, 0.0, 0.1, 0.0, 0.5, -0.1, 0.2, -0.3, 0.0, -0.1, 0.0, 0.8],
        dtype=np.float64,
    )
    return Observation(images=images, state={"joint_pos": state})


def _gateway_reply(rows: int, *, step: int = 0, session_id: str = "ep-1", next_frames: int = 4):
    chunk = {
        "session_id": session_id,
        "step": step,
        "dt": 1 / 30,
        "action_dim": 14,
        "action_space": "joint_position",
        "actions": [[0.05 * i] * 14 for i in range(rows)],
        "next_frames_per_camera": next_frames,
        "timing_ms": {"denoise": 14.0},
        "cache": {"mode": "action_tail", "hit_rate": 0.5},
        "context": {"latent_frames": step + 1, "window_frames": 21},
    }
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps(chunk)}}]}


def test_top_level_package_exports_core_harness_and_registers_abstract_policy():
    for name in abstract_harness.__all__:
        assert hasattr(abstract_harness, name), name
    pol = resolve("policy", "policy", api_key="sk-test")
    assert isinstance(pol, AbstractGatewayPolicy)
    assert pol.model == "abstract/dreamzero-astra:yam"


def test_abstract_gateway_policy_rolls_4_frame_window_for_dreamzero():
    calls: list[dict[str, Any]] = []

    def fake_post(url, payload, headers, timeout_s):
        assert url == "https://api.abstract.inc/v1/chat/completions"
        assert headers["authorization"] == "Bearer sk-test"
        calls.append(dict(payload))
        step = len(calls) - 1
        return _gateway_reply(24, step=step, session_id="ep-dz", next_frames=4)

    policy = AbstractGatewayPolicy(
        model="abstract/dreamzero-astra:yam",
        api_key="sk-test",
        post_fn=fake_post,
    )
    policy.reset(Scene(id="s0", instruction="fold the towel"))

    # Step 0: pass 1 frame per camera -> sends 1 text + 3 images.
    chunk0 = policy.act(_obs(0))
    assert len(chunk0) == 24 and chunk0.control_hz == pytest.approx(30.0)
    assert chunk0.meta["cache"]["mode"] == "action_tail"
    assert len(calls[0]["messages"][0]["content"]) == 1 + 3 * 1
    assert calls[0]["expected_step"] == 0
    assert calls[0]["robot_state"]["gripper_position"] == [0.5, 0.8]
    assert len(calls[0]["robot_state"]["joint_position"]) == 12

    # Continuation observes every action step, sampled across the completed chunk.
    for t in range(1, 24):
        policy.observe(_obs(t))
    chunk1 = policy.act(_obs(24))
    assert len(chunk1) == 24
    assert len(calls[1]["messages"][0]["content"]) == 1 + 3 * 4
    assert calls[1]["expected_step"] == 1
    assert calls[1]["session_id"] == "ep-dz"


def test_abstract_gateway_policy_runs_stateless_models_with_single_frame():
    calls: list[dict[str, Any]] = []

    def fake_post(url, payload, headers, timeout_s):
        calls.append(dict(payload))
        return _gateway_reply(30, step=0, session_id="stateless", next_frames=1)

    policy = AbstractGatewayPolicy(
        model="physical-intelligence/pi0.5:yam",
        api_key="sk-test",
        post_fn=fake_post,
    )
    policy.reset(Scene(id="s0", instruction="pick up the cup"))
    chunk0 = policy.act(_obs(0))
    chunk1 = policy.act(_obs(1))
    assert len(chunk0) == 30 and len(chunk1) == 30
    assert "session_id" not in calls[0] and "session_id" not in calls[1]
    assert len(calls[1]["messages"][0]["content"]) == 1 + 3 * 1


def test_yam_abstract_rig_config_loads_and_builds_policy_args():
    rig_file = Path(__file__).resolve().parents[1] / "rigs" / "yam-abstract.toml"
    cfg = RobotConfig.load(rig_file)
    assert cfg.policy_name == "policy"
    assert cfg.policy_args()["model"] == "abstract/dreamzero-astra:yam"
    assert cfg.policy_args()["server_url"] == "https://api.abstract.inc/v1"


def test_cli_eval_runs_scripted_on_cubepick(tmp_path):
    with pytest.raises(SystemExit) as exited:
        cli_main(["eval", "--no-prompt", "--log-dir", str(tmp_path / "eval")])
    assert exited.value.code == 0
    assert list((tmp_path / "eval").glob("*.json"))


def test_local_policy_adapts_to_any_embodiment_in_process_and_via_cli(tmp_path):
    # 1. Direct Python callable: automatically adopts cubepick's 3-D action space on bind().
    task = Task(
        name="local-reach",
        scenes=[Scene(id="s0", instruction="reach the cube", init_seed=0)],
        scorer=success_at_end(),
        max_steps=40,
    )
    pol = local_policy(sample_local_vla, name="inproc-vla", action_horizon=2)
    assert isinstance(pol, CallablePolicy)
    logs = eval(task, policy=pol, embodiment="cubepick", log_dir=str(tmp_path / "inproc"))
    assert logs[0].status == "success"
    assert logs[0].results.metrics["success_at_end"] == 1.0

    # 2. Resolved by string target via `policy="policy"` on the CLI.
    with pytest.raises(SystemExit) as exited:
        cli_main(
            [
                "eval",
                "--no-prompt",
                "--policy",
                "policy",
                "-P",
                "target=test_gateway_policy:sample_local_vla",
                "--log-dir",
                str(tmp_path / "cli-local"),
            ]
        )
    assert exited.value.code == 0


def test_gateway_policy_supports_keyless_local_servers_and_custom_action_dims(monkeypatch):
    monkeypatch.delenv("ABSTRACT_API_KEY", raising=False)
    calls: list[dict[str, Any]] = []

    def fake_local_post(url, payload, headers, timeout_s):
        assert url == "http://127.0.0.1:8000/v1/chat/completions"
        assert "authorization" not in headers
        calls.append(dict(payload))
        chunk = {
            "session_id": "franka-ep",
            "step": 0,
            "dt": 1 / 15,
            "action_dim": 8,
            "action_space": "joint_position",
            "actions": [[0.1] * 8 for _ in range(16)],
            "next_frames_per_camera": 1,
        }
        return {"choices": [{"message": {"role": "assistant", "content": json.dumps(chunk)}}]}

    franka_policy = AbstractGatewayPolicy(
        model="local/openvla:franka",
        server_url="http://127.0.0.1:8000/v1",
        camera_order=("exterior", "wrist"),
        cam_height=180,
        cam_width=320,
        action_dim=8,
        gripper_indices=(7,),
        action_horizon=16,
        stateful=False,
        post_fn=fake_local_post,
    )
    franka_policy.reset(Scene(id="s0", instruction="pick up the banana"))
    obs = Observation(
        images={
            "exterior": np.zeros((180, 320, 3), dtype=np.uint8),
            "wrist": np.zeros((180, 320, 3), dtype=np.uint8),
        },
        state={"joint_pos": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])},
    )
    chunk = franka_policy.act(obs)
    assert len(chunk) == 16 and chunk.actions[0].data.shape == (8,)
    assert calls[0]["robot_state"] == {
        "joint_position": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "gripper_position": [1.0],
    }


def test_custom_non_yam_robot_runs_doctor_probe_and_episode_through_robot_config(tmp_path):
    from abstract_harness import (
        Action,
        CameraSpec,
        EmbodimentBase,
        StepResult,
        embodiment,
        joint_embodiment_info,
    )
    from abstract_harness.robot import cli as robot_cli
    from abstract_harness.robot import stack

    @embodiment("franka_test_arm")
    class FrankaTestArm(EmbodimentBase):
        def __init__(self, host: str = "127.0.0.1") -> None:
            self.host = host
            self.closed = False
            self.info = joint_embodiment_info(
                "franka_test_arm",
                low=[-2.8] * 7 + [0.0],
                high=[2.8] * 7 + [1.0],
                dim_labels=(*(f"panda_joint{i}" for i in range(1, 8)), "panda_gripper"),
                cameras=(CameraSpec(name="wrist", height=180, width=320),),
                control_hz=15.0,
            )

        def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
            del scene, seed
            return Observation(
                images={"wrist": np.zeros((180, 320, 3), dtype=np.uint8)},
                state={"joint_pos": np.zeros(8, dtype=np.float64)},
            )

        def step(self, action: Action) -> StepResult:
            obs = Observation(
                images={"wrist": np.zeros((180, 320, 3), dtype=np.uint8)},
                state={"joint_pos": np.asarray(action.data, dtype=np.float64)},
            )
            return StepResult(observation=obs, terminated=True)

        def close(self) -> None:
            self.closed = True

    toml_path = tmp_path / "franka.toml"
    toml_path.write_text(
        "[policy]\n"
        "[policy.args]\n"
        'target = "test_gateway_policy:franka_local_vla"\n'
        "[rig]\n"
        'name = "franka-1"\n'
        'type = "franka_test_arm"\n'
        "max_steps = 10\n"
        "[rig.embodiment]\n"
        'host = "192.0.2.10"\n'
    )
    cfg = RobotConfig.load(toml_path)
    assert cfg.rig.type == "franka_test_arm"
    assert cfg.setup()["embodiment"] == "franka_test_arm"

    # 1. Conformance doctor passes on the non-YAM embodiment.
    code, _ = stack.captured(stack.doctor, cfg, "franka.toml")
    assert code == 0

    # 2. policy_probe + policy_probe_result validate the 8-DoF chunk against franka_test_arm.
    probe = stack.policy_probe(cfg)
    ok, detail, _ = robot_cli.policy_probe_result(probe)
    assert ok, detail
    assert "4x8 chunk" in detail

    # 3. Full episode runs through stack.episode on franka_test_arm.
    ep_code = stack.episode(
        cfg,
        "pick up the bolt",
        tmp_path / "run-logs",
        source="franka.toml",
        prompt=False,
        store_frames=False,
        rerun_save=False,
        grade=False,
    )
    assert ep_code == 0


def franka_local_vla(obs: Observation, instruction: str) -> np.ndarray:
    """8-DoF local policy for `franka_test_arm`."""
    assert "wrist" in obs.images
    return np.full((4, 8), 0.05, dtype=np.float64)
