"""Regression cases from the public-release audit. All actuators and transports are fake."""

from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from abstract_harness import Action, ActionChunk, Observation, Scene, Task, eval, success_at_end
from abstract_harness.robot import cloud, formats, store
from abstract_harness.robot.cli import overrides, write_setup
from abstract_harness.robot.config import RobotConfig
from abstract_harness.robot.core.approver import GuardrailContribution
from abstract_harness.robot.core.compat import check_compatibility
from abstract_harness.robot.core.controller import DefaultController, EnsemblingController
from abstract_harness.robot.core.embodiment import EmbodimentInfo
from abstract_harness.robot.core.errors import ConfigError, PolicyError, SafetyAbort
from abstract_harness.robot.core.policy import PolicyConfig, PolicyInfo
from abstract_harness.robot.core.registry import resolve
from abstract_harness.robot.core.spaces import ActionSemantics, Box, ObservationSpace
from abstract_harness.robot.core.supervision import InferenceSupervisor
from abstract_harness.robot.core.types import StepResult
from abstract_harness.robot.vla import CallableVLAPolicy, GatewayVLAPolicy


@pytest.mark.parametrize("mode", ["joint_delta", "eef_abs_pose"])
def test_gateway_binding_cannot_relabel_fixed_model_actions(mode):
    policy = GatewayVLAPolicy()
    original = policy.info.action_space
    changed = replace(original, semantics=replace(original.semantics, control_mode=mode))
    info = EmbodimentInfo("fake", changed, policy.info.observation_space)
    policy.bind(info)
    assert policy.info.action_space is original
    assert "control_mode" in {
        i.code for i in check_compatibility(policy, SimpleNamespace(info=info)).errors
    }


def test_callable_explicit_semantics_and_terminal_metadata_survive():
    policy = CallableVLAPolicy(
        fn=lambda obs: Action(np.zeros(14), {"request_stop": True, "stop_reason": "done"}),
        control_mode="joint_delta",
    )
    fixed = GatewayVLAPolicy().info
    policy.bind(EmbodimentInfo("fake", fixed.action_space, fixed.observation_space))
    assert policy.info.action_space.semantics.control_mode == "joint_delta"
    result = policy.act(Observation())
    assert result.actions[0].meta == {"request_stop": True, "stop_reason": "done"}


@pytest.mark.parametrize("key", ["api_key", "access_token", "password", "client_secret", "apiKey"])
def test_credential_literals_are_rejected_without_echoing_values(key):
    secret = "synthetic-never-persist-this-value"
    with pytest.raises(ValidationError) as error:
        RobotConfig.model_validate({"vla": {"args": {key: secret}}, "rig": {"type": "cubepick"}})
    # CLI ValidationError handling renders only location/message, never Pydantic inputs.
    assert secret not in error.value.errors()[0]["msg"]
    with pytest.raises(ValueError) as error:
        overrides([f"{key}={secret}"], "-P")
    assert secret not in str(error.value)


def test_snapshot_redacts_defensively_and_old_artifacts_cannot_upload(tmp_path):
    config = RobotConfig.model_validate({"vla": {}, "rig": {"type": "cubepick"}})
    run = tmp_path / "20261016-120000"
    run.mkdir()
    path = write_setup(config, run / "setup.json", policy_overrides={"api_key": "synthetic-secret"})
    assert "synthetic-secret" not in path.read_text()
    path.write_text(json.dumps({"policy_args": {"api_key": "old-secret"}}))
    with pytest.raises(cloud.UploadError, match="credential fields"):
        cloud.prepare_run(run, "fake")


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000/v1", "https://workstation.example/v1"])
def test_endpoint_switch_does_not_forward_abstract_credential(monkeypatch, url):
    monkeypatch.setenv("ABSTRACT_API_KEY", "synthetic-cloud-secret")
    assert GatewayVLAPolicy(server_url=url)._resolve_api_key() == ""
    with pytest.raises(ConfigError, match="scoped"):
        GatewayVLAPolicy(server_url=url, api_key_env="ABSTRACT_API_KEY")
    monkeypatch.setenv("LOCAL_POLICY_KEY", "synthetic-local-key")
    assert (
        GatewayVLAPolicy(server_url=url, api_key_env="LOCAL_POLICY_KEY")._resolve_api_key()
        == "synthetic-local-key"
    )


@pytest.mark.parametrize(
    "url",
    ["http://remote.example/v1", "https://user:secret@host/v1", "https://host/v1?token=secret"],
)
def test_python_gateway_rejects_unsafe_urls_without_echoing_credentials(url):
    with pytest.raises(ValueError) as error:
        GatewayVLAPolicy(server_url=url)
    assert "secret" not in str(error.value)


def test_every_shipped_rig_has_compatible_actual_factories(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unused-key")
    for path in (Path(__file__).parents[1] / "rigs").glob("*.toml"):
        config = RobotConfig.load(path)
        policy = resolve(
            "policy",
            config.policy_name,
            **config.policy_args("http://127.0.0.1:1/v1" if config.agent else None),
        )
        embodiment = resolve("embodiment", config.rig.type, **config.embodiment_args())
        try:
            bind = getattr(policy, "bind", None)
            if callable(bind):
                bind(embodiment.info)
            assert check_compatibility(policy, embodiment).ok, path.name
        finally:
            embodiment.close()


def _reply(step=0, **overrides):
    chunk = {
        "actions": [[0.0] * 14] * 24,
        "action_dim": 14,
        "action_space": "joint_position",
        "step": step,
        "session_id": "synthetic-session",
        "next_frames_per_camera": 4,
        "dt": 1 / 30,
    }
    chunk.update(overrides)
    return {"choices": [{"message": {"content": json.dumps(chunk)}}]}


def _observation(index):
    return Observation(
        images={
            camera: np.full((1, 1, 3), index, np.uint8)
            for camera in ("top_cam", "left_cam", "right_cam")
        },
        state={"joint_pos": np.zeros(14)},
    )


def test_stateful_history_follows_control_observations_across_buffering(monkeypatch):
    calls = []

    def post(url, payload, headers, timeout):
        calls.append(payload)
        return _reply(len(calls) - 1)

    monkeypatch.setattr(
        "abstract_harness.robot.vla._encode_jpeg_data_url", lambda frame: f"frame-{frame[0, 0, 0]}"
    )
    policy = GatewayVLAPolicy(api_key="synthetic-key", cam_height=1, cam_width=1, post_fn=post)
    policy.reset(Scene("s", "hold"))
    controller, state = DefaultController(), {}
    for t in range(25):
        controller.next_action(policy, _observation(t), t, state)
    content = calls[1]["messages"][0]["content"]
    assert [part["image_url"]["url"] for part in content[1:5]] == [
        "frame-1",
        "frame-8",
        "frame-16",
        "frame-24",
    ]
    assert calls[1]["expected_step"] == 1


def test_stateful_early_replan_is_refused_instead_of_fabricating_frames():
    policy = GatewayVLAPolicy(
        api_key="synthetic-key", cam_height=1, cam_width=1, post_fn=lambda *a: _reply()
    )
    policy.reset(Scene("s", "hold"))
    policy.act(_observation(0))
    with pytest.raises(PolicyError, match="fresh control observations"):
        policy.act(_observation(1))


@pytest.mark.parametrize(
    "bad",
    [
        {"step": True},
        {"step": -1},
        {"step": 12},
        {"next_frames_per_camera": 0},
        {"next_frames_per_camera": 2},
        {"next_frames_per_camera": 9999},
        {"dt": float("nan")},
        {"dt": 0},
        {"action_dim": 8},
        {"action_space": "eef_position"},
        {"actions": []},
    ],
)
def test_invalid_gateway_chunk_never_advances_session(bad):
    policy = GatewayVLAPolicy(
        api_key="synthetic-key", cam_height=1, cam_width=1, post_fn=lambda *a: _reply(**bad)
    )
    policy.reset(Scene("s", "hold"))
    with pytest.raises((ValueError, ConfigError, PolicyError)):
        policy.act(_observation(0))
    assert policy.step == 0 and policy.num_inferences == 0


class _Store:
    def upload_run(self, run):
        cloud.prepare_run(run, "fake")
        return {"run": run.name, "bytes": 0}


def _recording_run(root, verified):
    run = root / "20261016-120000"
    (run / "frames").mkdir(parents=True)
    (run / "frames" / "camera.npy").write_bytes(b"irreplaceable-raw-frame")
    recording = run / "episode.rrd"
    recording.write_bytes(b"RRF2-synthetic-recording")
    if verified:
        recording.with_suffix(".complete.json").write_text(
            json.dumps(
                {
                    "file": recording.name,
                    "bytes": recording.stat().st_size,
                    "sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
                    "frames": {
                        "frames/camera.npy": hashlib.sha256(
                            (run / "frames" / "camera.npy").read_bytes()
                        ).hexdigest()
                    },
                }
            )
        )
    (run / "run-summary.json").write_text(json.dumps({"formats": formats.write(run, ["rrd"])}))
    return run


def test_unverified_recording_retains_raw_frames_after_successful_upload(tmp_path):
    run = _recording_run(tmp_path, verified=False)
    result = store.offload(_Store(), tmp_path, run)
    assert result["local_retained"] is True
    assert (run / "frames" / "camera.npy").read_bytes() == b"irreplaceable-raw-frame"
    assert not cloud.finished_runs(tmp_path)  # no repeated automatic uploads of retained runs


def test_verified_recording_can_be_offloaded_after_completion(tmp_path):
    run = _recording_run(tmp_path, verified=True)
    store.offload(_Store(), tmp_path, run)
    assert not run.exists()


def test_modified_export_cannot_authorize_deletion_using_stale_summary(tmp_path):
    run = _recording_run(tmp_path, verified=True)
    (run / "episode.rrd").write_bytes(b"interrupted replacement")
    assert store.offload(_Store(), tmp_path, run)["local_retained"]
    assert (run / "frames" / "camera.npy").exists()


def test_removed_only_format_result_cannot_authorize_deletion(tmp_path):
    run = _recording_run(tmp_path, verified=False)
    (run / "run-summary.json").write_text(
        json.dumps({"formats": {"rrd": {"ok": True, "removed": ["x"]}}})
    )
    assert store.offload(_Store(), tmp_path, run)["local_retained"]


def test_failed_export_retains_only_copy_of_camera_data(tmp_path):
    run = _recording_run(tmp_path, verified=False)
    (run / "episode.rrd").unlink()
    (run / "run-summary.json").write_text(json.dumps({"formats": formats.write(run, ["rrd"])}))
    store.offload(_Store(), tmp_path, run)
    assert (run / "frames" / "camera.npy").exists()


class _Arm:
    def __init__(self):
        self.executed = []
        self.contributions = 0
        self.closed = False
        box = Box((1,), -np.ones(1), np.ones(1), ActionSemantics("joint_pos"))
        self.info = EmbodimentInfo(
            "fake-arm", box, ObservationSpace(state_keys=frozenset({"joint_pos"}))
        )

    def reset(self, scene, *, seed=None):
        return Observation(state={"joint_pos": np.zeros(1)})

    def step(self, action):
        self.executed.append(action.data.copy())
        return StepResult(Observation(state={"joint_pos": action.data.copy()}), terminated=True)

    def contribute_guardrails(self, space):
        self.contributions += 1
        return GuardrailContribution()

    def close(self):
        self.closed = True


def _task():
    return Task("fake", [Scene("s", "hold")], success_at_end(), max_steps=1)


def test_python_eval_uses_shared_guardrails_and_measured_first_action(tmp_path):
    arm = _Arm()
    policy = CallableVLAPolicy(lambda obs: np.array([100.0]))
    eval(_task(), policy, arm, log_dir=str(tmp_path))
    assert arm.contributions == 1
    assert arm.executed[0] == pytest.approx([0.1])
    action_log = next((tmp_path / "actions").rglob("*.jsonl"))
    row = json.loads(action_log.read_text().splitlines()[1])
    assert row["requested_action"] == [100.0] and row["action"] == pytest.approx([0.1])


def test_absolute_limits_reanchor_to_measured_state_when_actuator_does_not_follow(tmp_path):
    class StalledArm(_Arm):
        def step(self, action):
            self.executed.append(action.data.copy())
            return StepResult(Observation(state={"joint_pos": np.zeros(1)}))

    arm = StalledArm()
    task = replace(_task(), max_steps=3)
    eval(task, CallableVLAPolicy(lambda obs: np.array([100.0])), arm, log_dir=str(tmp_path))
    assert np.asarray(arm.executed) == pytest.approx(np.full((3, 1), 0.1))


def test_trial_records_snapshot_reused_state_and_mutated_action_buffers(tmp_path):
    class ReusingArm(_Arm):
        def reset(self, scene, *, seed=None):
            self.buffer = np.zeros(1)
            return Observation(state={"joint_pos": self.buffer})

        def step(self, action):
            action.data[:] = 0.7
            self.buffer[:] = 0.7
            return StepResult(Observation(state={"joint_pos": self.buffer}), terminated=True)

    eval(
        _task(),
        CallableVLAPolicy(lambda obs: np.array([100.0])),
        ReusingArm(),
        log_dir=str(tmp_path),
    )
    row = json.loads(next((tmp_path / "actions").rglob("*.jsonl")).read_text().splitlines()[1])
    assert row["state"]["joint_pos"] == [0.0]
    assert row["action"] == pytest.approx([0.1])


@pytest.mark.parametrize("rate", [0, -1, float("inf"), float("nan"), True])
def test_action_chunk_refuses_invalid_playback_rate(rate):
    with pytest.raises(ValueError, match="control_hz"):
        ActionChunk([Action(np.zeros(1))], control_hz=rate)


def test_real_file_recording_emits_verifiable_completion_receipt(tmp_path):
    pytest.importorskip("rerun", reason="file recording requires the robot extra")
    from abstract_harness.robot.core.logging.rerun_sink import RerunSink

    sink = RerunSink(recording_dir=str(tmp_path))
    eval(
        _task(),
        policy="scripted",
        embodiment="cubepick",
        log_dir=str(tmp_path),
        sinks=[sink],
        store_frames=True,
    )
    assert sink.recording_complete
    results = formats.write(tmp_path, ["rrd"])
    assert results["rrd"]["verified"]
    assert formats.verified_exports(tmp_path, results)


def test_arm_initialization_timeout_closes_late_survivor_before_releasing_claim():
    from abstract_harness.robot.yam._i2rt import start_arms_concurrently

    release, closed, unclaimed = threading.Event(), threading.Event(), threading.Event()
    order = []

    def late():
        release.wait(2)

        def close():
            order.append("closed")
            closed.set()

        return SimpleNamespace(close=close)

    def unclaim():
        order.append("released")
        unclaimed.set()

    try:
        with pytest.raises(TimeoutError):
            start_arms_concurrently(
                late,
                lambda: SimpleNamespace(close=lambda: None),
                timeout_s=0.01,
                on_failed_cleanup=unclaim,
            )
        assert not unclaimed.is_set()
        release.set()
        assert closed.wait(2) and unclaimed.wait(2)
        assert order == ["closed", "released"]
    finally:
        release.set()


def test_native_arm_factory_refuses_unarmed_failsafe_before_any_constructor(monkeypatch):
    from abstract_harness.robot.core.errors import EmbodimentFault
    from abstract_harness.robot.yam import embodiment
    from abstract_harness.robot.yam.config import YamConfig

    released, reads, built = [], [], []
    monkeypatch.setattr(embodiment, "_load_i2rt", lambda: (lambda **kw: built.append(kw), object()))
    monkeypatch.setattr(
        "abstract_harness.robot.core._claims.claim_devices",
        lambda *a: SimpleNamespace(release=lambda: released.append(True)),
    )

    def timeouts(channel):
        reads.append(channel)
        return {str(i): 0 for i in range(1, 8)}

    monkeypatch.setattr("abstract_harness.robot.stack.motor_timeouts", timeouts)
    with pytest.raises(EmbodimentFault, match="not armed"):
        embodiment._default_driver_factory(YamConfig(left_channel="can7", right_channel="can8"))
    assert reads == ["can7"] and released == [True] and not built


def test_rpc_shutdown_deadline_reports_unverified_physical_state():
    from abstract_harness.robot.rpc import Controller

    controller = object.__new__(Controller)
    controller._lock = threading.Lock()
    controller.phase = "running"
    requested = []
    controller._terminal = SimpleNamespace(write=lambda value: requested.append(value))
    controller.preview = SimpleNamespace(close=lambda: None)
    controller.uploads = SimpleNamespace(close=lambda timeout: [])
    with pytest.raises(TimeoutError, match="physical state is unverified"):
        controller.shutdown(timeout=0, upload_timeout=0)
    assert requested


def test_timed_out_policy_cannot_dispatch_late_action_or_collect_live_transcript(tmp_path):
    release, returned = threading.Event(), threading.Event()
    arm = _Arm()

    def slow(obs):
        release.wait(5)
        returned.set()
        return np.array([0.5])

    policy = CallableVLAPolicy(slow)
    try:
        logs = eval(_task(), policy, arm, log_dir=str(tmp_path), policy_timeout_s=0.02)
        assert logs[0].status == "error" and not arm.executed
        release.set()
        assert returned.wait(2)
        assert not arm.executed
    finally:
        release.set()
        arm.close()


def test_supervisor_checks_health_during_blocked_inference():
    release = threading.Event()

    def fault():
        raise SafetyAbort("synthetic health fault")

    try:
        with pytest.raises(SafetyAbort, match="health fault"):
            InferenceSupervisor(1).call(lambda: release.wait(2), fault)
    finally:
        release.set()


def test_ensembling_preserves_terminal_intent():
    arm = _Arm()
    policy = SimpleNamespace(
        info=PolicyInfo("stop", arm.info.action_space),
        config=PolicyConfig(),
        act=lambda obs: ActionChunk(
            [Action(np.zeros(1), {"request_stop": True, "stop_reason": "done"})]
        ),
    )
    action = EnsemblingController(arm.info.action_space).next_action(policy, Observation(), 0, {})
    assert action.meta["request_stop"] and action.meta["stop_reason"] == "done"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX termination semantics")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_cli_termination_closes_its_fake_embodiment(tmp_path, signum):
    script = """
import sys, time
from pathlib import Path
from abstract_harness.robot.core.mock.cubepick import CubePickEmbodiment
from abstract_harness.robot.core.registry import embodiment
from abstract_harness.cli import main
root = Path(sys.argv[1])
@embodiment("signal-arm")
class Arm(CubePickEmbodiment):
    def step(self, action):
        (root / "ready").touch()
        time.sleep(10)
        return super().step(action)
    def close(self):
        (root / "closed").touch()
main(["eval", "--policy", "noop", "--embodiment", "signal-arm",
      "--instruction", "hold", "--no-prompt", "--log-dir", str(root / "logs")])
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "ready").exists() and child.poll() is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert (tmp_path / "ready").exists()
        child.send_signal(signum)
        assert child.wait(timeout=5) == 130
        assert (tmp_path / "closed").exists()
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)
