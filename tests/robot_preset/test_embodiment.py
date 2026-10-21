"""The preset embodiment at its boundaries: the driver lifecycle, readiness gates, the
home ramp, pacing, the watchdog, fall detection, guardrails and full runs."""

from __future__ import annotations

import json
import shutil
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from abstract_harness.robot.config import RobotConfig
from abstract_harness.robot.core import eval
from abstract_harness.robot.core.command_stream import CommandStream
from abstract_harness.robot.core.errors import ConfigError, EmbodimentFault
from abstract_harness.robot.core.registry import resolve
from abstract_harness.robot.core.safety import build_guardrails
from abstract_harness.robot.core.scene import Scene
from abstract_harness.robot.core.scorer import success_at_end
from abstract_harness.robot.core.task import Task
from abstract_harness.robot.core.types import Action
from abstract_harness.robot.preset import FALL, PresetEmbodiment, SimDriver

ROBOTS = Path(__file__).parents[2] / "robots"
ARM = ROBOTS / "mobile-manipulator" / "arm.toml"
BASE = ROBOTS / "mobile-manipulator" / "base.toml"
SCENE = Scene(id="s", instruction="go", init_seed=0)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_the_sim_arm_homes_steps_and_reports_commands() -> None:
    clock = Clock()
    arm = PresetEmbodiment(str(ARM), clock=clock, sleep=clock.sleep)
    assert arm.simulated and arm.info.is_simulated and arm.info.control_hz == 30.0
    observation = arm.reset(SCENE)
    # The home ramp took at least home_secs at the policy rate, no faster than max_step.
    assert sum(clock.sleeps) == pytest.approx(2.0, abs=0.05)
    np.testing.assert_allclose(observation.state["joint_pos"][1:3], [0.4, -0.8])
    assert observation.state["joint_pos"][6] == pytest.approx(0.04)
    assert set(observation.images) == {"head_cam", "wrist_cam"}
    target = observation.state["joint_pos"].copy()
    target[0] = 0.02
    result = arm.step(Action(data=target))
    np.testing.assert_allclose(result.command["joint_pos"], target)
    assert result.observation.state["joint_pos"][0] == pytest.approx(0.02)
    assert not result.terminated
    arm.close()


def test_pacing_holds_steps_to_the_policy_rate() -> None:
    clock = Clock()
    base = PresetEmbodiment(str(BASE), clock=clock, sleep=clock.sleep)
    base.reset(SCENE)
    clock.sleeps.clear()
    for _ in range(3):
        base.step(Action(data=np.zeros(2)))
    assert clock.sleeps == pytest.approx([0.02, 0.02])
    unpaced = PresetEmbodiment(str(BASE), pace="false", clock=clock, sleep=clock.sleep)
    assert "self_paced" not in unpaced.info.capabilities
    base.close()


def test_the_sim_base_drives_and_its_watchdog_stops_it() -> None:
    base = PresetEmbodiment(str(BASE), pace=False)
    base.reset(SCENE)
    driver = base._driver
    assert isinstance(driver, SimDriver)
    for _ in range(10):
        result = base.step(Action(data=np.array([0.5, 0.0])))
    assert result.observation.state["base_pose"][0] == pytest.approx(0.1)
    assert result.command == {"base_vel": pytest.approx(np.array([0.5, 0.0]))}
    stops = driver.stops
    deadline = time.monotonic() + 3
    while base.watchdog_stops == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert base.watchdog_stops == 1 and driver.stops == stops + 1
    np.testing.assert_array_equal(driver.base_velocity, np.zeros(3))
    base.step(Action(data=np.array([0.2, 0.0])))  # the next step resumes motion
    np.testing.assert_allclose(driver.base_velocity, [0.2, 0.0, 0.0])
    base.close()
    assert driver.closed


def test_a_fall_stops_the_driver_and_ends_the_episode() -> None:
    base = PresetEmbodiment(str(BASE), pace=False)
    base.reset(SCENE)
    driver = base._driver
    driver.tilt(20)
    assert not base.step(Action(data=np.zeros(2))).terminated
    driver.tilt(40)
    stops = driver.stops
    result = base.step(Action(data=np.array([0.3, 0.0])))
    assert result.terminated and result.termination_reason == FALL
    assert result.info["tilt_deg"] == pytest.approx(40.0)
    assert driver.stops == stops + 1
    base.close()


def test_base_guardrails_include_the_acceleration_limit() -> None:
    base = PresetEmbodiment(str(BASE))
    approver, active, _ = build_guardrails(base.info.action_space, None, base)
    assert active == ["clamp", "delta-limit", "accel-limit"]
    store: dict[str, Any] = {}
    approver.begin_trial(None, store)
    first = approver.review(Action(data=np.array([1.0, 1.5])), store)
    # 1.0 m/s^2 and 3.0 rad/s^2 at 50 Hz: 0.02 and 0.06 per step from rest.
    np.testing.assert_allclose(first.data, [0.02, 0.06])
    arm = PresetEmbodiment(str(ARM))
    assert arm.contribute_guardrails(arm.info.action_space).approvers == ()


class FakeRobot:
    """A physical driver stand-in that records its calls."""

    calls: list[str] = []

    def __init__(self, description: Any, *, port: str = "/dev/ttyUSB0") -> None:
        self.description = description
        self.port = port
        self.position = np.zeros(description.dim)
        FakeRobot.calls = ["init"]

    def connect(self) -> None:
        FakeRobot.calls.append("connect")

    def read(self) -> dict[str, Any]:
        return {"joint_pos": self.position.copy(), "joint_vel": np.zeros(self.description.dim)}

    def images(self) -> dict[str, Any]:
        return {
            c.name: np.zeros((c.height, c.width, c.channels), np.uint8)
            for c in self.description.cameras
        }

    def command(self, values: Any) -> None:
        self.position = np.asarray(values, float).copy()
        FakeRobot.calls.append("command")

    def stop(self) -> None:
        FakeRobot.calls.append("stop")

    def close(self) -> None:
        FakeRobot.calls.append("close")


@pytest.fixture
def physical_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The example arm preset with a fake physical driver in an importable module."""
    shutil.copy(ARM.parent / "mobile_manipulator.urdf", tmp_path)
    preset = ARM.read_text().replace('driver = "sim"', 'driver = "fake_robot_driver:FakeRobot"')
    (tmp_path / "arm.toml").write_text(preset)
    FakeRobot.calls = []
    module = type(sys)("fake_robot_driver")
    module.FakeRobot = FakeRobot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_robot_driver", module)
    return tmp_path / "arm.toml"


class Session:
    def __init__(self) -> None:
        self.gates: list[str] = []
        self.lines: list[str] = []

    def gate(self, prompt: str, *, hint: str | None = None) -> None:
        self.gates.append(prompt)

    def write_line(self, text: str) -> None:
        self.lines.append(text)


def test_a_physical_driver_needs_an_attended_run(physical_arm: Path) -> None:
    arm = PresetEmbodiment(str(physical_arm), sleep=lambda s: None)
    assert not arm.simulated and not arm.info.is_simulated
    assert FakeRobot.calls == []  # constructing the embodiment touches no driver
    with pytest.raises(EmbodimentFault, match="needs an attended run"):
        arm.reset(SCENE)
    arm.close()
    # Refused before the driver was even constructed: nothing was opened.
    assert FakeRobot.calls == []


def test_physical_gates_go_through_the_session(physical_arm: Path) -> None:
    arm = PresetEmbodiment(str(physical_arm), sleep=lambda s: None)
    session = Session()
    arm.connect_operator_session(session)
    arm.reset(SCENE)
    arm.reset(SCENE)
    # The home gate once per connection, the scene gate every trial.
    assert [g.split(" - ")[0].split(",")[0] for g in session.gates] == [
        "Robot will move to its home pose",
        "Position the scene",
        "Position the scene",
    ]
    assert FakeRobot.calls.count("connect") == 1
    arm.close()


def test_auto_start_replaces_gates_with_notices(physical_arm: Path) -> None:
    arm = PresetEmbodiment(str(physical_arm), auto_start="true", sleep=lambda s: None)
    session = Session()
    arm.connect_operator_session(session)
    arm.reset(SCENE)
    assert session.gates == []
    assert session.lines[0].startswith("auto_start: the robot will move to its home pose")
    arm.close()


def test_driver_arguments_are_checked_at_construction(physical_arm: Path) -> None:
    PresetEmbodiment(str(physical_arm), port="/dev/ttyACM0")
    with pytest.raises(ConfigError, match="does not take these arguments"):
        PresetEmbodiment(str(physical_arm), baud=115200)


def test_unknown_drivers_and_flags_are_refused(tmp_path: Path) -> None:
    shutil.copy(ARM.parent / "mobile_manipulator.urdf", tmp_path)
    (tmp_path / "arm.toml").write_text(ARM.read_text().replace('"sim"', '"nowhere:Nothing"'))
    with pytest.raises(ConfigError, match="import failed"):
        PresetEmbodiment(str(tmp_path / "arm.toml"))
    with pytest.raises(EmbodimentFault, match="pace must be true or false"):
        PresetEmbodiment(str(ARM), pace="maybe")
    # A physical preset dry-runs in the simulator with driver=sim.
    assert PresetEmbodiment(str(tmp_path / "arm.toml"), driver="sim").simulated


def test_a_driver_file_next_to_the_preset_loads(tmp_path: Path) -> None:
    shutil.copy(ARM.parent / "mobile_manipulator.urdf", tmp_path)
    (tmp_path / "my_driver.py").write_text(
        "from abstract_harness.robot.preset import SimDriver\n"
        "class Mine(SimDriver):\n    simulated = False\n"
    )
    (tmp_path / "arm.toml").write_text(ARM.read_text().replace('"sim"', '"my_driver.py:Mine"'))
    arm = PresetEmbodiment(str(tmp_path / "arm.toml"))
    assert not arm.simulated and arm._factory.__name__ == "Mine"


class BrokenRead(FakeRobot):
    def read(self) -> dict[str, Any]:
        return {"joint_pos": np.zeros(3)}


def test_bad_driver_state_is_an_embodiment_fault(
    physical_arm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.modules["fake_robot_driver"].FakeRobot = BrokenRead  # type: ignore[attr-defined]
    arm = PresetEmbodiment(str(physical_arm), auto_start=True, sleep=lambda s: None)
    with pytest.raises(EmbodimentFault, match="missing state 'joint_vel'|has shape"):
        arm.reset(SCENE)
    arm.close()


def test_the_registered_preset_embodiment_and_its_stream() -> None:
    base = resolve("embodiment", "preset", preset=str(BASE))
    stream = base.command_stream()
    assert isinstance(stream, CommandStream) and sorted(stream.axes) == ["vx", "yaw_rate"]
    assert stream.event_names == frozenset({"go", "stop"})
    stream.close()
    assert "preset: " in base.doctor_notes()


def test_teleop_drives_the_sim_base_from_live_commands(tmp_path: Path) -> None:
    base = PresetEmbodiment(str(BASE), pace=False)
    stream = CommandStream(list(base.command_stream().axes.values()))

    task = Task(
        name="drive",
        scenes=[Scene(id="s0", instruction="drive forward", init_seed=0)],
        scorer=success_at_end(),
        max_steps=40,
    )
    stream.set("vx", 1.0)

    from abstract_harness.robot.core.teleop import TeleopPolicy

    class Holding(TeleopPolicy):
        """Holds the stick forward as a gamepad would, inside the deadman timeout."""

        def act(self, observation: Any) -> Any:
            stream.set("vx", 1.0)
            return super().act(observation)

    approver, _, _ = build_guardrails(base.info.action_space, None, base)
    (log,) = eval(
        task,
        Holding(),
        base,
        log_dir=str(tmp_path),
        commands=stream,
        approver=approver,
    )
    base.close()
    actions = tmp_path / log.samples[0].trial_metadata[0]["actions"]
    rows = [json.loads(line) for line in actions.read_text().splitlines()[1:]]
    vx = [row["action"][0] for row in rows]
    # The trial starts from rest; then the acceleration limit ramps 0.02 m/s per step
    # toward the commanded 1.0 m/s.
    assert vx[:4] == pytest.approx([0.0, 0.02, 0.04, 0.06])
    assert rows[1]["requested_action"][0] == pytest.approx(1.0)
    assert rows[0]["commands"] == {"vx": 0.0, "yaw_rate": 0.0}
    assert all(row["commands"]["vx"] == 1.0 for row in rows[1:])
    assert rows[-1]["state"]["base_pose"][0] > 0.2


def test_rig_preset_paths_resolve_against_the_rig_file(tmp_path: Path) -> None:
    rig = tmp_path / "rigs" / "sim.toml"
    rig.parent.mkdir()
    rig.write_text(
        textwrap.dedent(
            f"""
            [vla]
            policy = "teleop"
            [rig]
            preset = "{BASE}"
            [rig.embodiment]
            pace = false
            """
        )
    )
    config = RobotConfig.load(rig)
    assert config.rig.type == "preset" and config.rig.preset == BASE
    assert config.embodiment_args() == {"preset": str(BASE), "pace": False}
    assert config.policy_args() == {} and config.vla is not None and config.vla.endpoint is None
    relative = tmp_path / "relative.toml"
    relative.write_text('[vla]\npolicy = "teleop"\n[rig]\npreset = "robots/base.toml"\n')
    assert RobotConfig.load(relative).rig.preset == tmp_path / "robots" / "base.toml"


@pytest.mark.parametrize(
    ("rig", "message"),
    [
        ('[rig]\ntype = "cubepick"\npreset = "x.toml"\n', 'needs type = "preset"'),
        ('[rig]\ntype = "preset"\n', "needs \\[rig\\] preset"),
        (
            '[rig]\ntype = "preset"\n[rig.embodiment]\npreset = "y.toml"\n',
            "not \\[rig.embodiment\\]",
        ),
        (
            '[rig]\npreset = "x.toml"\n[rig.embodiment]\npreset = "y.toml"\n',
            "not \\[rig.embodiment\\]",
        ),
    ],
)
def test_rig_preset_errors(tmp_path: Path, rig: str, message: str) -> None:
    from pydantic import ValidationError

    path = tmp_path / "rig.toml"
    path.write_text('[vla]\npolicy = "teleop"\n' + rig)
    with pytest.raises(ValidationError, match=message):
        RobotConfig.load(path)


def test_teleop_refuses_a_server_url(tmp_path: Path) -> None:
    from pydantic import ValidationError

    path = tmp_path / "rig.toml"
    path.write_text(
        f'[vla]\npolicy = "teleop"\nserver_url = "http://h:1"\n[rig]\npreset = "{BASE}"\n'
    )
    with pytest.raises(ValidationError, match="has no server_url"):
        RobotConfig.load(path)


def test_the_shipped_teleop_rig_probes_without_hardware() -> None:
    from abstract_harness.robot import cli as robot_cli
    from abstract_harness.robot import stack

    config = RobotConfig.load(Path(__file__).parents[2] / "rigs" / "mobile-base-teleop.toml")
    assert config.rig.strict_timing and config.rig.type == "preset"
    ok, detail, _ = robot_cli.policy_probe_result(stack.policy_probe(config))
    assert ok, detail
    assert detail.startswith("1x2 chunk")


def test_the_doctor_prints_the_preset_summary(capsys: Any) -> None:
    from abstract_harness.robot.core import commands

    assert commands.doctor("preset", {"preset": str(ARM)}, "test") == 0
    out = capsys.readouterr().out
    assert "action: joint_pos, 7 dims at 30 Hz (driver servo 250 Hz), driver sim" in out
    assert "mobile-arm: conformant" in out


def test_yam_tools_refuse_a_preset_rig(capsys: Any) -> None:
    import argparse

    from abstract_harness.robot import tools

    rig = Path(__file__).parents[2] / "rigs" / "mobile-base-teleop.toml"
    args = argparse.Namespace(tool="health", tool_parser=argparse.ArgumentParser(), config=rig)
    with pytest.raises(SystemExit):
        tools.main(args)
    assert "health is a YAM tool" in capsys.readouterr().err


def test_the_home_ramp_never_jumps_past_the_step_limit(
    physical_arm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[np.ndarray] = []

    class LeftOutside(FakeRobot):
        def __init__(self, description: Any, **kwargs: Any) -> None:
            super().__init__(description, **kwargs)
            # arm_j1 past even the model's limit, arm_j2 past the preset's narrowed one.
            self.position = np.array([3.0, 1.7, 0.0, 0.0, 0.0, 0.0, 0.0])

        def command(self, values: Any) -> None:
            sent.append(np.asarray(values, float).copy())
            super().command(values)

    monkeypatch.setattr(sys.modules["fake_robot_driver"], "FakeRobot", LeftOutside)
    arm = PresetEmbodiment(str(physical_arm), auto_start=True, sleep=lambda s: None)
    arm.reset(SCENE)
    path = np.array([[3.0, 1.7, 0.0, 0.0, 0.0, 0.0, 0.0], *sent])
    steps = np.abs(np.diff(path, axis=0)).max(axis=0)
    assert steps[1] <= 0.03 + 1e-9 and steps[6] <= 0.004 + 1e-9
    assert path[-1][1] == pytest.approx(0.4)  # arm_j2 walked home
    assert np.all(path[:, 0] == 3.0)  # arm_j1 has no home value: it never moved
    arm.close()


def test_a_failed_watchdog_stop_faults_the_episode(tmp_path: Path) -> None:
    base = PresetEmbodiment(str(BASE), pace=False)
    base.reset(SCENE)
    driver = base._driver

    def broken_stop() -> None:
        raise RuntimeError("bus timeout")

    driver.stop = broken_stop
    base.step(Action(data=np.array([0.5, 0.0])))
    deadline = time.monotonic() + 3
    while base._fault is None and time.monotonic() < deadline:
        time.sleep(0.02)
    with pytest.raises(EmbodimentFault, match="could not stop the driver: bus timeout"):
        base.check_health()
    with pytest.raises(EmbodimentFault, match="could not stop the driver"):
        base.step(Action(data=np.array([0.5, 0.0])))
    base._driver = None  # its stop is broken for good; nothing left to close here


def test_the_acceleration_limit_restarts_from_rest_after_a_watchdog_stop() -> None:
    base = PresetEmbodiment(str(BASE), pace=False)
    approver, _, _ = build_guardrails(base.info.action_space, None, base)
    store: dict[str, Any] = {}
    approver.begin_trial(None, store)
    base.reset(SCENE)
    for _ in range(5):
        approved = approver.review(Action(data=np.array([1.0, 0.0])), store)
        base.step(approved)
    assert approved.data[0] == pytest.approx(0.1)
    deadline = time.monotonic() + 3
    while base.watchdog_stops == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    resumed = approver.review(Action(data=np.array([1.0, 0.0])), store)
    assert resumed.data[0] == pytest.approx(0.02)  # from rest, not from 0.1
    base.close()


def test_delta_commands_stop_at_the_position_limit(tmp_path: Path) -> None:
    shutil.copy(ARM.parent / "mobile_manipulator.urdf", tmp_path)
    (tmp_path / "delta.toml").write_text(
        '[robot]\nname = "delta"\nmodel = "mobile_manipulator.urdf"\npolicy_hz = 20\n'
        'mode = "joint_delta"\n[robot.joints]\nselect = ["arm_j1"]\nmax_step = { arm_j1 = 0.5 }\n'
    )
    arm = PresetEmbodiment(str(tmp_path / "delta.toml"), pace=False)
    arm.reset(SCENE)
    clamped = []
    for _ in range(8):
        result = arm.step(Action(data=np.array([0.5])))
        clamped.append(bool(result.info.get("position_clamped")))
    assert result.observation.state["joint_pos"][0] == pytest.approx(2.9)
    assert clamped[:5] == [False] * 5 and all(clamped[6:])
    back = arm.step(Action(data=np.array([-0.5])))
    assert back.observation.state["joint_pos"][0] == pytest.approx(2.4)
    arm.close()


def test_a_fallen_robot_is_refused_at_reset() -> None:
    base = PresetEmbodiment(str(BASE), pace=False)
    base.reset(SCENE)
    base._driver.tilt(60)
    with pytest.raises(EmbodimentFault, match="tilted 60 degrees from upright"):
        base.reset(SCENE)
    base.close()


def test_a_driver_that_fails_to_connect_is_closed(
    physical_arm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoBus(FakeRobot):
        def connect(self) -> None:
            raise OSError("no such device")

    monkeypatch.setattr(sys.modules["fake_robot_driver"], "FakeRobot", NoBus)
    arm = PresetEmbodiment(str(physical_arm), auto_start=True, sleep=lambda s: None)
    with pytest.raises(OSError, match="no such device"):
        arm.reset(SCENE)
    assert FakeRobot.calls[-1] == "close" and arm._driver is None


def test_velocity_near_a_position_limit_is_slowed_to_stop_by_the_watchdog_horizon(
    tmp_path: Path,
) -> None:
    shutil.copy(ARM.parent / "mobile_manipulator.urdf", tmp_path)
    (tmp_path / "vel.toml").write_text(
        '[robot]\nname = "vel"\nmodel = "mobile_manipulator.urdf"\npolicy_hz = 20\n'
        'mode = "joint_vel"\ncommand_timeout_s = 0.25\n[robot.joints]\nselect = ["arm_j1"]\n'
    )
    arm = PresetEmbodiment(str(tmp_path / "vel.toml"), pace=False)
    arm.reset(SCENE)
    arm._driver.position[:] = 2.8  # 0.1 rad from the 2.9 limit
    result = arm.step(Action(data=np.array([2.0])))
    # Even if this velocity persisted until the watchdog fired, it would stop at the limit.
    assert result.command["joint_vel"][0] == pytest.approx(0.1 / 0.25)
    assert result.info["position_clamped"] is True
    arm.close()


def test_low_rate_position_presets_load_with_the_default_timeout(tmp_path: Path) -> None:
    shutil.copy(ARM.parent / "mobile_manipulator.urdf", tmp_path)
    (tmp_path / "slow.toml").write_text(
        '[robot]\nname = "slow"\nmodel = "mobile_manipulator.urdf"\npolicy_hz = 5\n'
        '[robot.joints]\nselect = ["arm_j1"]\n'
    )
    assert PresetEmbodiment(str(tmp_path / "slow.toml")).info.control_hz == 5.0


def test_an_action_approved_before_a_watchdog_stop_goes_out_as_rest() -> None:
    base = PresetEmbodiment(str(BASE), pace=False)
    approver, _, _ = build_guardrails(base.info.action_space, None, base)
    store: dict[str, Any] = {}
    approver.begin_trial(None, store)
    base.reset(SCENE)
    approved = approver.review(Action(data=np.array([1.0, 0.0])), store)
    base._rested = True  # the watchdog fired between review and step
    result = base.step(approved)
    np.testing.assert_array_equal(result.command["base_vel"], [0.0, 0.0])
    resumed = approver.review(Action(data=np.array([1.0, 0.0])), store)
    assert resumed.data[0] == pytest.approx(0.02)  # from rest again
    base.close()
