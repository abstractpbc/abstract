"""Strict timing and async inference from the robot TOML and ``robot run`` flags."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from abstract_harness.cli import parser
from abstract_harness.robot import cli as robot_cli
from abstract_harness.robot import stack
from abstract_harness.robot.config import RobotConfig
from abstract_harness.robot.core.timing import StrictTiming

VLA = '[vla]\npolicy = "molmoact2"\nserver_url = "http://policy-host:8202"\n'
RIG = (
    '[rig]\nlog_root = "logs"\n{extra}'
    '[rig.embodiment]\nleft_channel = "can1"\nright_channel = "can0"\n'
)


def write(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "robot.toml"
    path.write_text(VLA + RIG.format(extra=extra))
    return path


def test_timing_is_off_by_default_and_recorded_in_the_setup(tmp_path: Path) -> None:
    config = RobotConfig.load(write(tmp_path))
    assert config.rig.strict_timing is False and config.rig.async_inference is False
    assert config.rig.timing() is False
    setup = config.setup()
    assert setup["strict_timing"] is False and setup["async_inference"] is False


def test_strict_timing_from_the_rig_section(tmp_path: Path) -> None:
    config = RobotConfig.load(
        write(
            tmp_path, "strict_timing = true\nmax_consecutive_overruns = 5\nasync_inference = true\n"
        )
    )
    assert config.rig.timing() == StrictTiming(max_consecutive_overruns=5)
    assert config.setup()["strict_timing"] is True


def test_an_overrun_limit_needs_strict_timing(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="needs strict_timing"):
        RobotConfig.load(write(tmp_path, "max_consecutive_overruns = 5\n"))


def test_timing_values_are_strict_booleans(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        RobotConfig.load(write(tmp_path, 'strict_timing = "yes"\n'))


@pytest.fixture
def interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stack, "installed", lambda: (True, "robot extra (fake)"))
    monkeypatch.setattr(robot_cli, "interactive_terminal", lambda: True)
    monkeypatch.setenv("TMUX", "/tmp/tmux-test")


def test_run_flags_turn_timing_on_for_the_episode_and_its_setup(
    tmp_path: Path, interactive: None, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    seen: list[RobotConfig] = []

    def fake_episode(config: RobotConfig, instruction: str, log_dir: Path, **kwargs: Any) -> int:
        seen.append(config)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "standin.rrd").write_bytes(b"RRF2")
        return 0

    monkeypatch.setattr(stack, "episode", fake_episode)
    path = write(tmp_path)
    args = parser().parse_args(
        ["robot", "run", "--config", str(path), "--strict-timing", "--async-inference", "go"]
    )
    assert robot_cli.run(args) == 0
    (config,) = seen
    assert config.rig.strict_timing is True and config.rig.async_inference is True
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["summary"]
    log_dir = tmp_path / "logs" / summary["log_dir"].rsplit("/", 1)[-1]
    setup = json.loads((log_dir / robot_cli.SETUP).read_text())
    assert setup["strict_timing"] is True and setup["async_inference"] is True


def test_stack_episode_hands_timing_to_run_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from abstract_harness.robot.core import commands

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(commands, "components", lambda *a, **k: object())
    monkeypatch.setattr(
        commands, "run_task", lambda task, parts, **kwargs: calls.append(kwargs) or 0
    )
    config = RobotConfig.load(write(tmp_path, "strict_timing = true\nasync_inference = true\n"))
    assert stack.episode(config, "go", tmp_path / "run", source="robot.toml") == 0
    assert calls[0]["strict_timing"] == StrictTiming()
    assert calls[0]["async_inference"] is True
