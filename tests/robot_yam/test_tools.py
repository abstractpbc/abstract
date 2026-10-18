"""``abstract robot holdcheck|health|pose``: the YAM tools, set up from the robot TOML."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from abstract_harness.robot import tools
from abstract_harness.robot.config import RobotConfig
from abstract_harness.robot.yam.config import YamConfig

from .conftest import robot_cli, write_rig

RIGS = Path(__file__).resolve().parents[2] / "rigs"


def _parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="abstract robot")
    tools.configure_parser(root.add_subparsers(dest="robot_command", required=True))
    return root


def test_help_names_every_tool(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        _parser().parse_args(["--help"])
    assert caught.value.code == 0
    out = capsys.readouterr().out
    assert all(tool in out for tool in ("holdcheck", "health", "pose"))


@pytest.mark.parametrize("tool", tools.COMMANDS)
def test_each_tool_needs_a_robot_toml(tool: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        _parser().parse_args([tool])
    assert caught.value.code == 2
    assert "--config" in capsys.readouterr().err


def test_a_missing_toml_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        robot_cli("pose", ["list"], config=tmp_path / "missing.toml")
    assert caught.value.code == 2
    assert "missing.toml" in capsys.readouterr().err


def test_pose_subcommands_leave_the_top_level_command_alone() -> None:
    # The abstract CLI dispatches on args.command; pose's own subcommand must not reuse it.
    root = argparse.ArgumentParser(prog="abstract")
    robot = root.add_subparsers(dest="command").add_parser("robot")
    tools.configure_parser(robot.add_subparsers(dest="robot_command"))
    args = root.parse_args(["robot", "pose", "--config", "rig.toml", "list"])
    assert (args.command, args.robot_command, args.tool, args.pose_command) == (
        "robot",
        "pose",
        "pose",
        "list",
    )


def test_the_tools_get_the_embodiment_args_a_run_gets(tmp_path: Path) -> None:
    path = write_rig(tmp_path / "rig.toml", {"rest_pose": [0.0] * 14, "control_hz": 30})
    args = RobotConfig.load(path).embodiment_args()
    assert args == {
        "left_channel": "can0",
        "right_channel": "can1",
        "rest_pose": ", ".join(["0.0"] * 14),
        "control_hz": 30,
    }
    assert YamConfig.from_kwargs(**args).rest_pose == (0.0,) * 14


@pytest.mark.parametrize("rig", sorted(RIGS.glob("*.toml")), ids=lambda path: path.name)
def test_the_committed_rigs_configure_the_tools(rig: Path) -> None:
    # The tools see the rig a run sees: every [rig.embodiment] value, lists as tuples.
    config = RobotConfig.load(rig)
    cfg = YamConfig.from_kwargs(**config.embodiment_args())
    for key, value in config.rig.embodiment.items():
        assert getattr(cfg, key) == (tuple(value) if isinstance(value, list) else value), key
