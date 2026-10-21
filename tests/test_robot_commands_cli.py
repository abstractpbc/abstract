"""Live commands through the CLIs: ``--commands``, ``--commands-fd`` and serve's
``command`` RPC."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from abstract_harness.cli import parser
from abstract_harness.robot import cli as robot_cli
from abstract_harness.robot import rpc, stack
from abstract_harness.robot.core import commands
from abstract_harness.robot.core.command_stream import (
    AxisSpec,
    CommandStream,
    FdCommandSource,
    UdpCommandSource,
)
from abstract_harness.robot.core.mock import CubePickEmbodiment

VLA = (
    '[vla]\npolicy = "molmoact2"\nserver_url = "http://policy-host:8202"\n'
    '[rig]\nlog_root = "logs"\n[rig.embodiment]\nleft_channel = "can1"\nright_channel = "can0"\n'
)


def test_a_run_without_sources_or_a_hook_has_no_stream() -> None:
    assert commands._command_stream(CubePickEmbodiment(), ()) is None


def test_sources_alone_make_a_free_form_stream(capsys: Any) -> None:
    stream = commands._command_stream(CubePickEmbodiment(), ["udp:127.0.0.1:0"])
    assert stream is not None and stream.axes == {}
    assert "commands: free-form commands" in capsys.readouterr().out
    stream.close()


def test_the_embodiment_stream_gets_the_extra_sources() -> None:
    own = CommandStream([AxisSpec("vx")])

    class Commanded(CubePickEmbodiment):
        def command_stream(self) -> CommandStream:
            return own

    source = UdpCommandSource("127.0.0.1", 0)
    stream = commands._command_stream(Commanded(), [source])
    assert stream is own and source in own._sources
    stream.close()


def test_a_bad_hook_or_source_is_a_clean_refusal() -> None:
    class Wrong(CubePickEmbodiment):
        def command_stream(self) -> str:
            return "nope"

    with pytest.raises(SystemExit, match="expected CommandStream"):
        commands._command_stream(Wrong(), [])
    with pytest.raises(SystemExit, match="expected udp:HOST:PORT"):
        commands._command_stream(CubePickEmbodiment(), ["serial:/dev/ttyUSB0"])


def test_cli_eval_takes_a_udp_command_source(tmp_path: Path, capsys: Any) -> None:
    from abstract_harness.cli import _main

    with pytest.raises(SystemExit) as exit_info:
        _main(
            [
                "eval",
                "--instruction",
                "reach",
                "--max-steps",
                "2",
                "--no-prompt",
                "--commands",
                "udp:127.0.0.1:0",
                "--log-dir",
                str(tmp_path),
            ]
        )
    assert exit_info.value.code == 0
    assert "commands: free-form commands" in capsys.readouterr().out


def test_robot_run_hands_command_sources_to_the_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_episode(config: Any, instruction: str, log_dir: Path, **kwargs: Any) -> int:
        calls.append(kwargs)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "standin.rrd").write_bytes(b"RRF2")
        return 0

    monkeypatch.setattr(stack, "episode", fake_episode)
    monkeypatch.setattr(stack, "installed", lambda: (True, "robot extra (fake)"))
    monkeypatch.setattr(robot_cli, "interactive_terminal", lambda: True)
    monkeypatch.setenv("TMUX", "/tmp/tmux-test")
    config = tmp_path / "robot.toml"
    config.write_text(VLA)
    read_fd, write_fd = os.pipe()
    args = parser().parse_args(
        [
            "robot",
            "run",
            "--config",
            str(config),
            "--commands",
            "udp:127.0.0.1:9999",
            "--commands-fd",
            str(read_fd),
            "go",
        ]
    )
    assert robot_cli.run(args) == 0
    sources = calls[0]["command_sources"]
    assert sources[0] == "udp:127.0.0.1:9999" and isinstance(sources[1], FdCommandSource)
    os.close(write_fd)
    os.close(read_fd)
    # Without either flag, the episode gets no command sources at all. Another log root:
    # both runs may start within the second their directories are named by.
    calls.clear()
    (tmp_path / "plain").mkdir()
    plain_config = tmp_path / "plain" / "robot.toml"
    plain_config.write_text(VLA)
    plain = parser().parse_args(["robot", "run", "--config", str(plain_config), "go"])
    assert robot_cli.run(plain) == 0
    assert "command_sources" not in calls[0]


def _controller() -> Any:
    """serve's Controller without a config or a running episode."""
    controller = object.__new__(rpc.Controller)
    controller._lock = threading.RLock()
    controller._commands_fd = None
    return controller


def test_command_rpc_forwards_validated_json_lines() -> None:
    controller = _controller()
    with pytest.raises(ValueError, match="no run is active"):
        controller.cmd_command({"values": {"vx": 0.1}})
    read_fd, write_fd = os.pipe()
    controller._commands_fd = write_fd
    assert controller.cmd_command({"values": {"vx": 0.25}, "events": ["go"]}) == {
        "disposition": "sent"
    }
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as pipe:
        assert json.loads(pipe.readline()) == {"values": {"vx": 0.25}, "events": ["go"]}


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ({}, "values, events or both"),
        ({"values": [1]}, "values must be an object"),
        ({"values": {"9x": 1}}, "must match"),
        ({"values": {"vx": True}}, "must be a number"),
        ({"values": {"vx": float("nan")}}, "must be finite"),
        ({"events": [3]}, "must match"),
        ({"values": {f"a{i}": 0 for i in range(65)}}, "at most 64"),
    ],
)
def test_command_rpc_refuses_malformed_commands(command: dict[str, Any], message: str) -> None:
    controller = _controller()
    controller._commands_fd = 99_999  # never written: validation fails first
    with pytest.raises(ValueError, match=message):
        controller.cmd_command(command)


def test_command_rpc_reports_a_closed_or_stalled_run() -> None:
    controller = _controller()
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    controller._commands_fd = write_fd
    with pytest.raises(ValueError, match="closed"):
        controller.cmd_command({"events": ["go"]})
    os.close(write_fd)

    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    controller._commands_fd = write_fd
    with pytest.raises(ValueError, match="not reading"):
        for _ in range(100_000):  # fill the pipe nobody reads
            controller.cmd_command({"values": {"vx": 0.0}})
    os.close(read_fd)
    os.close(write_fd)


def test_serve_command_pipe_alone_is_silent(capsys: Any) -> None:
    read_fd, write_fd = os.pipe()
    stream = commands._command_stream(CubePickEmbodiment(), [FdCommandSource(read_fd)])
    assert stream is not None and not stream.active
    assert "commands:" not in capsys.readouterr().out
    os.close(write_fd)
    stream.close()


def test_a_self_paced_run_prints_no_timing_line(capsys: Any) -> None:
    from dataclasses import replace

    from abstract_harness.robot.core.embodiment import SELF_PACED
    from abstract_harness.robot.core.mock import ScriptedPolicy

    cube = CubePickEmbodiment()
    cube.info = replace(cube.info, capabilities=cube.info.capabilities | {SELF_PACED})
    commands._announce_timing(ScriptedPolicy(), cube, False, False)
    assert capsys.readouterr().out == ""
    commands._announce_timing(ScriptedPolicy(), cube, False, True)
    assert "the embodiment paces itself with async inference" in capsys.readouterr().out


def test_command_rpc_refuses_lines_that_could_split(monkeypatch: pytest.MonkeyPatch) -> None:
    import select

    controller = _controller()
    controller._commands_fd = 99_999
    monkeypatch.setattr(select, "PIPE_BUF", 64)
    with pytest.raises(ValueError, match="command too long"):
        controller.cmd_command({"values": {f"axis{i}": 0.5 for i in range(10)}})
