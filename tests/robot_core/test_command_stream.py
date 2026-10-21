"""Live commands: the stream's deadman semantics, its sources, and the rollout boundary
where commands reach a policy, its records and the end of a trial."""

from __future__ import annotations

import json
import os
import socket
import struct
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from abstract_harness.robot.core import eval
from abstract_harness.robot.core.approver import AutoApprover
from abstract_harness.robot.core.command_stream import (
    END_EPISODE,
    AxisBinding,
    AxisSpec,
    ButtonBinding,
    CommandStream,
    FdCommandSource,
    JoystickSource,
    UdpCommandSource,
    parse_source,
)
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
from abstract_harness.robot.core.types import OPERATOR_END, Action, ActionChunk, Observation


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


AXES = (
    AxisSpec("vx", low=-1.0, high=1.0, timeout_s=0.5),
    AxisSpec("speed", low=0.0, high=2.0, default=1.0, timeout_s=None),
)


def test_declared_axes_clamp_rest_and_time_out() -> None:
    clock = Clock()
    stream = CommandStream(AXES, clock=clock)
    assert stream.snapshot().values == {"vx": 0.0, "speed": 1.0}
    stream.update({"vx": 3.0, "speed": 1.5})
    assert stream.peek() == {"vx": 1.0, "speed": 1.5}
    clock.now = 0.4
    assert stream.peek()["vx"] == 1.0
    clock.now = 0.6  # no vx update for longer than its timeout: back to rest
    assert stream.peek() == {"vx": 0.0, "speed": 1.5}
    stream.begin_trial()
    assert stream.peek() == {"vx": 0.0, "speed": 1.0}


def test_declared_streams_refuse_unknown_names_and_bad_values() -> None:
    stream = CommandStream(AXES)
    with pytest.raises(ValueError, match="unknown command 'yaw'"):
        stream.set("yaw", 0.1)
    for bad in (float("nan"), float("inf"), True, "fast"):
        with pytest.raises(ValueError):
            stream.set("vx", bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        stream.update({"vx": 0.5, "nope": 1.0})
    assert stream.peek()["vx"] == 0.0  # nothing applied from a refused batch


def test_free_form_streams_accept_any_name_and_drop_stale_values() -> None:
    clock = Clock()
    stream = CommandStream(default_timeout_s=0.25, clock=clock)
    stream.set("height", 0.7)
    assert stream.peek() == {"height": 0.7}
    clock.now = 0.3
    assert stream.peek() == {}


def test_events_drain_once_and_end_episode_latches() -> None:
    stream = CommandStream(events=["go", "stop"])
    stream.press("go", source="pad")
    stream.press(END_EPISODE, source="pad")
    with pytest.raises(ValueError, match="unknown event 'jump'"):
        stream.press("jump")
    events = stream.snapshot().events
    assert [(e.name, e.source) for e in events] == [("go", "pad"), (END_EPISODE, "pad")]
    assert stream.snapshot().events == ()
    assert stream.end_requested()
    stream.begin_trial()
    assert not stream.end_requested()


def test_wire_messages_apply_and_count_rejections() -> None:
    stream = CommandStream(AXES)
    stream.apply({"values": {"vx": 0.2}, "events": ["go"]}, source="udp")
    stream.apply({"vx": -0.3}, source="udp")
    stream.apply({"values": [1, 2]}, source="udp")
    stream.apply({"values": {"bogus": 1}}, source="udp")
    assert stream.peek()["vx"] == -0.3
    assert stream.rejected == 2
    assert [e.name for e in stream.snapshot().events] == ["go"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "9bad"},
        {"name": "vx", "low": 1.0, "high": 0.0},
        {"name": "vx", "default": 5.0},
        {"name": "vx", "timeout_s": 0},
        {"name": "vx", "low": float("nan")},
    ],
)
def test_axis_specs_are_validated(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        AxisSpec(**kwargs)


def test_axis_names_are_distinct() -> None:
    with pytest.raises(ConfigError, match="distinct"):
        CommandStream([AxisSpec("vx"), AxisSpec("vx")])


def _wait_for(predicate: Any, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.005)


def test_udp_source_delivers_datagrams() -> None:
    stream = CommandStream(AXES)
    source = UdpCommandSource("127.0.0.1", 0)
    stream.add_source(source)
    stream.start()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(b"not json", source.address)
            sender.sendto(json.dumps({"values": {"vx": 0.4}}).encode(), source.address)
        _wait_for(lambda: stream.peek()["vx"] == 0.4)
        assert stream.rejected == 1
    finally:
        stream.close()


def test_udp_source_refuses_non_loopback_binds_by_default() -> None:
    with pytest.raises(ConfigError, match="unauthenticated"):
        UdpCommandSource("0.0.0.0", 0)


def test_parse_source_forms() -> None:
    source = parse_source("udp:127.0.0.1:0")
    assert isinstance(source, UdpCommandSource)
    source.close()
    for bad in ("udp:", "tcp:1:2", "udp:host:port"):
        with pytest.raises(ConfigError):
            parse_source(bad)


def test_fd_source_reads_json_lines() -> None:
    read_fd, write_fd = os.pipe()
    stream = CommandStream(AXES, sources=[FdCommandSource(read_fd)])
    stream.start()
    os.write(write_fd, b'{"values": {"vx": -0.5}, "events": ["end_episode"]}\nnope\n')
    os.close(write_fd)
    _wait_for(lambda: stream.end_requested())
    assert stream.peek()["vx"] == -0.5 and stream.rejected == 1
    stream.close()


def test_joystick_bindings_deadzone_deadman_and_buttons() -> None:
    stream = CommandStream([AxisSpec("vx"), AxisSpec("speed", low=0, high=2, timeout_s=None)])
    pad = JoystickSource(
        axes=[AxisBinding("vx", axis=1, scale=-0.8, deadzone=0.1)],
        buttons=[ButtonBinding(0, event="go"), ButtonBinding(3, set={"speed": 2.0})],
        deadman=4,
    )
    pad.handle(stream, -32767, 0x02, 1)  # stick fully up, deadman not held
    assert stream.peek()["vx"] == 0.0
    pad.handle(stream, 1, 0x01, 4)  # hold the deadman
    assert stream.peek()["vx"] == pytest.approx(0.8)
    pad.handle(stream, 2000, 0x02, 1)  # inside the deadzone
    assert stream.peek()["vx"] == 0.0
    pad.handle(stream, 1, 0x01 | 0x80, 0)  # initial-state events never press buttons
    pad.handle(stream, 1, 0x01, 0)
    pad.handle(stream, 1, 0x01, 3)
    assert [e.name for e in stream.snapshot().events] == ["go"]
    assert stream.peek()["speed"] == 2.0
    pad.handle(stream, 0, 0x01, 4)  # deadman released: the stick no longer counts
    pad.handle(stream, -32767, 0x02, 1)
    assert stream.peek()["vx"] == 0.0


def test_joystick_reads_linux_js_events_from_its_device(tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()
    stream = CommandStream([AxisSpec("vx", timeout_s=None)])
    pad = JoystickSource(
        "fake-js0",
        axes=[AxisBinding("vx", axis=0)],
        opener=lambda path: os.fdopen(read_fd, "rb", buffering=0),
    )
    stream.add_source(pad)
    stream.start()
    os.write(write_fd, struct.pack("IhBB", 0, 16384, 0x02, 0))
    _wait_for(lambda: stream.peek()["vx"] > 0.4)
    os.close(write_fd)  # unplugged: the reader exits
    _wait_for(lambda: not pad.connected)
    stream.close()


class CommandPolicy:
    """Records the commands it sees and stops when told to."""

    def __init__(self) -> None:
        self.info = PolicyInfo(
            name="commanded",
            action_space=Box(
                shape=(2,),
                semantics=ActionSemantics(control_mode="eef_delta_pos", frame="world"),
            ),
        )
        self.config = PolicyConfig()
        self.seen: list[tuple[dict[str, float], list[dict[str, Any]]]] = []
        self.stream: CommandStream | None = None

    def reset(self, scene: Scene) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        commands = dict(observation.extra["commands"])
        events = list(observation.extra["command_events"])
        self.seen.append((commands, events))
        step = int(observation.extra["env_step"])
        if self.stream is not None and step == 1:
            self.stream.set("vx", 0.5, source="test")
            self.stream.press("go", source="test")
        if self.stream is not None and step == 3:
            self.stream.press(END_EPISODE, source="test")
        return ActionChunk(actions=[Action(data=np.array([commands["vx"] * 0.01, 0.0]))])


def test_rollout_injects_records_and_ends_on_end_episode() -> None:
    stream = CommandStream([AxisSpec("vx", timeout_s=None)], events=["go"])
    policy = CommandPolicy()
    policy.stream = stream
    stream.set("vx", 0.9)  # set before the trial: the trial starts from rest
    record = rollout(
        policy,  # type: ignore[arg-type]
        CubePickEmbodiment(),
        Scene(id="s", instruction="drive", init_seed=0),
        max_steps=20,
        seed=0,
        epoch=0,
        controller=DefaultController(),
        approver=AutoApprover(),
        sink=NullSink(),
        commands=stream,
    )
    assert [seen[0]["vx"] for seen in policy.seen] == [0.0, 0.0, 0.5, 0.5]
    assert policy.seen[2][1] == [{"t": 2, "name": "go", "source": "test"}]
    assert policy.seen[3][1] == []  # events are delivered once
    assert record.termination_reason == OPERATOR_END and record.terminated
    assert [step.commands for step in record.steps] == [{"vx": v} for v in (0.0, 0.0, 0.5, 0.5)]
    kinds = [(e.kind, e.data.get("name")) for e in record.events if e.kind == "command"]
    assert kinds == [("command", "go"), ("command", END_EPISODE)]


def test_rollout_without_a_stream_injects_nothing() -> None:
    seen: list[dict[str, Any]] = []

    class Plain(CommandPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            seen.append(dict(observation.extra))
            return ActionChunk(actions=[Action(data=np.zeros(2))])

    record = rollout(
        Plain(),  # type: ignore[arg-type]
        CubePickEmbodiment(),
        Scene(id="s", instruction="x", init_seed=0),
        max_steps=2,
        seed=0,
        epoch=0,
        controller=DefaultController(),
        approver=AutoApprover(),
        sink=NullSink(),
    )
    assert "commands" not in seen[0] and "command_events" not in seen[0]
    assert all(step.commands is None for step in record.steps)


def test_end_episode_while_inference_is_pending_stops_a_physical_trial() -> None:
    stream = CommandStream()

    class Slow(CommandPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            stream.press(END_EPISODE, source="pad")
            time.sleep(0.3)
            return ActionChunk(actions=[Action(data=np.zeros(2))])

    cube = CubePickEmbodiment()
    cube.info = replace(cube.info, is_simulated=False)
    from abstract_harness.robot.core.errors import _CancelledTrial

    with pytest.raises(_CancelledTrial) as caught:
        rollout(
            Slow(),  # type: ignore[arg-type]
            cube,
            Scene(id="s", instruction="x", init_seed=0),
            max_steps=5,
            seed=0,
            epoch=0,
            controller=DefaultController(),
            approver=AutoApprover(),
            sink=NullSink(),
            commands=stream,
        )
    assert caught.value.record.termination_reason == OPERATOR_END
    assert caught.value.record.steps == []


def test_eval_writes_commands_into_the_action_log(tmp_path: Path) -> None:
    stream = CommandStream([AxisSpec("vx", timeout_s=None)], events=["go"])
    policy = CommandPolicy()
    policy.stream = stream
    (log,) = eval(
        Task(
            name="commanded",
            scenes=[Scene(id="s0", instruction="drive", init_seed=0)],
            scorer=success_at_end(),
            max_steps=10,
        ),
        policy,  # type: ignore[arg-type]
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        commands=stream,
    )
    actions = tmp_path / log.samples[0].trial_metadata[0]["actions"]
    rows = [json.loads(line) for line in actions.read_text().splitlines()[1:]]
    assert [row["commands"] for row in rows] == [{"vx": v} for v in (0.0, 0.0, 0.5, 0.5)]
    assert log.samples[0].termination_reasons == (OPERATOR_END,)


def test_eval_without_commands_keeps_the_action_log_unchanged(tmp_path: Path) -> None:
    from abstract_harness.robot.core.mock import ScriptedPolicy

    (log,) = eval(
        Task(
            name="plain",
            scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
            scorer=success_at_end(),
            max_steps=3,
        ),
        ScriptedPolicy(),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )
    actions = tmp_path / log.samples[0].trial_metadata[0]["actions"]
    rows = [json.loads(line) for line in actions.read_text().splitlines()[1:]]
    assert all("commands" not in row for row in rows)


def test_a_console_verdict_in_the_same_step_wins_over_end_episode() -> None:
    from abstract_harness.robot.core.console import ConsolePoll, EndRequest

    stream = CommandStream()

    class Console:
        def begin_trial(self) -> None:
            return None

        def poll(self) -> ConsolePoll:
            stream.press(END_EPISODE, source="pad")
            return ConsolePoll(end=EndRequest(verdict="y", note="clean"))

    record = rollout(
        CommandPolicy(),  # type: ignore[arg-type]
        CubePickEmbodiment(),
        Scene(id="s", instruction="x", init_seed=0),
        max_steps=5,
        seed=0,
        epoch=0,
        controller=DefaultController(),
        approver=AutoApprover(),
        sink=NullSink(),
        operator_input=Console(),  # type: ignore[arg-type]
        commands=stream,
    )
    assert record.termination_reason == OPERATOR_END
    assert (record.operator_judgement, record.operator_note) == ("y", "clean")


def test_an_unused_free_form_stream_leaves_observations_and_records_unchanged() -> None:
    stream = CommandStream()  # what every serve run gets: a channel nobody has used yet
    seen: list[dict[str, Any]] = []

    class Recording(CommandPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            seen.append(dict(observation.extra))
            if observation.extra["env_step"] == 2:
                stream.set("vx", 0.4, source="rpc")
            return ActionChunk(actions=[Action(data=np.zeros(2))])

    assert not stream.active
    record = rollout(
        Recording(),  # type: ignore[arg-type]
        CubePickEmbodiment(),
        Scene(id="s", instruction="x", init_seed=0),
        max_steps=4,
        seed=0,
        epoch=0,
        controller=DefaultController(),
        approver=AutoApprover(),
        sink=NullSink(),
        commands=stream,
    )
    assert ["commands" in extra for extra in seen] == [False, False, False, True]
    assert seen[3]["commands"] == {"vx": 0.4}
    assert [step.commands for step in record.steps] == [None, None, None, {"vx": 0.4}]
    assert stream.active


def test_declared_streams_are_active_from_the_start() -> None:
    assert CommandStream([AxisSpec("vx")]).active
    assert CommandStream(events=["go"]).active


def test_a_silent_wireless_pad_returns_its_sticks_to_rest() -> None:
    clock = Clock()
    stream = CommandStream([AxisSpec("vx", timeout_s=None)], clock=clock)
    pad = JoystickSource(axes=[AxisBinding("vx", axis=1)], max_silence_s=1.0, clock=clock)
    pad.handle(stream, 32767, 0x02, 1)
    assert stream.peek()["vx"] == pytest.approx(1.0)
    clock.now = 0.9
    pad._publish(stream)  # a republish inside the silence limit keeps the value
    assert stream.peek()["vx"] == pytest.approx(1.0)
    clock.now = 1.5  # no event from the pad for longer than max_silence_s
    pad._publish(stream)
    assert stream.peek()["vx"] == 0.0
    pad.handle(stream, 32767, 0x02, 1)  # it speaks again
    assert stream.peek()["vx"] == pytest.approx(1.0)


def test_end_episode_ends_the_trial_even_when_the_queue_dropped_it() -> None:
    from abstract_harness.robot.core import command_stream

    stream = CommandStream()

    class Flooding(CommandPolicy):
        def act(self, observation: Observation) -> ActionChunk:
            for _ in range(command_stream.MAX_QUEUED_EVENTS + 5):
                stream.press("noise")
            stream.press(END_EPISODE)
            return ActionChunk(actions=[Action(data=np.zeros(2))])

    record = rollout(
        Flooding(),  # type: ignore[arg-type]
        CubePickEmbodiment(),
        Scene(id="s", instruction="x", init_seed=0),
        max_steps=20,
        seed=0,
        epoch=0,
        controller=DefaultController(),
        approver=AutoApprover(),
        sink=NullSink(),
        commands=stream,
    )
    assert stream.dropped_events > 0
    assert record.termination_reason == OPERATOR_END and len(record.steps) == 1
