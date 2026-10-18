"""robot/events.py: the run's events for its supervisor, off unless asked for."""

import json
import os
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from abstract_harness.robot import events


@contextmanager
def captured():
    """Events sent inside the block, read from a real pipe as serve reads them."""
    read_fd, write_fd = os.pipe()
    got = []

    def read():
        with os.fdopen(read_fd, "rb") as pipe:
            got.extend(json.loads(line) for line in pipe)

    reader = threading.Thread(target=read)
    reader.start()
    try:
        with events.channel(write_fd):
            yield got
    finally:
        reader.join(10)


def kinds(got):
    return [event["type"] for event in got]


def test_off_unless_a_channel_is_open():
    assert not events.active()
    events.emit("status", text="nobody listens")  # a no-op, not an error
    with events.channel(None):
        assert not events.active()
        events.emit("status", text="still nobody")
    with events.dialog("gate", "press Enter") as dialog:
        dialog.answer = "confirmed"


def test_a_channel_says_hello_then_sends_events_in_order():
    with captured() as got:
        assert events.active()
        events.emit("log_dir", path="/logs/x")
        events.emit("status", text="homing: ramping arms to start pose")
        events.emit("transcript", message={"role": "assistant", "content": object()})
    assert not events.active()
    hello, log_dir, status, transcript = got
    assert (hello["type"], hello["version"], hello["pid"]) == ("hello", 1, os.getpid())
    assert log_dir == {"t": log_dir["t"], "type": "log_dir", "path": "/logs/x"}
    assert abs(status["t"] - time.time()) < 60
    assert transcript["message"]["content"].startswith("<object")  # anything becomes text


def test_a_refusal_goes_out_as_an_error_and_still_raises():
    with pytest.raises(ValueError, match="pass --allow-paid"), captured() as got:
        raise ValueError("robot run calls a paid model; pass --allow-paid")
    assert kinds(got) == ["hello", "error"]
    assert got[1]["message"] == "robot run calls a paid model; pass --allow-paid"


def test_dialogs_report_how_they_ended_never_what_was_typed():
    with captured() as got:
        with events.dialog("gate", "Position the scene, then press Enter to start...") as gate:
            gate.answer = "confirmed"
        with pytest.raises(KeyboardInterrupt), events.dialog("gate", "stand clear"):
            raise KeyboardInterrupt
        with pytest.raises(EOFError), events.dialog("verdict", "did it? ", ("y", "n")):
            raise EOFError
    opened = [event for event in got if event["type"] == "dialog"]
    resolved = [event for event in got if event["type"] == "dialog_resolved"]
    assert opened[0]["title"] == "Position the scene, then press Enter to start..."
    assert opened[2]["title"] == "did it?" and opened[2]["options"] == ["y", "n"]
    assert len({event["id"] for event in opened}) == 3
    assert [(event["id"], event["answer"]) for event in resolved] == [
        (opened[0]["id"], "confirmed"),
        (opened[1]["id"], "interrupted"),
        (opened[2]["id"], "failed"),
    ]


@pytest.mark.parametrize("fd", [0, 1, 2])
def test_the_pipe_is_never_a_standard_stream(fd):
    with pytest.raises(ValueError, match="not stdin, stdout or stderr"), events.channel(fd):
        pass


def test_a_closed_descriptor_is_a_clear_refusal():
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    os.close(write_fd)
    with pytest.raises(ValueError, match=f"--events-fd {write_fd}"), events.channel(write_fd):
        pass


def test_a_reader_that_stops_reading_never_blocks_the_run(monkeypatch):
    monkeypatch.setattr(events, "MAX_QUEUED", 8)
    monkeypatch.setattr(events, "CLOSE_TIMEOUT_S", 0.2)
    read_fd, write_fd = os.pipe()  # nobody reads: the pipe fills after ~64 KiB
    try:
        started = time.monotonic()
        with events.channel(write_fd):
            for step in range(2000):
                events.emit("progress", step=step, padding="x" * 1000)
        assert time.monotonic() - started < 3
    finally:
        os.close(read_fd)


def test_the_event_sink_reports_steps_transcript_and_the_trial_end():
    sink = events.EventSink()
    record = SimpleNamespace(
        status="success", termination_reason="operator_end", operator_judgement="y"
    )
    with captured() as got:
        sink.on_eval_start(None)
        sink.on_trial_start("scene-0", 0)
        sink.log_step(0, None, None, None)
        sink.log_policy_messages(1, [{"role": "assistant", "content": "moving"}])
        sink.log_step(1, None, None, None)
        sink.on_trial_end(record)
        sink.on_eval_end(None)
    assert [{k: v for k, v in event.items() if k != "t"} for event in got[1:]] == [
        {"type": "progress", "trial": "scene-0-e0", "step": 1},
        {
            "type": "transcript",
            "trial": "scene-0-e0",
            "step": 1,
            "message": {"role": "assistant", "content": "moving"},
        },
        {"type": "progress", "trial": "scene-0-e0", "step": 2},
        {
            "type": "trial_end",
            "trial": "scene-0-e0",
            "status": "success",
            "termination_reason": "operator_end",
            "verdict": "y",
        },
    ]
