"""The run's events from their sources: run_task, the operator session, the frame store,
the YAM gates and the episode, in process."""

from __future__ import annotations

import io
import json
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from abstract_harness.robot import events, stack
from abstract_harness.robot.config import RobotConfig
from abstract_harness.robot.core import commands
from abstract_harness.robot.core.errors import EmbodimentFault
from abstract_harness.robot.core.frames import FrameStore
from abstract_harness.robot.core.rollout import TrialRecord
from abstract_harness.robot.core.scene import Scene
from abstract_harness.robot.core.session import OperatorSession
from abstract_harness.robot.yam.operator import OperatorIO


@contextmanager
def captured() -> Iterator[list[dict[str, Any]]]:
    """Events sent inside the block, read from a real pipe as serve reads them."""
    read_fd, write_fd = os.pipe()
    got: list[dict[str, Any]] = []

    def read() -> None:
        with os.fdopen(read_fd, "rb") as pipe:
            got.extend(json.loads(line) for line in pipe)

    reader = threading.Thread(target=read)
    reader.start()
    try:
        with events.channel(write_fd):
            yield got
    finally:
        reader.join(10)


def of(got: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [event for event in got if event["type"] == kind]


def scripted_input(answers: list[str]) -> Any:
    prompts: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return answers.pop(0)

    answer.prompts = prompts  # type: ignore[attr-defined]
    return answer


def test_run_task_on_cubepick_reports_steps_frames_and_the_trial_end(tmp_path: Path) -> None:
    parts = commands.components("scripted", {}, "cubepick", {}, "test")
    task = commands.adhoc_task("reach the cube", 5, "success_at_end")
    with captured() as got:
        code = commands.run_task(
            task, parts, log_dir=str(tmp_path / "logs"), prompt=False, rerun_save=False
        )
    steps = [event["step"] for event in of(got, "progress")]
    assert steps == list(range(1, len(steps) + 1)) and 1 <= len(steps) <= 5
    frames = of(got, "frame")
    assert {event["camera"] for event in frames} == {"top"}
    assert [event["step"] for event in frames] == list(range(len(steps) + 1))
    for frame in frames:  # a whole frame on disk by the time its event is out
        assert Path(frame["path"]).is_absolute()
        assert np.load(frame["path"]).dtype == np.uint8
    (trial_end,) = of(got, "trial_end")
    assert trial_end["trial"] == "scene-0-e0" == frames[0]["trial"]
    assert trial_end["status"] == "success"
    assert code in (0, 1)
    assert of(got, "dialog") == []  # unattended: no prompt


def test_attended_run_task_asks_the_verdict_and_notes_as_dialogs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeTty(io.StringIO):
        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            raise io.UnsupportedOperation("no descriptor behind this terminal")

    monkeypatch.setattr(sys, "stdin", FakeTty())
    answer = scripted_input(["maybe", "y", "clean fold"])
    monkeypatch.setattr("builtins.input", answer)
    parts = commands.components("scripted", {}, "cubepick", {}, "test")
    task = commands.adhoc_task("reach the cube", 3)  # graded by the operator
    with captured() as got:
        assert commands.run_task(task, parts, log_dir=str(tmp_path / "logs"), rerun_save=False) == 0
    dialogs = [event for event in got if event["type"].startswith("dialog")]
    assert [(event["type"], event["kind"]) for event in dialogs] == [
        ("dialog", "verdict"),
        ("dialog_resolved", "verdict"),
        ("dialog", "verdict"),
        ("dialog_resolved", "verdict"),
        ("dialog", "notes"),
        ("dialog_resolved", "notes"),
    ]
    assert dialogs[0]["title"] == answer.prompts[0].strip()
    assert dialogs[0]["options"] == ["y", "n", "partial", "skip"]
    assert [event["answer"] for event in dialogs[1::2]] == ["unrecognized", "y", "text"]
    assert "clean fold" not in json.dumps(got)  # the note stays in the run's log
    assert of(got, "trial_end")[0]["verdict"] == "y"


def test_the_session_reports_gates_and_each_status_change() -> None:
    written: list[str] = []
    answer = scripted_input([""])
    session = OperatorSession(input_fn=answer, write=written.append, flush_fn=lambda: None)
    with captured() as got:
        session.status("homing: ramping arms to start pose")
        session.status("homing: ramping arms to start pose")  # redrawn, not news
        session.gate("Position the scene, then press Enter to start...")
        session.status("Running. Max ~120s.")
    shown = [(event["type"], event.get("text", event.get("answer"))) for event in got[1:]]
    assert shown == [
        ("status", "homing: ramping arms to start pose"),
        ("status", None),  # the gate closes the status line first
        ("dialog", None),
        ("dialog_resolved", "confirmed"),
        ("status", "Running. Max ~120s."),
    ]
    assert got[3]["kind"] == "gate"
    assert got[3]["title"] == "Position the scene, then press Enter to start..."
    assert answer.prompts == ["Position the scene, then press Enter to start..."]


def test_a_gate_without_a_terminal_fails_as_before_and_says_so() -> None:
    def closed(prompt: str) -> str:
        raise EOFError

    session = OperatorSession(input_fn=closed, write=lambda text: None, flush_fn=lambda: None)
    with captured() as got, pytest.raises(EmbodimentFault, match="readiness gate"):
        session.gate("Arms will move to the home pose - stand clear, then press Enter...")
    assert of(got, "dialog_resolved")[0]["answer"] == "failed"


def test_the_yam_operator_gate_is_a_dialog_too() -> None:
    answer = scripted_input([""])
    operator = OperatorIO(input_fn=answer, output_fn=lambda line: None, flush_fn=lambda: None)
    with captured() as got:
        operator.wait_ready("Position the scene, then press Enter to start...", drain=False)
    assert [(event["type"], event["kind"]) for event in got[1:]] == [
        ("dialog", "gate"),
        ("dialog_resolved", "gate"),
    ]
    assert got[2]["answer"] == "confirmed"


def test_a_plain_terminal_session_writes_the_same_bytes_with_or_without_a_channel() -> None:
    def session_output() -> list[str]:
        written: list[str] = []
        session = OperatorSession(
            input_fn=scripted_input(["", "n", ""]), write=written.append, flush_fn=lambda: None
        )
        session.status("homing: ramping arms to start pose")
        session.gate("Position the scene, then press Enter to start...")
        session.prompt_verdict(TrialRecord("scene-0", 0, 0), Scene("scene-0", "fold"))
        return written

    without = session_output()
    with captured():
        assert session_output() == without


def test_a_frame_event_names_the_saved_file(tmp_path: Path) -> None:
    store = FrameStore(str(tmp_path / "frames"))
    with captured() as got:
        ref = store.put("scene-0-e0", 7, "top_cam", np.zeros((2, 3, 3), dtype=np.uint8))
    (frame,) = of(got, "frame")
    assert frame["path"] == os.path.abspath(ref.path)
    assert (frame["trial"], frame["camera"], frame["step"]) == ("scene-0-e0", "top_cam", 7)


@pytest.mark.parametrize(
    ("outcome", "code"), [(0, 0), (3, 3), (KeyboardInterrupt(), 130), (SystemExit(2), 2)]
)
def test_the_episode_end_carries_the_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: object, code: int
) -> None:
    config = tmp_path / "robot.toml"
    config.write_text(
        '[vla]\nserver_url = "http://127.0.0.1:8202"\n'
        '[rig.embodiment]\nleft_channel = "can1"\nright_channel = "can0"\n'
    )

    def run_task(*args: object, **kwargs: object) -> int:
        if isinstance(outcome, BaseException):
            raise outcome
        return int(outcome)  # type: ignore[call-overload]

    monkeypatch.setattr(commands, "components", lambda *args: None)
    monkeypatch.setattr(commands, "run_task", run_task)
    with captured() as got:
        assert stack.episode(RobotConfig.load(config), "fold", tmp_path, source="x") == code
    assert [event["exit_code"] for event in of(got, "episode_end")] == [code]
