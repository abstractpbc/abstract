"""run_task without grading (`abstract robot run --no-verdict`): the run stays attended,
nobody is asked for the verdict or grader notes, and the trial stays unscored."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from abstract_harness.robot import episode
from abstract_harness.robot.core import commands


class FakeTty(io.StringIO):
    """A terminal on stdin with no descriptor behind it: attended, no raw input."""

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        raise io.UnsupportedOperation("no descriptor behind this terminal")


@pytest.mark.parametrize("grade", [True, False])
def test_without_grading_nobody_is_asked_and_the_trial_stays_unscored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, grade: bool
) -> None:
    monkeypatch.setattr(sys, "stdin", FakeTty())
    asked: list[str] = []
    answers = ["y", ""]
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or answers.pop(0))
    parts = commands.components("scripted", {}, "cubepick", {}, "test")
    task = commands.adhoc_task("reach the cube", 3)  # ends at the step limit
    log_dir = tmp_path / "logs"
    code = commands.run_task(task, parts, log_dir=str(log_dir), rerun_save=False, grade=grade)
    assert code == 0
    (log_path,) = log_dir.glob("*.json")
    log = json.loads(log_path.read_text())
    if grade:
        assert any("did the robot succeed" in prompt for prompt in asked)
        assert episode.outcome(log["samples"][0], log) == ("success", "y")
    else:
        assert asked == []
        assert episode.outcome(log["samples"][0], log) == ("unscored", None)
