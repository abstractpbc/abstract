"""Run with a fresh wheel-installed Python from outside the source checkout."""

from __future__ import annotations

import importlib.resources
import subprocess
import sys
import tempfile
from pathlib import Path

import abstract_harness
from abstract_harness import Scene, Task, eval, success_at_end

source = Path(__file__).resolve().parents[1] / "src"
installed = Path(abstract_harness.__file__).resolve()
assert not installed.is_relative_to(source), installed
assert importlib.resources.files("abstract_harness").joinpath("py.typed").is_file()
assets = importlib.resources.files("abstract_harness.robot.yam").joinpath("assets")
assert "<mujoco" in assets.joinpath("yam_collision.xml").read_text()
assert "Permission is hereby granted" in assets.joinpath("MENAGERIE_LICENSE").read_text()
with tempfile.TemporaryDirectory() as directory:
    logs = eval(
        Task(
            name="wheel-smoke",
            scenes=[Scene(id="s", instruction="reach")],
            scorer=success_at_end(),
            max_steps=2,
        ),
        policy="scripted",
        embodiment="cubepick",
        log_dir=directory,
    )
    assert len(logs) == 1 and logs[0].status == "success", logs
    subprocess.run(
        [sys.executable, "-I", "-c", "from abstract_harness.cli import main; main(['--help'])"],
        cwd=directory,
        check=True,
        stdout=subprocess.DEVNULL,
    )
print(f"installed-wheel core evaluation and CLI passed: {installed}")
