"""A finished run directory, read as robot-neutral episodes for the format writers.

Nothing here names a robot. The layout comes from the embodiment's own description,
which the eval log keeps under ``eval.embodiment_info.spaces`` (``core/eval.py``): the
action's shape and labels, the cameras with their sizes, and the state fields in their
declared order. Each trial is one episode. Its rows come from the trial's action log
(``actions/<stamp>/<trial>.jsonl``): at step ``t``, the state the policy saw and the
action it chose. Its images are the raw frames the episode stored for that step
(``frames/<stamp>/<trial>_<camera>_<t:06d>.npy``); the frame stored after the last
action has no row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from abstract_harness.robot import episode as cards


class RunError(ValueError):
    """The run directory lacks something a format needs."""


@dataclass(frozen=True)
class Camera:
    name: str
    height: int
    width: int
    channels: int = 3


@dataclass(frozen=True)
class StateField:
    key: str
    size: int


@dataclass
class Episode:
    scene_id: str
    epoch: int
    task: str
    actions: list[list[float]]
    # Per state key, one row of floats (None where the value was not finite) per step.
    states: dict[str, list[list[float | None]]]
    # Per camera, the frame of each step, in step order.
    frames: dict[str, list[Path]]

    @property
    def length(self) -> int:
        return len(self.actions)


@dataclass
class Run:
    run_dir: Path
    embodiment: str
    fps: float
    action_labels: list[str]
    state_fields: list[StateField]
    cameras: list[Camera]
    episodes: list[Episode] = field(default_factory=list)


def _rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("kind") == "header":
            header = row
        else:
            rows.append(row)
    return header, rows


def read(run_dir: Path) -> Run:
    """The run's episodes and layout. Raises RunError for what a format cannot do without:
    an eval log with the embodiment's spaces, action logs with state, every frame."""
    log = cards.eval_log(run_dir)
    spec = log.get("eval") or {}
    info = spec.get("embodiment_info") or {}
    spaces = info.get("spaces")
    if not isinstance(spaces, dict):
        raise RunError("the eval log has no embodiment spaces (runs before formats lack them)")
    fps = info.get("control_hz")
    if not isinstance(fps, int | float) or fps <= 0:
        raise RunError(f"the embodiment's control rate is {fps!r}, not a positive number")
    action = spaces.get("action") or {}
    dim = 1
    for n in action.get("shape") or []:
        dim *= int(n)
    labels = action.get("labels") or [f"action.{i}" for i in range(dim)]
    cameras = [Camera(**camera) for camera in spaces.get("cameras") or []]
    declared = [str(item["key"]) for item in spaces.get("state") or []]
    run = Run(run_dir, str(spec.get("embodiment") or "unknown"), float(fps), labels, [], cameras)
    try:
        # The frame store's names; core needs numpy, which comes with the robot extra.
        from abstract_harness.robot.core.frames import _safe
    except ImportError as exc:
        raise RunError(f"the robot extra is missing: {exc}") from None
    sizes: dict[str, int] = {}
    for sample in log.get("samples") or []:
        task = sample.get("instruction") or ""
        for trial in sample.get("trial_metadata") or []:
            relative = (trial or {}).get("actions")
            if not isinstance(relative, str):
                raise RunError("a trial has no action log")
            header, rows = _rows(run_dir / relative)
            if any("state" not in row for row in rows):
                raise RunError(f"{relative} has no state (runs before formats lack it)")
            if [row["t"] for row in rows] != list(range(len(rows))):
                raise RunError(f"{relative}: steps are not 0, 1, 2, ... in order")
            # A thermal trip can end a trial without issuing its predicted action.
            # Preserve the forensic row, but never train an action on that transition.
            rows = [row for row in rows if row.get("action_executed", True)]
            if not rows:
                continue
            scene, epoch, stamp = header["scene_id"], int(header["epoch"]), header["run_id"]
            states: dict[str, list[list[float | None]]] = {}
            for key in declared or sorted({key for row in rows for key in row["state"]}):
                values = [row["state"].get(key) for row in rows]
                if any(value is None for value in values):
                    raise RunError(f"{relative}: state {key!r} is missing at some steps")
                size = sizes.setdefault(key, len(values[0]))
                if any(len(value) != size for value in values):
                    raise RunError(f"{relative}: state {key!r} changes size")
                states[key] = values
            prefix = run_dir / "frames" / stamp / _safe(f"{scene}-e{epoch}")
            frames: dict[str, list[Path]] = {}
            for camera in cameras:
                paths = [Path(f"{prefix}_{_safe(camera.name)}_{row['t']:06d}.npy") for row in rows]
                missing = [path.name for path in paths if not path.is_file()]
                if missing:
                    raise RunError(f"{len(missing)} frames missing, first {missing[0]}")
                frames[camera.name] = paths
            actions = [
                [
                    float(x)
                    for x in (
                        row.get("command", {}).get("joint_pos", row["action"])
                        if action.get("control_mode") == "joint_pos"
                        else row["action"]
                    )
                ]
                for row in rows
            ]
            if any(len(row) != len(labels) for row in actions):
                raise RunError(f"{relative}: an action is not {len(labels)} values")
            run.episodes.append(Episode(scene, epoch, task, actions, states, frames))
    if not run.episodes:
        raise RunError("the run recorded no trials")
    run.state_fields = [StateField(key, size) for key, size in sizes.items()]
    return run
