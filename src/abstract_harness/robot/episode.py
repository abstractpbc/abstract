"""What one robot run was, so it can be found again in the bucket: labels and markers.

``card(run_dir, rig)`` reads what ``abstract robot run`` left in a run directory (the
eval log, the run or meter summary, the action log and, for an agent, its transcript) and
returns the run's episode card: the instruction, the policy that acted, how it ended and
what the operator said, what each part of the robot was commanded to do, timing, cost,
versions and tags. Its ``markers`` are the steps where something happened: an operator message, an
agent tool call (with the agent's note, up to the next call), a gripper closing or
opening, the end. A step indexes the action log and the ``step`` timeline of the run's
Rerun recording, so a marker finds its camera images there.

The upload stores the card in the run as ``episode.json`` and, without the markers, as one
line of ``<prefix>/episodes.jsonl``, the bucket's catalog of runs. Motion labels describe
the commanded actions; measured joint state is in the run's Rerun recording.

No robot is assumed. The parts of the robot (a bimanual YAM's two arms, a Franka's one, a
mobile base) and the control mode come from the embodiment's spaces in the eval log, which
runs record since the formats; a run from before them is read by its action labels, as YAM
runs always were. The card's ``arms_moved``, ``displacement``, ``gripper_closes`` and the
gripper markers' ``arm`` keep their names for the stored runs and hold parts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

EPISODE = "episode.json"
CATALOG = "episodes.jsonl"
OUTCOMES = ("success", "partial", "failure", "skipped", "error", "unscored")
# The operator console's verdicts (y, n, partial, skip), by first letter.
VERDICTS = {"y": "success", "p": "partial", "n": "failure", "s": "skipped"}
# Gripper actions run from 0 (closed) to 1 (open). The gap between the two keeps a VLA's
# jitter around the middle from reading as many closes.
GRIPPER_CLOSED_BELOW = 0.4
GRIPPER_OPEN_ABOVE = 0.6
# The smallest commanded displacement that counts as a part moving: radians on any joint
# in joint control, metres on x, y or z in cartesian control.
MOVED_RAD = 0.05
MOVED_M = 0.01
# A robot without declared parts and without a symmetric <part>_<name> layout is one part.
SINGLE_PART = "robot"
# A cartesian position axis by its name within a part: x, or a per-step dx or delta_x.
POSITION_AXIS = re.compile(r"(?:d|delta_)?[xyz]")
CARTESIAN_MODES = frozenset({"eef_abs_pose", "eef_delta_pose", "eef_delta_pos"})
DELTA_MODES = frozenset({"joint_delta", "eef_delta_pose", "eef_delta_pos"})
STEP = re.compile(r"\bstep (\d+)/\d+")
TAG = re.compile(r"[a-z0-9][a-z0-9._:=+-]{0,63}")
OVERRIDES = frozenset({"policy", "outcome"})


def tag(value: str) -> str:
    """A tag as stored: lower case, 1 to 64 of a-z 0-9 . _ : = + -, no spaces."""
    normal = value.strip().lower()
    if not TAG.fullmatch(normal):
        raise ValueError(f"tag {value!r}: use 1-64 of a-z 0-9 . _ : = + - (no spaces)")
    return normal


def slug(text: str) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", text.lower()))[:80].rstrip("-")


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _lines(path: Path) -> Iterable[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            yield row


def eval_log(run_dir: Path) -> dict[str, Any]:
    """The run's eval log: the top-level JSON with ``eval`` and ``samples``."""
    for path in sorted(run_dir.glob("*.json")):
        data = _load(path)
        if isinstance(data, dict) and isinstance(data.get("eval"), dict):
            if isinstance(data.get("samples"), list):
                return data
    return {}


def _first(values: Any) -> Any:
    return values[0] if isinstance(values, list) and values else None


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""


def agent_markers(transcript: Path) -> list[dict[str, Any]]:
    """One marker per agent tool call, at the step of the observation it answered."""
    markers: list[dict[str, Any]] = []
    step = 0
    for message in _lines(transcript):
        if message.get("role") == "user" and (found := STEP.search(_text(message.get("content")))):
            step = int(found.group(1))
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except ValueError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            marker: dict[str, Any] = {"t": step, "kind": "tool_call", "tool": function.get("name")}
            if note := arguments.pop("note", None):
                marker["note"] = note
            if arguments:
                marker["args"] = arguments
            markers.append(marker)
    return markers


def parts(labels: list[str], declared: list[str] | None = None) -> dict[str, list[int]]:
    """Each part of the robot and its action columns, in order of first appearance: as the
    embodiment declares them (``ActionSemantics.dim_parts``), else the shared prefix of a
    symmetric layout (``left_j0``..``left_gripper`` and ``right_j0``..``right_gripper``: one
    part per side), else one part, ``robot``. A prefix alone proves nothing: a UR's
    ``shoulder_pan`` and ``wrist_1`` are joints of one arm."""
    groups: dict[str, list[int]] = {}
    if declared is not None and len(declared) == len(labels):
        for index, part in enumerate(declared):
            groups.setdefault(str(part), []).append(index)
        return groups
    split = [label.split("_", 1) for label in labels]
    if all(len(pair) == 2 for pair in split):
        for index, (prefix, _) in enumerate(split):
            groups.setdefault(prefix, []).append(index)
        names = {tuple(split[i][1] for i in columns) for columns in groups.values()}
        if len(groups) >= 2 and len(names) == 1:
            return groups
    return {SINGLE_PART: list(range(len(labels)))}


def _name(label: str, part: str) -> str:
    """A dimension's name within its part: ``left_gripper`` in part ``left`` is ``gripper``."""
    return label[len(part) + 1 :] if label.startswith(f"{part}_") else label


def _travel(
    rows: list[tuple[int, list[float]]], columns: list[int], mode: str | None, dt: float | None
) -> float | None:
    """The farthest any of ``columns`` was commanded from where it started: the change from
    the first action for absolute targets, the running sum of per-step changes, the running
    sum of velocities times ``dt``. None for velocities without a control rate."""
    if mode == "joint_vel" and dt is None:
        return None
    first, sums, farthest = rows[0][1], [0.0] * len(columns), 0.0
    for _, action in rows:
        for k, i in enumerate(columns):
            if mode in DELTA_MODES:
                sums[k] += action[i]
                value = sums[k]
            elif mode == "joint_vel":
                sums[k] += action[i] * (dt or 0.0)
                value = sums[k]
            else:
                value = action[i] - first[i]
            farthest = max(farthest, abs(value))
    return farthest


def motion(
    actions: Path, spaces: Mapping[str, Any] | None = None, control_hz: float | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Labels and gripper markers from the action log (one commanded action per step), per
    part of the robot (see the module docstring). ``spaces`` is the eval log's
    ``embodiment_info.spaces``; without it, labels decide, as for YAM runs before it."""
    labels: list[str] = []
    rows: list[tuple[int, list[float]]] = []
    for row in _lines(actions):
        if row.get("kind") == "header":
            labels = [str(label) for label in row.get("labels") or []]
        elif isinstance(row.get("t"), int) and isinstance(row.get("action"), list):
            rows.append((row["t"], row["action"]))
    if not labels or not rows:
        return {}, []
    action = (spaces or {}).get("action") or {}
    mode = action.get("control_mode")
    cartesian = mode in CARTESIAN_MODES if mode else any(label.endswith("_x") for label in labels)
    layout = parts(labels, action.get("parts"))
    dt = 1 / control_hz if control_hz else None
    out: dict[str, Any] = {
        "control_interface": "eef_pos" if cartesian else "joints",
        "arms_moved": [],
        "displacement": {},
        "gripper_closes": {},
    }
    markers: list[dict[str, Any]] = []
    first = rows[0][1]
    for part, columns in layout.items():
        names = {i: _name(labels[i], part) for i in columns}
        grippers = [i for i in columns if "gripper" in names[i]]
        if action.get("gripper") == "none":
            grippers = []
        if cartesian:
            position = [i for i in columns if POSITION_AXIS.fullmatch(names[i])]
        else:
            position = [i for i in columns if i not in grippers]
        farthest = _travel(rows, position, mode, dt)
        if farthest is not None:
            out["displacement"][part] = round(farthest, 3)
            if farthest > (MOVED_M if cartesian else MOVED_RAD):
                out["arms_moved"].append(part)
        if not grippers:
            continue
        # Gripper actions run from 0 (closed) to 1 (open), the core's convention.
        column, closes = grippers[0], 0
        closed = first[column] < GRIPPER_CLOSED_BELOW
        for t, values in rows:
            if not closed and values[column] < GRIPPER_CLOSED_BELOW:
                closed, closes = True, closes + 1
                markers.append({"t": t, "kind": "gripper_close", "arm": part, "part": part})
            elif closed and values[column] > GRIPPER_OPEN_ABOVE:
                closed = False
                markers.append({"t": t, "kind": "gripper_open", "arm": part, "part": part})
        out["gripper_closes"][part] = closes
    out.update(
        parts=list(layout),
        displacement_unit="m" if cartesian else "rad",
        control_mode=mode,
    )
    return out, markers


def outcome(sample: Mapping[str, Any], log: Mapping[str, Any]) -> tuple[str, str | None]:
    """(outcome, the operator's verdict as typed)."""
    verdict = _first(sample.get("operator_judgements"))
    if sample.get("error") or sample.get("status") == "error" or log.get("status") == "error":
        return "error", verdict
    if isinstance(verdict, str) and verdict[:1].lower() in VERDICTS:
        return VERDICTS[verdict[:1].lower()], verdict
    return "unscored", verdict


def card(run_dir: Path, rig: str) -> dict[str, Any]:
    """The episode card of one run directory (see the module docstring)."""
    log = eval_log(run_dir)
    spec, stats = log.get("eval") or {}, log.get("stats") or {}
    sample = _first(log.get("samples")) or {}
    trial = _first(sample.get("trial_metadata")) or {}
    meter = _load(run_dir / "meter-summary.json")
    summary = meter if isinstance(meter, dict) else _load(run_dir / "run-summary.json")
    summary = summary if isinstance(summary, dict) else {}
    agent = isinstance(meter, dict) or spec.get("policy") == "agent"
    policy_config = spec.get("policy_config") or {}
    result, verdict = outcome(sample, log)
    steps = stats.get("total_steps")
    instruction = sample.get("instruction")
    out: dict[str, Any] = {
        "run": run_dir.name,
        "rig": rig,
        "instruction": instruction,
        "task": slug(instruction) if isinstance(instruction, str) else None,
        "policy": (summary.get("model") or policy_config.get("model")) if agent else None,
        "kind": "agent" if agent else "vla",
        "client": "agent" if agent else summary.get("policy"),
        "config": summary.get("config"),
        "outcome": result,
        "score": (sample.get("reduced") or {}).get("operator"),
        "verdict": verdict,
        "note": _first(sample.get("operator_notes")),
        "termination": _first(sample.get("termination_reasons")),
        "error": sample.get("error") or log.get("error"),
        "exit_code": summary.get("exit_code"),
        "started_at": stats.get("started_at"),
        "ended_at": stats.get("completed_at"),
        "duration_s": round(stats["duration_s"], 1) if stats.get("duration_s") else None,
        "steps": steps,
        "control_hz": (spec.get("embodiment_info") or {}).get("control_hz"),
    }
    if agent:
        out.update(
            effort=policy_config.get("effort"),
            spent_usd=float(summary["spent_usd"]) if summary.get("spent_usd") else None,
            model_calls=summary.get("model_calls"),
            hindsight=trial.get("hindsight"),
        )
    else:
        # The /act client's `name` argument labels the checkpoint (isaac-0.5 behind a
        # molmoact2-style server); without it, the eval log carries the client's name.
        out.update(
            policy=spec.get("policy") or summary.get("policy"),
            server_url=summary.get("server_url"),
            inference_latency_s=stats.get("mean_inference_latency_s"),
        )
    markers: list[dict[str, Any]] = []
    if isinstance(trial.get("actions"), str):
        info = spec.get("embodiment_info") or {}
        labels, gripper = motion(
            run_dir / trial["actions"], info.get("spaces"), info.get("control_hz")
        )
        out.update(labels)
        markers += gripper
    if agent and isinstance(trial.get("transcript"), str):
        markers += agent_markers(run_dir / trial["transcript"])
    for message in _first(sample.get("operator_messages")) or []:
        if isinstance(message, dict):
            markers.append(
                {
                    "t": message.get("t"),
                    "kind": "operator_message",
                    "text": message.get("text"),
                    "source": message.get("source"),
                }
            )
    last = max(steps - 1, 0) if isinstance(steps, int) else 0
    end: dict[str, Any] = {"t": last, "kind": "end", "outcome": result}
    for key in ("termination", "error", "note"):
        if out.get(key):
            end[key] = out[key]
    markers.append(end)
    markers.sort(key=lambda marker: marker["t"] if isinstance(marker.get("t"), int) else last)
    calls = [marker for marker in markers if marker["kind"] == "tool_call"]
    for i, call in enumerate(calls):
        call["until"] = calls[i + 1]["t"] if i + 1 < len(calls) else last + 1
    out.update(
        git_commit=spec.get("git_commit"),
        # Runs from before the episode code moved into this package have none: they ran
        # on the packages the commit pins.
        harness_version=spec.get("harness_version"),
        tags=sorted(
            {
                value.strip().lower()
                for value in summary.get("tags") or []
                if isinstance(value, str) and TAG.fullmatch(value.strip().lower())
            }
        ),
        overrides={},
        markers=markers,
    )
    return out


def edit(
    episode: dict[str, Any],
    tags: Iterable[str],
    overrides: Mapping[str, str],
) -> dict[str, Any]:
    """Set an episode's tags and apply label overrides (a corrected policy name, or an
    outcome decided on review); the overrides are kept so a rebuilt card keeps them."""
    unknown = set(overrides) - OVERRIDES
    if unknown:
        raise ValueError(f"unknown label overrides: {', '.join(sorted(unknown))}")
    if overrides.get("outcome") not in (None, *OUTCOMES):
        raise ValueError(f"outcome must be one of {', '.join(OUTCOMES)}")
    episode["tags"] = sorted({tag(value) for value in tags})
    episode["overrides"] = dict(sorted(overrides.items()))
    episode.update(overrides)
    return episode


def row(episode: Mapping[str, Any]) -> dict[str, Any]:
    """The card without its markers: one catalog line."""
    out = {key: value for key, value in episode.items() if key != "markers"}
    out["markers"] = len(episode.get("markers") or [])
    return out


def dumps(episode: Mapping[str, Any]) -> bytes:
    return (json.dumps(episode, indent=2) + "\n").encode()
