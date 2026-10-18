"""The episode card's motion labels, per part of whatever robot ran: a bimanual YAM keeps
its labels, and a Franka, a UR, a mobile manipulator or the cubepick sim get theirs."""

import json

import pytest

from abstract_harness.robot import episode

YAM = [
    f"{side}_{name}"
    for side in ("left", "right")
    for name in (*(f"j{i}" for i in range(6)), "gripper")
]
STILL = [0.0] * 6 + [1.0]


def log(tmp_path, labels, actions):
    path = tmp_path / "actions.jsonl"
    rows = [{"kind": "header", "labels": labels}]
    rows += [{"t": t, "action": action} for t, action in enumerate(actions)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def spaces(mode, parts=None, gripper="continuous"):
    return {"action": {"control_mode": mode, "parts": parts, "gripper": gripper}}


def yam_actions():
    # The right arm turns 0.3 rad and its gripper closes, then opens.
    return [STILL + STILL, STILL + [0.3, *[0.0] * 5, 0.1], STILL + [0.3, *[0.0] * 5, 0.9]]


def test_a_yam_keeps_its_labels_with_or_without_the_layout(tmp_path):
    path = log(tmp_path, YAM, yam_actions())
    before, before_markers = episode.motion(path)
    declared = ["left"] * 7 + ["right"] * 7
    after, after_markers = episode.motion(path, spaces("joint_pos", declared), 30.0)
    for labels in (before, after):
        assert labels["control_interface"] == "joints"
        assert labels["arms_moved"] == ["right"]
        assert labels["displacement"] == {"left": 0.0, "right": 0.3}
        assert labels["gripper_closes"] == {"left": 0, "right": 1}
        assert labels["parts"] == ["left", "right"] and labels["displacement_unit"] == "rad"
    assert (before["control_mode"], after["control_mode"]) == (None, "joint_pos")
    assert (
        before_markers
        == after_markers
        == [
            {"t": 1, "kind": "gripper_close", "arm": "right", "part": "right"},
            {"t": 2, "kind": "gripper_open", "arm": "right", "part": "right"},
        ]
    )


def test_a_yam_in_cartesian_control_measures_metres_on_x_y_z(tmp_path):
    labels = [
        f"{side}_{name}"
        for side in ("left", "right")
        for name in ("x", "y", "z", "yaw", "pitch", "roll", "gripper")
    ]
    pose = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
    turned = [0.3, 0.0, 0.2, 1.5, 0.0, 0.0, 1.0]  # yaw only: not a move in metres
    moved = [0.3, 0.05, 0.2, 0.0, 0.0, 0.0, 1.0]
    path = log(tmp_path, labels, [pose + pose, turned + moved])
    for given in (None, spaces("eef_abs_pose", ["left"] * 7 + ["right"] * 7)):
        labels_out, _ = episode.motion(path, given)
        assert labels_out["control_interface"] == "eef_pos"
        assert labels_out["displacement"] == {"left": 0.0, "right": 0.05}
        assert labels_out["arms_moved"] == ["right"] and labels_out["displacement_unit"] == "m"


def test_a_franka_is_one_part(tmp_path):
    labels = [f"j{i}" for i in range(7)] + ["gripper"]
    start = [0.0] * 7 + [1.0]
    path = log(tmp_path, labels, [start, [0.0, 0.2, *[0.0] * 5, 0.2], [0.0, 0.4, *[0.0] * 5, 0.9]])
    labels_out, markers = episode.motion(path, spaces("joint_pos"), 15.0)
    assert labels_out["parts"] == ["robot"]
    assert labels_out["displacement"] == {"robot": 0.4} and labels_out["arms_moved"] == ["robot"]
    assert labels_out["gripper_closes"] == {"robot": 1}
    assert [(m["t"], m["kind"], m["part"]) for m in markers] == [
        (1, "gripper_close", "robot"),
        (2, "gripper_open", "robot"),
    ]


def test_a_ur_without_gripper_is_one_arm_not_three(tmp_path):
    labels = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    path = log(tmp_path, labels, [[0.0] * 6, [0.0, 0.0, 0.0, 0.02, 0.0, 0.0]])
    labels_out, markers = episode.motion(path, spaces("joint_pos", gripper="none"))
    assert labels_out["parts"] == ["robot"] and labels_out["arms_moved"] == []
    assert labels_out["displacement"] == {"robot": 0.02} and labels_out["gripper_closes"] == {}
    assert markers == []
    # Even labelled with underscores everywhere, an asymmetric layout stays one part.
    assert episode.parts(["shoulder_pan", "shoulder_lift", "wrist_1"]) == {"robot": [0, 1, 2]}


def test_declared_parts_split_a_mobile_manipulator(tmp_path):
    labels = ["vx", "vy", "wz", *(f"j{i}" for i in range(7)), "gripper"]
    declared = ["base"] * 3 + ["arm"] * 8
    actions = [[0.5, 0.0, 0.0, *[0.0] * 7, 1.0] for _ in range(4)]
    path = log(tmp_path, labels, actions)
    labels_out, _ = episode.motion(path, spaces("joint_vel", declared), 10.0)
    assert labels_out["parts"] == ["base", "arm"]
    # Velocities integrated at 10 Hz: 4 steps of 0.5 for 0.1 s.
    assert labels_out["displacement"] == {"base": 0.2, "arm": 0.0}
    assert labels_out["arms_moved"] == ["base"] and labels_out["gripper_closes"] == {"arm": 0}
    # Without a control rate a velocity cannot be integrated, so no displacement is claimed.
    no_rate, _ = episode.motion(path, spaces("joint_vel", declared))
    assert no_rate["displacement"] == {} and no_rate["arms_moved"] == []


def test_per_step_changes_add_up(tmp_path):
    path = log(tmp_path, ["dx", "dy"], [[0.004, 0.0]] * 5)
    labels_out, _ = episode.motion(path, spaces("eef_delta_pos", gripper="none"))
    assert labels_out["displacement"] == {"robot": 0.02} and labels_out["arms_moved"] == ["robot"]
    assert labels_out["displacement_unit"] == "m"


def test_a_symmetric_layout_is_split_by_side_without_a_declaration():
    labels = [f"{side}_{name}" for side in ("left", "right") for name in ("j0", "j1", "gripper")]
    assert episode.parts(labels) == {"left": [0, 1, 2], "right": [3, 4, 5]}
    # A declaration of the wrong length is ignored rather than trusted.
    assert episode.parts(labels, ["arm"] * 5) == {"left": [0, 1, 2], "right": [3, 4, 5]}


def test_a_cubepick_card_reads_the_layout_its_run_recorded(tmp_path):
    pytest.importorskip("numpy")
    from abstract_harness.robot.core import eval
    from abstract_harness.robot.core.mock import CubePickEmbodiment, ScriptedPolicy
    from abstract_harness.robot.core.scene import Scene
    from abstract_harness.robot.core.scorer import success_at_end
    from abstract_harness.robot.core.task import Task

    task = Task(
        name="reach",
        scenes=[Scene(id="s0", instruction="reach the cube", init_seed=0)],
        scorer=success_at_end(),
        max_steps=12,
    )
    eval(task, ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    card = episode.card(tmp_path, "sim")
    assert (card["parts"], card["control_mode"], card["control_interface"]) == (
        ["robot"],
        "eef_delta_pos",
        "eef_pos",
    )
    assert card["displacement"]["robot"] > 0 and card["gripper_closes"] == {}
