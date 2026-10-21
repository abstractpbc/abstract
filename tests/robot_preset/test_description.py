"""Preset resolution: joint selection, limits, frames, state, commands and controllers,
each refusal naming the field and the fix."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import numpy as np
import pytest

from abstract_harness.robot.core.command_stream import JoystickSource
from abstract_harness.robot.core.conformance import check_embodiment
from abstract_harness.robot.preset import PresetError, load_preset

ROBOTS = Path(__file__).parents[2] / "robots"
URDF = ROBOTS / "mobile-manipulator" / "mobile_manipulator.urdf"


def write(tmp_path: Path, body: str, *, model: bool = True) -> Path:
    if model:
        shutil.copy(URDF, tmp_path / "robot.urdf")
    header = '[robot]\nname = "toy"\npolicy_hz = 20\n'
    if model:
        header += 'model = "robot.urdf"\n'
    path = tmp_path / "preset.toml"
    path.write_text(header + textwrap.dedent(body))
    return path


def test_default_selection_is_every_movable_joint_with_model_limits(tmp_path: Path) -> None:
    with pytest.raises(PresetError, match="left_wheel_joint .continuous. has no position limits"):
        load_preset(write(tmp_path, ""))
    description = load_preset(
        write(tmp_path, '[robot.joints]\nlimits = { "*_wheel_joint" = [-1, 1] }\n')
    )
    assert description.labels[:2] == ("left_wheel_joint", "right_wheel_joint")
    assert description.dim == 9 and description.control_hz == 20.0


def test_select_orders_globs_and_exclude_removes(tmp_path: Path) -> None:
    description = load_preset(
        write(tmp_path, '[robot.joints]\nselect = ["claw_joint", "arm_j*"]\nexclude = ["arm_j6"]\n')
    )
    assert description.labels == ("claw_joint", "arm_j1", "arm_j2", "arm_j3", "arm_j4", "arm_j5")
    np.testing.assert_allclose(description.low[:2], [0.0, -2.9])


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('[robot.joints]\nselect = ["arm_j9"]\n', "matches no movable joint"),
        ('[robot.joints]\nselect = ["arm_jl"]\n', "did you mean"),
        ('[robot.joints]\nselect = ["imu_joint"]\n', "a fixed or unsupported joint"),
        ('[robot.joints]\nselect = ["arm_j1"]\nexclude = ["nope"]\n', "exclude 'nope'"),
        ('[robot.joints]\nselect = ["arm_j1"]\nexclude = ["arm_j1"]\n', "selects no joints"),
        (
            '[robot.joints]\nselect = ["arm_j1"]\nlimits = { arm_j1 = [-3.5, 1] }\n',
            "exceed the model's position limits",
        ),
        (
            '[robot.joints]\nselect = ["arm_j1"]\nmax_step = { arm_j1 = 9.0 }\n',
            "not smaller than its range",
        ),
        ('[robot.joints]\nselect = ["arm_j1"]\nhome = { arm_j1 = 4.0 }\n', "outside its limits"),
        ('[robot.joints]\nselect = ["arm_j1"]\nhome = { arm_j2 = 0.1 }\n', "matches no selected"),
        (
            '[robot.joints]\nselect = ["arm_j1"]\n[robot.parts]\narm = ["arm_j7"]\n',
            "matches no selected joint",
        ),
        (
            '[robot.joints]\nselect = ["arm_j1"]\n[[robot.end_effectors]]\nname = "claw"\n'
            'frame = "claw_bse_link"\n',
            "did you mean 'claw_base_link'",
        ),
        (
            '[robot.joints]\nselect = ["arm_j1"]\n[[robot.end_effectors]]\nname = "claw"\n'
            'frame = "claw_base_link"\njoints = ["claw_joint"]\n',
            "not a selected joint",
        ),
        (
            '[robot.joints]\nselect = ["arm_j1"]\n[robot.state]\nfields = ["joint_vel"]\n',
            "joint_pos",
        ),
        ('[robot.joints]\nselect = ["arm_j1"]\n[robot.state]\nfields = ["tau"]\n', "not known"),
        (
            '[robot.joints]\nselect = ["arm_j1"]\n[robot.base]\nmax_tilt_deg = 30\n',
            "max_tilt_deg needs projected_gravity",
        ),
        (
            '[robot.joints]\nselect = ["arm_j1"]\n'
            '[[robot.cameras]]\nname = "c"\nframe = "x"\nwidth = 4\nheight = 4\n',
            "frame 'x'",
        ),
        ('[robot.joints]\nselect = ["arm_j1"]\npolicy = 3\n', "Extra inputs"),
    ],
)
def test_resolution_errors_name_the_field(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(PresetError, match=message):
        load_preset(write(tmp_path, body))


def test_limits_narrow_and_parts_fill_with_body(tmp_path: Path) -> None:
    description = load_preset(
        write(
            tmp_path,
            """
            [robot.joints]
            select = ["arm_j1", "arm_j2", "claw_joint"]
            limits = { arm_j2 = [-1.0, 1.0] }
            max_step = { "arm_j*" = 0.05 }
            home = { arm_j2 = 0.5 }
            [robot.parts]
            arm = ["arm_j*"]
            """,
        )
    )
    np.testing.assert_allclose(description.low, [-2.9, -1.0, 0.0])
    assert description.max_step == (0.05, 0.05, None)
    assert description.home == (None, 0.5, None)
    assert description.parts == ("arm", "arm", "body")


def test_velocity_and_delta_modes_take_their_limits(tmp_path: Path) -> None:
    velocity = load_preset(
        write(tmp_path, 'mode = "joint_vel"\n[robot.joints]\nselect = ["arm_j1", "claw_joint"]\n')
    )
    np.testing.assert_allclose(velocity.high, [2.0, 0.1])
    np.testing.assert_allclose(velocity.low, [-2.0, -0.1])
    with pytest.raises(PresetError, match="needs \\[robot.joints\\] max_step for every joint"):
        load_preset(write(tmp_path, 'mode = "joint_delta"\n[robot.joints]\nselect = ["arm_j1"]\n'))
    delta = load_preset(
        write(
            tmp_path,
            'mode = "joint_delta"\n[robot.joints]\nselect = ["arm_j1"]\n'
            "max_step = { arm_j1 = 0.02 }\n",
        )
    )
    np.testing.assert_allclose([delta.low[0], delta.high[0]], [-0.02, 0.02])
    assert delta.max_step is None


def test_joints_declared_without_a_model(tmp_path: Path) -> None:
    description = load_preset(
        write(
            tmp_path,
            """
            [[robot.joint]]
            name = "lift"
            type = "prismatic"
            lower = 0.0
            upper = 0.8
            """,
            model=False,
        )
    )
    assert description.labels == ("lift",) and description.model is None
    assert "joints declared in the preset" in description.summary()
    with pytest.raises(PresetError, match="needs a model file or"):
        load_preset(write(tmp_path, "", model=False))


def test_declared_joints_cannot_shadow_model_joints(tmp_path: Path) -> None:
    with pytest.raises(PresetError, match="already joints of the model"):
        load_preset(write(tmp_path, '[[robot.joint]]\nname = "arm_j1"\nlower = 0\nupper = 1\n'))


def test_base_presets(tmp_path: Path) -> None:
    with pytest.raises(PresetError, match="needs a \\[robot.base\\] section"):
        load_preset(write(tmp_path, 'mode = "base_vel"\n'))
    with pytest.raises(PresetError, match="needs \\[robot.base\\] limits"):
        load_preset(write(tmp_path, 'mode = "base_vel"\n[robot.base]\n'))
    with pytest.raises(PresetError, match="must contain zero"):
        load_preset(
            write(
                tmp_path,
                'mode = "base_vel"\n[robot.base]\ncommand = ["vx"]\nlimits = { vx = [0.1, 1] }\n',
            )
        )
    with pytest.raises(PresetError, match="not in command"):
        load_preset(
            write(
                tmp_path,
                'mode = "base_vel"\n[robot.base]\ncommand = ["vx"]\nlimits = { wz = [-1, 1] }\n',
            )
        )


def test_the_shipped_examples_are_conformant() -> None:
    for name in ("arm", "base"):
        description = load_preset(ROBOTS / "mobile-manipulator" / f"{name}.toml")
        info = description.embodiment_info(simulated=True, pace=True)
        assert check_embodiment(info).ok, name
    base = load_preset(ROBOTS / "mobile-manipulator" / "base.toml")
    assert base.labels == ("vx", "yaw_rate") and base.parts == ("base", "base")
    assert [f.key for f in base.state_fields] == [
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "base_pose",
    ]


def test_the_preset_revision_follows_the_model(tmp_path: Path) -> None:
    path = write(tmp_path, '[robot.joints]\nselect = ["arm_j1"]\n')
    before = load_preset(path).digest
    (tmp_path / "robot.urdf").write_text(URDF.read_text().replace('upper="2.9"', 'upper="2.8"'))
    assert load_preset(path).digest != before


def test_commands_controllers_and_bindings(tmp_path: Path) -> None:
    body = """
    command_events = ["go"]
    [robot.joints]
    select = ["arm_j1"]
    [[robot.commands]]
    name = "vx"
    [[robot.commands]]
    name = "speed"
    low = 0
    high = 2
    default = 1
    hold = true
    [[robot.controllers]]
    name = "pad"
    kind = "joystick"
    deadman = "lb"
    [robot.controllers.axes]
    vx = { axis = "ly", scale = -1 }
    [robot.controllers.buttons]
    a = "go"
    y = { set = { speed = 2.0 } }
    start = "end_episode"
    """
    description = load_preset(write(tmp_path, body))
    stream = description.command_stream()
    assert stream is not None
    assert stream.axes["speed"].timeout_s is None and stream.axes["vx"].timeout_s == 0.5
    (pad,) = stream._sources
    assert isinstance(pad, JoystickSource)
    assert pad.deadman == 4 and pad.axes[0].axis == 1 and pad.axes[0].scale == -1.0
    assert sorted(pad.buttons) == [0, 3, 7]
    stream.close()


@pytest.mark.parametrize(
    ("controller", "message"),
    [
        (
            '[robot.controllers.axes]\nvz = { axis = "ly" }\n',
            "not a \\[\\[robot.commands\\]\\] name",
        ),
        ('[robot.controllers.axes]\nvx = { axis = "lyy" }\n', "unknown axis 'lyy'"),
        ('[robot.controllers.buttons]\nzz = "go"\n', "unknown button 'zz'"),
        ('[robot.controllers.buttons]\na = "jump"\n', "not in command_events"),
        ("[robot.controllers.buttons]\na = { set = { nope = 1 } }\n", "sets 'nope'"),
    ],
)
def test_controller_errors(tmp_path: Path, controller: str, message: str) -> None:
    body = (
        'command_events = ["go"]\n[robot.joints]\nselect = ["arm_j1"]\n'
        '[[robot.commands]]\nname = "vx"\n'
        '[[robot.controllers]]\nname = "pad"\nkind = "joystick"\n' + controller
    )
    with pytest.raises(PresetError, match=message):
        load_preset(write(tmp_path, body))


def test_udp_controllers_take_udp_fields_only(tmp_path: Path) -> None:
    body = (
        '[robot.joints]\nselect = ["arm_j1"]\n'
        '[[robot.controllers]]\nname = "bridge"\nkind = "udp"\ndevice = "/dev/input/js0"\n'
    )
    with pytest.raises(PresetError, match="not udp fields"):
        load_preset(write(tmp_path, body))


G1_JOINTS = [
    *(
        f"{side}_{name}"
        for side in ("left", "right")
        for name in (
            "hip_pitch_joint",
            "hip_roll_joint",
            "hip_yaw_joint",
            "knee_joint",
            "ankle_pitch_joint",
            "ankle_roll_joint",
        )
    ),
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    *(
        f"{side}_{name}"
        for side in ("left", "right")
        for name in (
            "shoulder_pitch_joint",
            "shoulder_roll_joint",
            "shoulder_yaw_joint",
            "elbow_joint",
            "wrist_roll_joint",
            "wrist_pitch_joint",
            "wrist_yaw_joint",
        )
    ),
]


def _fake_g1(path: Path) -> None:
    """A stand-in with the G1 29-DoF URDF's joint and frame names and generous limits."""
    links = ["pelvis", "imu_in_pelvis", "d435_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    links += [f"{joint}_child" for joint in G1_JOINTS]
    joints = "\n".join(
        f'<joint name="{joint}" type="revolute"><parent link="pelvis"/>'
        f'<child link="{joint}_child"/>'
        f'<limit lower="-2.9" upper="2.9" effort="88" velocity="32"/></joint>'
        for joint in G1_JOINTS
    )
    path.write_text(
        '<robot name="g1_29dof">'
        + "".join(f'<link name="{link}"/>' for link in links)
        + joints
        + "</robot>"
    )


def test_the_g1_template_resolves_against_a_g1_model(tmp_path: Path) -> None:
    for name in ("preset.toml", "g1_driver.py"):
        shutil.copy(ROBOTS / "unitree-g1" / name, tmp_path / name)
    _fake_g1(tmp_path / "g1_29dof.urdf")
    description = load_preset(tmp_path / "preset.toml")
    assert description.labels[: len(G1_JOINTS)] == tuple(G1_JOINTS)
    assert description.labels[-2:] == ("left_ee_joint", "right_ee_joint")
    assert description.dim == 31 and description.policy_hz == 50 and description.control_hz == 500
    assert [ee.name for ee in description.end_effectors] == ["left_gripper", "right_gripper"]
    assert description.end_effectors[0].joints == ("left_ee_joint",)
    assert set(description.parts or ()) == {
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "right_arm",
        "left_ee",
        "right_ee",
    }
    info = description.embodiment_info(simulated=True, pace=False)
    assert check_embodiment(info).ok
    assert "left_gripper (gripper) at frame left_wrist_yaw_link" in (info.docs or "")
    stream = description.command_stream()
    assert stream is not None and sorted(stream.axes) == ["vx", "vy", "yaw_rate"]
    stream.close()


def test_the_watchdog_timeout_must_cover_two_policy_periods(tmp_path: Path) -> None:
    with pytest.raises(PresetError, match="shorter than two policy periods"):
        load_preset(
            write(
                tmp_path,
                'mode = "joint_vel"\ncommand_timeout_s = 0.05\n'
                '[robot.joints]\nselect = ["arm_j1"]\n',
            )
        )
    description = load_preset(
        write(
            tmp_path,
            'mode = "joint_vel"\ncommand_timeout_s = 0.5\n[robot.joints]\nselect = ["arm_j1"]\n',
        )
    )
    assert description.command_timeout_s == 0.5
    assert description.position_limits is not None
    np.testing.assert_allclose(description.position_limits[1], [2.9])
