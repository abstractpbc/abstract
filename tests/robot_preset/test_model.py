"""Model files: what a preset reads from URDF and MJCF, and the errors it gives."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from abstract_harness.robot.preset.model import ModelError, load_model

ROBOTS = Path(__file__).parents[2] / "robots"


def test_the_example_urdf() -> None:
    model = load_model(ROBOTS / "mobile-manipulator" / "mobile_manipulator.urdf")
    assert model.format == "urdf" and model.name == "mobile_manipulator"
    movable = [joint.name for joint in model.movable]
    assert movable == [
        "left_wheel_joint",
        "right_wheel_joint",
        *(f"arm_j{i}" for i in range(1, 7)),
        "claw_joint",
    ]
    wheel = model.joint("left_wheel_joint")
    assert wheel is not None and wheel.lower is None and wheel.velocity == 12.0
    claw = model.joint("claw_joint")
    assert claw is not None and (claw.lower, claw.upper, claw.unit) == (0.0, 0.04, "m")
    assert {"claw_base_link", "head_camera_link", "imu_link"} <= model.frames
    assert len(model.digest) == 64


MJCF = """
<mujoco model="toy">
  <compiler angle="{angle}"/>
  <default>
    <joint range="-90 90"/>
    <default class="wrist"><joint range="-45 45"/></default>
  </default>
  <worldbody>
    <camera name="overhead"/>
    <body name="torso">
      <freejoint/>
      <site name="imu_site"/>
      <body name="shoulder">
        <joint name="shoulder" type="hinge"/>
        <body name="wrist" childclass="wrist">
          <joint name="wrist"/>
          <camera name="wrist_cam"/>
          <include file="fingers.xml"/>
        </body>
      </body>
      <body name="rail">
        <joint name="slider" type="slide" range="0 0.5"/>
        <joint name="free_spin" limited="false" range="-1 1"/>
        <joint name="ball" type="ball"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="wrist"/>
    <position joint="shoulder"/>
    <motor joint="slider"/>
  </actuator>
</mujoco>
"""

FINGERS = """
<mujoco>
  <body name="finger"><joint name="finger" type="slide" range="0 0.03"/></body>
</mujoco>
"""


def _mjcf(tmp_path: Path, angle: str = "degree") -> Path:
    (tmp_path / "fingers.xml").write_text(FINGERS)
    path = tmp_path / "toy.xml"
    path.write_text(MJCF.format(angle=angle))
    return path


def test_mjcf_ranges_defaults_frames_and_actuator_order(tmp_path: Path) -> None:
    model = load_model(_mjcf(tmp_path))
    assert model.format == "mjcf" and model.name == "toy"
    shoulder, wrist = model.joint("shoulder"), model.joint("wrist")
    assert shoulder is not None and shoulder.upper == pytest.approx(math.pi / 2)
    # childclass="wrist" applies the class's narrower range.
    assert wrist is not None and wrist.upper == pytest.approx(math.pi / 4)
    slider = model.joint("slider")
    assert slider is not None and slider.upper == 0.5  # slides are never converted
    spin = model.joint("free_spin")
    assert spin is not None and spin.lower is None  # limited="false"
    assert model.joint("finger") is not None  # from the included file
    # Actuated joints in actuator order; the ball joint and the free base are not movable.
    assert [j.name for j in model.movable] == ["wrist", "shoulder", "slider"]
    assert {"torso", "imu_site", "overhead", "wrist_cam", "finger"} <= model.frames
    assert model.frame_kinds["imu_site"] == "site"


def test_mjcf_radian_compiler(tmp_path: Path) -> None:
    model = load_model(_mjcf(tmp_path, angle="radian"))
    shoulder = model.joint("shoulder")
    assert shoulder is not None and shoulder.upper == 90.0


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "<robot><joint name='a' type='revolute'/><joint name='a' type='revolute'/></robot>",
            "twice",
        ),
        ("<robot><joint type='revolute'/></robot>", "name and a type"),
        (
            "<robot><joint name='a' type='revolute'><limit lower='x' upper='1'/></joint></robot>",
            "not a number",
        ),
        ("<sdf/>", "expected a URDF"),
        ("<robot", "not well-formed"),
        ("<mujoco/>", "worldbody"),
        ("<mujoco><worldbody><body><joint/></body></worldbody></mujoco>", "needs a name"),
    ],
)
def test_model_errors(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.xml"
    path.write_text(text)
    with pytest.raises(ModelError, match=message):
        load_model(path)


def test_a_missing_model_file(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="cannot be read"):
        load_model(tmp_path / "absent.urdf")
