"""hold_check: the 6.4 hold-behavior verification (plan 0008 §6.4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from abstract_harness.robot.yam import hold_check
from abstract_harness.robot.yam.hold_check import HoldResult, run_hold_check

from .conftest import robot_cli, write_rig


def main(argv: list[str], **kwargs: Any) -> int:
    return robot_cli("holdcheck", argv, **kwargs)


class _FakeArm:
    """A single 7-D arm whose pose drifts by `drift_per_read` each get."""

    def __init__(self, drift_per_read: float = 0.0):
        self.pose = np.zeros(7)
        self.drift = drift_per_read
        self.commands: list[np.ndarray] = []
        self.reads = 0

    def get_joint_pos(self) -> np.ndarray:
        self.reads += 1
        self.pose = self.pose + self.drift
        return self.pose.copy()

    def command_joint_pos(self, target: np.ndarray) -> None:
        self.commands.append(np.asarray(target).copy())


def _run(arm: _FakeArm, **kwargs: object) -> HoldResult:
    sleeps: list[float] = []
    result = run_hold_check(
        robot=arm,
        duration_s=20.0,
        interval_s=5.0,
        sleep_fn=sleeps.append,
        emit=lambda _line: None,
        **kwargs,  # type: ignore[arg-type]
    )
    assert sleeps == [5.0] * 4
    return result


def test_holding_arm_passes() -> None:
    arm = _FakeArm(drift_per_read=0.0)
    result = _run(arm)
    assert result.max_drift == pytest.approx(0.0)
    assert result.passed is True
    assert len(arm.commands) == 1  # exactly one command: the current pose


def test_drifting_arm_fails_and_reports_worst_joint() -> None:
    arm = _FakeArm(drift_per_read=0.01)
    result = _run(arm)
    assert result.passed is False
    assert result.max_drift > 0.01
    assert result.samples  # per-interval history retained for the report


def test_thresholds_are_configurable() -> None:
    arm = _FakeArm(drift_per_read=0.001)
    assert _run(arm, settle_rad=1.0, trend_rad=1.0).passed is True


def test_settle_and_trend_are_judged_separately() -> None:
    class _SettlingArm(_FakeArm):
        """Settles 0.03 on the first read after command, then holds flat."""

        def get_joint_pos(self):  # type: ignore[no-untyped-def]
            self.reads += 1
            if self.reads > 1:
                self.pose = np.full(7, 0.03)
            return self.pose.copy()

    settled = _run(_SettlingArm())
    assert settled.settle == pytest.approx(0.03)
    assert settled.trend == pytest.approx(0.0)
    assert settled.passed is True  # one-time settle within the generous limit

    drifting = _run(_FakeArm(drift_per_read=0.02))
    assert drifting.trend > 0.01
    assert drifting.passed is False  # growth after the first sample = sag


def test_main_wires_argv_and_exit_codes() -> None:
    lines: list[str] = []
    holding = _FakeArm()

    def factory(channel: str, zero_gravity_mode: bool) -> _FakeArm:
        assert channel == "can0" and zero_gravity_mode is True
        return holding

    rc = main(
        ["can0", "--zero-gravity", "true", "--duration-s", "10", "--interval-s", "5"],
        robot_factory=factory,
        sleep_fn=lambda _s: None,
        emit=lines.append,
    )
    assert rc == 0
    assert any("PASS" in line for line in lines)

    drifting = _FakeArm(drift_per_read=0.05)
    rc = main(
        ["can1", "--zero-gravity", "false"],
        robot_factory=lambda channel, zero_gravity_mode: drifting,
        sleep_fn=lambda _s: None,
        emit=lines.append,
    )
    assert rc == 1
    assert any("FAIL" in line for line in lines)


@pytest.mark.parametrize(
    ("side", "key", "resolved"),
    [
        ("left", "left_channel", "can_left"),
        ("right", "right_channel", "can_right"),
    ],
)
def test_main_resolves_toml_side_and_prints_resolved_channel(
    side: str,
    key: str,
    resolved: str,
    tmp_path: Path,
) -> None:
    config = write_rig(tmp_path / "rig.toml", {key: resolved})
    lines: list[str] = []
    calls: list[tuple[str, bool]] = []

    def factory(channel: str, zero_gravity: bool) -> _FakeArm:
        calls.append((channel, zero_gravity))
        return _FakeArm()

    assert (
        main(
            [side, "--zero-gravity", "false", "--duration-s", "0"],
            config=config,
            robot_factory=factory,
            sleep_fn=lambda _seconds: None,
            emit=lines.append,
        )
        == 0
    )
    assert calls == [(resolved, False)]
    assert lines[0] == f"{resolved} ({side}) zero_gravity=false: watching for 0s"


def test_default_robot_factory_gets_the_toml_gripper_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool, str]] = []

    def factory(channel: str, zero_gravity_mode: bool, *, gripper_type: str) -> _FakeArm:
        calls.append((channel, zero_gravity_mode, gripper_type))
        return _FakeArm()

    monkeypatch.setattr(hold_check, "_default_robot_factory", factory)
    config = write_rig(tmp_path / "rig.toml", {"gripper_type": "CRANK_4310"})
    assert (
        main(
            ["can0", "--zero-gravity", "true", "--duration-s", "0"],
            config=config,
            sleep_fn=lambda _seconds: None,
            emit=lambda _line: None,
        )
        == 0
    )
    assert calls == [("can0", True, "CRANK_4310")]


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        (
            "[vla]\nserver_url = 'http://127.0.0.1:8202'\n"
            "[rig.embodiment]\nleft_channel = 'can0'\n",
            "must set left_channel and right_channel",
        ),
        ("[vla\n", "rig.toml"),
    ],
    ids=["invalid", "malformed"],
)
def test_a_robot_toml_that_does_not_load_is_a_usage_error(
    toml: str, message: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "rig.toml"
    config.write_text(toml, encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        main(["can0", "--zero-gravity", "true"], config=config)
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_an_embodiment_yam_rejects_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_rig(tmp_path / "rig.toml", {"gripper_type": "CLAW"})
    with pytest.raises(SystemExit) as exc_info:
        main(["left", "--zero-gravity", "true"], config=config)
    assert exc_info.value.code == 2
    assert "gripper_type 'CLAW' is not supported" in capsys.readouterr().err


def test_channel_help_explains_claimed_side_literals(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "interface literally named left or right must be renamed" in output


def test_main_rejects_bad_zero_gravity_value() -> None:
    with pytest.raises(SystemExit):
        main(["can0", "--zero-gravity", "maybe"], sleep_fn=lambda _s: None, emit=lambda _l: None)


def test_main_closes_the_robot_even_on_failure() -> None:
    class _ClosableArm(_FakeArm):
        closed = 0

        def close(self) -> None:
            _ClosableArm.closed += 1

    arm = _ClosableArm(drift_per_read=0.5)  # guaranteed FAIL verdict
    rc = main(
        ["can0", "--zero-gravity", "true"],
        robot_factory=lambda channel, zero_gravity_mode: arm,
        sleep_fn=lambda _s: None,
        emit=lambda _l: None,
    )
    assert rc == 1
    assert _ClosableArm.closed == 1  # released regardless of the verdict


def test_the_robot_is_released_through_close_robot_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # i2rt's own close() shuts the CAN socket while its control thread still reads it
    # (the fd -1 traceback after every verdict); close_robot_safely joins the thread
    # first.
    released: list[object] = []
    monkeypatch.setattr(hold_check, "close_robot_safely", released.append)

    class _ClosableArm(_FakeArm):
        def close(self) -> None:
            raise AssertionError("closed directly, not through close_robot_safely")

    arm = _ClosableArm()
    rc = main(
        ["can0", "--zero-gravity", "false", "--duration-s", "0"],
        robot_factory=lambda channel, zero_gravity_mode: arm,
        sleep_fn=lambda _s: None,
        emit=lambda _l: None,
    )
    assert rc == 0 and released == [arm]


def test_default_emit_flushes(capsys: pytest.CaptureFixture[str]) -> None:
    from abstract_harness.robot.yam.hold_check import _print_flushed

    _print_flushed("hello")
    assert capsys.readouterr().out == "hello\n"
