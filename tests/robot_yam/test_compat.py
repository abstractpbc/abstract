"""YAM arms + MolmoAct2 are provably compatible: zero errors, zero warnings."""

from __future__ import annotations

from abstract_harness.robot.core.compat import check_compatibility
from abstract_harness.robot.core.policy import PolicyConfig, PolicyInfo
from abstract_harness.robot.core.spaces import ActionSemantics, Box
from abstract_harness.robot.yam.config import action_box, observation_space
from abstract_harness.robot.yam.embodiment import YAMEmbodiment
from abstract_harness.robot.yam.policy import MolmoAct2Policy


def test_molmoact2_yam_compatible_no_errors_no_warnings() -> None:
    report = check_compatibility(MolmoAct2Policy(), YAMEmbodiment())
    assert report.ok is True
    assert report.errors == []
    assert report.warnings == []


def test_state_key_mismatch_trips_state_error() -> None:
    # A non-default state_key must surface in compat (the policy would KeyError in
    # act() otherwise), and disappear again when both sides agree.
    from abstract_harness.robot.yam.config import MolmoActConfig

    mismatched = MolmoAct2Policy(MolmoActConfig(state_key="proprio"))
    report = check_compatibility(mismatched, YAMEmbodiment())
    assert report.ok is False
    assert any(i.code == "missing_state" for i in report.errors)

    matched = MolmoAct2Policy(MolmoActConfig(state_key="joint_pos"))
    ok_report = check_compatibility(matched, YAMEmbodiment())
    assert ok_report.ok is True and ok_report.errors == []


def test_negative_wrong_dim_policy_trips_action_dim_error() -> None:
    bad = PolicyInfo(
        name="bad",
        action_space=Box(shape=(8,), semantics=ActionSemantics(control_mode="joint_pos")),
    )

    class _Bad:
        info = bad
        config = PolicyConfig()

        def reset(self, scene: object) -> None: ...
        def act(self, obs: object) -> object: ...  # pragma: no cover

    report = check_compatibility(_Bad(), YAMEmbodiment())  # type: ignore[arg-type]
    assert report.ok is False
    assert any(i.code == "action_dim" for i in report.errors)


def test_negative_policy_advertising_rate_trips_control_rate_warning() -> None:
    # Locks the load-bearing detail: leaving PolicyInfo.control_hz=None is the only
    # reason the real pairing is warning-free. A policy that advertises a rate warns.
    info = PolicyInfo(
        name="rated",
        action_space=action_box(),
        observation_space=observation_space(224, 224, ("top_cam", "left_cam", "right_cam")),
        control_hz=100.0,
    )

    class _Rated:
        config = PolicyConfig()

        def __init__(self) -> None:
            self.info = info

        def reset(self, scene: object) -> None: ...
        def act(self, obs: object) -> object: ...  # pragma: no cover

    report = check_compatibility(_Rated(), YAMEmbodiment())  # type: ignore[arg-type]
    assert report.ok is True  # only a warning, not an error
    assert any(i.code == "control_rate" for i in report.warnings)


def test_delta_pairing_passes_compat_in_lockstep() -> None:
    from abstract_harness.robot.yam import MolmoAct2Policy, YamConfig, YAMEmbodiment
    from abstract_harness.robot.yam.config import MolmoActConfig

    delta_emb = YAMEmbodiment(YamConfig(joints_are_delta=True))
    delta_pol = MolmoAct2Policy(MolmoActConfig(joints_are_delta=True))
    assert check_compatibility(delta_pol, delta_emb).ok is True

    # A mismatch (delta rig, absolute-declaring policy) is a hard compat error.
    absolute_pol = MolmoAct2Policy(MolmoActConfig())
    report = check_compatibility(absolute_pol, delta_emb)
    assert report.ok is False
    assert any(i.code == "control_mode" for i in report.errors)
