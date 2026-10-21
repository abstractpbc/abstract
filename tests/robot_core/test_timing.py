"""Strict timing: the pacer's schedule, its reports, and eval()/CLI integration."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from abstract_harness.cli import _main as cli_main
from abstract_harness.robot.core import eval
from abstract_harness.robot.core.approver import AutoApprover
from abstract_harness.robot.core.controller import DefaultController
from abstract_harness.robot.core.embodiment import SELF_PACED
from abstract_harness.robot.core.errors import ConfigError, SafetyAbort
from abstract_harness.robot.core.logging.sink import NullSink
from abstract_harness.robot.core.mock import CubePickEmbodiment, ScriptedPolicy
from abstract_harness.robot.core.rollout import rollout
from abstract_harness.robot.core.scene import Scene
from abstract_harness.robot.core.scorer import success_at_end
from abstract_harness.robot.core.task import Task
from abstract_harness.robot.core.timing import Pacer, StrictTiming, resolve_strict_timing


class FakeClock:
    """A clock that only moves when slept on or advanced."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _pacer(clock: FakeClock, hz: float = 10.0, **kwargs: Any) -> Pacer:
    return Pacer(hz, clock=clock, sleep=clock.sleep, **kwargs)


def test_first_tick_anchors_and_later_ticks_wait_for_their_deadline() -> None:
    clock = FakeClock()
    pacer = _pacer(clock)
    assert pacer.wait() == 0.0
    assert clock.sleeps == []
    clock.now += 0.03  # the step and the next inference took 30 ms
    pacer.wait()
    assert clock.sleeps == [pytest.approx(0.07)]
    assert clock.now == pytest.approx(100.2 - 0.1)
    pacer.wait()  # no work at all: a full period
    assert clock.sleeps[-1] == pytest.approx(0.1)
    report = pacer.report()
    assert report["ticks"] == 3 and report["overruns"] == 0
    assert report["effective_hz"] == pytest.approx(10.0)


def test_late_steps_count_as_overruns_and_never_burst() -> None:
    clock = FakeClock()
    pacer = _pacer(clock)
    pacer.wait()
    clock.now += 0.15  # 50 ms late for tick 1: an overrun inside one period
    assert pacer.wait() == pytest.approx(0.05)
    clock.now += 0.35  # tick 2 was due 50 ms after the previous dispatch: 300 ms late
    assert pacer.wait() == pytest.approx(0.3)
    # More than one period late: the schedule re-anchors at this dispatch, so the next
    # tick is a full period later rather than an immediate catch-up burst.
    assert pacer.missed_ticks == 3
    pacer.wait()
    assert clock.sleeps[-1] == pytest.approx(0.1)
    report = pacer.report()
    assert report["overruns"] == 2 and report["missed_ticks"] == 3
    assert report["max_late_ms"] == pytest.approx(300.0)
    assert pacer.consecutive_overruns == 0


def test_lateness_within_tolerance_is_on_time() -> None:
    clock = FakeClock()
    pacer = _pacer(clock, tolerance_s=0.02)
    pacer.wait()
    clock.now += 0.115
    pacer.wait()
    assert pacer.overruns == 0


def test_consecutive_overruns_abort() -> None:
    clock = FakeClock()
    pacer = _pacer(clock, max_consecutive_overruns=2)
    pacer.wait()
    clock.now += 0.2
    pacer.wait()
    clock.now += 0.2
    with pytest.raises(SafetyAbort, match="2 consecutive steps missed their 10 Hz deadline"):
        pacer.wait()


def test_an_on_time_step_resets_the_consecutive_count() -> None:
    clock = FakeClock()
    pacer = _pacer(clock, max_consecutive_overruns=2)
    pacer.wait()
    for _ in range(5):
        clock.now += 0.2  # late
        pacer.wait()
        pacer.wait()  # on time again
    assert pacer.overruns == 5


@pytest.mark.parametrize("hz", [0, -1, float("nan"), float("inf"), True])
def test_rates_must_be_finite_and_positive(hz: Any) -> None:
    with pytest.raises(ConfigError):
        Pacer(hz)
    with pytest.raises(ConfigError):
        StrictTiming(hz=hz)


@pytest.mark.parametrize("limit", [0, -3, 1.5, True])
def test_overrun_limits_are_positive_integers(limit: Any) -> None:
    with pytest.raises(ConfigError):
        StrictTiming(max_consecutive_overruns=limit)


def test_strict_timing_argument_forms() -> None:
    assert resolve_strict_timing(False) is None
    assert resolve_strict_timing(None) is None
    assert resolve_strict_timing(True) == StrictTiming()
    config = StrictTiming(hz=5)
    assert resolve_strict_timing(config) is config
    with pytest.raises(ConfigError):
        resolve_strict_timing("yes")  # type: ignore[arg-type]
    assert config.pacer(50).hz == 5.0
    assert StrictTiming().pacer(50).hz == 50.0


def _task(max_steps: int = 12) -> Task:
    return Task(
        name="paced",
        scenes=[Scene(id="s0", instruction="reach", init_seed=0)],
        scorer=success_at_end(),
        max_steps=max_steps,
    )


class _FastCube(CubePickEmbodiment):
    """CubePick declared at 200 Hz so paced tests stay fast."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from dataclasses import replace

        self.info = replace(self.info, control_hz=200.0)


def test_eval_paces_each_trial_at_the_embodiment_rate(tmp_path: Any) -> None:
    started = time.monotonic()
    (log,) = eval(
        _task(),
        ScriptedPolicy(chunk_size=1, max_step=0.01),
        _FastCube(),
        log_dir=str(tmp_path),
        strict_timing=True,
    )
    elapsed = time.monotonic() - started
    timing = log.samples[0].trial_metadata[0]["timing"]
    assert timing["strict"] is True and timing["hz"] == 200.0 and timing["auto"] is False
    assert timing["ticks"] == 12
    # Eleven periods separate twelve dispatches.
    assert elapsed >= 11 / 200.0
    assert 50 <= timing["effective_hz"] <= 201


def test_eval_without_strict_timing_records_no_timing(tmp_path: Any) -> None:
    (log,) = eval(_task(), ScriptedPolicy(), CubePickEmbodiment(), log_dir=str(tmp_path))
    assert "timing" not in log.samples[0].trial_metadata[0]


def test_explicit_rate_wins_over_the_embodiment(tmp_path: Any) -> None:
    (log,) = eval(
        _task(4),
        ScriptedPolicy(chunk_size=1),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
        strict_timing=StrictTiming(hz=400),
    )
    assert log.samples[0].trial_metadata[0]["timing"]["hz"] == 400.0


def test_strict_timing_needs_a_rate(tmp_path: Any) -> None:
    from dataclasses import replace

    cube = CubePickEmbodiment()
    cube.info = replace(cube.info, control_hz=None)
    with pytest.raises(ConfigError, match="needs a tick rate"):
        eval(_task(), ScriptedPolicy(), cube, log_dir=str(tmp_path), strict_timing=True)


class _NeedsPacing(ScriptedPolicy):
    requires_paced_loop = True


def test_a_policy_that_needs_a_paced_loop_gets_one(tmp_path: Any) -> None:
    (log,) = eval(_task(4), _NeedsPacing(chunk_size=1), _FastCube(), log_dir=str(tmp_path))
    timing = log.samples[0].trial_metadata[0]["timing"]
    assert timing["auto"] is True and timing["hz"] == 200.0


def test_a_self_paced_embodiment_is_not_paced_again(tmp_path: Any) -> None:
    from dataclasses import replace

    cube = _FastCube()
    cube.info = replace(cube.info, capabilities=cube.info.capabilities | {SELF_PACED})
    (log,) = eval(_task(4), _NeedsPacing(chunk_size=1), cube, log_dir=str(tmp_path))
    assert "timing" not in log.samples[0].trial_metadata[0]


class _SlowPolicy(ScriptedPolicy):
    def act(self, observation: Any) -> Any:
        time.sleep(0.03)
        return super().act(observation)


def test_sustained_overruns_halt_the_eval(tmp_path: Any) -> None:
    (log,) = eval(
        _task(40),
        _SlowPolicy(chunk_size=1),
        _FastCube(),
        log_dir=str(tmp_path),
        strict_timing=StrictTiming(max_consecutive_overruns=3),
    )
    assert log.status == "error"
    assert "SafetyAbort" in (log.error or "")
    assert "consecutive steps missed" in (log.error or "")
    timing = log.samples[0].trial_metadata[0]["timing"]
    assert timing["overruns"] == 3


def test_rollout_records_the_pacer_report_and_dispatches_after_waiting() -> None:
    clock = FakeClock()
    dispatched: list[float] = []

    class Recording(CubePickEmbodiment):
        def step(self, action: Any) -> Any:
            dispatched.append(clock.now)
            return super().step(action)

    record = rollout(
        ScriptedPolicy(chunk_size=1, max_step=0.001),
        Recording(),
        Scene(id="s", instruction="reach", init_seed=0),
        max_steps=4,
        seed=0,
        epoch=0,
        controller=DefaultController(),
        approver=AutoApprover(),
        sink=NullSink(),
        pacer=Pacer(10, clock=clock, sleep=clock.sleep),
    )
    assert np.diff(dispatched) == pytest.approx([0.1, 0.1, 0.1])
    assert record.metadata["timing"]["ticks"] == 4


def test_cli_eval_takes_strict_timing_and_async_inference(tmp_path: Any, capsys: Any) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli_main(
            [
                "eval",
                "--instruction",
                "reach the cube",
                "--max-steps",
                "3",
                "--no-prompt",
                "--strict-timing",
                "--async-inference",
                "--log-dir",
                str(tmp_path),
            ]
        )
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "timing: strict, 10 Hz with async inference" in out
