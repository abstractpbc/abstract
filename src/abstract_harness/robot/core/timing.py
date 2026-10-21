"""Strict timing: hold a rollout to a fixed tick rate.

Without a pacer the rollout applies no wall-clock pacing (R1): ``embodiment.step()`` is
called as soon as the policy, controller and approver produce an action, and an
embodiment that needs real-time cadence paces itself (``"self_paced"``). Strict timing
moves that cadence into the loop for any embodiment. A ``Pacer`` dispatches step ``k``
of a trial no earlier than ``start + k / hz``, so a fast policy cannot run the robot
faster than its configured rate and a step's timing no longer depends on how long the
previous inference took.

A step that cannot be dispatched on time is an *overrun*. The pacer never bursts to
catch up: a late step re-anchors the schedule at the moment it is dispatched, so the
next step still waits a full period, and the ticks it skipped are counted. Every trial
records its timing report (``TrialRecord.metadata["timing"]``): rate, ticks, overruns,
missed ticks, the worst lateness and the effective rate. ``max_consecutive_overruns``
turns sustained lateness into a ``SafetyAbort`` (the eval halts and a physical embodiment
parks), for robots whose controllers are unsafe at an irregular rate.

Strict timing paces dispatch; it does not make inference faster. Pair it with
``AsyncController`` (``core/controller.py``) so slow inference runs while buffered actions
keep playing on time. An embodiment that already paces itself to the same rate composes
with a pacer: whichever waits longer sets the step time.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from abstract_harness.robot.core.errors import ConfigError, SafetyAbort

# A step dispatched within this fraction of a period after its deadline is on time:
# sleep granularity and scheduler noise, not lateness.
DEFAULT_TOLERANCE_FRACTION = 0.1


@dataclass(frozen=True)
class StrictTiming:
    """How ``eval()`` paces each trial; ``eval(strict_timing=True)`` uses the defaults.

    ``hz`` defaults to the embodiment's declared ``control_hz`` (the policy rate in its
    robot config), then the policy's. ``max_consecutive_overruns`` (``None``: never)
    halts the eval with ``SafetyAbort`` after that many late steps in a row.
    ``tolerance_s`` is the lateness still counted as on time (default: a tenth of a
    period).
    """

    hz: float | None = None
    max_consecutive_overruns: int | None = None
    tolerance_s: float | None = None

    def __post_init__(self) -> None:
        if self.hz is not None:
            _check_hz(self.hz)
        limit = self.max_consecutive_overruns
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)):
            raise ConfigError("max_consecutive_overruns must be an integer >= 1 or None")
        if limit is not None and limit < 1:
            raise ConfigError("max_consecutive_overruns must be an integer >= 1 or None")
        if self.tolerance_s is not None and (
            isinstance(self.tolerance_s, bool)
            or not math.isfinite(self.tolerance_s)
            or self.tolerance_s < 0
        ):
            raise ConfigError("tolerance_s must be finite and >= 0")

    def pacer(
        self,
        hz: float,
        *,
        auto: bool = False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Pacer:
        """A fresh pacer for one trial at ``hz`` (this config's ``hz`` when set).
        ``auto`` records that the policy, not the caller, asked for the paced loop."""
        return Pacer(
            self.hz if self.hz is not None else hz,
            max_consecutive_overruns=self.max_consecutive_overruns,
            tolerance_s=self.tolerance_s,
            auto=auto,
            clock=clock,
            sleep=sleep,
        )


def _check_hz(hz: float) -> float:
    if isinstance(hz, bool) or not isinstance(hz, int | float) or not math.isfinite(hz) or hz <= 0:
        raise ConfigError(f"strict timing needs a finite, positive tick rate, got {hz!r}")
    return float(hz)


def resolve_strict_timing(value: bool | StrictTiming | None) -> StrictTiming | None:
    """``eval()``'s ``strict_timing`` argument as a config, or ``None`` when off."""
    if value is None or value is False:
        return None
    if value is True:
        return StrictTiming()
    if isinstance(value, StrictTiming):
        return value
    raise ConfigError(f"strict_timing must be a bool or StrictTiming, got {value!r}")


class Pacer:
    """Dispatch one trial's steps on a fixed schedule and account for lateness.

    ``wait()`` is called immediately before each ``embodiment.step()``. The first call
    anchors the schedule and returns at once; call ``k`` returns no earlier than
    ``anchor + k / hz``. ``clock`` and ``sleep`` are injectable for tests.
    """

    def __init__(
        self,
        hz: float,
        *,
        max_consecutive_overruns: int | None = None,
        tolerance_s: float | None = None,
        auto: bool = False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.hz = _check_hz(hz)
        self.auto = auto
        self.period_s = 1.0 / self.hz
        self.max_consecutive_overruns = max_consecutive_overruns
        self.tolerance_s = (
            tolerance_s if tolerance_s is not None else DEFAULT_TOLERANCE_FRACTION * self.period_s
        )
        self._clock = clock
        self._sleep = sleep
        self._anchor: float | None = None
        self._k = 0
        self.ticks = 0
        self.overruns = 0
        self.consecutive_overruns = 0
        self.missed_ticks = 0
        self.max_late_s = 0.0
        self._late_total_s = 0.0
        self._first: float | None = None
        self._last: float | None = None

    def wait(self) -> float:
        """Block until this step's deadline; return how late it is dispatched (seconds).

        Raises ``SafetyAbort`` once ``max_consecutive_overruns`` steps in a row were late.
        """
        now = self._clock()
        slack = True
        if self._anchor is None:
            self._anchor = now
            self._k = 0
            late = 0.0
        else:
            self._k += 1
            deadline = self._anchor + self._k * self.period_s
            slack = now <= deadline
            if now < deadline:
                self._sleep(deadline - now)
                now = self._clock()
            late = max(0.0, now - deadline)
            if late > self.tolerance_s:
                # Never burst: re-anchor at this dispatch, so the next step waits a full
                # period, and count the ticks it skipped.
                self.missed_ticks += math.floor(late * self.hz + 1e-9)
                self._anchor = now
                self._k = 0
        self.ticks += 1
        if self._first is None:
            self._first = now
        self._last = now
        if late > self.tolerance_s:
            self.overruns += 1
            self.consecutive_overruns += 1
            self._late_total_s += late
            self.max_late_s = max(self.max_late_s, late)
            limit = self.max_consecutive_overruns
            if limit is not None and self.consecutive_overruns >= limit:
                raise SafetyAbort(
                    f"strict timing: {self.consecutive_overruns} consecutive steps missed "
                    f"their {self.hz:g} Hz deadline (last {late * 1000:.1f} ms late); "
                    "evaluation halted"
                )
        elif slack:
            # Only a step that had time to spare ends a run of overruns: one that is a
            # little late (within the tolerance) is not keeping the rate either.
            self.consecutive_overruns = 0
        return late

    def report(self) -> dict[str, Any]:
        """The trial's timing, JSON-ready, for ``TrialRecord.metadata["timing"]``."""
        effective = None
        if self._first is not None and self._last is not None and self.ticks > 1:
            elapsed = self._last - self._first
            if elapsed > 0:
                effective = round((self.ticks - 1) / elapsed, 3)
        return {
            "strict": True,
            "hz": self.hz,
            "ticks": self.ticks,
            "overruns": self.overruns,
            "missed_ticks": self.missed_ticks,
            "max_late_ms": round(self.max_late_s * 1000, 3),
            "mean_overrun_late_ms": round(self._late_total_s / self.overruns * 1000, 3)
            if self.overruns
            else 0.0,
            "effective_hz": effective,
            "max_consecutive_overruns": self.max_consecutive_overruns,
            "auto": self.auto,
        }
