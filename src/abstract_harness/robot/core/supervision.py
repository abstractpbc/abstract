"""Bound inference waits without putting actuator access on the inference worker."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from abstract_harness.robot.core.errors import ConfigError, SafetyAbort

T = TypeVar("T")


class InferenceSupervisor:
    """Keep the owning thread responsive to signals, operator input and health checks.

    A timed-out worker is abandoned and the evaluation must halt. Python cannot kill
    a blocked thread; no later result is dispatched and the policy must not be reused.
    Policy implementations must never access actuators themselves.
    """

    def __init__(self, timeout_s: float):
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ConfigError("policy_timeout_s must be finite and positive")
        self.timeout_s = timeout_s
        self.abandoned = False

    def call(self, function: Callable[[], T], tick: Callable[[], None]) -> T:
        result: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                value = function()
            except BaseException as exc:
                result.put((False, exc))
            else:
                result.put((True, value))

        worker = threading.Thread(target=run, name="abstract-policy", daemon=True)
        deadline = time.monotonic() + self.timeout_s
        worker.start()
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SafetyAbort("policy inference deadline exceeded; evaluation halted")
                try:
                    ok, value = result.get(timeout=min(0.05, remaining))
                except queue.Empty:
                    tick()
                    continue
                tick()
                if time.monotonic() >= deadline:
                    raise SafetyAbort("policy inference completed after its deadline")
                if not ok:
                    assert isinstance(value, BaseException)
                    raise value
                return value  # type: ignore[return-value]
        except BaseException:
            self.abandoned = worker.is_alive()
            raise
