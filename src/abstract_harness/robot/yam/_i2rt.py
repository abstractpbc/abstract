"""Load the optional I2RT driver lazily with actionable installation guidance."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

# The build constraint is required while every published ruckig (i2rt's pinned
# dependency) is a source-only release that no longer builds under
# scikit-build-core 1.0; drop it once ruckig ships the fix from pantor/ruckig#261
# and i2rt moves off ruckig==0.15.3 (#47).
# The commit is pinned to the one the rigs run, not to i2rt's newest release.
# ac096928 defaults enable_auto_recovery=False, so a motor error fails the
# episode fast instead of the driver cleaning and re-enabling the errored motor
# inside the control loop. Every rig hour behind yam_arms is on that behavior;
# the previous pin (db582eaa, 2026-07-07) predates the change and self-heals
# silently under load. Treat a bump as a rig-evaluation decision (#118).
I2RT_INSTALL_COMMAND = (
    "echo 'scikit-build-core<0.10' > build-constraints.txt && "
    "uv pip install --build-constraints build-constraints.txt "
    '"i2rt @ git+https://github.com/i2rt-robotics/'
    'i2rt@ac096928d6899ddf852a71c5e8fbaa6055cd9745"'
)
_CONTROL_THREAD_JOIN_TIMEOUT = 5.0
_CONTROL_THREAD_GRACE_PERIOD = 0.05

logger = logging.getLogger(__name__)


def close_robot_safely(robot: Any) -> None:
    """Close an I2RT robot without racing its control loop against the CAN socket.

    I2RT discards its control-thread handle and closes the CAN socket without joining
    that thread, so the loop crashes with ``fd=-1`` during every teardown. This helper
    works around upstream YAM issue #28 by discovering the discarded thread
    and interposing its join between I2RT setting ``running = False`` and closing the
    socket.
    """
    chain = getattr(robot, "motor_chain", None)
    if chain is None or getattr(chain, "motor_interface", None) is None:
        # No single-chain interface to guard (unknown driver shape, or a
        # multi-chain aggregate) — fall back to the driver's own teardown.
        robot.close()
        return

    control_threads = [
        thread
        for thread in threading.enumerate()
        if getattr(getattr(thread, "_target", None), "__self__", None) is chain
    ]
    original_close = chain.motor_interface.close
    close_lock = threading.Lock()
    closed = False

    def close_motor_interface_safely() -> None:
        nonlocal closed
        with close_lock:
            if closed:
                return

            if control_threads:
                for thread in control_threads:
                    thread.join(timeout=_CONTROL_THREAD_JOIN_TIMEOUT)
                    if thread.is_alive():
                        logger.warning(
                            "I2RT control thread %s did not stop within %.1f seconds",
                            thread.name,
                            _CONTROL_THREAD_JOIN_TIMEOUT,
                        )
            else:
                logger.debug(
                    "No I2RT control thread was discoverable; waiting %.2f seconds "
                    "before closing the motor interface",
                    _CONTROL_THREAD_GRACE_PERIOD,
                )
                time.sleep(_CONTROL_THREAD_GRACE_PERIOD)

            original_close()
            closed = True

    chain.motor_interface.close = close_motor_interface_safely
    robot.close()


def start_arms_concurrently(
    left_factory: Callable[[], Any],
    right_factory: Callable[[], Any],
    *,
    timeout_s: float = 60.0,
    on_failed_cleanup: Callable[[], None] | None = None,
) -> tuple[Any, Any]:
    """Bound bring-up waits; close survivors, including arms that finish after cancellation.

    Factories run on daemon workers. A wedged SDK constructor cannot be killed by Python;
    the operator must use the hardware stop. A late successful constructor is closed and
    never returned after a timeout or interrupt.
    """
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("arm initialization timeout must be finite and positive")
    condition = threading.Condition()
    cancelled = False
    robots: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    completed: set[str] = set()
    cleanup_done = False
    cleanup_ready = False

    def finish_cleanup() -> None:
        nonlocal cleanup_done
        with condition:
            release = cancelled and cleanup_ready and len(completed) == 2 and not cleanup_done
            if release:
                cleanup_done = True
        if release and on_failed_cleanup is not None:
            on_failed_cleanup()

    def build(side: str, factory: Callable[[], Any]) -> None:
        try:
            robot = _bring_up_one_arm(side, factory)
        except BaseException as exc:
            with condition:
                errors[side] = exc
                completed.add(side)
                condition.notify_all()
            finish_cleanup()
        else:
            with condition:
                discard = cancelled
                if not discard:
                    robots[side] = robot
                if not discard:
                    completed.add(side)
                    condition.notify_all()
            if discard:
                _close_after_failed_bring_up(side, robot)
                with condition:
                    completed.add(side)
                    condition.notify_all()
            finish_cleanup()

    workers = [
        threading.Thread(
            target=build, args=(side, factory), name=f"yam-arm-init-{side}", daemon=True
        )
        for side, factory in (("left", left_factory), ("right", right_factory))
    ]
    started: list[threading.Thread] = []
    deadline = time.monotonic() + timeout_s
    try:
        for worker in workers:
            worker.start()
            started.append(worker)
        with condition:
            while len(completed) < 2:
                for error in errors.values():
                    if not isinstance(error, Exception):
                        raise error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "arm initialization deadline exceeded; use the hardware stop"
                    )
                condition.wait(min(remaining, 0.05))
        for error in errors.values():
            if not isinstance(error, Exception):
                raise error
        for side in ("left", "right"):
            if side in errors:
                raise errors[side]
        return robots["left"], robots["right"]
    except BaseException:
        with condition:
            cancelled = True
            completed.update(
                side
                for side, worker in zip(("left", "right"), workers, strict=True)
                if worker not in started
            )
            survivors = dict(robots)
            robots.clear()
        for side, error in list(errors.items()):
            logger.error("%s arm bring-up failed: %s", side, error)
        for side, robot in survivors.items():
            _close_after_failed_bring_up(side, robot)
        cleanup_ready = True
        # Briefly let already-completing workers finish their late-result cleanup.
        for worker in started:
            worker.join(timeout=0.05)
        finish_cleanup()
        raise


def _bring_up_one_arm(side: str, factory: Callable[[], Any]) -> Any:
    logger.info("%s arm bring-up starting", side)
    robot = factory()
    logger.info("%s arm bring-up complete", side)
    return robot


def _close_after_failed_bring_up(side: str, robot: Any) -> None:
    try:
        close_robot_safely(robot)
    except Exception:
        logger.warning("closing the %s arm after a failed bring-up failed", side, exc_info=True)


def _load_i2rt() -> tuple[Any, Any]:
    """Load the git-only YAM driver symbols with actionable installation guidance."""
    try:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType
    except ModuleNotFoundError as exc:
        if exc.name != "i2rt" and not (exc.name or "").startswith("i2rt."):
            raise
        raise ModuleNotFoundError(
            "i2rt is the I2RT YAM arm driver. It is git-only and not on PyPI. "
            f"Install or update it with: {I2RT_INSTALL_COMMAND}",
            name=exc.name,
        ) from exc
    return get_yam_robot, GripperType


def _load_i2rt_kinematics() -> tuple[Any, Any, Any, Any, Any]:
    """Load optional i2rt kinematics symbols without affecting package imports."""
    try:
        from i2rt.robots.kinematics import Kinematics
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
        from mink import NoSolutionFound
    except ModuleNotFoundError as exc:
        if exc.name != "i2rt" and not (exc.name or "").startswith("i2rt."):
            raise
        raise ModuleNotFoundError(
            "i2rt is the I2RT YAM arm driver. It is git-only and not on PyPI. "
            f"Install or update it with: {I2RT_INSTALL_COMMAND}",
            name=exc.name,
        ) from exc
    return Kinematics, ArmType, GripperType, combine_arm_and_gripper_xml, NoSolutionFound
