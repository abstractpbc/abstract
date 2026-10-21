"""``preset``: any robot described by a preset TOML, driven through its driver.

The embodiment is built from the resolved preset (``description.py``): its action space,
proprioception, cameras, rate and operating notes all come from the TOML and the model
file, so a new robot needs a preset and a driver, not a new embodiment class. It owns
what the driver must not decide alone:

* **Lifecycle.** The driver is constructed and connected at the first reset, never in
  the constructor, and stopped and closed in ``close()``.
* **Readiness gates.** On a physical driver, the operator confirms before the home ramp
  and before the episode starts (``auto_start = true`` replaces both with a notice).
  Without an attended terminal a physical preset refuses to reset unless it auto-starts.
* **Home.** ``[robot.joints] home`` ramps the measured pose there over ``home_secs``, no
  faster than the per-step limits.
* **Pacing.** ``step()`` waits out the policy period (``pace = true``, the default), so
  steps never run faster than ``policy_hz`` whatever the policy does.
* **Watchdog.** For rate modes (``base_vel``, ``joint_vel``), a step that does not arrive
  within ``command_timeout_s`` stops the driver; motion resumes with the next step, and
  the acceleration limit ramps it up again from rest. A driver whose ``stop()`` fails
  faults the episode.
* **Fall detection.** Past ``max_tilt_deg`` from upright, the driver stops and the episode
  ends (``fall_detected``); a reset refuses to start from a fallen robot.
* **Position limits.** In ``joint_delta`` and ``joint_vel`` modes, whose action bounds
  limit steps and velocities, a command that would carry a joint past the model's
  position limits is cut to stop at the limit.

Every action reaching the driver has passed the approver chain: the bounds clamp, the
per-step limit, and for rate modes the acceleration limit from ``max_accel``.
"""

from __future__ import annotations

import inspect
import math
import sys
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from abstract_harness.robot.core import registry
from abstract_harness.robot.core.approver import AccelerationLimitApprover, GuardrailContribution
from abstract_harness.robot.core.command_stream import CommandStream
from abstract_harness.robot.core.embodiment import EmbodimentBase
from abstract_harness.robot.core.errors import ConfigError, EmbodimentFault
from abstract_harness.robot.core.scene import Scene
from abstract_harness.robot.core.spaces import RATE_CONTROL_MODES, Box
from abstract_harness.robot.core.timing import Pacer
from abstract_harness.robot.core.types import Action, Observation, StepResult
from abstract_harness.robot.preset.description import PresetError, RobotDescription, load_preset

FALL = "fall_detected"


def _flag(value: object, name: str) -> bool | None:
    """A boolean constructor argument, also from its -E string form."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise EmbodimentFault(f"{name} must be true or false, got {value!r}")


class PresetEmbodiment(EmbodimentBase):
    """A preset robot. ``preset`` is the TOML path; ``driver``, ``pace`` and
    ``auto_start`` override the preset's (``-E driver=sim`` dry-runs a physical preset);
    other keyword arguments override ``[robot.driver_args]``."""

    def __init__(
        self,
        preset: str,
        *,
        driver: str | None = None,
        pace: bool | str | None = None,
        auto_start: bool | str | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        **driver_args: Any,
    ) -> None:
        self.description: RobotDescription = load_preset(preset)
        section = self.description.section
        self.driver_spec = driver or section.driver
        try:
            self._factory = registry.lookup(
                "driver", self.driver_spec, base=self.description.path.parent
            )
        except KeyError as exc:
            raise PresetError(f"driver: {exc.args[0]}") from None
        except ConfigError as exc:
            raise PresetError(f"driver: {exc}") from exc
        self.simulated = getattr(self._factory, "simulated", False) is True
        paced = _flag(pace, "pace")
        self._pace = section.pace if paced is None else paced
        started = _flag(auto_start, "auto_start")
        self._auto_start = section.auto_start if started is None else started
        self._driver_args = {**self.description.driver_args, **driver_args}
        try:
            inspect.signature(self._factory).bind(self.description, **self._driver_args)
        except TypeError as exc:
            raise ConfigError(
                f"driver {self.driver_spec!r} does not take these arguments "
                f"({sorted(self._driver_args)}): {exc}"
            ) from None
        except ValueError:
            pass  # no introspectable signature (a builtin or C extension): checked at reset
        self.info = self.description.embodiment_info(simulated=self.simulated, pace=self._pace)
        self._clock = clock
        self._sleep = sleep
        self._driver: Any = None
        self._lock = threading.RLock()
        self._session: Any = None
        self._pacer: Pacer | None = None
        self._last_command: float | None = None
        self._watchdog_stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        self._halted = False
        self.watchdog_stops = 0
        base = section.base
        self._rate_timeout = self.description.command_timeout_s
        self._max_tilt = base.max_tilt_deg if base is not None else None
        self._home_confirmed = False
        self._fault: str | None = None
        self._rested = False
        self._accel_limited = False

    # -- optional hooks ------------------------------------------------------------

    def connect_operator_session(self, session: Any) -> None:
        """Gates and notices go through the framework session; never stdin directly."""
        self._session = session

    def command_stream(self) -> CommandStream | None:
        """The preset's live command axes and controllers (``[[robot.commands]]``)."""
        return self.description.command_stream()

    def contribute_guardrails(self, action_space: Box) -> GuardrailContribution:
        """The acceleration limit, for a rate mode with ``max_accel``."""
        limits = self.description.max_accel
        if limits is None:
            return GuardrailContribution()
        span = self.description.high - self.description.low
        per_step = [
            limit / self.description.policy_hz if limit is not None else float(span[index]) or 1.0
            for index, limit in enumerate(limits)
        ]
        approver = AccelerationLimitApprover(action_space, per_step, rest=self._take_rest)
        self._accel_limited = True
        return GuardrailContribution(approvers=(("accel-limit", approver),))

    def _take_rest(self) -> bool:
        """Whether the driver was stopped outside the policy's control since the last
        call (watchdog, fall): the acceleration limit then ramps from rest again."""
        with self._lock:
            rested, self._rested = self._rested, False
        return rested

    def doctor_notes(self) -> str:
        """What the preset resolved to, for ``abstract robot check``."""
        return self.description.summary()

    def check_health(self) -> None:
        """The driver's optional health check, run while inference is pending."""
        if self._fault is not None:
            raise EmbodimentFault(self._fault)
        driver = self._driver
        health = getattr(driver, "health", None) if driver is not None else None
        if callable(health):
            try:
                health()
            except EmbodimentFault:
                raise
            except Exception as exc:
                raise EmbodimentFault(f"driver health check failed: {exc}") from exc

    # -- lifecycle -----------------------------------------------------------------

    def _ensure_driver(self) -> Any:
        if self._driver is None:
            try:
                driver = self._factory(self.description, **self._driver_args)
            except TypeError as exc:
                raise EmbodimentFault(f"driver {self.driver_spec!r}: {exc}") from exc
            missing = [
                name
                for name in ("connect", "read", "command", "stop", "close")
                if not callable(getattr(driver, name, None))
            ]
            if missing:
                raise EmbodimentFault(f"driver {self.driver_spec!r} lacks {missing}")
            try:
                driver.connect()
            except BaseException:
                # A half-opened driver must not leak its ports or SDK session.
                try:
                    driver.close()
                except Exception:  # the connect failure is the one to report
                    pass
                raise
            self._driver = driver
        return self._driver

    def _notice(self, text: str) -> None:
        if self._session is not None:
            self._session.write_line(text)
        else:
            print(text, file=sys.stderr, flush=True)

    def _require_attended(self) -> None:
        if not self.simulated and not self._auto_start and self._session is None:
            raise EmbodimentFault(
                f"preset {self.description.name!r} drives a physical robot and needs an "
                "attended run to confirm readiness (abstract robot run), or auto_start = "
                "true with someone at the hardware stop"
            )

    def _gate(self, prompt: str, notice: str) -> None:
        if self.simulated:
            return
        if self._auto_start:
            self._notice(f"auto_start: {notice}")
            return
        self._require_attended()
        self._session.gate(prompt, hint="Set auto_start = true in the preset to skip.")

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self._stop_watchdog()
        # Refused before the driver connects: an unattended physical run opens nothing.
        self._require_attended()
        if self._fault is not None:
            raise EmbodimentFault(self._fault)
        driver = self._ensure_driver()
        with self._lock:
            driver.stop()
            self._halted = False
            self._rested = True
        state = self._state(driver)
        tilt = self._tilt(state)
        if self._max_tilt is not None and tilt is not None and tilt > self._max_tilt:
            raise EmbodimentFault(
                f"the robot is tilted {tilt:.0f} degrees from upright (max_tilt_deg "
                f"{self._max_tilt:g}): stand it up before starting an episode"
            )
        home = self.description.home
        if home is not None and any(value is not None for value in home):
            if not self._home_confirmed:
                self._gate(
                    "Robot will move to its home pose - stand clear, then press Enter...",
                    "the robot will move to its home pose - stand clear.",
                )
                self._home_confirmed = True
            self._ramp_home(driver, home)
        self._gate(
            "Position the scene, then press Enter to start...",
            "the episode starts now.",
        )
        self._pacer = (
            Pacer(self.description.policy_hz, clock=self._clock, sleep=self._sleep)
            if self._pace
            else None
        )
        self._last_command = None
        if self.description.mode in RATE_CONTROL_MODES:
            self._start_watchdog()
        return self._observe(driver, scene.instruction)

    def _ramp_home(self, driver: Any, home: tuple[float | None, ...]) -> None:
        """Straight to the home pose from the measured one, every step within the per-step
        limits. Waypoints are not clipped to the limits: a joint the operator left past a
        narrowed limit walks back in at the step limit instead of jumping to the bound,
        and a joint without a home value does not move."""
        measured = self._state(driver)["joint_pos"]
        target = np.array([measured[i] if value is None else value for i, value in enumerate(home)])
        description = self.description
        limit = np.array(
            [
                step if step is not None else 0.05 * span
                for step, span in zip(
                    description.max_step or (None,) * description.dim,
                    description.high - description.low,
                    strict=True,
                )
            ]
        )
        distance = np.abs(target - measured)
        movable = limit > 0
        by_limit = int(np.ceil(np.max(distance[movable] / limit[movable]))) if movable.any() else 1
        steps = max(
            1, math.ceil(description.section.joints.home_secs * description.policy_hz), by_limit
        )
        period = 1.0 / description.policy_hz
        for fraction in np.linspace(1.0 / steps, 1.0, steps):
            with self._lock:
                driver.command(measured + (target - measured) * fraction)
            self._sleep(period)

    def step(self, action: Action) -> StepResult:
        driver = self._driver
        if driver is None:
            raise EmbodimentFault("step() before reset()")
        if self._fault is not None:
            raise EmbodimentFault(self._fault)
        data = np.asarray(action.data, dtype=np.float64).reshape(self.description.dim)
        if self._pacer is not None:
            self._pacer.wait()
        position_clamped = False
        with self._lock:
            if self.description.position_limits is not None:
                # Read just before commanding: the joint has moved since the last step.
                fresh = self._state(driver)["joint_pos"]
                data, position_clamped = self._within_position_limits(data, fresh)
            if self._rested and self._accel_limited:
                # The watchdog stopped the base after this action was approved against
                # the old velocity: send rest now; the next one ramps up from it.
                data = np.zeros_like(data)
            driver.command(data)
            self._last_command = self._clock()
            self._halted = False
        command_time = self._last_command
        observation = self._observe(driver, None)
        fallen = self._fallen(observation)
        if fallen is not None:
            with self._lock:
                driver.stop()
                self._halted = True
                self._rested = True
        key = "joint_pos" if self.description.mode == "joint_pos" else self.description.mode
        info: dict[str, Any] = {"tilt_deg": fallen} if fallen is not None else {}
        if position_clamped:
            info["position_clamped"] = True
        return StepResult(
            observation=observation,
            terminated=fallen is not None,
            termination_reason=FALL if fallen is not None else None,
            info=info,
            command={key: data.copy()},
            command_time=command_time,
        )

    def _within_position_limits(
        self, data: npt.NDArray[np.float64], measured: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], bool]:
        """For delta and velocity joint modes: shorten a command that would carry a joint
        past the model's position limits so it stops at the limit. A command is only ever
        shortened, never reversed or enlarged, so a joint already past a limit can move
        back in but is never pushed. A delta applies to the measured position; a velocity
        is assumed to last until the watchdog would stop it (``command_timeout_s``), the
        longest it can persist without another step."""
        limits = self.description.position_limits
        if limits is None:
            return data, False
        low, high = limits
        dt = 1.0 if self.description.mode == "joint_delta" else self._rate_timeout
        with np.errstate(invalid="ignore"):
            room_up = np.where(np.isnan(high), np.inf, np.maximum(high - measured, 0.0)) / dt
            room_down = np.where(np.isnan(low), -np.inf, np.minimum(low - measured, 0.0)) / dt
        limited = np.where(data > 0, np.minimum(data, room_up), np.maximum(data, room_down))
        if np.array_equal(limited, data):
            return data, False
        return limited, True

    def close(self) -> None:
        self._stop_watchdog()
        driver, self._driver = self._driver, None
        if driver is None:
            return
        try:
            with self._lock:
                driver.stop()
        finally:
            driver.close()

    # -- observation and safety ------------------------------------------------------

    def _state(self, driver: Any) -> dict[str, npt.NDArray[np.float64]]:
        raw = driver.read()
        if not isinstance(raw, Mapping):
            raise EmbodimentFault(f"driver read() returned {type(raw).__name__}, not a mapping")
        state: dict[str, npt.NDArray[np.float64]] = {}
        for state_field in self.description.state_fields:
            if state_field.key not in raw:
                raise EmbodimentFault(f"driver read() is missing state {state_field.key!r}")
            try:
                value = np.asarray(raw[state_field.key], dtype=np.float64).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise EmbodimentFault(f"driver state {state_field.key!r}: {exc}") from exc
            if value.shape != state_field.shape:
                raise EmbodimentFault(
                    f"driver state {state_field.key!r} has shape {value.shape}, "
                    f"expected {state_field.shape}"
                )
            if not np.isfinite(value).all():
                raise EmbodimentFault(f"driver state {state_field.key!r} is not finite")
            state[state_field.key] = value
        return state

    def _observe(self, driver: Any, instruction: str | None) -> Observation:
        state = self._state(driver)
        state_time = self._clock()
        images: dict[str, Any] = {}
        image_times: dict[str, float] = {}
        if self.description.cameras:
            grab = getattr(driver, "images", None)
            frames = grab() if callable(grab) else {}
            for camera in self.description.cameras:
                frame = frames.get(camera.name) if isinstance(frames, Mapping) else None
                if frame is None:
                    raise EmbodimentFault(f"driver gave no frame for camera {camera.name!r}")
                frame = np.asarray(frame)
                expected = (camera.height, camera.width, camera.channels)
                if frame.shape != expected or frame.dtype != np.uint8:
                    raise EmbodimentFault(
                        f"camera {camera.name!r} frame is {frame.dtype}{frame.shape}, "
                        f"expected uint8{expected}"
                    )
                images[camera.name] = frame
                image_times[camera.name] = state_time
        return Observation(
            images=images,
            state=state,
            instruction=instruction,
            image_times=image_times,
            state_time=state_time,
        )

    def _fallen(self, observation: Observation) -> float | None:
        """The base's tilt in degrees when it exceeds ``max_tilt_deg``, else ``None``."""
        if self._max_tilt is None:
            return None
        tilt = self._tilt(observation.state)
        return tilt if tilt is not None and tilt > self._max_tilt else None

    @staticmethod
    def _tilt(state: Mapping[str, Any]) -> float | None:
        """Degrees from upright, from projected gravity or the base quaternion."""
        gravity = state.get("projected_gravity")
        if gravity is not None:
            norm = float(np.linalg.norm(gravity))
            if norm == 0:
                return None
            return math.degrees(math.acos(max(-1.0, min(1.0, -float(gravity[2]) / norm))))
        quat = state.get("base_quat")
        if quat is None:
            return None
        _, x, y, _ = (float(v) for v in quat)
        return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))

    def _start_watchdog(self) -> None:
        self._watchdog_stop.clear()
        self._watchdog = threading.Thread(target=self._watch, name="preset-watchdog", daemon=True)
        self._watchdog.start()

    def _stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        thread, self._watchdog = self._watchdog, None
        if thread is not None:
            thread.join(timeout=2)

    def _watch(self) -> None:
        interval = min(0.05, self._rate_timeout / 4)
        while not self._watchdog_stop.wait(interval):
            last = self._last_command
            if last is None or self._halted:
                continue
            if self._clock() - last > self._rate_timeout:
                with self._lock:
                    driver = self._driver
                    if driver is None or self._halted or self._last_command != last:
                        continue
                    try:
                        driver.stop()
                    except Exception as exc:
                        # Keep trying every interval; the next step or health check faults
                        # the episode, so it cannot run on past a failed stop.
                        self._fault = f"the watchdog could not stop the driver: {exc}"
                        continue
                    self._halted = True
                    self._rested = True
                    self.watchdog_stops += 1
