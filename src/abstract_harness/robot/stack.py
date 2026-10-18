"""The rig's software, in this process: the robot core with its agent and YAM components,
i2rt and librealsense.

The hardware, camera and recording libraries come with the ``robot`` extra (``uv sync
--extra robot``). Every import of the core and of them is local to the function that
needs it, so the rest of the harness and its tests run without the extra.
"""

from __future__ import annotations

import io
import os
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from abstract_harness.robot import events
from abstract_harness.robot.config import EMBODIMENT, RobotConfig

CORE_ROBOT_PACKAGES = (
    "numpy",
    "rerun-sdk",
    "opencv-python-headless",
)
PACKAGES = (
    "numpy",
    "rerun-sdk",
    "json-numpy",
    "opencv-python-headless",
    "mujoco",
    "i2rt",
    "pyrealsense2",
)
YAM_MOTOR_IDS = range(1, 8)  # six joints and the gripper, per arm


def installed() -> tuple[bool, str]:
    """Whether the YAM hardware stack is installed, with its package versions."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            return False, f"{name} is not installed: uv sync --extra robot"
    return True, ", ".join(f"{name} {value}" for name, value in versions.items())


def installed_for(embodiment: str = EMBODIMENT) -> tuple[bool, str]:
    """Whether the runtime packages required by ``embodiment`` are installed."""
    if embodiment == EMBODIMENT:
        return installed()
    versions = {}
    for name in CORE_ROBOT_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            return False, f"{name} is not installed: uv sync --extra robot"
    try:
        from abstract_harness.robot.core.conformance import missing_runtime_requirements
        from abstract_harness.robot.core.registry import registered

        factory = registered("embodiment").get(embodiment)
        missing = missing_runtime_requirements(factory) if factory is not None else {}
        if missing:
            mod, remedy = next(iter(missing.items()))
            return False, f"{mod} is not installed: {remedy}"
    except Exception:
        pass
    return True, ", ".join(f"{name} {value}" for name, value in versions.items())


@contextmanager
def environment(**values: str) -> Iterator[None]:
    """Set environment variables for one episode (the agent's API key), then restore them."""
    saved = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _exit_code(exc: SystemExit) -> int:
    """A refusal from the core (an unknown component, invalid arguments) as an exit code,
    its message on stderr."""
    if isinstance(exc.code, str):
        print(exc.code, file=sys.stderr, flush=True)
        return 1
    return exc.code or 0


def episode(
    config: RobotConfig,
    instruction: str,
    log_dir: Path,
    *,
    source: str,
    base_url: str | None = None,
    policy_args: Mapping[str, Any] | None = None,
    embodiment_args: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    embodiment: str | None = None,
    max_steps: int | None = None,
    prompt: bool = True,
    store_frames: bool = True,
    rerun_save: bool = True,
    grade: bool = True,
) -> int:
    """One episode of ``instruction``: the config's policy on ``config.rig.type`` (or
    ``embodiment``, which then gets only ``embodiment_args``), with ``env`` set for its
    duration. The operator's ``policy_args`` and ``embodiment_args`` come last and win.
    ``source`` names the config in the run header.
    """
    from abstract_harness.robot.core import commands

    target_embodiment = config.rig.type if embodiment is None else embodiment
    policy_kwargs = {**config.policy_args(base_url), **(policy_args or {})}
    defaults = (
        config.embodiment_args() if embodiment is None or embodiment == config.rig.type else {}
    )
    embodiment_kwargs = {**defaults, **(embodiment_args or {})}
    with environment(**(env or {})):
        try:
            task = commands.adhoc_task(instruction, max_steps or config.rig.max_steps)
            parts = commands.components(
                config.policy_name, policy_kwargs, target_embodiment, embodiment_kwargs, source
            )
            code = commands.run_task(
                task,
                parts,
                log_dir=str(log_dir),
                prompt=prompt,
                store_frames=store_frames,
                rerun_save=rerun_save,
                grade=grade,
            )
        except KeyboardInterrupt:
            code = 130
        except SystemExit as exc:
            code = _exit_code(exc)
    events.emit("episode_end", exit_code=code)
    return code


def doctor(config: RobotConfig, source: str) -> int:
    """The conformance audit of the configured embodiment: constructed, never connected
    or moved."""
    from abstract_harness.robot.core import commands

    try:
        return commands.doctor(config.rig.type, config.embodiment_args(), source)
    except SystemExit as exc:
        return _exit_code(exc)


def captured(function: Callable[..., int], *args: Any, **kwargs: Any) -> tuple[int, str]:
    """``function``'s exit code with its terminal output captured, for unattended calls."""
    output = io.StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        code = function(*args, **kwargs)
    return code, output.getvalue()


def motor_timeouts(channel: str) -> dict[str, int | str]:
    """Each motor's CAN loss-of-comms TIMEOUT register on one arm, through i2rt's
    register-tool code; a string value means that motor did not answer.

    Parameter reads only: nothing is enabled, disabled, written or saved. TIMEOUT 0 turns
    the failsafe off, and the motor keeps executing its last command after its controller
    dies (seen on a real arm).
    """
    try:
        from i2rt.motor_config_tool.dm_motor_registers import read_register
        from i2rt.motor_config_tool.utils import RawCanInterface

        bus = RawCanInterface(channel=channel, bustype="socketcan", name="abstract-robot-check")
    except Exception as exc:
        return {"read": f"{type(exc).__name__}: {exc}"}
    values: dict[str, int | str] = {}
    try:
        for motor_id in YAM_MOTOR_IDS:
            try:
                values[str(motor_id)] = int(read_register(bus, motor_id, "TIMEOUT"))
            except Exception as exc:
                values[str(motor_id)] = type(exc).__name__
    finally:
        bus.close()
    return values


def realsense_devices() -> list[dict[str, Any]] | str:
    """Connected RealSense devices through librealsense, or why they could not be listed.
    Enumeration and device info only; no stream is started. These serials differ from the
    USB descriptor serials."""
    try:
        import pyrealsense2 as rs

        devices = []
        for device in rs.context().query_devices():

            def info(key: Any, device: Any = device) -> Any:
                return device.get_info(key) if device.supports(key) else None

            devices.append(
                {
                    "serial": info(rs.camera_info.serial_number),
                    "name": info(rs.camera_info.name),
                    "usb": info(rs.camera_info.usb_type_descriptor),
                }
            )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return devices


def policy_probe(config: RobotConfig) -> dict[str, Any]:
    """One real inference through the configured action policy without touching hardware:
    the embodiment is constructed (inert until reset) for the compatibility check, and
    the observation is synthetic (black frames at the policy's camera specs, and the
    embodiment's home/zero state)."""
    out: dict[str, Any] = {}
    try:
        import numpy as np

        from abstract_harness.robot.core import Observation, Scene, resolve
        from abstract_harness.robot.core.compat import check_compatibility
        from abstract_harness.robot.yam.config import DEFAULT_JOINT_HOME_POSE
    except ImportError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    try:
        embodiment = resolve("embodiment", config.rig.type, **config.embodiment_args())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    try:
        policy = resolve("policy", config.policy_name, **config.policy_args())
        bind_fn = getattr(policy, "bind", None)
        if callable(bind_fn):
            bind_fn(embodiment.info)
        report = check_compatibility(policy, embodiment)
        out["compat_errors"] = [f"{i.code}: {i.message}" for i in report.warnings if False] + [
            f"{i.code}: {i.message}" for i in report.errors
        ]
        out["compat_warnings"] = [f"{i.code}: {i.message}" for i in report.warnings]
        cams = policy.info.observation_space.cameras or embodiment.info.observation_space.cameras
        images = {c.name: np.zeros((c.height, c.width, c.channels), np.uint8) for c in cams}
        space = embodiment.info.action_space
        dim = int(space.dim)
        if config.rig.type == EMBODIMENT and dim == len(DEFAULT_JOINT_HOME_POSE):
            state = np.array(DEFAULT_JOINT_HOME_POSE, np.float64)
        elif space.low is not None and space.high is not None:
            state = np.clip(np.zeros(dim, np.float64), space.low, space.high)
        else:
            state = np.zeros(dim, np.float64)
        state_keys = list(
            policy.info.observation_space.state_keys
            or embodiment.info.observation_space.state_keys
            or ("joint_pos",)
        )
        obs_state = {key: state.copy() for key in state_keys}
        policy.reset(Scene(id="abstract-robot-check", instruction="hold still"))
        started = time.perf_counter()
        chunk = policy.act(Observation(images=images, state=obs_state))
        out["latency_s"] = round(time.perf_counter() - started, 3)
        actions = np.array([action.data for action in chunk.actions], np.float64)
        out["expected_dim"] = dim
        out["chunk_shape"] = list(actions.shape)
        out["first_step_rad"] = round(float(np.abs(actions[0] - state).max()), 4)
        if space.low is not None and space.high is not None:
            out["outside_bounds"] = int(((actions < space.low) | (actions > space.high)).sum())
        else:
            out["outside_bounds"] = 0
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        embodiment.close()
    return out


vla_probe = policy_probe
