"""Unified VLA policy (`policy("vla")`): cloud `/v1` gateway, local `/v1` or `/act` HTTP
server, or an in-process local Python model/callable.

One entry point (`vla_policy`) dispatches based on how the VLA is hosted:

1. **In-process local Python callable / model (`fn` or `target="module:callable"`):**
   Runs the model directly in this process and automatically adopts the embodiment's
   action and observation spaces via `bind()`.
2. **Cloud or local `/v1` gateway (`model="origin/model:embodiment"` or `/v1` URL):**
   Calls `POST /v1/chat/completions` (e.g. `https://api.abstract.inc/v1` or a local
   server at `http://127.0.0.1:8000/v1`), maintaining a per-camera rolling frame ring
   buffer (`history_frames`) for stateful world-action models like DreamZero.
3. **Local `/act` HTTP server (`server_url="http://127.0.0.1:8202"` without `model`):**
   Calls `ActServerPolicy` over `json_numpy`.
"""

from __future__ import annotations

import base64
import importlib
import inspect
import io
import json
import math
import os
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
from PIL import Image

from abstract_harness.robot.config import plain_api_url
from abstract_harness.robot.core.embodiment import EmbodimentInfo
from abstract_harness.robot.core.errors import ConfigError, PolicyError
from abstract_harness.robot.core.policy import Policy, PolicyBase, PolicyConfig, PolicyInfo
from abstract_harness.robot.core.scene import Scene
from abstract_harness.robot.core.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ControlMode,
    ObservationSpace,
    StateField,
    StateSpec,
)
from abstract_harness.robot.core.types import Action, ActionChunk, Observation
from abstract_harness.robot.yam import packing
from abstract_harness.robot.yam.config import action_box

DEFAULT_GATEWAY_URL = "https://api.abstract.inc/v1"
DEFAULT_GATEWAY_MODEL = "abstract/dreamzero-astra:yam"
API_KEY_ENV = "ABSTRACT_API_KEY"

GATEWAY_MODELS: dict[str, dict[str, Any]] = {
    "abstract/dreamzero-astra:yam": {
        "history_offsets": (-23, -16, -8, 0),
        "history_frames": 4,
        "action_horizon": 24,
        "stateful": True,
        "cameras": ("top", "left", "right"),
    },
    "stealth/picasso:yam": {
        "history_offsets": (-23, -16, -8, 0),
        "history_frames": 4,
        "action_horizon": 24,
        "stateful": True,
        "cameras": ("top", "left", "right"),
    },
    "allenai/molmoact2:yam": {
        "history_frames": 1,
        "action_horizon": 30,
        "stateful": False,
        "cameras": ("top", "left", "right"),
    },
    "physical-intelligence/pi0.5:yam": {
        "history_frames": 1,
        "action_horizon": 30,
        "stateful": False,
        "cameras": ("top", "left", "right"),
    },
    "perceptron/isaac-0.5:yam": {
        "history_frames": 1,
        "action_horizon": 30,
        "stateful": False,
        "cameras": ("top", "left", "right"),
    },
}

GatewayPostFn = Callable[[str, Mapping[str, Any], Mapping[str, str], float], Mapping[str, Any]]


def _encode_jpeg_data_url(image: np.ndarray, quality: int = 85) -> str:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(f"camera frame must be HxWx3 uint8 RGB, got {arr.shape} {arr.dtype}")
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _default_gateway_post(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_s: float,
) -> Mapping[str, Any]:
    response = httpx.post(url, json=payload, headers=dict(headers), timeout=timeout_s)
    response.raise_for_status()
    data: Mapping[str, Any] = response.json()
    return data


def _parse_gripper_indices(value: tuple[int, ...] | list[int] | str) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        try:
            return tuple(int(p) for p in parts)
        except ValueError:
            raise ConfigError("indices must be integers") from None
    if any(type(x) is not int for x in value):
        raise ConfigError("indices must be integers")
    return tuple(value)


def _is_cloud_abstract_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "api.abstract.inc"
        and parsed.port in (None, 443)
    )


def _positive_int(name: str, value: object, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ConfigError(f"{name} must be an integer in 1..{maximum}")
    return value


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{name} must be finite and positive")
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{name} must be finite and positive")
    return float(value)


class GatewayVLAPolicy:
    """`Policy` driving any VLA on the Abstract Gateway or a local `/v1` server."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_GATEWAY_MODEL,
        server_url: str = DEFAULT_GATEWAY_URL,
        api_key: str | None = None,
        api_key_env: str | None = None,
        cam_height: int = 360,
        cam_width: int = 640,
        camera_order: tuple[str, ...] | str = ("top_cam", "left_cam", "right_cam"),
        state_key: str = packing.STATE_KEY,
        action_dim: int = packing.TOTAL_DIM,
        gripper_indices: tuple[int, ...] | str = (6, 13),
        action_horizon: int | None = None,
        history_frames: int | None = None,
        history_offsets: tuple[int, ...] | str | None = None,
        stateful: bool | None = None,
        num_inference_steps: int | None = None,
        return_video: bool = False,
        timeout_s: float = 30.0,
        post_fn: GatewayPostFn | None = None,
        policy_kind: str = "vla",
    ) -> None:
        preset = GATEWAY_MODELS.get(model, {})
        if isinstance(camera_order, str):
            cameras = tuple(part.strip() for part in camera_order.split(",") if part.strip())
        else:
            cameras = tuple(camera_order)
        if not cameras or len(set(cameras)) != len(cameras) or any(not c for c in cameras):
            raise ConfigError("camera_order must contain distinct, non-empty camera names")

        self.model = model
        self.policy_kind = policy_kind
        self.server_url = plain_api_url(server_url)
        if urlparse(self.server_url).path not in ("", "/v1"):
            raise ConfigError("server_url must have no path or the /v1 path")
        cloud = _is_cloud_abstract_url(self.server_url)
        if api_key_env == API_KEY_ENV and not cloud:
            raise ConfigError("ABSTRACT_API_KEY is scoped to https://api.abstract.inc")
        self.api_key_env = api_key_env or (API_KEY_ENV if cloud else None)
        self._explicit_key = api_key
        self.cam_height = _positive_int("cam_height", cam_height, 8192)
        self.cam_width = _positive_int("cam_width", cam_width, 8192)
        self.camera_order = cameras
        self.state_key = state_key
        self.action_dim = _positive_int("action_dim", action_dim, 1024)
        self.gripper_indices = _parse_gripper_indices(gripper_indices)
        if len(set(self.gripper_indices)) != len(self.gripper_indices) or any(
            index < 0 or index >= self.action_dim for index in self.gripper_indices
        ):
            raise ConfigError("gripper_indices must be distinct indices within action_dim")
        default_history = 4 if policy_kind == "world_model" else 1
        self.history_frames = _positive_int(
            "history_frames",
            history_frames
            if history_frames is not None
            else preset.get("history_frames", default_history),
            128,
        )
        self.stateful = bool(
            stateful
            if stateful is not None
            else preset.get("stateful", policy_kind == "world_model" or self.history_frames > 1)
        )
        offsets = history_offsets if history_offsets is not None else preset.get("history_offsets")
        self.history_offsets = (
            _parse_gripper_indices(offsets)
            if offsets is not None
            else tuple(range(1 - self.history_frames, 1))
        )
        if (
            len(self.history_offsets) != self.history_frames
            or self.history_offsets[-1] != 0
            or tuple(sorted(set(self.history_offsets))) != self.history_offsets
            or self.history_offsets[0] < -5999
        ):
            raise ConfigError("history_offsets must be distinct ascending offsets ending at zero")
        self._history_capacity = 1 - self.history_offsets[0]
        horizon = _positive_int(
            "action_horizon",
            action_horizon if action_horizon is not None else preset.get("action_horizon", 24),
            6000,
        )
        if num_inference_steps is not None:
            _positive_int("num_inference_steps", num_inference_steps, 1000)
        self.num_inference_steps = num_inference_steps
        self.return_video = bool(return_video)
        self.timeout_s = _positive_float("timeout_s", timeout_s)
        self._post_fn: GatewayPostFn = post_fn or _default_gateway_post

        self._instruction: str = ""
        self.session_id: str | None = None
        self.step: int = 0
        self.next_frames: int = 1
        self.num_inferences: int = 0
        self._last_observation: Observation | None = None
        self._new_observations = 0
        self._ring: dict[str, deque[str]] = {
            cam: deque(maxlen=self._history_capacity) for cam in self.camera_order
        }

        if self.action_dim == packing.TOTAL_DIM:
            act_space = action_box(joints_are_delta=False)
        else:
            act_space = Box(
                shape=(self.action_dim,),
                low=-np.full(self.action_dim, np.pi, dtype=np.float64),
                high=np.full(self.action_dim, np.pi, dtype=np.float64),
                semantics=ActionSemantics(
                    control_mode="joint_pos",
                    gripper="continuous" if self.gripper_indices else "none",
                ),
            )
        self.info = PolicyInfo(
            name=f"{policy_kind}:{model}",
            action_space=act_space,
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(c, self.cam_height, self.cam_width) for c in cameras),
                state_keys=frozenset({state_key}),
                state=StateSpec(
                    (StateField(state_key, (self.action_dim,), unit="rad+normalized"),)
                ),
            ),
            control_hz=None,
        )
        self.config = PolicyConfig(action_horizon=horizon)

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Fixed model semantics never change to fit a robot; compatibility checks them."""

    @property
    def remedy(self) -> str:
        return (
            f"configure the endpoint credential environment variable and verify "
            f"{self.server_url} serves {self.model!r}"
        )

    def _resolve_api_key(self) -> str:
        key = self._explicit_key or (
            os.environ.get(self.api_key_env, "").strip() if self.api_key_env else ""
        )
        if not key:
            if not _is_cloud_abstract_url(self.server_url):
                return ""
            raise ValueError(
                f"missing API key for {self.model!r}: set ${self.api_key_env} or pass api_key"
            )
        return key

    def reset(self, scene: Scene) -> None:
        self._instruction = scene.instruction or "perform the task"
        self.session_id = uuid.uuid4().hex if self.stateful else None
        self.step = 0
        self.next_frames = 1
        self.num_inferences = 0
        self._last_observation = None
        self._new_observations = 0
        cap = self._history_capacity
        self._ring = {cam: deque(maxlen=cap) for cam in self.camera_order}

    def observe(self, observation: Observation) -> None:
        """Collect every control observation, including steps served from a buffered chunk.

        History samples the configured offsets in control observations. Direct act()
        callers must feed intermediate observations through this hook.
        """
        if observation is self._last_observation:
            return
        expected_shape = (self.cam_height, self.cam_width, 3)
        staged: dict[str, deque[str]] = {
            cam: deque(self._ring[cam], maxlen=self._history_capacity) for cam in self.camera_order
        }
        for cam in self.camera_order:
            if cam not in observation.images:
                raise ValueError(f"observation missing camera {cam!r} required by {self.model}")
            img = observation.images[cam]
            if img is None:
                raise ValueError(f"camera {cam!r} has no frame, expected {expected_shape}")
            if img.shape != expected_shape:
                raise ValueError(f"camera {cam!r} has shape {img.shape}, expected {expected_shape}")
            staged[cam].append(_encode_jpeg_data_url(img))

        self._ring = staged
        self._last_observation = observation
        self._new_observations += 1

    def act(self, observation: Observation) -> ActionChunk:
        self.observe(observation)
        if self.state_key not in observation.state:
            raise ValueError(f"observation missing state key {self.state_key!r}")
        raw_state = np.asarray(observation.state[self.state_key], dtype=np.float64)
        if raw_state.shape != (self.action_dim,) or not np.isfinite(raw_state).all():
            raise ValueError(
                f"state {self.state_key!r} must be a finite ({self.action_dim},) vector, "
                f"got {raw_state.shape}"
            )
        g_set = set(self.gripper_indices)
        joint_vals = [float(raw_state[i]) for i in range(self.action_dim) if i not in g_set]
        grip_vals = [float(raw_state[i]) for i in self.gripper_indices]
        robot_state = {
            "joint_position": joint_vals,
            "gripper_position": grip_vals,
        }

        needed = self.next_frames
        if self.stateful and self.step > 0 and self._new_observations < self._history_capacity:
            raise PolicyError(
                "stateful continuation requires fresh control observations at every history offset"
            )
        image_parts: list[dict[str, Any]] = []
        for cam in self.camera_order:
            history = list(self._ring[cam])
            offsets = (0,) if needed == 1 else self.history_offsets
            frames = [history[len(history) - 1 + offset] for offset in offsets]
            for url in frames:
                image_parts.append({"type": "image_url", "image_url": {"url": url}})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._instruction or "perform the task"},
                        *image_parts,
                    ],
                }
            ],
            "robot_state": robot_state,
            "observation_id": uuid.uuid4().hex,
        }
        if self.stateful:
            if self.session_id is None:
                self.session_id = uuid.uuid4().hex
            body["session_id"] = self.session_id
            body["expected_step"] = self.step
        if self.config.action_horizon is not None:
            body["action_horizon"] = self.config.action_horizon
        if self.num_inference_steps is not None:
            body["num_inference_steps"] = self.num_inference_steps
        if self.return_video:
            body["return_video"] = True

        headers = {
            "content-type": "application/json",
            "accept": "application/json",
        }
        if key := self._resolve_api_key():
            headers["authorization"] = f"Bearer {key}"
        base = self.server_url if self.server_url.endswith("/v1") else f"{self.server_url}/v1"
        endpoint = f"{base}/chat/completions"
        t0 = time.perf_counter()
        reply = self._post_fn(endpoint, body, headers, self.timeout_s)
        elapsed = time.perf_counter() - t0

        try:
            content = reply["choices"][0]["message"]["content"]
            chunk = json.loads(content) if isinstance(content, str) else content
            raw_actions = np.asarray(chunk["actions"], dtype=np.float64)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"gateway returned unreadable action chunk: {exc!r}") from exc

        if raw_actions.ndim != 2 or raw_actions.shape[1] != self.action_dim:
            raise ValueError(
                f"gateway returned actions of shape {raw_actions.shape}; "
                f"expected (N, {self.action_dim})"
            )
        if (
            not 1 <= raw_actions.shape[0] <= self.config.action_horizon
            or not np.isfinite(raw_actions).all()
        ):
            raise ValueError("gateway returned empty or non-finite actions")
        if chunk.get("action_dim", self.action_dim) != self.action_dim:
            raise PolicyError("gateway returned an incompatible action_dim")
        if chunk.get("action_space", "joint_position") != "joint_position":
            raise PolicyError("gateway returned incompatible action semantics")
        dt = chunk.get("dt")
        chunk_hz = 1.0 / _positive_float("gateway dt", dt) if dt is not None else None
        next_frames = chunk.get("next_frames_per_camera", self.history_frames)
        if type(next_frames) is not int or next_frames != self.history_frames:
            raise PolicyError("gateway returned an invalid next_frames_per_camera")
        if self.stateful:
            returned_step = chunk.get("step")
            if type(returned_step) is not int or returned_step != self.step:
                raise PolicyError("gateway returned an unexpected session step")
            session_id = chunk.get("session_id", self.session_id)
            if not isinstance(session_id, str) or not session_id or len(session_id) > 256:
                raise PolicyError("gateway returned an invalid session_id")
            # Some gateways assign the public session id at the first step.
            if self.step > 0 and session_id != self.session_id:
                raise PolicyError("gateway changed session_id during an episode")
            self.session_id = session_id
            self.step += 1
            self.next_frames = next_frames
        self.num_inferences += 1
        self._new_observations = 0

        meta = {
            k: chunk[k]
            for k in (
                "session_id",
                "step",
                "timing_ms",
                "cache",
                "context",
                "predicted_video",
            )
            if k in chunk
        }
        return ActionChunk(
            actions=[Action(data=row.copy()) for row in raw_actions],
            control_hz=chunk_hz,
            inference_latency_s=elapsed,
            meta=meta,
        )


def _load_target(target: str | Callable[..., Any]) -> Callable[..., Any]:
    if callable(target):
        return target
    if not isinstance(target, str) or ":" not in target:
        raise ConfigError(
            f"local VLA `target` must be a callable or a 'module:attribute' string, got {target!r}"
        )
    mod_name, _, attr_name = target.partition(":")
    if not mod_name or not attr_name:
        raise ConfigError(f"invalid target {target!r}; expected 'module:attribute'")
    try:
        module = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        for candidate in (str(Path.cwd()), str(Path.cwd() / "tests")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
        try:
            module = importlib.import_module(mod_name)
        except Exception as exc:
            raise ConfigError(f"failed to import local VLA module {mod_name!r}: {exc}") from exc
    except Exception as exc:
        raise ConfigError(f"failed to import local VLA module {mod_name!r}: {exc}") from exc
    obj = getattr(module, attr_name, None)
    if obj is None:
        raise ConfigError(f"module {mod_name!r} has no attribute {attr_name!r}")
    if isinstance(obj, type):
        obj = obj()
    if not callable(obj):
        raise ConfigError(f"target {target!r} is not callable")
    return obj


def _accepts_two_positional(fn: Callable[..., Any]) -> bool:
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return False
    pos_count = 0
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            pos_count += 1
    return pos_count >= 2


class CallableVLAPolicy(PolicyBase):
    """Embodiment-adaptive `Policy` wrapping any in-process Python callable or model object."""

    def __init__(
        self,
        fn: str | Callable[..., Any] | None = None,
        *,
        target: str | Callable[..., Any] | None = None,
        name: str | None = None,
        action_dim: int | None = None,
        action_horizon: int = 1,
        control_mode: ControlMode | None = None,
        control_hz: float | None = None,
        action_space: Box | None = None,
        observation_space: ObservationSpace | None = None,
    ) -> None:
        resolved = fn if fn is not None else target
        if resolved is None:
            raise ConfigError("in-process VLA requires a callable `fn` or `target=module:callable`")
        self._fn = _load_target(resolved)
        self._passes_instruction = _accepts_two_positional(self._fn)
        self._explicit_action_space = action_space is not None
        self._explicit_obs_space = observation_space is not None
        self._explicit_action_dim = action_dim
        self._explicit_control_mode = control_mode
        self._control_hz = (
            _positive_float("control_hz", control_hz) if control_hz is not None else None
        )
        self._instruction: str = ""

        dim = _positive_int("action_dim", action_dim, 1024) if action_dim is not None else 14
        default_box = action_space or Box(
            shape=(dim,),
            low=-np.full(dim, np.pi, dtype=np.float64),
            high=np.full(dim, np.pi, dtype=np.float64),
            semantics=ActionSemantics(control_mode=control_mode or "joint_pos"),
        )
        inferred_name = name or (
            resolved
            if isinstance(resolved, str)
            else getattr(self._fn, "__name__", type(self._fn).__name__)
        )
        self.info = PolicyInfo(
            name=f"vla:{inferred_name}",
            action_space=default_box,
            observation_space=observation_space or ObservationSpace(),
            control_hz=self._control_hz,
        )
        self.config = PolicyConfig(
            action_horizon=_positive_int("action_horizon", action_horizon, 6000)
        )

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        new_action_space = (
            self.info.action_space
            if self._explicit_action_space
            or self._explicit_control_mode is not None
            or (
                self._explicit_action_dim is not None
                and self._explicit_action_dim != embodiment_info.action_space.dim
            )
            else embodiment_info.action_space
        )
        new_obs_space = (
            self.info.observation_space
            if self._explicit_obs_space
            else embodiment_info.observation_space
        )
        if (
            not self._explicit_action_space
            and self._explicit_control_mode is not None
            and self._explicit_action_dim is None
        ):
            dim = embodiment_info.action_space.dim
            new_action_space = Box(
                shape=(dim,),
                low=-np.full(dim, np.pi, dtype=np.float64),
                high=np.full(dim, np.pi, dtype=np.float64),
                semantics=ActionSemantics(control_mode=self._explicit_control_mode),
            )
        self.info = PolicyInfo(
            name=self.info.name,
            action_space=new_action_space,
            observation_space=new_obs_space,
            control_hz=self._control_hz,
            checkpoint=self.info.checkpoint,
        )

    def reset(self, scene: Scene) -> None:
        self._instruction = scene.instruction or ""
        reset_hook = getattr(self._fn, "reset", None)
        if callable(reset_hook):
            try:
                sig = inspect.signature(reset_hook)
            except (ValueError, TypeError):
                reset_hook()
            else:
                if len(sig.parameters) >= 1:
                    reset_hook(scene)
                else:
                    reset_hook()

    def act(self, observation: Observation) -> ActionChunk:
        t0 = time.perf_counter()
        if self._passes_instruction:
            raw = self._fn(observation, self._instruction)
        else:
            raw = self._fn(observation)
        elapsed = time.perf_counter() - t0

        if isinstance(raw, ActionChunk):
            if raw.inference_latency_s is None:
                return ActionChunk(
                    actions=raw.actions,
                    control_hz=raw.control_hz or self._control_hz,
                    inference_latency_s=elapsed,
                    meta=raw.meta,
                )
            return raw

        if isinstance(raw, Action):
            arr = np.asarray(raw.data, dtype=np.float64)[None, :]
        else:
            arr = np.asarray(raw, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[None, :]

        expected_dim = self._explicit_action_dim or self.info.action_space.dim
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != expected_dim:
            raise PolicyError(
                f"VLA policy {self.info.name!r} returned shape {arr.shape}; "
                f"expected (H, {expected_dim}) or ({expected_dim},)"
            )
        if not np.isfinite(arr).all():
            raise PolicyError(f"VLA policy {self.info.name!r} returned non-finite actions")

        return ActionChunk(
            actions=[
                Action(data=row.copy(), meta=dict(raw.meta) if isinstance(raw, Action) else {})
                for row in arr
            ],
            control_hz=self._control_hz,
            inference_latency_s=elapsed,
        )


def vla_policy(
    fn: str | Callable[..., Any] | None = None,
    *,
    target: str | Callable[..., Any] | None = None,
    model: str | None = None,
    server_url: str | None = None,
    **kwargs: Any,
) -> Policy:
    """Unified VLA factory (`policy("vla")`):

    - `fn` or `target="module:callable"` -> in-process `CallableVLAPolicy`
    - `model="..."` or `/v1` `server_url` (or default) -> `GatewayVLAPolicy`
    - `/act` `server_url` without `model` -> `ActServerPolicy`
    """
    if fn is not None or target is not None:
        return CallableVLAPolicy(fn, target=target, **kwargs)
    if (
        model is None
        and server_url is not None
        and not server_url.rstrip("/").endswith("/v1")
        and not _is_cloud_abstract_url(server_url)
    ):
        from abstract_harness.robot.yam.policy import ActServerPolicy

        return ActServerPolicy(server_url=server_url, **kwargs)
    return GatewayVLAPolicy(
        model=model or DEFAULT_GATEWAY_MODEL,
        server_url=server_url or DEFAULT_GATEWAY_URL,
        **kwargs,
    )


class WorldModelPolicy(GatewayVLAPolicy):
    """Stateful World-Action Model (`policy("world_model")`) with rolling video history,
    causal KV-cache tracking, and optional predicted-video output."""

    def __init__(self, *, model: str = DEFAULT_GATEWAY_MODEL, **kwargs: Any) -> None:
        kwargs.setdefault("policy_kind", "world_model")
        super().__init__(model=model, **kwargs)


def world_model_policy(
    fn: str | Callable[..., Any] | None = None,
    *,
    target: str | Callable[..., Any] | None = None,
    model: str | None = None,
    server_url: str | None = None,
    **kwargs: Any,
) -> Policy:
    """Unified World-Model factory (`policy("world_model")`):

    - `fn` or `target="module:callable"` -> in-process `CallableVLAPolicy`
    - `model="..."` or `/v1` `server_url` (or default) -> stateful `WorldModelPolicy`
    """
    if fn is not None or target is not None:
        return CallableVLAPolicy(fn, target=target, **kwargs)
    return WorldModelPolicy(
        model=model or DEFAULT_GATEWAY_MODEL,
        server_url=server_url or DEFAULT_GATEWAY_URL,
        **kwargs,
    )


# Convenience aliases for direct imports
AbstractGatewayPolicy = GatewayVLAPolicy
CallablePolicy = CallableVLAPolicy
local_policy = vla_policy
abstract_policy = vla_policy
