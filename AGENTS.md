# AGENTS.md

Instructions for coding agents working in `abstract-harness`.

## Overview

`abstract-harness` runs and evaluates robotics policies across simulated and physical embodiments. The core loop lives in `src/abstract_harness/robot/core/rollout.py`:

```text
Observation (from Embodiment.reset / step)
  -> Controller.next_action(Policy, Observation, t, store)
  -> Approver.review(Action, store)   # bounds clamp + per-step delta limit + collision check
  -> Embodiment.step(Action) -> StepResult
  -> LogSink (JSON, Live HTML, Rerun .rrd, LeRobot v3 export)
```

## Repository Map

- `src/abstract_harness/__init__.py` — Lazy PEP 562 top-level re-exports of the public API (`Policy`, `Embodiment`, `Controller`, `Action`, `ActionChunk`, `Observation`, `Scene`, `Task`, `eval`, `rollout`, `local_policy`, `joint_embodiment_info`, etc.).
- `src/abstract_harness/cli.py` — Root CLI (`abstract eval` and `abstract robot <subcommand>`).
- `src/abstract_harness/robot/core/` — Embodiment-agnostic evaluation and rollout engine:
  - `policy.py` — `Policy` protocol (`info`, `config`, `reset(scene)`, `act(obs) -> ActionChunk`, optional `bind(embodiment_info)`).
  - `embodiment.py` — `Embodiment` protocol (`info`, `reset(scene, *, seed=None)`, `step(action)`, `close()`) and `joint_embodiment_info()` helper.
  - `controller.py` — `DefaultController` (open-loop chunk execution with `replan_interval`), `SmoothingController` (EMA), `EnsemblingController` (ALOHA-style temporal chunk blending).
  - `approver.py` & `safety.py` — `ClampApprover`, `DeltaLimitApprover`, `ChainApprover`.
  - `conformance.py` — `check_embodiment(info)` static validation used by `abstract robot check` (`doctor`).
  - `registry.py` & `_builtins.py` — Component registry (`@policy`, `@embodiment`, `@task`, `@scorer`, `@grader`, `@sink`). `BUNDLED["policy"]` exposes two primary modes: `"agent"` and `"policy"`.
- `src/abstract_harness/robot/vla.py` — Unified `"policy"` implementation (`vla_policy`):
  - `GatewayVLAPolicy` / `WorldModelPolicy`: calls `POST /v1/chat/completions` on `https://api.abstract.inc/v1` or any local `/v1` server, maintaining a per-camera rolling frame ring buffer (`history_frames`) for stateful world-action models (`abstract/dreamzero-astra:yam`, `stealth/picasso:yam`) and 1-frame requests for stateless VLAs (`physical-intelligence/pi0.5:yam`, `allenai/molmoact2:yam`, `perceptron/isaac-0.5:yam`).
  - `CallableVLAPolicy` (`local_policy`): wraps any in-process Python function or callable object (`fn` or `target="module:callable"`) and auto-binds to the target embodiment's spaces via `bind()`.
  - Routes `/act` `server_url`s (without `model`) to `ActServerPolicy`.
- `src/abstract_harness/robot/agent/` — Tool-calling VLM/LLM agent policy (`move_to`, `done`, `give_up`) over OpenAI Responses, Anthropic, or Gemini Live, paired with the loopback spend meter (`robot/meter.py`).
- `src/abstract_harness/robot/yam/` — Physical bimanual I2RT YAM embodiment (`yam_arms`), RealSense multiprocess capture, MuJoCo self-collision guardrails, `/act` client, and operator tools (`health`, `holdcheck`, `pose`).
- `src/abstract_harness/robot/formats/` — Post-episode exporters for Rerun (`.rrd`) and LeRobot v3.0 datasets (`lerobot/`).
- `src/abstract_harness/robot/config.py`, `stack.py`, `cli.py`, `rpc.py`, `store.py`, `cloud.py` — TOML configuration (`[policy]`, `[agent]`, `[rig]`, `[data]`, `[store]`), motion-free `check` (`doctor` + `policy_probe` / `rehearse`), `run`, `serve` (HTTP RPC + SSE), and run uploads.

## Dependency Tiers

1. `uv sync --locked` — Base dependencies (`httpx`, `numpy`, `pillow`, `pydantic`, `websockets`). Runs `abstract eval`, `eval()`, `rollout()`, `policy`, and `agent`.
2. `uv sync --locked --extra robot` — Embodiment-agnostic robot runtime (`rerun-sdk`, `pyarrow`, `av`, `opencv-python-headless`, `mujoco`, `requests`, `json-numpy`). Standard PyPI wheels only; no hardware-specific git dependencies.
3. `uv sync --locked --extra yam --group yam-driver` — Adds Linux RealSense and the separate immutable Git-pinned i2rt driver group. The wheel extra never resolves a bare i2rt.

## Invariants & Guardrails for Agents

- **Never hardcode a single embodiment in `robot/core`, `robot/config.py`, `robot/stack.py`, or `robot/cli.py`.**
  - `RigConfig.type` defaults to `"yam_arms"` (which requires `left_channel` and `right_channel`), but any other registered embodiment (`type = "franka_arm"`, `"so100"`, `"cubepick"`, etc.) must work without YAM CAN channels or 14-DoF assumptions.
  - `stack.installed_for(rig.type)` only requires `i2rt` and `pyrealsense2` when `rig.type == "yam_arms"`.
- **Public API Snapshot Tests:**
  - `tests/robot_core/test_api_snapshot.py` and `tests/robot_yam/test_api_snapshot.py` assert the exact `__all__` sets of `abstract_harness.robot.core` and `abstract_harness.robot.yam`. Do not add or remove symbols in those two `__all__` lists without updating their snapshot tests; export new top-level helpers in `src/abstract_harness/__init__.py`.
- **Doc Heading Anchor Tests:**
  - `tests/robot_yam/test_policy.py` (`test_remedy_docs_anchors_resolve_to_readme_headings`) verifies that `#vla-runs` in `docs/ROBOT.md` matches the heading slug referenced by `ActServerConfig.remedy`. Keep `## VLA runs` in `docs/ROBOT.md`.
- **No Hardware, GPU, or Network in Tests:**
  - All tests in `tests/` must run offline against fakes/mocks without touching CAN interfaces, cameras, GPUs, or live APIs.

## Verification Commands

Run these before committing any change:

```bash
uv sync --locked --group typing
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest -q
uv build --no-sources
uv run --no-sync python scripts/check_dist.py dist
```

Physical lifecycle ownership, termination limits, measured action references and raw-data
retention are specified in `docs/SAFETY.md`; release validation is in `CONTRIBUTING.md`.
Agents never energize or move the arms: native reset, run, holdcheck, health and pose
capture/goto are operator-run actions. Core tests always collect; only actual optional
integrations skip absent dependencies.
