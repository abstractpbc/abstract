"""Controllers — the rollout middleware layer.

A ``Controller`` owns the per-control-step decision
of which action to send
to the embodiment. It internally decides *when* to call ``policy.act()`` (a slow
VLA inference returning an ``ActionChunk``), buffers the
returned chunk, and pops the next action each step. This single-method, stateful
shape (R3) is what lets advanced controllers — e.g. a temporal-ensembling
controller that re-infers every step and blends overlapping predictions —
compose without forking the rollout loop.

``DefaultController`` plays the first ``replan_interval`` actions of each chunk,
then re-infers (``replan_interval=None`` ⇒ play the whole chunk before replanning).
``AsyncController`` runs inference on a worker thread while buffered actions keep
playing, for loops held to a fixed rate (strict timing, ``core/timing.py``).
"""

from __future__ import annotations

import math
import queue
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import count
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from abstract_harness.robot.core.policy import Policy
from abstract_harness.robot.core.spaces import ABSOLUTE_CONTROL_MODES, Box
from abstract_harness.robot.core.types import Action, ActionChunk, Observation

_BUFFER_KEY = "_controller_action_buffer"
# Each entry is (inference_latency_s | None, chunk_len): one per policy.act() call.
_INFER_KEY = "_controller_inferences"
# A controller may publish a small JSON-ready report of its trial here; the rollout
# copies it into ``TrialRecord.metadata["controller"]`` when the trial ends.
_REPORT_KEY = "_controller_report"


@runtime_checkable
class Controller(Protocol):
    """Decides the next action to execute, calling the policy as needed."""

    def next_action(
        self, policy: Policy, observation: Observation, t: int, store: dict[str, Any]
    ) -> Action:
        """Choose step ``t``'s action, using ``store`` for trial-local controller state."""
        ...


class DefaultController:
    """Open-loop chunk execution with periodic replanning."""

    def __init__(self, replan_interval: int | None = None):
        if replan_interval is not None and replan_interval < 1:
            raise ValueError("replan_interval must be >= 1 or None")
        self.replan_interval = replan_interval

    def next_action(
        self, policy: Policy, observation: Observation, t: int, store: dict[str, Any]
    ) -> Action:
        """Reuse buffered chunk actions until the configured replanning boundary."""
        observe = getattr(policy, "observe", None)
        if callable(observe):
            observe(observation)
        buffer: deque[Action] = store.setdefault(_BUFFER_KEY, deque())
        if not buffer:
            chunk = policy.act(observation)
            take = self.replan_interval or len(chunk)
            taken = list(chunk.actions)[:take]
            buffer.extend(taken)
            # Record the number of actions actually buffered (the chunk may be
            # shorter than replan_interval), so the transcript never overstates.
            store.setdefault(_INFER_KEY, []).append((chunk.inference_latency_s, len(taken)))
        return buffer.popleft()


_SMOOTH_KEY = "_smoothing_prev_action"
# Distinguishes stacked instances' keys; process-local and never persisted.
_smooth_serial = count()


class SmoothingController:
    """Wrap another controller and exponentially smooth its action stream.

    Demonstrates the middleware composition the single-method interface enables:
    the wrapped controller owns inference/replanning while this layer applies an
    exponential moving average (``alpha`` toward the new action) on top. Only
    valid for additive/continuous action spaces (the caller's responsibility).
    """

    def __init__(self, inner: Controller, alpha: float = 0.5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.inner = inner
        self.alpha = alpha
        # Per instance, not per module: this is the one controller state written
        # destructively rather than appended to, so two smoothing layers in the
        # same chain would otherwise overwrite each other's previous action.
        self._store_key = f"{_SMOOTH_KEY}:{next(_smooth_serial)}"

    def next_action(
        self, policy: Policy, observation: Observation, t: int, store: dict[str, Any]
    ) -> Action:
        """Apply an exponential moving average to the wrapped controller's output."""
        action = self.inner.next_action(policy, observation, t, store)
        raw = np.asarray(action.data, dtype=np.float64)
        prev = store.get(self._store_key)
        smoothed = raw if prev is None else self.alpha * raw + (1 - self.alpha) * prev
        store[self._store_key] = smoothed
        return replace(action, data=smoothed)


_ENSEMBLE_KEY = "_ensemble_chunks"
# Control modes whose actions live in a vector space and may be linearly averaged.
_AVERAGEABLE_MODES = frozenset(
    {
        "joint_pos",
        "joint_vel",
        "joint_delta",
        "eef_delta_pos",
        "eef_abs_pose",
        "eef_delta_pose",
        "base_vel",
    }
)
# Rotation representations that survive linear averaging. "none" has no rotation;
# "rot6d" is averaged un-normalized here on the assumption the consumer applies
# Gram-Schmidt on decode (true for the standard rot6d->matrix path).
_AVERAGEABLE_ROT = frozenset({"none", "rot6d"})


class EnsemblingController:
    """ACT/ALOHA-style temporal ensembling over overlapping action chunks.

    Queries the policy every control step and blends, for the current step, the
    predictions of all still-relevant recent chunks. A chunk queried at global
    step ``q`` predicts step ``t`` via its action at index ``t - q`` (valid while
    ``0 <= t - q < len(chunk)``). Predictions are weighted ``exp(-m * i)`` with
    ``i = 0`` for the **oldest** contributing chunk (ALOHA's convention: older
    predictions dominate, which smooths motion); larger ``m`` favors the oldest.

    Only valid for additive action representations: the constructor refuses
    rotation reps and binary grippers that cannot be linearly averaged (R8).
    """

    def __init__(self, action_space: Box, m: float = 0.1):
        if not np.isfinite(m) or m < 0:
            raise ValueError("m must be >= 0")
        self.action_space = action_space
        self.m = m
        sem = action_space.semantics
        if sem is None:
            warnings.warn(
                "EnsemblingController: action space has no semantics; cannot "
                "verify that actions are safe to average.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            if sem.control_mode not in _AVERAGEABLE_MODES:  # pragma: no cover
                # Defensive: every valid ControlMode literal is currently averageable.
                raise ValueError(
                    f"EnsemblingController cannot average control_mode {sem.control_mode!r}"
                )
            if sem.rotation_repr not in _AVERAGEABLE_ROT:
                raise ValueError(
                    f"EnsemblingController cannot linearly average rotation_repr "
                    f"{sem.rotation_repr!r}; only {sorted(_AVERAGEABLE_ROT)} are safe"
                )
            if sem.gripper == "binary":
                raise ValueError(
                    "EnsemblingController cannot average a binary gripper; threshold "
                    "it downstream or use a continuous gripper"
                )

    def next_action(
        self, policy: Policy, observation: Observation, t: int, store: dict[str, Any]
    ) -> Action:
        """Query once for step ``t`` and blend every retained prediction for that step."""
        observe = getattr(policy, "observe", None)
        if callable(observe):
            observe(observation)
        chunk = policy.act(observation)
        if not chunk.actions:
            from abstract_harness.robot.core.errors import PolicyError

            raise PolicyError(f"policy {policy.info.name!r} returned an empty ActionChunk")
        store.setdefault(_INFER_KEY, []).append((chunk.inference_latency_s, len(chunk)))

        buffer: list[tuple[int, list[Any], list[dict[str, Any]], dict[str, Any]]] = (
            store.setdefault(_ENSEMBLE_KEY, [])
        )
        buffer.append(
            (
                t,
                [np.asarray(a.data, dtype=np.float64) for a in chunk.actions],
                [dict(a.meta) for a in chunk.actions],
                dict(chunk.meta),
            )
        )
        # Keep only chunks that predict the current step; evict the stale ones.
        buffer[:] = [
            (q, acts, am, meta) for (q, acts, am, meta) in buffer if 0 <= t - q < len(acts)
        ]
        # Oldest first (ascending query time) so weight index 0 is the oldest.
        buffer.sort(key=lambda e: e[0])

        predictions = [acts[t - q] for (q, acts, _am, _meta) in buffer]
        weights = np.exp(-self.m * np.arange(len(predictions)))
        weights = weights / weights.sum()
        blended = np.average(np.stack(predictions), axis=0, weights=weights)
        newest_meta = {**buffer[-1][3], **dict(chunk.actions[0].meta)}
        # Terminal intent is not motion data. Any current prediction requesting a stop
        # survives blending, even if another chunk supplies the dominant motion.
        for q, _acts, action_metas, chunk_meta in buffer:
            meta = {**chunk_meta, **action_metas[t - q]}
            if meta.get("request_stop"):
                newest_meta.update(
                    request_stop=True, stop_reason=meta.get("stop_reason", "policy_stop")
                )
        return Action(data=blended, meta=newest_meta)


_ASYNC_KEY = "_async_controller_trial"
#: Set by a controller to the step whose observation it just handed the policy. The
#: rollout then marks that step's approvals, operator messages and command events as
#: delivered; without it, delivery is assumed at every collected inference.
_DISPATCH_KEY = "_controller_dispatch_step"


@dataclass
class _AsyncTrial:
    """One trial's state for ``AsyncController``, kept in the rollout store."""

    # (the step an action is for, the action), oldest first.
    buffer: deque[tuple[int, Action]] = field(default_factory=deque)
    result: queue.Queue[tuple[bool, object, float, int]] = field(
        default_factory=lambda: queue.Queue(maxsize=1)
    )
    worker: threading.Thread | None = None
    dispatched_at: float | None = None
    last_dispatch_t: int | None = None
    last: Action | None = None
    last_chunk_len: int = 1
    # A buffered action requested the trial's end: nothing replaces it.
    stop_buffered: bool = False
    # A stop requested by an action that was dropped as stale, still to be delivered.
    stop_meta: dict[str, Any] = field(default_factory=dict)
    inferences: int = 0
    holds: int = 0
    late_actions: int = 0
    stale_actions: int = 0


def _stop_meta(actions: Any) -> dict[str, Any]:
    for action in actions:
        if action.meta.get("request_stop"):
            return {
                "request_stop": True,
                "stop_reason": action.meta.get("stop_reason", "policy_stop"),
                **(
                    {"stop_detail": action.meta["stop_detail"]}
                    if "stop_detail" in action.meta
                    else {}
                ),
            }
    return {}


class AsyncController:
    """Overlap inference with execution: the robot keeps moving while the policy thinks.

    The first call of a trial runs one inference and waits for it. After that, whenever
    ``prefetch`` or fewer buffered actions remain (an int, or a fraction of the last
    chunk's length), the next inference starts on a worker thread from the current
    observation while the buffer keeps playing; no observation is sent twice. A chunk
    computed from the observation of step ``q`` predicts steps ``q, q+1, ...``: when it
    arrives at step ``t`` its first ``t - q`` actions are already stale and are dropped,
    and the rest replace whatever is still buffered (the newer prediction wins). When the
    whole chunk is stale (a one-action reactive policy, or latency longer than the
    chunk), its last action runs at ``t``, one or more steps late, marked
    ``meta["late"]``.

    If the buffer runs dry while inference is still running (an *underrun*), the
    controller holds: an absolute-target action space repeats the last action, any
    other mode (displacements, velocities) sends zeros, so the robot stays where it is.
    ``underrun="wait"`` blocks instead, as ``DefaultController`` would; that is also the
    behavior when the action space declares no semantics. Hold actions carry
    ``meta["hold"] = True``. A stop request is never lost: a buffered one is not
    replaced by a later prediction, and one on a dropped stale action rides the next
    action out.

    Use it on a paced loop (strict timing or a self-paced embodiment): on an unpaced
    simulator the loop would spin through hold steps while inference runs. Only
    ``act()`` runs on the worker, so policies whose ``observe()`` or
    ``transcript_delta()`` hooks would run concurrently with it (stateful world models,
    the LLM agent) are refused. An inference running longer than ``timeout_s`` raises
    ``SafetyAbort``. The policy is never read or reset while ``act()`` runs: at the end
    of a trial the rollout calls ``end_trial``, which waits for an inference still in
    flight until that inference's own deadline (``timeout_s`` from its dispatch), and
    one that never returns marks the policy abandoned; before the next trial's
    ``policy.reset()`` it calls ``prepare_trial``, which refuses (``PolicyError``) while
    a previous inference still runs. The trial's counts (inferences, holds, late and
    stale actions) land in ``TrialRecord.metadata["controller"]``.
    """

    def __init__(
        self,
        *,
        prefetch: int | float = 0.5,
        underrun: Literal["hold", "wait"] = "hold",
        timeout_s: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        from abstract_harness.robot.core.errors import ConfigError

        if isinstance(prefetch, bool) or not isinstance(prefetch, int | float):
            raise ConfigError("prefetch must be an int >= 0 or a fraction in (0, 1]")
        if isinstance(prefetch, int) and prefetch < 0:
            raise ConfigError("prefetch must be an int >= 0 or a fraction in (0, 1]")
        if isinstance(prefetch, float) and not 0.0 < prefetch <= 1.0:
            raise ConfigError("prefetch must be an int >= 0 or a fraction in (0, 1]")
        if underrun not in ("hold", "wait"):
            raise ConfigError("underrun must be 'hold' or 'wait'")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ConfigError("timeout_s must be finite and positive")
        self.prefetch = prefetch
        self.underrun = underrun
        self.timeout_s = float(timeout_s)
        self._clock = clock
        self._previous: _AsyncTrial | None = None

    def next_action(
        self, policy: Policy, observation: Observation, t: int, store: dict[str, Any]
    ) -> Action:
        """Step ``t``'s action from the freshest prediction, inferring in the background."""
        state: _AsyncTrial | None = store.get(_ASYNC_KEY)
        if state is None:
            state = self._begin_trial(policy)
            store[_ASYNC_KEY] = state
        self._collect(state, t, store, block=False)
        if not state.buffer and state.worker is None and not state.stop_buffered:
            self._dispatch(policy, observation, t, state, store)
        if not state.buffer and state.worker is not None:
            if state.last is None or self._hold_data(policy, state) is None:
                self._collect(state, t, store, block=True)
        while state.buffer and state.buffer[0][0] < t:
            _, dropped = state.buffer.popleft()
            state.stale_actions += 1
            state.stop_meta = state.stop_meta or _stop_meta([dropped])
        if state.buffer:
            _, action = state.buffer.popleft()
            state.last = action
        else:
            hold = self._hold_data(policy, state)
            if hold is None:
                from abstract_harness.robot.core.errors import PolicyError

                raise PolicyError(f"policy {policy.info.name!r} left no action to execute")
            action = Action(data=hold, meta={"hold": True})
            state.holds += 1
        if state.stop_meta:
            action = replace(action, meta={**dict(action.meta), **state.stop_meta})
            state.stop_meta = {}
        if (
            state.worker is None
            and not state.stop_buffered
            and len(state.buffer) <= self._threshold(state)
        ):
            self._dispatch(policy, observation, t, state, store)
        store[_REPORT_KEY] = {
            "controller": "async",
            "inferences": state.inferences,
            "holds": state.holds,
            "late_actions": state.late_actions,
            "stale_actions": state.stale_actions,
        }
        return action

    def end_trial(self, store: dict[str, Any]) -> bool:
        """Wait for an inference still in flight, until its own deadline; ``False`` when
        it did not return, so the rollout must not reset or read the policy."""
        state: _AsyncTrial | None = store.get(_ASYNC_KEY)
        return self._settle(state)

    def prepare_trial(self, policy: Policy) -> None:
        """Before the rollout resets ``policy``: refuse a policy this controller cannot
        drive, and one whose previous inference is still running."""
        from abstract_harness.robot.core.errors import ConfigError, PolicyError

        for hook in ("observe", "transcript_delta"):
            if callable(getattr(policy, hook, None)):
                raise ConfigError(
                    f"AsyncController cannot drive policy {policy.info.name!r}: its {hook}() "
                    "hook would run concurrently with act() on the inference worker; use "
                    "DefaultController"
                )
        if not self._settle(self._previous):
            raise PolicyError(
                "the previous trial's inference is still running past its deadline; this "
                "policy cannot be reused until it returns"
            )

    def _settle(self, state: _AsyncTrial | None) -> bool:
        worker = state.worker if state is not None else None
        if state is None or worker is None:
            return True
        started = state.dispatched_at if state.dispatched_at is not None else self._clock()
        # Past its deadline it still gets a moment to finish returning.
        worker.join(timeout=max(0.05, self.timeout_s - (self._clock() - started)))
        return not worker.is_alive()

    def _begin_trial(self, policy: Policy) -> _AsyncTrial:
        self.prepare_trial(policy)  # again, for callers that drive next_action directly
        state = _AsyncTrial()
        self._previous = state
        return state

    def _threshold(self, state: _AsyncTrial) -> int:
        if isinstance(self.prefetch, float):
            return max(0, math.floor(self.prefetch * state.last_chunk_len))
        return self.prefetch

    def _hold_data(self, policy: Policy, state: _AsyncTrial) -> Any:
        """What an underrun step sends, or ``None`` when the controller must wait."""
        if self.underrun == "wait" or state.last is None:
            return None
        semantics = policy.info.action_space.semantics
        if semantics is None:
            return None
        last = np.asarray(state.last.data, dtype=np.float64)
        if semantics.control_mode in ABSOLUTE_CONTROL_MODES:
            return last.copy()
        return np.zeros_like(last)

    def _dispatch(
        self,
        policy: Policy,
        observation: Observation,
        t: int,
        state: _AsyncTrial,
        store: dict[str, Any],
    ) -> None:
        if state.last_dispatch_t == t:
            return  # this step's observation already went to the policy
        results = state.result

        def run() -> None:
            started = time.perf_counter()
            try:
                chunk = policy.act(observation)
            except BaseException as exc:
                results.put((False, exc, time.perf_counter() - started, t))
            else:
                results.put((True, chunk, time.perf_counter() - started, t))

        state.worker = threading.Thread(target=run, name="abstract-async-inference", daemon=True)
        state.dispatched_at = self._clock()
        state.last_dispatch_t = t
        store[_DISPATCH_KEY] = t
        state.worker.start()

    def _collect(self, state: _AsyncTrial, t: int, store: dict[str, Any], *, block: bool) -> None:
        from abstract_harness.robot.core.errors import PolicyError, SafetyAbort

        if state.worker is None:
            return
        assert state.dispatched_at is not None
        while True:
            try:
                if block:
                    remaining = self.timeout_s - (self._clock() - state.dispatched_at)
                    item = state.result.get(timeout=min(max(remaining, 0.0), 0.05))
                else:
                    item = state.result.get_nowait()
                break
            except queue.Empty:
                if self._clock() - state.dispatched_at >= self.timeout_s:
                    raise SafetyAbort(
                        f"async inference exceeded {self.timeout_s:g}s; evaluation halted"
                    ) from None
                if not block:
                    return
        ok, value, elapsed, query_t = item
        state.worker = None
        state.dispatched_at = None
        if not ok:
            assert isinstance(value, BaseException)
            raise value
        if not isinstance(value, ActionChunk) or not value.actions:
            raise PolicyError("policy returned an empty or invalid ActionChunk")
        chunk = value
        state.inferences += 1
        state.last_chunk_len = len(chunk)
        latency = chunk.inference_latency_s if chunk.inference_latency_s is not None else elapsed
        actions = list(chunk.actions)
        fresh = [(query_t + i, a) for i, a in enumerate(actions) if query_t + i >= t]
        stale = actions[: len(actions) - len(fresh)]
        if not fresh:
            # All of it is late: its newest action is still the closest prediction to now.
            last = actions[-1]
            fresh = [(t, replace(last, meta={**dict(last.meta), "late": True}))]
            stale = actions[:-1]
            state.late_actions += 1
        store.setdefault(_INFER_KEY, []).append((latency, len(fresh)))
        state.stale_actions += len(stale)
        state.stop_meta = state.stop_meta or _stop_meta(stale)
        if state.stop_buffered:
            return
        state.buffer = deque(fresh)
        state.stop_buffered = bool(state.stop_meta) or any(
            action.meta.get("request_stop") for _, action in fresh
        )
