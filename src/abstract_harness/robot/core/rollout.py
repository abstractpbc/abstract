"""The rollout engine — the closed control loop at the heart of the robot harness.

One ``rollout`` runs a single trial (one scene, one epoch): it
drives the policy↔embodiment loop through the ``Controller``
(open-loop chunk execution) and the ``Approver`` safety
gate, logging each step to the sinks, and returns an immutable
``TrialRecord`` that scorers consume.
"""

from __future__ import annotations

import json
import warnings
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from abstract_harness.robot.core.approver import Approver
from abstract_harness.robot.core.controller import _INFER_KEY, Controller
from abstract_harness.robot.core.embodiment import Embodiment
from abstract_harness.robot.core.errors import (
    EmbodimentFault,
    PolicyError,
    RobotError,
    SafetyAbort,
    _CancelledTrial,
)
from abstract_harness.robot.core.frames import FrameRef, FrameStore
from abstract_harness.robot.core.policy import Policy
from abstract_harness.robot.core.scene import Scene
from abstract_harness.robot.core.supervision import InferenceSupervisor
from abstract_harness.robot.core.transcript import (
    Event,
    approval_event,
    error_event,
    inference_event,
    operator_event,
    operator_message_event,
    reset_event,
    step_event,
)
from abstract_harness.robot.core.types import OPERATOR_END, Action, Observation, StepResult

if TYPE_CHECKING:
    from abstract_harness.robot.core.console import OperatorInput
    from abstract_harness.robot.core.logging.sink import LogSink

_TRANSCRIPT_BYTE_LIMIT = 2 * 1024 * 1024
_APPROVALS_KEY = "_rollout_approvals"
_OPERATOR_MSGS_KEY = "_rollout_operator_messages"


class _OperatorStopped(Exception):
    """Internal control flow: an operator stopped while inference was pending."""


def derive_seed(eval_seed: int | None, scene_seed: int | None, epoch: int) -> int:
    """Deterministically combine eval/scene seeds and the epoch index (R2).

    Distinct epochs of the same scene get distinct seeds so repeats actually vary
    for stochastic policies, while a fixed ``(eval_seed, scene_seed, epoch)``
    reproduces bitwise. ``None`` and ``0`` hash differently, so an unseeded
    input does not silently alias ``seed=0``.
    """
    payload = f"{eval_seed}:{scene_seed}:{epoch}".encode()
    return zlib.crc32(payload) & 0xFFFFFFFF


@dataclass(frozen=True, eq=False)
class StepRecord:
    """One step of a recorded trajectory.

    When a ``FrameStore`` is used, both
    ``observation`` and ``result.observation`` have their images stripped.
    ``image_refs`` and ``result_image_refs`` hold the corresponding on-disk
    handles instead (R5).
    """

    t: int
    observation: Observation
    action: Action
    result: StepResult
    image_refs: Mapping[str, FrameRef] | None = None
    result_image_refs: Mapping[str, FrameRef] | None = None
    requested_action: Action | None = None


@dataclass
class TrialRecord:
    """The full record of one trial — the unit scorers consume."""

    scene_id: str
    epoch: int
    seed: int | None
    steps: list[StepRecord] = field(default_factory=list)
    # In-memory only: EvalLog embeds SceneResult aggregates and never TrialRecord,
    # so no asdict path reaches the numpy images and the log schema is untouched.
    parked_observation: Observation | None = None
    terminated: bool = False
    truncated: bool = False
    termination_reason: str | None = None
    status: str = "success"  # "success" (ran to completion) | "error" | "cancelled"
    error: str | None = None
    inference_latencies: list[float] = field(default_factory=list)
    # Human operator's success verdict, captured once during rollout (R6). Read
    # by OperatorScorer; remains None for unattended/CI runs.
    operator_judgement: str | None = None
    # Qualitative context from the operator; read by nothing that scores.
    operator_note: str | None = None
    # Typed transcript of what happened during the trial.
    events: list[Event] = field(default_factory=list)
    # Extensible metadata for the trial (e.g. populated by policies).
    metadata: dict[str, Any] = field(default_factory=dict)
    # The policy's optional per-trial audit record (e.g. an LLM conversation),
    # collected via the duck-typed transcript() hook and normalized to plain
    # JSON types by _collect_transcript. None when the policy has no hook.
    policy_transcript: Any = None


def _collect_transcript(policy: object) -> Any:
    """Normalize a policy's optional audit hook without affecting trial outcome."""
    try:
        transcript = getattr(policy, "transcript", None)
        if not callable(transcript):
            return None
        raw = transcript()
        if raw is None:
            return None
        dumped = json.dumps(raw, default=str)
        normalized = json.loads(dumped)
        size = len(dumped.encode())
        if size > _TRANSCRIPT_BYTE_LIMIT:
            return {
                "transcript_dropped": True,
                "bytes": size,
                "note": "exceeds inline limit; policies must not embed binary data",
            }
        return normalized
    except Exception as exc:
        try:
            detail = f"{type(exc).__name__}: {exc}"
        except Exception:
            detail = type(exc).__name__
        return {"transcript_error": detail}


def _record_failure(record: TrialRecord, exc: RobotError, t: int) -> RobotError:
    """Mark ``record`` failed and attach it to ``exc`` (see ``RobotError.record``).

    The partial record — steps walked and transcript events up to the failure —
    is forensic data the orchestrator preserves in the eval log.
    """
    message = str(exc)
    record.events.append(error_event(t, type(exc).__name__, message))
    record.status = "error"
    record.error = f"{type(exc).__name__}: {message}"
    exc.record = record
    return exc


def _connection_failure(exc: Exception) -> bool:
    """Return whether an exception chain contains a connection failure."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ConnectionError) or type(current).__name__ in {
            "ConnectionError",
            "NewConnectionError",
            "ConnectError",
        }:
            return True
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return False


def _policy_error(policy: Policy, exc: Exception) -> PolicyError:
    """Wrap a generic policy exception, appending a hint on connection failures."""
    message = str(exc)
    if _connection_failure(exc):
        try:
            url = getattr(policy, "server_url", None)
            remedy = getattr(policy, "remedy", None)
            name = policy.info.name
            if url:
                # "up and healthy", not "running": builtin ConnectionError
                # matches also cover reset/broken-pipe, where the server
                # answered and then dropped the connection.
                message += (
                    f"\nhint: policy {name!r} could not hold a connection to its "
                    f"action server at {url} — is the server up and healthy? "
                    "Start (or restart) it, then rerun."
                )
            else:
                message += (
                    f"\nhint: policy {name!r} hit a connection failure — "
                    "a backend it depends on may be down or unreachable."
                )
            if remedy:
                message += f"\nhint: {remedy}"
        except Exception:
            pass  # a raising server_url/remedy property must not mask the trial error
    return PolicyError(message)


def _non_finite_detail(data: object) -> str | None:
    """Describe why action data is not finite, or return ``None`` when it is."""
    try:
        array = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        return f"is not numeric: {exc}"
    if np.isfinite(array).all():
        return None
    if np.isnan(array).any():
        return "contains nan"
    return "contains inf"


def _store_frames(
    frame_store: FrameStore | None, trial_id: str, t: int, obs: Observation
) -> tuple[Observation, Mapping[str, FrameRef] | None]:
    """Snapshot numeric buffers before their owner can reuse them; optionally persist images."""
    refs = (
        {cam: frame_store.put(trial_id, t, cam, image) for cam, image in obs.images.items()}
        if frame_store is not None and obs.images
        else None
    )
    return replace(
        obs,
        state={key: np.asarray(value).copy() for key, value in obs.state.items()},
        images={} if refs else {key: value.copy() for key, value in obs.images.items()},
        image_times=dict(obs.image_times),
    ), refs


def _end_operator_trial(operator_input: OperatorInput | None) -> None:
    """Best-effort call the optional operator-input trial teardown hook."""
    if operator_input is not None:
        try:
            end_trial = getattr(operator_input, "end_trial", None)
            if callable(end_trial):
                end_trial()
        except Exception as exc:
            # stacklevel=3 skips this helper so the warning points at rollout's
            # caller, like the disable-site warnings issued from rollout() itself.
            warnings.warn(
                f"Operator console disabled for this trial after {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )


def rollout(
    policy: Policy,
    embodiment: Embodiment,
    scene: Scene,
    *,
    max_steps: int,
    seed: int | None,
    epoch: int,
    controller: Controller,
    approver: Approver,
    sink: LogSink,
    frame_store: FrameStore | None = None,
    operator_input: OperatorInput | None = None,
    policy_timeout_s: float = 120.0,
) -> TrialRecord:
    """Run a single trial and return its record.

    Generic exceptions raised by the policy, or a non-finite action, are
    reported as ``PolicyError``; generic
    exceptions raised by the embodiment are reported as
    ``EmbodimentFault``; by the approver as
    ``SafetyAbort`` (an approver that crashed cannot
    vouch for safety), as is a non-finite action introduced by the approver.
    Already-typed harness errors (incl.
    ``SafetyAbort``) propagate unchanged, so the
    eval orchestrator can apply the correct continue-vs-halt policy. Every error
    raised from inside the trial carries the partial ``TrialRecord`` on
    ``exc.record`` for the orchestrator to preserve.

    The loop applies no wall-clock pacing of its own: ``embodiment.step()`` is
    called as fast as the policy/controller/approver can produce actions. An
    embodiment that needs real-time cadence paces itself inside ``step()`` and
    declares the ``"self_paced"`` capability to document that it does (see
    ``Embodiment``).
    """
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError(f"max_steps must be an integer >= 1, got {max_steps!r}")
    trial_id = f"{scene.id}-e{epoch}"
    record = TrialRecord(scene_id=scene.id, epoch=epoch, seed=seed)
    record.events.append(reset_event(seed))
    store: dict[str, Any] = {}
    expected_dim = embodiment.info.action_space.dim
    policy_reset_ok = False
    delta_hook: Any = getattr(policy, "transcript_delta", None)
    messages_hook: Any = getattr(sink, "log_policy_messages", None)
    stream_ok = callable(delta_hook) and callable(messages_hook)
    console_ok = operator_input is not None
    supervisor = InferenceSupervisor(policy_timeout_s)

    try:
        t = -1
        try:
            policy.reset(scene)
            policy_reset_ok = True
        except RobotError as exc:
            _record_failure(record, exc, -1)
            raise
        except Exception as exc:
            raise _record_failure(record, _policy_error(policy, exc), -1) from exc
        try:
            obs = embodiment.reset(scene, seed=seed)
        except RobotError as exc:
            _record_failure(record, exc, -1)
            raise
        except Exception as exc:
            raise _record_failure(record, EmbodimentFault(str(exc)), -1) from exc

        begin = getattr(approver, "begin_trial", None)
        if callable(begin):
            reference_hook = getattr(embodiment, "action_reference", None)
            reference = reference_hook(obs) if callable(reference_hook) else None
            semantics = embodiment.info.action_space.semantics
            if reference is None and semantics is not None:
                key = "joint_pos" if semantics.control_mode == "joint_pos" else "eef_state"
                reference = obs.state.get(key)
            try:
                begin(reference, store)
            except RobotError as exc:
                _record_failure(record, exc, -1)
                raise
            except Exception as exc:
                raise _record_failure(record, SafetyAbort(str(exc)), -1) from exc

        if operator_input is not None:
            try:
                operator_input.begin_trial()
            except Exception as exc:
                console_ok = False
                _end_operator_trial(operator_input)
                warnings.warn(
                    f"Operator console disabled for this trial after {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        obs_rec, refs = _store_frames(frame_store, trial_id, 0, obs)
        t = 0
        while t < max_steps:
            poll = None
            if operator_input is not None and console_ok:
                try:
                    poll = operator_input.poll()
                except Exception as exc:
                    console_ok = False
                    _end_operator_trial(operator_input)
                    warnings.warn(
                        "Operator console disabled for this trial after "
                        f"{type(exc).__name__}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if poll is not None:
                for i, text in enumerate(poll.messages):
                    source = poll.sources[i] if i < len(poll.sources) else "console"
                    store.setdefault(_OPERATOR_MSGS_KEY, []).append(
                        {"t": t, "text": text, "source": source}
                    )
                    record.events.append(operator_message_event(t, text, source))

            prev_inferences = len(store.get(_INFER_KEY, []))
            all_approvals = store.get(_APPROVALS_KEY, [])
            last_approvals_idx = store.get("_rollout_last_approvals_idx", 0)
            tail_approvals = [dict(a) for a in all_approvals[last_approvals_idx:]]
            all_operator_msgs = store.get(_OPERATOR_MSGS_KEY, [])
            last_operator_msgs_idx = store.get("_rollout_last_operator_msgs_idx", 0)
            tail_operator_msgs = [
                dict(message) for message in all_operator_msgs[last_operator_msgs_idx:]
            ]

            if poll is not None and poll.end is not None:
                record.terminated = True
                record.termination_reason = OPERATOR_END
                if poll.end.verdict is not None:
                    record.operator_judgement = poll.end.verdict
                    record.operator_note = poll.end.note
                    record.events.append(
                        operator_event(
                            t=t,
                            verdict=poll.end.verdict,
                            source="console",
                            note=poll.end.note,
                        )
                    )
                break

            try:
                policy_extra = {**obs.extra, "env_step": t, "approvals": tail_approvals}
                if operator_input is not None:
                    policy_extra["operator_messages"] = tail_operator_msgs
                obs_with_extra = replace(
                    obs,
                    extra=policy_extra,
                )
                observe = getattr(policy, "observe", None)
                if callable(observe):
                    observe(obs_with_extra)

                def during_inference(step: int = t, console_enabled: bool = console_ok) -> None:
                    health = getattr(embodiment, "check_health", None)
                    if callable(health):
                        health()
                    if operator_input is not None and console_enabled:
                        waiting = operator_input.poll()
                        for i, text in enumerate(waiting.messages):
                            source = waiting.sources[i] if i < len(waiting.sources) else "console"
                            store.setdefault(_OPERATOR_MSGS_KEY, []).append(
                                {"t": step, "text": text, "source": source}
                            )
                            record.events.append(operator_message_event(step, text, source))
                        if waiting.end is not None:
                            record.terminated = True
                            record.termination_reason = OPERATOR_END
                            record.operator_judgement = waiting.end.verdict
                            record.operator_note = waiting.end.note
                            if waiting.end.verdict is not None:
                                record.events.append(
                                    operator_event(
                                        step, waiting.end.verdict, "console", waiting.end.note
                                    )
                                )
                            raise _OperatorStopped

                if not embodiment.info.is_simulated:
                    action = supervisor.call(
                        lambda observation=obs_with_extra, step=t: controller.next_action(
                            policy, observation, step, store
                        ),
                        during_inference,
                    )
                else:
                    action = controller.next_action(policy, obs_with_extra, t, store)
            except _OperatorStopped:
                policy_reset_ok = False
                record.metadata["policy_abandoned"] = supervisor.abandoned
                raise KeyboardInterrupt("operator stopped during inference") from None
            except RobotError as exc:
                if supervisor.abandoned:
                    policy_reset_ok = False
                    record.metadata["policy_abandoned"] = True
                _record_failure(record, exc, t)
                raise
            except Exception as exc:
                raise _record_failure(record, _policy_error(policy, exc), t) from exc

            inferences = store.get(_INFER_KEY, [])
            if len(inferences) > prev_inferences:
                store["_rollout_last_approvals_idx"] = len(all_approvals)
                if operator_input is not None:
                    store["_rollout_last_operator_msgs_idx"] = len(all_operator_msgs)
                latency, chunk_len = inferences[-1]
                record.events.append(inference_event(t, latency, chunk_len))
                if stream_ok:
                    try:
                        delta = delta_hook()
                        entries = list(delta) if delta is not None else []
                        if entries:
                            messages_hook(t, entries)
                    except Exception as exc:
                        stream_ok = False
                        warnings.warn(
                            "Live policy transcript streaming disabled for this "
                            f"trial after {type(exc).__name__}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )

            # A malformed action is the policy's fault; catching it here keeps it
            # from surfacing inside the approver/embodiment as a halting fault.
            emitted_dim = int(np.asarray(action.data).size)
            if emitted_dim != expected_dim:
                raise _record_failure(
                    record,
                    PolicyError(
                        f"policy emitted a {emitted_dim}-D action but embodiment "
                        f"{embodiment.info.name!r} expects {expected_dim}-D"
                    ),
                    t,
                )
            non_finite_detail = _non_finite_detail(action.data)
            if non_finite_detail is not None:
                raise _record_failure(
                    record,
                    PolicyError(
                        "policy emitted a non-finite action "
                        f"({non_finite_detail}) for embodiment {embodiment.info.name!r}"
                    ),
                    t,
                )

            # Policy-requested stop (plan 0008 §3d), captured from the
            # PRE-review action so an approver rewrite cannot erase the
            # intent, including under chunk buffering and temporal ensembling.
            requested_stop = bool(action.meta.get("request_stop"))
            stop_reason = str(action.meta.get("stop_reason", "policy_stop"))

            try:
                requested_action = replace(
                    action, data=np.asarray(action.data).copy(), meta=dict(action.meta)
                )
                update = getattr(approver, "update_reference", None)
                if callable(update):
                    reference_hook = getattr(embodiment, "action_reference", None)
                    semantics = embodiment.info.action_space.semantics
                    reference = reference_hook(obs) if callable(reference_hook) else None
                    if reference is None and semantics is not None:
                        key = "joint_pos" if semantics.control_mode == "joint_pos" else "eef_state"
                        reference = obs.state.get(key)
                    update(reference, store)
                reviewed = approver.review(action, store)  # may raise SafetyAbort
            except RobotError as exc:
                _record_failure(record, exc, t)
                raise
            except Exception as exc:
                raise _record_failure(record, SafetyAbort(str(exc)), t) from exc
            if reviewed is not action:
                flags = [k for k in ("clamped", "delta_clamped") if reviewed.meta.get(k)]
                detail = ", ".join(flags) or None
                record.events.append(approval_event(t, modified=True, detail=detail))
                store.setdefault(_APPROVALS_KEY, []).append({"t": t, "detail": detail})
            action = reviewed

            # Recheck because an approver may mutate the array in place and return it.
            non_finite_detail = _non_finite_detail(action.data)
            if non_finite_detail is not None:
                raise _record_failure(
                    record,
                    SafetyAbort(
                        f"approver {type(approver).__name__} returned a non-finite "
                        f"action ({non_finite_detail})"
                    ),
                    t,
                )

            approved_action = replace(
                action, data=np.asarray(action.data).copy(), meta=dict(action.meta)
            )
            sink_observation = (
                replace(obs_rec, images={key: value.copy() for key, value in obs.images.items()})
                if refs
                else obs_rec
            )
            try:
                result: StepResult = embodiment.step(action)
            except RobotError as exc:
                _record_failure(record, exc, t)
                raise
            except Exception as exc:
                raise _record_failure(record, EmbodimentFault(str(exc)), t) from exc

            sink.log_step(t, sink_observation, approved_action, result)
            result_obs_rec, result_refs = _store_frames(
                frame_store, trial_id, t + 1, result.observation
            )
            result_rec = replace(
                result,
                observation=result_obs_rec,
                command={key: np.asarray(value).copy() for key, value in result.command.items()},
            )
            record.steps.append(
                StepRecord(
                    t=t,
                    observation=obs_rec,
                    action=approved_action,
                    result=result_rec,
                    image_refs=refs,
                    result_image_refs=result_refs,
                    requested_action=requested_action,
                )
            )
            record.events.append(
                step_event(t, result.terminated, result.truncated, result.termination_reason)
            )
            t += 1

            if result.terminated:
                record.terminated = True
                record.termination_reason = result.termination_reason
                break
            if result.truncated:
                record.truncated = True
                record.termination_reason = result.termination_reason or "truncated"
                break
            if requested_stop:
                # Embodiment-reported termination above wins (ground truth);
                # otherwise the policy's stop ends the trial as a truncation —
                # scoring stays the scorer's job, done() is not success.
                record.truncated = True
                record.termination_reason = stop_reason
                break
            obs = result.observation
            obs_rec = result_obs_rec
            refs = result_refs
        else:
            record.truncated = True
            record.termination_reason = "max_steps"
    except KeyboardInterrupt as exc:
        if supervisor.abandoned:
            policy_reset_ok = False
            record.metadata["policy_abandoned"] = True
        record.status = "cancelled"
        record.error = "cancelled by user (KeyboardInterrupt)"
        record.events.append(error_event(t, "KeyboardInterrupt", "cancelled by user"))
        raise _CancelledTrial(record.error, record) from exc
    finally:
        _end_operator_trial(operator_input)
        # Preserve measured latencies even when the trial ends in an error.
        record.inference_latencies = [
            lat for lat, _ in store.get(_INFER_KEY, []) if lat is not None
        ]
        if policy_reset_ok:  # pragma: no branch - false only while an exception unwinds
            record.policy_transcript = _collect_transcript(policy)
    return record
