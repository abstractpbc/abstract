"""The robot commands' shared machinery: one attended episode, the doctor, and the log
tools behind ``abstract robot show``, ``view`` and ``video``.

``components`` constructs a policy and an embodiment by registry name and claims the
embodiment's devices; ``run_task`` runs a task on them with the guardrails (bounds clamp
and per-step delta limit), the operator console and grader when a terminal is attended,
and the JSON, live and Rerun logs, then closes the embodiment and releases the claim on
every exit path. ``doctor`` audits an embodiment's declared spaces without connecting it.
``configure_parser`` adds the log tools to ``abstract robot``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn, cast

from abstract_harness.robot import events
from abstract_harness.robot.config import parse_value as _parse_value
from abstract_harness.robot.core._claims import DeviceClaim, claim_devices
from abstract_harness.robot.core._html import (
    _chat_content,
    _display_status,
    _is_chat_transcript,
    _status_class,
    render_html,
)
from abstract_harness.robot.core._html_index import IndexEntry, render_index
from abstract_harness.robot.core._pointers import (
    derive_blob_dir,
    read_jsonl_prefix,
    resolve_log_pointer,
)
from abstract_harness.robot.core.conformance import device_slots
from abstract_harness.robot.core.console import USAGE, USAGE_END_ONLY
from abstract_harness.robot.core.registry import registered
from abstract_harness.robot.core.session import ECHO_INTERVAL_S, OperatorSession
from abstract_harness.robot.core.types import OPERATOR_END

if TYPE_CHECKING:
    from abstract_harness.robot.core.approver import Approver
    from abstract_harness.robot.core.embodiment import Embodiment
    from abstract_harness.robot.core.log import EvalLog
    from abstract_harness.robot.core.logging.sink import LogSink
    from abstract_harness.robot.core.spaces import Box
    from abstract_harness.robot.core.task import Task


def _styled(text: str, code: str) -> str:
    """Wrap ``text`` in an ANSI style when stdout is an interactive terminal.

    Plain text is returned when piped/redirected or when ``NO_COLOR`` is set,
    so scripts and CI logs never see escape codes.
    """
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _format_metric(value: float | int | None) -> str:
    """Format numeric metric value or return 'n/a' when None."""
    return "n/a" if value is None else f"{value:.4g}"


_BOLD = "1"

_BOLD_BRIGHT_MAGENTA = "1;95"

_DIM = "2"

_CYAN = "36"

_GREEN = "32"

_RED = "31"

_YELLOW = "33"

_OUTCOME_PHRASES = {
    "success": "succeeded",
    "failure": "failed",
    "max_steps": "hit step limit",
    OPERATOR_END: "ended by operator",
    "give_up": "gave up",
    "done": "reported done",
    "policy_stop": "stopped by policy",
    "truncated": "truncated",
}

_SERVE_RERENDER_SECONDS = 60

_SERVE_LIVE_RERENDER_SECONDS = 2

# Two-level directory stamp (plan 0060): a suppressed-tier page (serve pass or
# --no-video) is stamped this far below the source log's mtime so a later
# full-tier pass can recognize and upgrade it. The delta spans many filesystem
# timestamp ticks (NTFS rounds to 100 ns) and the gate compares with slack so
# tick rounding can never turn a skip into a per-tick re-render.
_SUPPRESSED_STAMP_DELTA_NS = 2_000

_STAMP_TICK_SLACK_NS = 1_000

_serve_sleep = time.sleep


def _parse_kvs(pairs: Sequence[str] | None) -> dict[str, Any]:
    from abstract_harness._security import sensitive_argument

    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"expected key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        if sensitive_argument(key):
            raise SystemExit("credentials must use an environment-variable reference")
        out[key] = _parse_value(value)
    return out


def _resolve_or_exit(
    kind: str, name: str, args_section: str | None = None, /, **kwargs: Any
) -> Any:
    """Registry resolution with a clean error instead of a traceback.

    Unknown names raise ``KeyError``; a factory that cannot construct itself
    (e.g. the agent policy with no model/key configured) raises a guided
    ``ConfigError``. Invalid constructor arguments raise ``TypeError``. All
    three become user-facing messages rather than tracebacks. ``args_section``
    can identify a config section whose name differs from the registry kind.
    """
    from abstract_harness.robot.core.errors import ConfigError
    from abstract_harness.robot.core.registry import resolve

    try:
        return resolve(kind, name, **kwargs)
    except KeyError as exc:
        message = str(exc.args[0])
        # Only the registry's own unknown-name error earns the install hint; a
        # KeyError escaping a factory must not masquerade as a missing plugin.
        raise SystemExit(message) from exc
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    except TypeError as exc:
        if args_section is None:
            args_section = f"{kind}.args"
        flag = {
            "task": "-T",
            "policy": "-P",
            "embodiment": "-E",
            "grader": "-G",
            "sink": "-S",
            "operator_input": "-V",
        }.get(kind, "the CLI args flag")
        raise SystemExit(
            f"invalid arguments for {kind} {name!r}: {exc}; check [{args_section}] and {flag} k=v"
        ) from exc


def _build_guardrails(
    space: Box, max_action_delta: float | None, embodiment: Embodiment | None = None
) -> tuple[Approver, list[str], list[str]]:
    from abstract_harness.robot.core.errors import ConfigError
    from abstract_harness.robot.core.safety import build_guardrails

    try:
        return build_guardrails(space, max_action_delta, embodiment)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc


def _build_operator_session(
    policy: object, embodiment: Embodiment
) -> tuple[OperatorSession, OperatorSession | None]:
    """Build the prompt owner and enable its live input channel only when safe."""
    accepts_messages = bool(getattr(policy, "accepts_operator_messages", False))
    session = OperatorSession(console_usage=None if accepts_messages else USAGE_END_ONLY)
    connect_hook = getattr(embodiment, "connect_operator_session", None)
    if callable(connect_hook):
        if sys.platform == "win32":
            session.write_line(
                _styled(
                    "operator console unavailable: select cannot watch stdin on Windows; "
                    "feedback typing stays off",
                    _YELLOW,
                )
            )
            return session, None
        connect_hook(session)
        usage = USAGE if accepts_messages else USAGE_END_ONLY
        session.enable_footer(
            label="sent" if accepts_messages else "noted", echo_interval_s=ECHO_INTERVAL_S
        )
        label = "operator console:"
        session.write_line(f"{_styled(label, _CYAN)} {usage.removeprefix(label + ' ')}")
        return session, session

    if not accepts_messages:
        return session, None
    if sys.platform == "win32":
        session.write_line(
            _styled(
                "operator console unavailable: select cannot watch stdin on Windows; "
                "feedback typing stays off",
                _YELLOW,
            )
        )
        return session, None

    defer_hook = getattr(embodiment, "defer_operator_end", None)
    if callable(defer_hook):
        defer_hook()
    elif not embodiment.info.is_simulated:
        session.write_line(
            _styled(
                "operator console unavailable: this embodiment predates the operator console "
                "and still owns the end-of-episode keypress; feedback typing stays off",
                _YELLOW,
            )
        )
        return session, None

    label = "operator console:"
    session.enable_footer(label="sent", echo_interval_s=ECHO_INTERVAL_S)
    session.write_line(f"{_styled(label, _CYAN)} {USAGE.removeprefix(label + ' ')}")
    return session, session


def _step_limit_count(log: EvalLog) -> int:
    """Count recorded trials whose termination reason is the step horizon."""
    return sum(
        reason == "max_steps" for scene in log.samples for reason in scene.termination_reasons
    )


def _seconds_horizon(log: EvalLog) -> tuple[float, int, float | None] | None:
    """Return a validated declared/resolved seconds horizon from a saved log."""
    max_seconds = log.eval.max_seconds
    max_steps = log.eval.max_steps
    if not (
        isinstance(max_seconds, (int, float))
        and not isinstance(max_seconds, bool)
        and math.isfinite(max_seconds)
        and max_seconds > 0
        and isinstance(max_steps, int)
        and not isinstance(max_steps, bool)
        and max_steps > 0
    ):
        return None

    rate = log.eval.embodiment_info.get("control_hz")
    valid_rate = (
        float(rate)
        if isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and math.isfinite(rate)
        and rate > 0
        else None
    )
    return float(max_seconds), max_steps, valid_rate


def _seconds_horizon_text(log: EvalLog) -> str | None:
    """Format a seconds-based horizon for compact CLI metadata surfaces."""
    horizon = _seconds_horizon(log)
    if horizon is None:
        return None
    max_seconds, max_steps, rate = horizon
    text = f"{max_seconds:g}s -> {max_steps} steps"
    return f"{text} at {rate:g} Hz" if rate is not None else text


def _outcome_line(log: EvalLog) -> tuple[str, bool] | None:
    """Return an outcome digest and unmapped flag, or ``None`` with no reasons."""
    reasons: list[object] = [
        reason for scene in log.samples for reason in scene.termination_reasons
    ]
    if not reasons:
        return None

    counts: dict[str, int] = {}
    has_unmapped = False
    for reason in reasons:
        text = "" if reason is None else str(reason)
        if not text:
            phrase = "no reason recorded"
        else:
            phrase = _OUTCOME_PHRASES.get(text, text)
            if text not in _OUTCOME_PHRASES:
                has_unmapped = True
        counts[phrase] = counts.get(phrase, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(reasons) == 1:
        return ordered[0][0], has_unmapped
    return ", ".join(f"{count} {phrase}" for phrase, count in ordered), has_unmapped


def _print_step_limit_notice(log: EvalLog, is_adhoc: bool) -> None:
    """Print the shared timeout note and the horizon-ownership hint when needed."""
    count = _step_limit_count(log)
    if count == 0:
        return

    note = f"note: {count}/{log.results.total_trials} trials hit the step limit before terminating"
    max_steps = log.eval.max_steps
    seconds_horizon = _seconds_horizon(log)
    if seconds_horizon is not None:
        max_seconds, resolved_steps, rate = seconds_horizon
        parenthetical = f"max_seconds={max_seconds:g}, resolved max_steps={resolved_steps}"
        if rate is not None:
            parenthetical += f" at {rate:g} Hz"
        note += f" ({parenthetical})"
    # Guards below reject bool/str values a hand-edited log can smuggle past
    # from_dict (bool is an int subclass, so isinstance alone lets True in).
    elif isinstance(max_steps, int) and not isinstance(max_steps, bool):
        parenthetical = f"max_steps={max_steps}"
        rate = log.eval.embodiment_info.get("control_hz")
        if (
            isinstance(rate, (int, float))
            and not isinstance(rate, bool)
            and math.isfinite(rate)
            and rate > 0
        ):
            parenthetical += f", ~{max_steps / rate:g}s at {rate:g} Hz"
        note += f" ({parenthetical})"
    print(_styled(note, _YELLOW))
    if is_adhoc:
        hint = "hint: raise max_steps in the robot config's [rig] section"
    elif seconds_horizon is not None:
        hint = f"hint: task {log.eval.task!r} defines its own max_seconds"
    else:
        hint = f"hint: task {log.eval.task!r} defines its own max_steps"
    print(_styled(hint, _DIM))


def _has_policy_transcripts(log: EvalLog) -> bool:
    """Whether any recorded trial carries a policy audit record."""
    return any(
        transcript is not None for scene in log.samples for transcript in scene.policy_transcripts
    )


def _print_degraded(line: str) -> None:
    """Print transcript-derived text, replacing unencodable code points.

    Transcripts are foreign data: a hostile or buggy model server can put lone
    UTF-16 surrogates in message content, and they survive the log's JSON
    round-trip but crash ``print`` on a strict-UTF-8 stdout. The forensic
    reader must degrade, never crash, on the episodes it exists to explain.
    """
    print(line.encode("utf-8", errors="replace").decode("utf-8"))


def _render_chat_transcript(transcript: list[object]) -> None:
    """Print roles, text, tool calls, and their indented results."""
    for raw_message in transcript:
        if not isinstance(raw_message, dict):
            continue
        role = str(raw_message["role"])
        content = _chat_content(raw_message.get("content"))
        if role == "tool":
            suffix = "" if content is None else f" {content}"
            _print_degraded(f"        tool:{suffix}")
            continue
        suffix = "" if content is None else f" {content}"
        _print_degraded(f"    {role}:{suffix}")
        tool_calls = raw_message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for raw_call in tool_calls:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function")
            if not isinstance(function, dict):
                continue
            name = str(function.get("name", "unknown"))
            arguments = function.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            _print_degraded(f"      -> {name}({arguments})")


def _print_policy_transcripts(log: EvalLog) -> None:
    """Append each available policy audit record in a tolerant human-readable form."""
    if not _has_policy_transcripts(log):
        print("no policy transcripts recorded")
        return
    print("policy transcripts:")
    for scene in log.samples:
        for trial, transcript in enumerate(scene.policy_transcripts):
            if transcript is None:
                continue
            _print_degraded(f"scene {scene.scene_id}, trial {trial}:")
            if _is_chat_transcript(transcript):
                _render_chat_transcript(transcript)
                continue
            for line in json.dumps(transcript, indent=2).splitlines():
                _print_degraded(f"    {line}")


class _WireTrial(NamedTuple):
    """A readable wire sidecar and its trial/run filesystem context."""

    trial_id: str
    rows: list[dict[str, Any]]
    blob_dir: Path


_WIRE_BLOB_RE = re.compile(r"\$blob:([0-9a-f]{64})")


def _wire_blob_shas(value: object) -> list[str]:
    """Return every valid symbolic blob reference in a captured JSON value."""
    if isinstance(value, str):
        return [match.group(1) for match in _WIRE_BLOB_RE.finditer(value)]
    if isinstance(value, list):
        return [sha for item in value for sha in _wire_blob_shas(item)]
    if isinstance(value, dict):
        return [sha for item in value.values() for sha in _wire_blob_shas(item)]
    return []


def _load_wire_trials(log: EvalLog, log_path: Path) -> list[_WireTrial]:
    """Load every readable guarded wire sidecar referenced by an eval log."""
    trials: list[_WireTrial] = []
    for scene in log.samples:
        for epoch in range(len(scene.epochs)):
            if epoch >= len(scene.trial_metadata):
                continue
            target = resolve_log_pointer(log_path, scene.trial_metadata[epoch].get("wire_capture"))
            if target is None:
                continue
            loaded = read_jsonl_prefix(target)
            if loaded is None:
                continue
            blob_dir = derive_blob_dir(log_path, target)
            if blob_dir is None:
                continue
            trials.append(
                _WireTrial(
                    f"{scene.scene_id}-e{epoch}",
                    loaded,
                    blob_dir,
                )
            )
    return trials


def _wire_request(row: dict[str, Any]) -> dict[str, Any]:
    """Return one row's request mapping, degrading malformed foreign data."""
    request = row.get("request")
    return cast(dict[str, Any], request) if isinstance(request, dict) else {}


def _print_wire_table(trials: list[_WireTrial]) -> None:
    """Print the compact all-trials wire attempt table."""
    if not trials:
        print("no wire capture recorded")
        return
    print("wire calls:")
    print("  trial  call  attempt  endpoint  status  duration  images  new blob bytes")
    for trial in trials:
        seen: set[str] = set()
        for row in trial.rows:
            shas = _wire_blob_shas(_wire_request(row))
            first_shas = set(shas).difference(seen)
            seen.update(shas)
            new_bytes = 0
            for sha in first_shas:
                with suppress(OSError):
                    new_bytes += (trial.blob_dir / f"{sha}.png").stat().st_size
            duration = row.get("duration_s")
            shown_duration = (
                f"{duration:.3f}s"
                if isinstance(duration, (int, float)) and not isinstance(duration, bool)
                else "-"
            )
            status = "-" if row.get("status") is None else str(row["status"])
            print(
                f"  {trial.trial_id}  {row.get('call', '-')}  "
                f"{row.get('attempt', '-')}  {row.get('endpoint', '-')}  "
                f"{status}  {shown_duration}  {len(shas)}  {new_bytes}"
            )


def _available_wire_trials(trials: list[_WireTrial]) -> str:
    """Format captured trial ids for a guided CLI error."""
    return ", ".join(trial.trial_id for trial in trials)


def _print_wire_call(trials: list[_WireTrial], call: int, selected_trial: str | None) -> None:
    """Pretty-print every captured attempt for one logical call index."""
    if not trials:
        raise SystemExit("no wire capture recorded; omit CALL to print an empty table")
    if selected_trial is None:
        if len(trials) > 1:
            raise SystemExit(
                "--trial is required for --wire CALL; available trials: "
                f"{_available_wire_trials(trials)}"
            )
        trial = trials[0]
    else:
        matches = [trial for trial in trials if trial.trial_id == selected_trial]
        if not matches:
            raise SystemExit(
                f"wire trial {selected_trial!r} not found; available trials: "
                f"{_available_wire_trials(trials)}"
            )
        trial = matches[0]

    rows = [row for row in trial.rows if row.get("call") == call]
    if not rows:
        raise SystemExit(f"wire call {call} not found in trial {trial.trial_id}")
    print(f"wire trial {trial.trial_id}, call {call}:")
    for row in rows:
        print(f"attempt {row.get('attempt', '-')}:")
        print(f"  endpoint: {row.get('endpoint', '-')}")
        print(f"  status: {row.get('status')}")
        print(f"  duration_s: {row.get('duration_s', '-')}")
        if "error" in row:
            _print_degraded(f"  error: {row['error']}")
        for label in ("request", "response"):
            _print_degraded(f"  {label}:")
            dumped = json.dumps(row.get(label), indent=2, sort_keys=True, ensure_ascii=False)
            for line in dumped.splitlines():
                _print_degraded(f"    {line}")


def _print_wire_capture(
    log: EvalLog,
    log_path: Path,
    wire: int | bool,
    selected_trial: str | None,
) -> None:
    """Print either the capture table or every attempt for one call."""
    trials = _load_wire_trials(log, log_path)
    if wire is True:
        if selected_trial is not None:
            raise SystemExit("--trial requires an integer --wire CALL")
        _print_wire_table(trials)
        return
    _print_wire_call(trials, wire, selected_trial)


def _print_run_summary(log: EvalLog, log_path: str, is_adhoc: bool) -> None:
    """Print the compact post-run summary and failure diagnostics."""
    failed = log.status != "success"
    errored_count = log.results.errored_trials
    status_color = _RED if failed else _GREEN
    print(f"{_styled('run status:', _CYAN)} {_styled(_display_status(log.status), status_color)}")
    outcome = _outcome_line(log)
    if outcome is not None:
        digest, has_unmapped = outcome
        line = f"{_styled('outcome:', _CYAN)} {digest}"
        if has_unmapped:
            _print_degraded(line)
        else:
            print(line)
    if failed and log.error is not None:
        print(f"{_styled('error:', _CYAN)} {_styled(log.error, _RED)}")
    if failed or errored_count:
        # Non-successful scenes are failure context. Errored scenes stay visible
        # even when the run succeeded (issue #73).
        for scene in log.samples:
            if scene.status != "success":
                detail = "" if scene.error in (None, log.error) else f": {scene.error}"
                print(f"  [{_styled(scene.status, _RED)}] {scene.scene_id}{detail}")
    _print_step_limit_notice(log, is_adhoc)
    trials = f"trials: {log.results.total_trials}"
    if errored_count:
        trials += f" ({errored_count} errored)"
    print(f"{_styled('scenes:', _CYAN)} {log.results.total_scenes}  {trials}")
    for name, value in sorted(log.results.metrics.items()):
        print(f"  {name}: {_styled(_format_metric(value), _BOLD)}")
    print(f"{_styled('log:', _CYAN)} {_styled(log_path, _DIM)}")
    # Every run ends with the copy-pasteable read-back command (issue #90):
    # a bare path teaches a first-time user nothing about what to do next.
    print(_styled(f"hint: read it with: abstract robot show {log_path}", _DIM))
    print(_styled(f"hint: HTML viewer: abstract robot view {log_path}", _DIM))
    if _has_policy_transcripts(log):
        print(
            _styled(
                f"hint: agent conversation: abstract robot show {log_path} --transcript",
                _DIM,
            )
        )
    if log.stats.frames_dir is not None:
        from abstract_harness.robot.core._video import count_frames, resolve_frames_dir

        # Gate on frames actually existing: a camera-less --store-frames run
        # records a frames_dir but writes nothing, and the hint must not
        # point at a command that would exit "no frames found".
        root = resolve_frames_dir(log.stats.frames_dir, Path(log_path))
        if root is not None and count_frames(root):
            print(_styled(f"hint: render videos with: abstract robot video {log_path}", _DIM))
    log_dir = Path(log_path).parent
    print(_styled(f"hint: browse all logs: abstract robot view {log_dir}", _DIM))


class Components(NamedTuple):
    """A constructed policy and embodiment, with their names and the config they came
    from for the run header, and the claim on the embodiment's devices."""

    policy: Any
    policy_name: str
    policy_source: str
    embodiment: Any
    embodiment_name: str
    embodiment_source: str
    claim: DeviceClaim


def _headless_session(env: Mapping[str, str], platform: str = sys.platform) -> bool:
    """Return whether the operator is expected to be remote from this process."""
    # _setup.py asks if a viewer window can open here; this asks if the human is elsewhere.
    if env.get("SSH_CONNECTION"):
        return True
    return platform.startswith("linux") and not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))


def _cmd_show(
    path: str,
    *,
    transcript: bool = False,
    wire: int | bool | None = None,
    trial: str | None = None,
) -> int:
    from abstract_harness.robot.core import read_eval_log

    log = read_eval_log(path)
    _print_step_limit_notice(log, log.eval.task in ("adhoc", "auto"))
    print(f"task:        {log.eval.task}")
    # One shared instruction (the adhoc case) reads as run-level identity;
    # differing instructions print per scene below instead. Instructions are
    # foreign text (dataset/operator-supplied), so printing degrades.
    instructions = {scene.instruction for scene in log.samples}
    shared = next(iter(instructions)) if len(instructions) == 1 else None
    if shared:
        _print_degraded(f"instruction: {shared}")
    print(f"policy:      {log.eval.policy}")
    print(f"embodiment:  {log.eval.embodiment}")
    print(f"run status:  {_display_status(log.status)}")
    outcome = _outcome_line(log)
    if outcome is not None:
        digest, has_unmapped = outcome
        line = f"outcome:     {digest}"
        if has_unmapped:
            _print_degraded(line)
        else:
            print(line)
    print(f"created:     {log.eval.created}")
    print(f"git:         {log.eval.git_commit}")
    horizon = _seconds_horizon_text(log)
    if horizon is not None:
        print(f"horizon:     {horizon}")
    trials = f"trials: {log.results.total_trials}"
    if log.results.errored_trials:
        trials += f" ({log.results.errored_trials} errored)"
    print(f"scenes:      {log.results.total_scenes}   {trials}")
    if log.stats.frames_dir is not None:
        from abstract_harness.robot.core._video import count_frames, resolve_frames_dir

        root = resolve_frames_dir(log.stats.frames_dir, Path(path))
        if root is None:
            print(f"frames:      {log.stats.frames_dir} (not found from this directory)")
        else:
            # The resolved path, not the stored string: after a machine move
            # the stored string is exactly the path that does not work.
            n_frames = count_frames(root)
            plural = "frame" if n_frames == 1 else "frames"
            print(f"frames:      {root} ({n_frames} {plural})")
            if n_frames:
                print(_styled(f"hint: render videos with: abstract robot video {path}", _DIM))
    print("metrics:")
    for name, value in sorted(log.results.metrics.items()):
        print(f"  {name}: {_format_metric(value)}")
    print("scenes:")
    for scene in log.samples:
        reduced = "  ".join(f"{k}={_format_metric(v)}" for k, v in sorted(scene.reduced.items()))
        step_limit_count = sum(reason == "max_steps" for reason in scene.termination_reasons)
        details = [reduced] if reduced else []
        if step_limit_count:
            details.append(f"({step_limit_count}/{len(scene.epochs)} trials hit max_steps)")
        print(f"  [{scene.status}] {scene.scene_id}: {'  '.join(details)}")
        if not shared and scene.instruction:
            _print_degraded(f"      instruction: {scene.instruction}")
    if log.error:
        print(f"error: {log.error}")
    if transcript:
        _print_policy_transcripts(log)
    elif _has_policy_transcripts(log):
        print("policy transcripts: recorded (--transcript to print)")
        print(_styled(f"hint: HTML viewer: abstract robot view {path}", _DIM))
    if wire is not None:
        _print_wire_capture(log, Path(path), wire, trial)
    elif trial is not None:
        raise SystemExit("--trial requires --wire CALL")
    return 0 if log.status == "success" else 1


def _write_html(document: str, out_path: Path | None) -> int:
    """Write one document as UTF-8, returning the encoded byte count."""
    encoded = document.encode("utf-8", errors="replace")
    if out_path is None:
        sys.stdout.write(encoded.decode("utf-8"))
        return len(encoded)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(document)
    return len(encoded)


def _write_html_atomic(document: str, out_path: Path) -> int:
    """Replace one UTF-8 document atomically and return its encoded byte count."""
    encoded = document.encode("utf-8", errors="replace")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(f"{out_path.suffix}.tmp")
    with tmp.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(document)
    os.replace(tmp, out_path)
    return len(encoded)


def _render_log_page(
    log: EvalLog,
    log_path: Path,
    out_path: Path | None,
    *,
    no_frames: bool,
    frames_budget: float,
    live_frames_budget: float | None = None,
    refresh_seconds: int | None = None,
    atomic: bool = False,
    no_video: bool = False,
    serve_pass: bool = False,
) -> int:
    """Render one already-parsed log through the shared single-page pipeline."""
    frames_dir = None
    if not no_frames and log.stats.frames_dir is not None:
        from abstract_harness.robot.core._video import resolve_frames_dir

        frames_dir = resolve_frames_dir(log.stats.frames_dir, log_path)
    document = render_html(
        log,
        title=f"{log.eval.task} - {log_path.name}",
        log_path=log_path,
        frames_dir=frames_dir,
        frames_budget_bytes=int(frames_budget * 1_000_000),
        live_frames_budget_bytes=(
            None if live_frames_budget is None else int(live_frames_budget * 1_000_000)
        ),
        wire_media_elided=log.status == "started",
        refresh_seconds=refresh_seconds,
        no_video=no_video,
        serve_pass=serve_pass,
    )
    if atomic and out_path is not None:
        return _write_html_atomic(document, out_path)
    return _write_html(document, out_path)


def _open_browser(uri: str, display: str | Path) -> None:
    """Open one rendered artifact, degrading browser failures to warnings."""
    import webbrowser

    try:
        opened = webbrowser.open(uri)
    except Exception as exc:
        print(f"warning: could not open browser for {display}: {exc}", file=sys.stderr)
    else:
        if not opened:
            print(f"warning: could not open browser for {display}", file=sys.stderr)


def _index_instruction(log: EvalLog) -> str:
    """Return the shared instruction or the multi-scene fallback."""
    instructions = {scene.instruction for scene in log.samples}
    shared = next(iter(instructions)) if len(instructions) == 1 else None
    if shared:
        return shared
    count = len(log.samples)
    return f"{count} scene" if count == 1 else f"{count} scenes"


def _index_termination(log: EvalLog) -> str:
    """Return the order-preserved union of human-facing termination reasons."""
    seen: set[str] = set()
    phrases: list[str] = []
    for scene in log.samples:
        for reason in scene.termination_reasons:
            if reason is None:
                continue
            text = str(reason)
            if text in seen:
                continue
            seen.add(text)
            phrases.append(_OUTCOME_PHRASES.get(text, text))
    return ", ".join(phrases)


def _index_entry(log: EvalLog, log_path: Path, page: str) -> IndexEntry:
    """Build one directory-index row from a successfully parsed log."""
    model = log.eval.policy_config.get("model")
    return IndexEntry(
        name=log_path.name,
        page=page,
        created=log.eval.created,
        instruction=_index_instruction(log),
        policy=log.eval.policy,
        model=None if model is None else str(model),
        status=_display_status(log.status),
        status_class=_status_class(log.status),
        metrics=log.results.metrics,
        errored_trials=log.results.errored_trials,
        termination=_index_termination(log),
        error=log.error,
    )


def _unreadable_index_entry(log_path: Path, error: Exception) -> IndexEntry:
    """Build an error row for a file that is not a readable EvalLog."""
    try:
        created = datetime.fromtimestamp(log_path.stat().st_mtime, tz=UTC).isoformat()
    except OSError:
        created = ""
    return IndexEntry(
        name=log_path.name,
        page=None,
        created=created,
        instruction="",
        policy="",
        model=None,
        status="error",
        status_class="status-error",
        metrics={},
        errored_trials=0,
        termination="",
        error=f"unreadable: {error}",
    )


def _directory_page_names(log_paths: Sequence[Path]) -> dict[Path, str]:
    """Assign collision-free report names without consulting existing files."""
    names = {path: f"{path.stem}.html" for path in log_paths if path.stem != "index"}
    reserved = set(names.values())
    for path in log_paths:
        if path.stem != "index":
            continue
        candidate = "index_log.html"
        suffix = 2
        while candidate in reserved:
            candidate = f"index_log_{suffix}.html"
            suffix += 1
        names[path] = candidate
        reserved.add(candidate)
    return names


class _DirectoryRenderResult(NamedTuple):
    """Paths produced by one logs-directory render pass."""

    out_dir: Path
    index_path: Path


_LIVE_REDIRECT_SENTINEL = "<!-- abstract robot live redirect stub -->"


def _write_live_redirect_stub(path: Path) -> int:
    """Atomically replace an orphaned live page with an idempotent index redirect."""
    document = f"""{_LIVE_REDIRECT_SENTINEL}
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=index.html">
<title>Run completed</title></head>
<body><a href="index.html">Continue to the run index</a></body></html>
"""
    return _write_html_atomic(document, path)


def _render_view_directory(
    args: argparse.Namespace,
    log_dir: Path,
    *,
    force: bool,
    quiet: bool,
    refresh_seconds: int | None,
) -> _DirectoryRenderResult:
    """Render one incremental logs-directory pass and return its output paths."""
    from abstract_harness.robot.core import read_eval_log

    if args.out == "-":
        raise SystemExit("-o - cannot be used with a logs directory; pass an output directory")

    log_paths = sorted(path for path in log_dir.glob("*.json") if path.is_file())
    if not log_paths and not args.serve:
        raise SystemExit(f"no top-level *.json logs found in {log_dir}")

    out_dir = log_dir / "html" if args.out is None else Path(args.out)
    if (out_dir.exists() or out_dir.is_symlink()) and not out_dir.is_dir():
        if args.out is not None:
            raise SystemExit(f"--out {out_dir} is not a directory; pass an output directory")
        raise SystemExit(
            f"default output path {out_dir} exists and is not a directory; move it or pass -o DIR"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    page_names = _directory_page_names(log_paths)
    # Derived from this pass's own glob, not the caller's earlier one: a live
    # log appearing between the two globs must not have its freshly rendered
    # page immediately stubbed as an orphan.
    backed_live_pages = {page_names[path] for path in log_paths if path.name.endswith(".live.json")}
    entries: list[IndexEntry] = []
    pages_written = 0
    bytes_written = 0
    total = len(log_paths)
    # One pass, one parse per log, nothing retained beyond its row: a log that
    # fails to read or render (e.g. replaced mid-run by a concurrent writer)
    # becomes an error row instead of sinking the index.
    for index, log_path in enumerate(log_paths, start=1):
        page_name = page_names[log_path]
        page_path = out_dir / page_name
        is_live_path = log_path.name.endswith(".live.json")
        try:
            source_stat = log_path.stat()
            log = read_eval_log(str(log_path))
            suppressed_tier = args.serve or args.no_video
            # The suppressed-tier stamp sits a full 2 microseconds below the
            # source mtime, and the gate allows 1 microsecond of slack:
            # filesystems round timestamps to their tick (100 ns on NTFS), so
            # a 1 ns delta would floor below its own target and re-render
            # suppressed pages on every serve tick on Windows.
            stamp_ns = max(
                0,
                source_stat.st_mtime_ns - (_SUPPRESSED_STAMP_DELTA_NS if suppressed_tier else 0),
            )
            render_page = (
                force
                or log.status == "started"
                or not page_path.exists()
                or page_path.stat().st_mtime_ns < stamp_ns - _STAMP_TICK_SLACK_NS
            )
            if render_page:
                if not quiet:
                    print(f"[{index}/{total}] rendering {log_path.name}", file=sys.stderr)
                bytes_written += _render_log_page(
                    log,
                    log_path,
                    page_path,
                    no_frames=args.no_frames,
                    frames_budget=args.frames_budget,
                    live_frames_budget=(
                        args.live_frames_budget if args.serve and log.status == "started" else None
                    ),
                    refresh_seconds=(
                        _SERVE_LIVE_RERENDER_SECONDS
                        if args.serve and log.status == "started"
                        else None
                    ),
                    atomic=log.status == "started",
                    no_video=args.no_video,
                    serve_pass=args.serve,
                )
                if log.status != "started":
                    # Started pages re-render every pass by design (plan 0058);
                    # the two-level stamp is a completed-page contract only.
                    os.utime(
                        page_path,
                        ns=(source_stat.st_atime_ns, stamp_ns),
                    )
                pages_written += 1
            entries.append(_index_entry(log, log_path, page_name))
        except FileNotFoundError:
            if is_live_path:
                continue
            exc = FileNotFoundError(log_path)
            print(f"warning: could not read or render {log_path.name}: {exc}", file=sys.stderr)
            entries.append(_unreadable_index_entry(log_path, exc))
        except Exception as exc:
            print(f"warning: could not read or render {log_path.name}: {exc}", file=sys.stderr)
            entries.append(_unreadable_index_entry(log_path, exc))

    for live_page in out_dir.glob("*.live.html"):
        if live_page.name in backed_live_pages:
            continue
        try:
            with live_page.open(encoding="utf-8", errors="replace") as handle:
                if handle.readline().rstrip("\r\n") == _LIVE_REDIRECT_SENTINEL:
                    continue
            bytes_written += _write_live_redirect_stub(live_page)
        except OSError as exc:
            print(f"warning: could not redirect {live_page.name}: {exc}", file=sys.stderr)

    index_path = out_dir / "index.html"
    bytes_written += _write_html_atomic(
        render_index(entries, refresh_seconds=refresh_seconds),
        index_path,
    )
    if not quiet:
        print(
            f"index: {index_path} "
            f"({total} logs, {pages_written} pages, {bytes_written / 1_000_000:.1f} MB)"
        )
    return _DirectoryRenderResult(out_dir=out_dir, index_path=index_path)


class _QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Serve static reports without per-request stderr logging."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress the standard request log."""
        return None


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


def _serve_display_urls(bind_host: str, port: int) -> tuple[str, ...]:
    """Return the URLs shown for one bound address and socket-derived port."""
    display_host = socket.gethostname() if bind_host == "0.0.0.0" else bind_host
    primary = f"http://{display_host}:{port}/"
    if bind_host == "0.0.0.0":
        return primary, f"http://localhost:{port}/"
    return (primary,)


def _serve_open_url(bind_host: str, port: int) -> str:
    """Return the reachable URL that ``--open`` should target."""
    open_host = "127.0.0.1" if bind_host in _LOOPBACK_HOSTS | {"0.0.0.0"} else bind_host
    return f"http://{open_host}:{port}/"


def _raise_keyboard_interrupt(_signum: int, _frame: FrameType | None) -> NoReturn:
    """Map a termination signal onto the normal Ctrl-C shutdown path."""
    raise KeyboardInterrupt


def _serve_view_directory(
    args: argparse.Namespace,
    log_dir: Path,
    render_result: _DirectoryRenderResult,
) -> int:
    """Serve rendered logs and refresh the artifacts incrementally until stopped."""
    bind_host = cast(str, args.host)
    port = cast(int, args.port)
    handler = partial(_QuietHTTPRequestHandler, directory=str(render_result.out_dir))
    try:
        server = ThreadingHTTPServer((bind_host, port), handler)
    except OSError as exc:
        raise SystemExit(str(exc)) from exc

    actual_port = int(server.server_address[1])
    urls = _serve_display_urls(bind_host, actual_port)
    # ThreadingHTTPServer.__init__ has already bound and started listening, so
    # printing the URLs and opening one before serve_forever cannot race a request.
    print(f"serving logs at: {urls[0]}")
    for url in urls[1:]:
        print(f"                 {url}")
    if bind_host in _LOOPBACK_HOSTS:
        print(
            _styled(
                "serving to this machine only; pass --host 0.0.0.0 to serve to your network",
                _DIM,
            )
        )
    else:
        print(
            _styled(
                "serving to your network: anyone who can reach this machine can view these logs",
                _YELLOW,
            )
        )
    print("press Ctrl-C to stop")

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    sigterm_installed = False
    server_thread_started = False
    previous_live_set: set[str] = set()
    accumulated_seconds = 0
    try:
        signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
        sigterm_installed = True
        server_thread.start()
        server_thread_started = True
        if args.open:
            open_url = _serve_open_url(bind_host, actual_port)
            _open_browser(open_url, open_url)
        while True:
            _serve_sleep(_SERVE_LIVE_RERENDER_SECONDS)
            accumulated_seconds += _SERVE_LIVE_RERENDER_SECONDS
            live_set = {path.name for path in log_dir.glob("*.live.json")}
            if not (
                live_set != previous_live_set
                or live_set
                or accumulated_seconds >= _SERVE_RERENDER_SECONDS
            ):
                continue
            try:
                _render_view_directory(
                    args,
                    log_dir,
                    force=False,
                    quiet=True,
                    refresh_seconds=(
                        _SERVE_LIVE_RERENDER_SECONDS if live_set else _SERVE_RERENDER_SECONDS
                    ),
                )
            except (Exception, SystemExit) as exc:
                print(f"warning: re-render failed: {exc}", file=sys.stderr)
                # Reset the accumulator on failure too: an idle directory with
                # a persistent render error must warn once per baseline period,
                # not once per 2s tick.
                accumulated_seconds = 0
            else:
                previous_live_set = live_set
                accumulated_seconds = 0
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            # shutdown() waits on an event only serve_forever sets; calling it
            # when the serve thread never started would hang forever.
            if server_thread_started:
                server.shutdown()
        finally:
            try:
                server.server_close()
            finally:
                if sigterm_installed:
                    # A non-Python-installed prior handler reads back as None,
                    # which signal.signal rejects; SIG_DFL is the sane restore.
                    restore = signal.SIG_DFL if previous_sigterm is None else previous_sigterm
                    signal.signal(signal.SIGTERM, restore)


def _cmd_view_directory(args: argparse.Namespace, log_dir: Path) -> int:
    """Render a logs directory and optionally serve it until stopped."""
    live_set = {path.name for path in log_dir.glob("*.live.json")}
    render_result = _render_view_directory(
        args,
        log_dir,
        force=args.force,
        quiet=False,
        refresh_seconds=(
            (_SERVE_LIVE_RERENDER_SECONDS if live_set else _SERVE_RERENDER_SECONDS)
            if args.serve
            else None
        ),
    )
    if args.serve:
        return _serve_view_directory(args, log_dir, render_result)
    if args.open:
        index_uri = render_result.index_path.resolve().as_uri()
        _open_browser(index_uri, render_result.index_path)
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    """Render one saved log or a logs directory to UTF-8 HTML artifacts."""
    if args.port is not None and not args.serve:
        raise SystemExit("--port requires --serve")
    if args.host is not None and not args.serve:
        raise SystemExit("--host requires --serve")
    if args.port is not None and not (0 <= args.port <= 65535):
        raise SystemExit(f"--port must be between 0 and 65535, got {args.port}")
    args.port = 8300 if args.port is None else args.port
    args.host = "127.0.0.1" if args.host is None else args.host

    if not (math.isfinite(args.frames_budget) and args.frames_budget >= 0):
        raise SystemExit("--frames-budget must be a non-negative finite number")
    if not (math.isfinite(args.live_frames_budget) and args.live_frames_budget >= 0):
        raise SystemExit("--live-frames-budget must be a non-negative finite number")

    log_path = Path(args.log)
    if args.serve and not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)
    if log_path.is_dir():
        return _cmd_view_directory(args, log_path)
    if args.serve:
        raise SystemExit("--serve requires a logs directory")

    stdout_mode = args.out == "-"
    if stdout_mode and args.open:
        raise SystemExit("--open cannot be used with -o -: no file to open")
    out_path = (
        None
        if stdout_mode
        else (log_path.with_suffix(".html") if args.out is None else Path(args.out))
    )
    if out_path is not None and out_path.exists() and out_path.is_dir():
        raise SystemExit(f"--out {out_path} is a directory; pass an HTML file path")
    if out_path is not None and out_path.resolve() == log_path.resolve():
        # The one data-loss path in this command: rendering over the log it reads.
        raise SystemExit(f"--out {out_path} would overwrite the input log; pass a different path")

    from abstract_harness.robot.core import read_eval_log

    document_size = _render_log_page(
        read_eval_log(str(log_path)),
        log_path,
        out_path,
        no_frames=args.no_frames,
        frames_budget=args.frames_budget,
        no_video=args.no_video,
    )
    if stdout_mode:
        return 0

    file_path = cast(Path, out_path)
    size_suffix = f" ({document_size / 1_000_000:.1f} MB)" if document_size > 1_000_000 else ""
    print(f"wrote {file_path}{size_suffix}")
    if args.open:
        _open_browser(file_path.resolve().as_uri(), file_path)
    return 0


def _cmd_video(args: argparse.Namespace) -> int:
    """Render a log's stored frames to one MP4 per (trial, camera) stream.

    Per-stream failures are isolated: each is reported on stderr, the
    remaining streams still encode, and the exit code is 1 if any failed.
    Results (``wrote ...`` lines, the fps note, the final summary) go to
    stdout; warnings and failure reports go to stderr.
    """
    from abstract_harness.robot.core import read_eval_log
    from abstract_harness.robot.core._video import (
        default_fps,
        discover_streams,
        encode_stream,
        frames_dir_candidates,
        resolve_frames_dir,
    )

    log = read_eval_log(args.log)
    frames_dir = log.stats.frames_dir
    if frames_dir is None:
        raise SystemExit("this log has no stored frames (re-run with --store-frames)")
    log_path = Path(args.log)
    root = resolve_frames_dir(frames_dir, log_path)
    if root is None:
        as_is, fallback = frames_dir_candidates(frames_dir, log_path)
        raise SystemExit(f"frames directory not found; tried {as_is} and {fallback}")

    streams, strays = discover_streams(root)
    for stray in strays:
        print(
            f"warning: skipping {stray.name}: does not match the frame filename pattern",
            file=sys.stderr,
        )
    if not streams:
        raise SystemExit(f"no frames found in {root}")

    if args.fps is not None:
        if not (math.isfinite(args.fps) and args.fps > 0):
            raise SystemExit("--fps must be a positive finite number")
        fps, fps_source = args.fps, "--fps"
    else:
        fps, fps_source = default_fps(log.eval.embodiment_info)

    if args.ffmpeg is not None:
        if not (os.path.isfile(args.ffmpeg) and os.access(args.ffmpeg, os.X_OK)):
            raise SystemExit(f"--ffmpeg {args.ffmpeg} is not an executable file")
        ffmpeg = args.ffmpeg
    else:
        which = shutil.which("ffmpeg")
        if which is None:
            raise SystemExit(
                "ffmpeg not found on PATH; install it (e.g. apt install ffmpeg) "
                "or pass --ffmpeg PATH"
            )
        ffmpeg = which

    out_dir = root if args.out is None else Path(args.out)
    if out_dir.exists() and not out_dir.is_dir():
        raise SystemExit(f"--out {out_dir} exists and is not a directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    # All validation is done: only now may result-looking stdout appear.
    print(f"fps: {fps:g} ({fps_source})")
    failed = 0
    for prefix, frames in streams.items():
        out_path = out_dir / f"{prefix}.mp4"
        result = encode_stream(frames, out_path, fps, ffmpeg)
        if result.skipped_empty:
            plural = "frame" if result.skipped_empty == 1 else "frames"
            print(
                f"warning: {prefix}: skipped {result.skipped_empty} empty {plural}",
                file=sys.stderr,
            )
        if result.error is not None:
            failed += 1
            print(f"failed: {prefix}: {result.error}", file=sys.stderr)
        else:
            plural = "frame" if result.piped == 1 else "frames"
            print(f"wrote {out_path} ({result.piped} {plural})")
    total = len(streams)
    summary = f"wrote {total - failed}/{total} streams"
    if failed:
        summary += f", {failed} failed"
    print(summary)
    return 1 if failed else 0


def components(
    policy: str,
    policy_args: Mapping[str, Any],
    embodiment: str,
    embodiment_args: Mapping[str, Any],
    source: str,
) -> Components:
    """Construct a registered policy and embodiment and claim the embodiment's devices.

    The embodiment is constructed last, so the caller's ``run_task`` (or its own
    ``try``/``finally`` around ``close()``) follows immediately and no constructed
    embodiment can leak past a later failure. A failing construction releases the claim.
    """
    built_policy = _resolve_or_exit("policy", policy, **policy_args)
    factories = registered("embodiment")
    slots = device_slots(factories[embodiment]) if embodiment in factories else ()
    if embodiment in factories and getattr(factories[embodiment], "MANAGES_DEVICE_CLAIMS", False):
        slots = ()
    claim = claim_devices(slots, embodiment_args, os.environ)
    try:
        built_embodiment = _resolve_or_exit("embodiment", embodiment, **embodiment_args)
        bind_fn = getattr(built_policy, "bind", None)
        if callable(bind_fn):
            bind_fn(built_embodiment.info)
    except BaseException:
        claim.release()
        raise
    return Components(built_policy, policy, source, built_embodiment, embodiment, source, claim)


def adhoc_task(instruction: str, max_steps: int, scorer: str = "operator") -> Task:
    """One scene of an operator-written instruction."""
    from abstract_harness.robot.core.scene import Scene
    from abstract_harness.robot.core.task import Task

    return Task(
        name="adhoc",
        scenes=[Scene(id="scene-0", instruction=instruction)],
        scorer=_resolve_or_exit("scorer", scorer),
        max_steps=max_steps,
        metadata={"instruction": instruction, "adhoc": True},
    )


def _build_and_announce_guardrails(
    space: Box, embodiment: Embodiment, max_action_delta: float | None = None
) -> Approver:
    """Build the guardrail chain for a run, announcing what is active.

    The approver chain sits below the policy in the rollout, so nothing the policy emits
    (a wild VLA action or a misbehaving LLM) reaches the hardware unchecked.
    ``max_action_delta`` overrides the per-step limit derived from the space's bounds.
    """
    approver, active, guard_warnings = _build_guardrails(space, max_action_delta, embodiment)
    for warning in guard_warnings:
        print(f"guardrails warning: {warning}", file=sys.stderr)
    print(f"guardrails: {' + '.join(active) if active else 'none active'}")
    return approver


def _announce_live_view(
    policy_name: str, log_dir: str, env: Mapping[str, str] = os.environ
) -> None:
    """Print the copyable browser-view command for agent runs."""
    if policy_name != "agent":
        return
    quoted = shlex.quote(log_dir)
    headless = _headless_session(env)
    command_tail = "--serve --host 0.0.0.0" if headless else "--serve --open"
    print(
        _styled(
            f"live: watch this run in your browser →  abstract robot view {quoted} {command_tail}",
            _BOLD_BRIGHT_MAGENTA,
        )
    )
    url = ""
    if headless:
        fields = env.get("SSH_CONNECTION", "").split()
        host = fields[2] if len(fields) == 4 else socket.gethostname()
        if ":" in host and not (host.startswith("[") and host.endswith("]")):
            host = f"[{host}]"
        url = f"; open http://{host}:8300/"
    print(
        _styled(
            "      each agent turn, notes, and operator/voice input, updating live" + url,
            _DIM,
        )
    )


def run_task(
    task: Task,
    resolved: Components,
    *,
    log_dir: str,
    prompt: bool = True,
    store_frames: bool = True,
    rerun_save: bool = True,
    seed: int = 0,
    grade: bool = True,
) -> int:
    """Run ``task`` on the constructed components and return the exit code: 0 when the
    run succeeds, 1 when it does not, 130 on Ctrl-C.

    With ``prompt`` and a terminal on stdin the run is attended: the operator console
    takes messages and Esc, and, with ``grade``, the operator grader asks for the
    verdict. The JSON log and its live snapshot go to ``log_dir``, with stored frames and
    a Rerun recording when asked. The embodiment closes (releasing motor torque on real
    hardware) and the device claim is released in the ``finally``, whatever ends the run.
    """
    from abstract_harness.robot.core import eval
    from abstract_harness.robot.core.logging import JsonLogSink, LiveLogSink

    embodiment = resolved.embodiment
    live_sink: LiveLogSink | None = None
    rerun_sink: LogSink | None = None
    try:
        print(f"policy: {resolved.policy_name} ({resolved.policy_source})")
        print(f"embodiment: {resolved.embodiment_name} ({resolved.embodiment_source})")
        approver = _build_and_announce_guardrails(embodiment.info.action_space, embodiment)
        _announce_live_view(resolved.policy_name, log_dir)

        operator_input = None
        operator_session = None
        attended = prompt and sys.stdin.isatty()
        if attended:
            operator_session, operator_input = _build_operator_session(resolved.policy, embodiment)
        grader = None
        # Without ``grade`` nobody is asked for a verdict or notes after the trial. The
        # console's /y /n /p still record one; otherwise the trial stays unscored.
        if attended and grade:
            grader = _resolve_or_exit("grader", "operator")
            connect = getattr(grader, "connect_session", None)
            if operator_session is not None and callable(connect):
                connect(operator_session)

        sink = JsonLogSink(log_dir)
        live_sink = LiveLogSink(log_dir)
        sinks: list[LogSink] = [sink, live_sink]
        if rerun_save:
            from abstract_harness.robot.core.logging.rerun_sink import RerunSink

            rerun_sink = RerunSink(recording_dir=log_dir)
            sinks.append(rerun_sink)
            print(f"{_styled('rerun:', _CYAN)} recording .rrd")
        if events.active():
            sinks.append(events.EventSink())
        try:
            logs = eval(
                task,
                resolved.policy,
                embodiment,
                log_dir=log_dir,
                seed=seed,
                sinks=sinks,
                fail_on_error=False,
                approver=approver,
                store_frames=store_frames,
                operator_input=operator_input,
                grader=grader,
            )
        except KeyboardInterrupt:
            if sink.path is not None and sink.path.exists():
                _print_degraded(f"cancelled: partial log written to {sink.path}")
                print(_styled(f"hint: read it with: abstract robot show {sink.path}", _DIM))
            else:
                _print_degraded("cancelled: no log written")
            return 130
    finally:
        # Real-hardware embodiments release motor torque in close(); skipping it leaves
        # a robot energized. A failing unlink (read-only remount, full disk) must not
        # skip the close or replace the run's real exception.
        try:
            if live_sink is not None and live_sink.path is not None:
                with suppress(OSError):
                    live_sink.path.unlink(missing_ok=True)
        finally:
            try:
                embodiment.close()
            finally:
                resolved.claim.release()
    log = logs[0]
    _print_run_summary(log, str(sink.path), task.name in ("adhoc", "auto"))
    resolved_recording_path = getattr(rerun_sink, "resolved_recording_path", None)
    if resolved_recording_path:
        print(f"{_styled('rrd:', _CYAN)} {_styled(str(resolved_recording_path), _DIM)}")
    return 0 if log.status == "success" else 1


def doctor(embodiment: str, embodiment_args: Mapping[str, Any], source: str) -> int:
    """Check runtime requirements and the conformance of an embodiment's declared spaces.

    Purely declarative: the embodiment is constructed (constructors stay hardware-free
    by convention) but never reset or stepped.
    """
    from abstract_harness.robot.core.conformance import (
        check_embodiment,
        missing_runtime_requirements,
    )

    print(f"embodiment: {embodiment} ({source})")
    missing = missing_runtime_requirements(registered("embodiment").get(embodiment))
    for module, remedy in missing.items():
        print(f"  [error] runtime-requirement: {module} missing → {remedy}")
    built = _resolve_or_exit("embodiment", embodiment, **embodiment_args)
    try:
        report = check_embodiment(built.info)
    finally:
        built.close()
    print(report.summary())
    return 1 if not report.ok or missing else 0


def main(args: argparse.Namespace) -> int:
    """Run ``show``, ``view`` or ``video``."""
    if args.robot_command == "show":
        return _cmd_show(args.log, transcript=args.transcript, wire=args.wire, trial=args.trial)
    if args.robot_command == "view":
        return _cmd_view(args)
    return _cmd_video(args)
