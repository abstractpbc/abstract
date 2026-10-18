"""A run's events for whatever supervises it (``abstract robot serve``); off by default.

``abstract robot run --events-fd N`` opens the channel on an inherited pipe, and the run
writes one JSON object per line to it: ``{"t": <unix seconds>, "type": ..., ...}``.
Without the flag nothing is opened and ``emit`` returns at once, so a run in a terminal
behaves and prints exactly as before. The events describe what the terminal shows; the
operator's answers still go through the terminal, so nothing here changes how a run
moves, stops or parks the arms.

=================  ==========================================  ============================
type               fields                                      emitted by
=================  ==========================================  ============================
hello              version, pid                                ``channel``
log_dir            path                                        ``robot.cli.run``
dialog             id, kind (gate, verdict, notes), title,     ``OperatorSession``,
                   options                                     ``OperatorIO.wait_ready``
dialog_resolved    id, kind, answer                            the same
status             text (null when the status line closes)     ``OperatorSession.status``
progress           trial, step (control steps completed)       ``EventSink`` (``run_task``)
frame              trial, camera, step, path (a whole .npy)    ``FrameStore.put``
transcript         trial, step, message                        ``EventSink`` (``run_task``)
trial_end          trial, status, termination_reason,          ``EventSink`` (``run_task``)
                   verdict
episode_end        exit_code (the arms are parked by then)     ``robot.stack.episode``
summary            summary (the ``{"summary"}`` line)          ``robot.cli.run``
upload             upload (one ``{"upload"}`` line)            ``robot.cli``
error              message (a refusal or a crash)              ``channel``
dropped            count (events lost to a stalled reader)     ``channel``
=================  ==========================================  ============================

A dialog's ``answer`` says how it ended, never what was typed beyond a verdict:
``confirmed`` (a gate), the verdict as typed (``y``, ``n``, ``partial``, ``skip``,
``yes``, ``no``) or ``unrecognized``, ``text`` or ``empty`` (the notes), ``interrupted``
(Ctrl-C) or ``failed`` (no terminal to read).

A writer thread owns the pipe, so a supervisor that stops reading can never block the
episode: past ``MAX_QUEUED`` waiting events, new ones are dropped and counted.
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from typing import IO, Any

VERSION = 1
MAX_QUEUED = 10_000
CLOSE_TIMEOUT_S = 5.0


class _Channel:
    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream
        self._queue: queue.Queue[bytes | None] = queue.Queue(MAX_QUEUED)
        self.dropped = 0
        self._writer = threading.Thread(target=self._write, name="run-events", daemon=True)
        self._writer.start()

    def put(self, line: bytes) -> None:
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            self.dropped += 1

    def _write(self) -> None:
        while (line := self._queue.get()) is not None:
            try:
                self._stream.write(line)
                self._stream.flush()
            except (OSError, ValueError):  # the reader is gone; the run goes on without it
                return

    def close(self) -> None:
        """Flush what is queued, for at most ``CLOSE_TIMEOUT_S`` in all, then close. A
        writer still stuck on a reader that stopped reading keeps the pipe; the process
        exit closes it."""
        deadline = time.monotonic() + CLOSE_TIMEOUT_S
        with suppress(queue.Full):
            self._queue.put(None, timeout=CLOSE_TIMEOUT_S)
        self._writer.join(max(0.0, deadline - time.monotonic()))
        if not self._writer.is_alive():
            with suppress(OSError):
                self._stream.close()


_channel: _Channel | None = None
_ids = itertools.count(1)


def active() -> bool:
    """Whether a supervisor is listening (``--events-fd`` was given)."""
    return _channel is not None


def emit(type: str, **fields: Any) -> None:  # noqa: A002 - the event's own field name
    """Send one event; a no-op without a channel, and never raises or blocks."""
    channel = _channel
    if channel is None:
        return
    try:
        line = json.dumps({"t": round(time.time(), 3), "type": type, **fields}, default=str)
    except (TypeError, ValueError):  # unserializable even as text: not worth a crash
        return
    channel.put(line.encode() + b"\n")


@contextmanager
def channel(fd: int | None) -> Iterator[None]:
    """Events to the inherited pipe ``fd`` while the block runs; ``None`` leaves them off.

    The block's exception, if any, goes out as an ``error`` event first. On the way out
    the queued events are flushed (for at most ``CLOSE_TIMEOUT_S``) and the pipe closed.
    """
    global _channel
    if fd is None:
        yield
        return
    if fd < 3:
        raise ValueError("--events-fd must be a pipe, not stdin, stdout or stderr")
    try:
        os.set_inheritable(fd, False)  # this run's children do not hold the supervisor's pipe
        stream = os.fdopen(fd, "wb")
    except OSError as exc:
        raise ValueError(f"--events-fd {fd}: {exc.strerror}") from None
    opened = _channel = _Channel(stream)
    emit("hello", version=VERSION, pid=os.getpid())
    try:
        yield
    except Exception as exc:
        emit("error", message=str(exc) or type(exc).__name__)
        raise
    finally:
        if opened.dropped:
            dropped, opened.dropped = opened.dropped, 0
            emit("dropped", count=dropped)
        _channel = None
        opened.close()


class Dialog:
    """One blocking prompt as the supervisor sees it; set ``answer`` once it is read."""

    def __init__(self, kind: str) -> None:
        self.id = f"d{next(_ids)}"
        self.kind = kind
        self.answer = "answered"


@contextmanager
def dialog(kind: str, title: str, options: Sequence[str] = ()) -> Iterator[Dialog]:
    """Bracket one blocking prompt with ``dialog`` and ``dialog_resolved`` events."""
    prompt = Dialog(kind)
    emit("dialog", id=prompt.id, kind=kind, title=title.strip(), options=list(options))
    try:
        yield prompt
    except KeyboardInterrupt:
        emit("dialog_resolved", id=prompt.id, kind=kind, answer="interrupted")
        raise
    except BaseException:
        emit("dialog_resolved", id=prompt.id, kind=kind, answer="failed")
        raise
    emit("dialog_resolved", id=prompt.id, kind=kind, answer=prompt.answer)


class EventSink:
    """A log sink for ``run_task``: steps, policy transcript messages and trial ends as
    events. Rollout calls ``log_policy_messages`` only on sinks that define it."""

    def __init__(self) -> None:
        self._trial = ""

    def on_eval_start(self, spec: Any) -> None:
        """Nothing to send before the first trial."""

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        self._trial = f"{scene_id}-e{epoch}"

    def log_step(self, t: int, observation: Any, action: Any, result: Any) -> None:
        emit("progress", trial=self._trial, step=t + 1)

    def log_policy_messages(self, t: int, messages: Sequence[Any]) -> None:
        for message in messages:
            emit("transcript", trial=self._trial, step=t, message=message)

    def on_trial_end(self, record: Any) -> None:
        emit(
            "trial_end",
            trial=self._trial,
            status=record.status,
            termination_reason=record.termination_reason,
            verdict=record.operator_judgement,
        )

    def on_eval_end(self, log: Any) -> None:
        """The episode's end goes out from ``stack.episode``, with its exit code."""
