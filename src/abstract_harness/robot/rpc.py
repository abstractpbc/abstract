"""``abstract robot serve``: command the rig over HTTP and watch its runs.

The protocol follows Pi's RPC mode (pi.dev/docs/latest/rpc). A client POSTs one command
``{"id", "type", ...}`` to ``/rpc`` and gets ``{"id", "type": "response", "command",
"success", "data" | "error"}`` back. What happens after a command is accepted arrives on
the event stream (``/events``, server-sent events), never as a second response. Prompts
that block a run (the YAM readiness gates, the verdict and grader notes) are
``extension_ui_request`` events, answered with an ``extension_ui_response`` command, as Pi
does for extension dialogs.

Each episode is ``abstract robot run`` itself (the same refusals, meter, summary and
upload) on a pseudo-terminal this server owns. Its gates, Esc and Ctrl-C keep their
meaning: confirming a gate over HTTP is pressing Enter at the rig. No client chooses -E/-P
overrides and nothing here sets ``unattended``. The one override is the operator's, made
when starting serve: ``--auto-start`` gives every run ``-E auto_start=true``, which drops
yam_arms' two readiness gates and keeps Esc, End episode and Ctrl-C (demo mode, for runs
started by a relay with nobody clicking; someone stays at the rig with the e-stop).
``hold`` pauses new runs so the scene can be reset. The serving terminal mirrors the
episode and forwards keys typed there, so the operator at the rig keeps the usual console.
One episode at a time.

What the run is doing comes from its events, not its screen: the run writes them to a
pipe given as ``--events-fd`` (``robot/events.py``: dialogs, status, progress, stored
frames, transcript, episode end, summary, upload). Answers still go to the run's
terminal, typed as an operator would. A run that sends no ``hello`` (an older harness)
still runs and can be watched and stopped, but has no dialogs over RPC; its ``run_end``
says ``events: false``.

The feed (``/feed/<camera>``) streams the newest frame the running episode stored (each
``frame`` event names the file); between runs, a RealSense preview (the ``robot`` extra's
pyrealsense2), which is stopped before a run or check opens the cameras.
"""

from __future__ import annotations

import argparse
import ast
import codecs
import fcntl
import hmac
import json
import os
import re
import secrets
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
import traceback
import zlib
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from abstract_harness.robot import cli, episode
from abstract_harness.robot.config import RobotConfig

TOKEN_ENV = "ABSTRACT_RPC_TOKEN"
MAX_BODY_BYTES = 64 * 1024
MAX_EVENTS = 5000
MAX_OUTPUT_LINES = 1000
MAX_TEXT = 4000  # characters kept per transcript string or output line in an event
PROGRESS_INTERVAL_S = 0.25  # at most this often a progress event; a run steps at 30 Hz
TERMINAL_SIZE = (40, 120)  # rows, columns, when the serving process has no terminal
CTRL_C = b"\x03"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
CONTROL = re.compile(r"[\x00-\x1f\x7f]")
VERDICTS = ("y", "n", "partial", "skip")
# How each kind of run dialog (robot/events.py) is asked over RPC, and the bytes a cancel
# types into the run's terminal: Ctrl-C at a gate, as at the rig.
DIALOGS: dict[str, tuple[dict[str, Any], bytes]] = {
    "gate": ({"method": "confirm"}, CTRL_C),
    "verdict": ({"method": "select"}, b"skip\n"),
    "notes": ({"method": "input"}, b"\n"),
}
# Captures colour from the config's RealSense serials and keeps the newest JPEG of each
# at <out>/<camera>.jpg, at 1280x720 where the camera's link allows it (15 fps, then 6 fps
# on a USB 2 link), else 640x480. Streams only: no arm is touched. The first second of each camera
# is dropped while auto-exposure and white balance settle (it comes out dark and tinted).
# Exits when the server that started it goes away.
PREVIEW = """
import json, os, signal, sys, time
import cv2
import numpy as np
import pyrealsense2 as rs
out, parent = sys.argv[1], os.getppid()
stop = []
signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
pipelines, seen = {}, {}
MODES = ((1280, 720, 15), (1280, 720, 6), (640, 480, 15))
for arg in sys.argv[2:]:
    camera, serial = arg.split("=", 1)
    error = None
    for width, height, fps in MODES:
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        pipeline = rs.pipeline()
        try:
            pipeline.start(config)
        except RuntimeError as exc:
            error = str(exc)
            continue
        pipelines[camera] = pipeline
        break
    else:
        print(json.dumps({"camera": camera, "error": error}), file=sys.stderr, flush=True)
try:
    while pipelines and not stop and os.getppid() == parent:
        for camera, pipeline in pipelines.items():
            frames = pipeline.poll_for_frames()
            color = frames.get_color_frame() if frames else None
            if not color:
                continue
            seen[camera] = seen.get(camera, 0) + 1
            if seen[camera] <= 15:
                continue
            image = np.asanyarray(color.get_data())
            ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                path = os.path.join(out, camera + ".jpg")
                with open(path + ".tmp", "wb") as f:
                    f.write(jpeg.tobytes())
                os.replace(path + ".tmp", path)
        time.sleep(0.02)
finally:
    for pipeline in pipelines.values():
        pipeline.stop()
"""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def abstract_command() -> list[str]:
    """This harness's ``abstract`` command, from the environment serving."""
    script = Path(sys.executable).with_name("abstract")
    if script.is_file():
        return [str(script)]
    return [sys.executable, "-c", "from abstract_harness.cli import main; main()"]


def load_token(path: Path) -> str:
    """The bearer token from ``path``, written there first (0600) when the file is new."""
    path = path.expanduser()
    if token := cli.env_value(TOKEN_ENV, path):
        return token
    if path.exists():
        raise ValueError(f"{path} has no {TOKEN_ENV}=... line")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"{TOKEN_ENV}={token}\n")
    return token


def clip(value: Any) -> Any:
    """``value`` with long strings shortened, for events."""
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + "…"
    if isinstance(value, list):
        return [clip(item) for item in value]
    if isinstance(value, dict):
        return {key: clip(item) for key, item in value.items()}
    return value


def one_line(value: Any, name: str, *, limit: int = 2000) -> str:
    """A single line of operator text, safe to type into the episode's terminal."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if CONTROL.search(value) or len(value) > limit:
        raise ValueError(f"{name} must be one line of at most {limit} characters")
    return value.strip()


class EventLog:
    """Numbered events in a bounded buffer; readers wait for the ones after a number."""

    def __init__(self, limit: int = MAX_EVENTS) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=limit)
        self._seq = 0
        self._changed = threading.Condition()

    @property
    def seq(self) -> int:
        return self._seq

    def emit(self, event: dict[str, Any]) -> dict[str, Any]:
        with self._changed:
            self._seq += 1
            event = {**event, "seq": self._seq, "time": now()}
            self._events.append(event)
            self._changed.notify_all()
        return event

    def after(self, seq: int, timeout: float = 0.0) -> list[dict[str, Any]]:
        with self._changed:
            if self._seq <= seq and timeout > 0:
                self._changed.wait(timeout)
            return [event for event in self._events if event["seq"] > seq]


class Terminal:
    """A child process on its own pseudo-terminal, which it has as controlling terminal,
    so Ctrl-C written here reaches it as SIGINT, exactly as a keypress would."""

    def __init__(
        self,
        argv: list[str],
        env: dict[str, str],
        on_output: Callable[[bytes], None],
        *,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        setsid = shutil.which("setsid")
        if setsid is None:
            raise OSError("serve needs util-linux setsid to give the run a controlling terminal")
        master, slave = os.openpty()
        rows, columns = terminal_size()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
        try:
            # Not a process group leader, so setsid execs in place: self.process.pid is
            # `abstract robot run` itself. --ctty makes the pty its controlling terminal.
            self.process = subprocess.Popen(
                [setsid, "--ctty", *argv],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                env={"TERM": "xterm-256color", **env},
                close_fds=True,
                pass_fds=pass_fds,
            )
        except BaseException:
            os.close(master)
            raise
        finally:
            os.close(slave)
        self._master = master
        self._on_output = on_output
        self._closed = threading.Event()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        while not self._closed.is_set():
            ready, _, _ = select.select([self._master], [], [], 0.2)
            if not ready:
                continue
            try:
                data = os.read(self._master, 65536)
            except OSError:  # EIO: every holder of the slave side closed it
                return
            if not data:
                return
            try:
                self._on_output(data)
            except Exception:  # keep draining: an undrained pty blocks the run mid-episode
                with suppress(Exception):
                    traceback.print_exc()

    def write(self, data: bytes) -> None:
        os.write(self._master, data)

    def wait(self) -> int:
        code = self.process.wait()
        # Drain the last output; a lingering grandchild must not hold the reader forever.
        self._reader.join(timeout=2)
        self._closed.set()
        self._reader.join()
        os.close(self._master)
        return code


def terminal_size() -> tuple[int, int]:
    if sys.stdout.isatty():
        size = os.get_terminal_size(sys.stdout.fileno())
        return size.lines, size.columns
    return TERMINAL_SIZE


@dataclass
class Dialog:
    id: str
    kind: str
    request: dict[str, Any]
    cancel: bytes
    # "rpc" once a client's answer went to the terminal; the run's dialog_resolved
    # event closes it.
    answered_by: str | None = None


class EventReader:
    """The run's events pipe, one JSON object per line, each handed to ``on_event``.

    Read to the end whatever a handler does, like the pty: the run writes through a queue
    and never blocks on this, but its events would stop arriving.
    """

    def __init__(self, fd: int, on_event: Callable[[dict[str, Any]], None]) -> None:
        self._file = os.fdopen(fd, "rb")
        self._on_event = on_event
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        with self._file:
            for line in self._file:  # until every holder of the write end closed it
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                try:
                    self._on_event(event)
                except Exception:
                    with suppress(Exception):
                        traceback.print_exc()

    def wait(self, timeout: float) -> None:
        """The last events, once the run has exited and its end of the pipe is closed."""
        self._thread.join(timeout)


class Screen:
    """Turns terminal output into ``output`` events, one per completed line."""

    def __init__(self, emit: Callable[[dict[str, Any]], Any]) -> None:
        self._emit = emit
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._partial = ""
        self.lines: deque[str] = deque(maxlen=MAX_OUTPUT_LINES)
        self._lock = threading.Lock()

    @staticmethod
    def _visible(text: str) -> str:
        # A carriage return redraws the line: only what follows the last one shows. The
        # pty ends every line with CR LF, so a trailing CR is not a redraw.
        return ANSI.sub("", text).rstrip("\r").rsplit("\r", 1)[-1].rstrip()

    def feed(self, data: bytes) -> None:
        """One read from the terminal; only the pty reader thread calls this."""
        with self._lock:
            self._partial += self._decoder.decode(data)
            *complete, self._partial = self._partial.split("\n")
            for raw in complete:
                line = self._visible(raw)
                if line.strip():
                    self.lines.append(line)
                    self._emit({"type": "output", "text": clip(line)})


def read_npy_rgb(path: Path) -> tuple[int, int, bytes] | None:
    """(width, height, RGB bytes) of a uint8 HxWx3 ``.npy`` frame; None while it is
    still being written or when it is not one."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if data[:6] != b"\x93NUMPY" or len(data) < 10:
        return None
    if data[6] == 1:
        start, length = 10, int.from_bytes(data[8:10], "little")
    else:
        start, length = 12, int.from_bytes(data[8:12], "little")
    try:
        header = ast.literal_eval(data[start : start + length].decode("latin-1"))
    except (ValueError, SyntaxError, UnicodeDecodeError):
        return None
    shape = header.get("shape", ())
    if header.get("descr") != "|u1" or header.get("fortran_order") or len(shape) != 3:
        return None
    height, width, channels = shape
    pixels = data[start + length :]
    if channels != 3 or len(pixels) != height * width * 3:
        return None
    return width, height, pixels


def png(width: int, height: int, rgb: bytes, *, halve: bool = True) -> bytes:
    """An RGB PNG, at half size by default; zlib only, the harness has no image library."""
    if halve:
        stride, width, height = width * 3, width // 2, height // 2
        out = bytearray(width * height * 3)
        for y in range(height):
            row = rgb[2 * y * stride : 2 * y * stride + stride]
            base = y * width * 3
            for channel in range(3):
                out[base + channel : base + width * 3 : 3] = row[channel : 6 * width : 6]
        rgb = bytes(out)
    stride = width * 3
    raw = b"".join(b"\x00" + rgb[y * stride : (y + 1) * stride] for y in range(height))

    def chunk(tag: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(tag + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 3))
        + chunk(b"IEND", b"")
    )


def jpeg(width: int, height: int, rgb: bytes, quality: int = 85) -> bytes | None:
    """A full-size JPEG of an RGB frame, or None without OpenCV (the robot extra has it)."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    bgr = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3)[:, :, ::-1]
    ok, data = cv2.imencode(".jpg", np.ascontiguousarray(bgr), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return data.tobytes() if ok else None


def shrunk(data: bytes, width: int, quality: int = 80) -> bytes:
    """A JPEG at most ``width`` wide, keeping its aspect; unchanged when it is narrower
    already or OpenCV is missing."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return data
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[1] <= width:
        return data
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return encoded.tobytes() if ok else data


def query_int(query: dict[str, list[str]], name: str, default: int, low: int, high: int) -> int:
    """An integer query parameter, clamped to [low, high]; the default when absent or bad."""
    try:
        value = int((query.get(name) or [""])[0])
    except ValueError:
        return default
    return min(high, max(low, value))


@dataclass
class Frame:
    key: str
    content_type: str
    data: bytes


class RunFrames:
    """The newest stored frame per camera of the running episode, from its ``frame``
    events, encoded once per file: a camera frame (640 wide or more) as a full-size JPEG,
    else, or without OpenCV, as a PNG (halved from 640 wide)."""

    def __init__(self) -> None:
        self._newest: dict[str, str] = {}
        self._cache: dict[str, Frame] = {}
        self._lock = threading.Lock()

    def add(self, camera: str, path: str) -> None:
        with self._lock:
            self._newest[camera] = path

    def cameras(self) -> list[str]:
        with self._lock:
            return sorted(self._newest)

    def frame(self, camera: str) -> Frame | None:
        with self._lock:
            path = self._newest.get(camera)
            if path is None:
                return None
            cached = self._cache.get(camera)
            if cached is not None and cached.key == path:
                return cached
            decoded = read_npy_rgb(Path(path))
            if decoded is None:
                return cached
            width, height, pixels = decoded
            data = jpeg(width, height, pixels) if width >= 640 else None
            if data is not None:
                self._cache[camera] = Frame(path, "image/jpeg", data)
            else:
                data = png(width, height, pixels, halve=width >= 640)
                self._cache[camera] = Frame(path, "image/png", data)
            return self._cache[camera]


class Preview:
    """Between runs, the cameras through ``PREVIEW`` while anyone watches the feed."""

    GRACE_S = 5.0

    def __init__(
        self, python: Path, serials: dict[str, str], emit: Callable[[dict[str, Any]], Any]
    ) -> None:
        self._python = python
        self._cameras = {f"{slot}_cam": serial for slot, serial in serials.items()}
        self._emit = emit
        self._dir = Path(tempfile.mkdtemp(prefix="abstract-preview-"))
        self._process: subprocess.Popen[bytes] | None = None
        self._viewers = 0
        self._blocked = 0
        self._stop_timer: threading.Timer | None = None
        self._lock = threading.Lock()

    @property
    def on(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def viewers(self) -> int:
        return self._viewers

    def _start(self) -> None:
        if self.on or self._blocked or not self._cameras:
            return
        for stale in self._dir.glob("*.jpg"):
            stale.unlink()
        cameras = [f"{camera}={serial}" for camera, serial in self._cameras.items()]
        log = (self._dir / "preview.log").open("ab")
        self._process = subprocess.Popen(
            [str(self._python), "-c", PREVIEW, str(self._dir), *cameras],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )
        log.close()
        self._emit({"type": "preview", "on": True})

    def _stop(self) -> None:
        if self._stop_timer is not None:
            self._stop_timer.cancel()
            self._stop_timer = None
        process, self._process = self._process, None
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        self._emit({"type": "preview", "on": False})

    def _stop_if_unwatched(self) -> None:
        with self._lock:
            if self._viewers == 0:
                self._stop()

    @contextmanager
    def watching(self) -> Iterator[None]:
        with self._lock:
            self._viewers += 1
            if self._stop_timer is not None:
                self._stop_timer.cancel()
                self._stop_timer = None
            self._start()
        try:
            yield
        finally:
            with self._lock:
                self._viewers -= 1
                if self._viewers == 0 and self._process is not None:
                    self._stop_timer = threading.Timer(self.GRACE_S, self._stop_if_unwatched)
                    self._stop_timer.daemon = True
                    self._stop_timer.start()

    @contextmanager
    def released(self) -> Iterator[None]:
        """The cameras free for a run or check; the preview resumes after if watched."""
        with self._lock:
            self._blocked += 1
            self._stop()
        try:
            yield
        finally:
            with self._lock:
                self._blocked -= 1
                if self._viewers:
                    self._start()

    def frame(self, camera: str) -> Frame | None:
        if not self.on:
            return None
        path = self._dir / f"{camera}.jpg"
        try:
            stat = path.stat()
            return Frame(f"{camera}:{stat.st_mtime_ns}", "image/jpeg", path.read_bytes())
        except OSError:
            return None

    def close(self) -> None:
        with self._lock:
            self._blocked += 1
            self._stop()
        shutil.rmtree(self._dir, ignore_errors=True)


class Uploads:
    """Finished runs going to the [store], one at a time and each after its run has exited,
    so the next run can start while the last one uploads. Each is ``abstract robot upload``
    itself (its checks, its delete only after success, its runs.jsonl line): a failure
    leaves the run on disk for the next ``abstract robot upload``."""

    def __init__(
        self, command: Callable[[str], list[str]], emit: Callable[[dict[str, Any]], Any]
    ) -> None:
        self._command = command
        self._emit = emit
        self._queue: deque[str] = deque()
        self._current: str | None = None
        self._last: dict[str, Any] | None = None
        self._closing = False
        self._worker: threading.Thread | None = None
        self._changed = threading.Condition()

    def state(self) -> dict[str, Any]:
        with self._changed:
            return {"queued": list(self._queue), "current": self._current, "last": self._last}

    def add(self, run: str) -> None:
        with self._changed:
            self._queue.append(run)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, daemon=True)
                self._worker.start()
            self._changed.notify_all()
        self._emit({"type": "upload_queued", "run": run})

    def _work(self) -> None:
        while True:
            with self._changed:
                while not self._queue and not self._closing:
                    self._changed.wait()
                if self._closing:
                    return
                run = self._current = self._queue.popleft()
            self._emit({"type": "upload_start", "run": run})
            result = self._upload(run)
            with self._changed:
                self._current, self._last = None, result
                self._changed.notify_all()
            self._emit({"type": "upload_end", **result})

    def _upload(self, run: str) -> dict[str, Any]:
        result: dict[str, Any] = {"run": run, "ok": False, "error": None, "bytes": None}
        tail: deque[str] = deque(maxlen=6)
        try:
            # Its own session: a Ctrl-C meant for the server or a run never cuts it short.
            process = subprocess.Popen(
                self._command(run),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip()
                value: Any = None
                if line.startswith("{"):
                    with suppress(ValueError):
                        value = json.loads(line)
                record = value.get("upload") if isinstance(value, dict) else None
                if isinstance(record, dict):
                    ok = record.get("ok") is True
                    result.update(ok=ok, error=record.get("error"), bytes=record.get("bytes"))
                elif line:
                    tail.append(line)
            code = process.wait()
        except OSError as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            return result
        if not result["ok"] and not result["error"]:
            result["error"] = clip(f"exit {code}: " + " | ".join(tail))
        return result

    def close(self, timeout: float) -> list[str]:
        """Wait up to ``timeout`` for the upload under way; returns the runs left on disk
        (queued, and the current one if it outlasted the wait)."""
        deadline = time.monotonic() + timeout
        with self._changed:
            self._closing = True
            left = list(self._queue)
            self._queue.clear()
            self._changed.notify_all()
            while self._current is not None and (wait := deadline - time.monotonic()) > 0:
                self._changed.wait(wait)
            if self._current is not None:
                left.insert(0, self._current)
        return left


@dataclass
class Run:
    id: str
    instruction: str
    started_at: str
    tags: list[str] = field(default_factory=list)
    log_dir: Path | None = None
    pid: int | None = None
    step: int | None = None
    step_sent: tuple[float, int | None] = (0.0, None)
    frames: RunFrames = field(default_factory=RunFrames)
    # The run said hello on its events pipe.
    events: bool = False
    # Why the run refused or failed, from its error event.
    error: str | None = None
    summary: dict[str, Any] | None = None
    # The run's upload records, each as its {"upload": ...} line.
    upload: list[dict[str, Any]] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "instruction": self.instruction,
            "tags": self.tags,
            "startedAt": self.started_at,
            "logDir": str(self.log_dir) if self.log_dir else None,
            "pid": self.pid,
            "step": self.step,
        }


class Controller:
    """The rig's one episode at a time, its terminal, its dialogs and the event log."""

    def __init__(
        self,
        config_path: Path,
        *,
        allow_paid: bool,
        env_file: Path | None,
        allow_no_tmux: bool = False,
        no_verdict: bool = False,
        auto_start: bool = False,
        mirror: Callable[[bytes], None] | None = None,
    ) -> None:
        self.config_path = config_path.resolve()
        self.config = RobotConfig.load(self.config_path)
        self.allow_paid = allow_paid
        self.env_file = env_file
        self.allow_no_tmux = allow_no_tmux
        # Demo mode: runs end without the verdict and grader-notes prompts.
        self.no_verdict = no_verdict
        # Demo mode: runs start without the readiness gates (-E auto_start=true).
        self.auto_start = auto_start
        # The operator paused new runs (``hold``); a running episode goes on.
        self.held = False
        self.events = EventLog()
        # With a [store], runs end without uploading; this sends them afterwards.
        self.uploads = Uploads(self._upload_command, self.events.emit)
        self._mirror = mirror
        self._lock = threading.RLock()
        self.phase = "idle"
        self.run: Run | None = None
        self.last_run: dict[str, Any] | None = None
        self._terminal: Terminal | None = None
        self._screen = Screen(self.events.emit)
        # The running episode's open dialogs (by their id here) and status line, as its
        # events report them.
        self._dialogs: dict[str, Dialog] = {}
        self.status = ""
        serials = cli.depth_serials(self.config.rig)
        self.cameras = [f"{slot}_cam" for slot in serials]
        self.preview = Preview(Path(sys.executable), serials, self.events.emit)

    # -- commands ------------------------------------------------------------------

    def handle(self, command: Any) -> dict[str, Any]:
        """One RPC command to its response, in Pi's shape."""
        if not isinstance(command, dict) or not isinstance(command.get("type"), str):
            return {
                "type": "response",
                "command": "parse",
                "success": False,
                "error": "a command is a JSON object with a string type",
            }
        name = command["type"]
        response: dict[str, Any] = {"type": "response", "command": name}
        if isinstance(command.get("id"), str):
            response = {"id": command["id"], **response}
        handler = getattr(self, f"cmd_{name}", None)
        if handler is None:
            return {**response, "success": False, "error": f"unknown command {name!r}"}
        try:
            data = handler(command)
        except ValueError as exc:
            return {**response, "success": False, "error": str(exc)}
        except Exception as exc:  # reported to the client; the server keeps serving
            return {**response, "success": False, "error": f"{type(exc).__name__}: {exc}"}
        return {**response, "success": True, **({"data": data} if data is not None else {})}

    def cmd_get_state(self, command: dict[str, Any]) -> dict[str, Any]:
        agent, vla = self.config.agent, self.config.vla
        with self._lock:
            return {
                "phase": self.phase,
                "policy": f"agent:{agent.model}" if agent else (vla.policy if vla else None),
                "config": str(self.config_path),
                "allowPaid": self.allow_paid,
                "noVerdict": self.no_verdict,
                "autoStart": self.auto_start,
                "held": self.held,
                "uploads": self.uploads.state(),
                "cameras": self.camera_names(),
                "run": self.run.describe() if self.run else None,
                "pendingDialogs": [
                    {"id": d.id, **d.request}
                    for d in self._dialogs.values()
                    if d.answered_by is None
                ],
                "status": self.status,
                "lastRun": self.last_run,
                "preview": {"on": self.preview.on, "viewers": self.preview.viewers},
                "eventSeq": self.events.seq,
            }

    def cmd_get_commands(self, command: dict[str, Any]) -> dict[str, Any]:
        return {"commands": sorted(n[4:] for n in dir(self) if n.startswith("cmd_"))}

    def cmd_get_output(self, command: dict[str, Any]) -> dict[str, Any]:
        lines = command.get("lines", 200)
        if not isinstance(lines, int) or not 0 < lines <= MAX_OUTPUT_LINES:
            raise ValueError(f"lines must be 1 to {MAX_OUTPUT_LINES}")
        return {"lines": list(self._screen.lines)[-lines:]}

    def cmd_prompt(self, command: dict[str, Any]) -> dict[str, Any]:
        """Start ``abstract robot run`` with ``message`` as its instruction and ``tags`` as
        its episode labels (``--tag``)."""
        if self.phase == "running" and command.get("streamingBehavior") == "steer":
            return self.cmd_steer(command)
        message = one_line(command.get("message"), "message")
        tags = command.get("tags") or []
        if not isinstance(tags, list) or not all(isinstance(value, str) for value in tags):
            raise ValueError("tags must be a list of strings")
        # Checked here as run checks them, so a bad tag is refused before the rig is touched.
        tags = sorted({episode.tag(value) for value in tags})
        if self.config.agent is not None and not self.allow_paid:
            raise ValueError("this config calls a paid model; restart serve with --allow-paid")
        # No -E/-P from the network: either could switch off a gate or a guardrail.
        argv = [*abstract_command(), "robot", "run", "--config", str(self.config_path)]
        if self.env_file is not None:
            argv += ["--env-file", str(self.env_file)]
        if self.config.agent is not None:
            argv.append("--allow-paid")
        if self.allow_no_tmux:
            argv.append("--allow-no-tmux")
        if self.no_verdict:
            argv.append("--no-verdict")
        if self.auto_start:
            # The operator's choice when starting serve, never a client's: no readiness gates.
            argv += ["-E", "auto_start=true"]
        if self.config.store is not None:
            # The run ends at its summary; the upload goes on here while the next one runs.
            argv.append("--no-upload")
        for value in tags:
            argv += ["--tag", value]
        argv += ["--", message]
        with self._lock:
            if self.held:
                raise ValueError("held: the operator paused new runs (hold off to start one)")
            if self.phase != "idle":
                raise ValueError(f"the rig is {self.phase}; steer, stop or abort the run first")
            self.phase = "running"
            self.run = Run(
                id=f"run-{secrets.token_hex(4)}", instruction=message, started_at=now(), tags=tags
            )
            run = self.run
            self._screen = Screen(self.events.emit)
            self._dialogs, self.status = {}, ""
        threading.Thread(target=self._episode, args=(argv, run), daemon=True).start()
        return {"disposition": "started", "run": run.id}

    def cmd_hold(self, command: dict[str, Any]) -> dict[str, Any]:
        """Pause (``{"on": true}``) or resume (``{"on": false}``) new runs. While held,
        ``prompt`` is refused, so the scene can be reset between runs without a prompt
        homing the arms; a running episode goes on."""
        on = command.get("on")
        if not isinstance(on, bool):
            raise ValueError("on must be true or false")
        with self._lock:
            self.held = on
        self.events.emit({"type": "hold", "on": on})
        return {"held": on}

    def cmd_steer(self, command: dict[str, Any]) -> dict[str, Any]:
        """Type a message to the policy, as the operator console does."""
        message = one_line(command.get("message"), "message")
        if message.startswith("/"):
            raise ValueError("console commands go through stop, not steer")
        self._type(message.encode() + b"\n")
        return {"disposition": "sent"}

    def cmd_stop(self, command: dict[str, Any]) -> None:
        """End the episode from the console (``/stop``, or ``/y`` ``/n`` ``/p`` with a
        verdict); the episode parks the arms as at any episode end."""
        verdict = command.get("verdict")
        if verdict not in (None, "y", "n", "partial"):
            raise ValueError("verdict must be y, n or partial")
        note = command.get("note")
        note = one_line(note, "note", limit=500) if note else ""
        token = {"y": "/y", "n": "/n", "partial": "/p"}.get(verdict or "", "/stop")
        self._type(f"{token} {note}".rstrip().encode() + b"\n")

    def cmd_abort(self, command: dict[str, Any]) -> None:
        """Ctrl-C: the episode stops at once and parks the arms."""
        self._type(CTRL_C)

    def cmd_extension_ui_response(self, command: dict[str, Any]) -> None:
        """Answer a dialog by typing into the run's terminal; the run's own
        dialog_resolved event closes it."""
        dialog_id = command.get("id")
        with self._lock:
            dialog = self._dialogs.get(dialog_id) if isinstance(dialog_id, str) else None
            if dialog is None or dialog.answered_by is not None:
                raise ValueError(f"no pending dialog {dialog_id!r}")
            if command.get("cancelled"):
                answer = dialog.cancel
            elif dialog.kind == "gate":
                if command.get("confirmed") is not True:
                    raise ValueError("a gate takes confirmed: true or cancelled: true")
                answer = b"\n"
            elif dialog.kind == "verdict":
                if command.get("value") not in VERDICTS:
                    raise ValueError(f"value must be one of {', '.join(VERDICTS)}")
                answer = command["value"].encode() + b"\n"
            else:
                value = command.get("value") or ""
                answer = (one_line(value, "value", limit=500) if value else "").encode() + b"\n"
            self._type(answer)
            dialog.answered_by = "rpc"

    def cmd_check(self, command: dict[str, Any]) -> dict[str, Any]:
        """``abstract robot check`` (motion-free) with its results as events."""
        with self._lock:
            if self.phase != "idle":
                raise ValueError(f"the rig is {self.phase}")
            self.phase = "checking"
        threading.Thread(target=self._check, daemon=True).start()
        return {"disposition": "started"}

    # -- episodes --------------------------------------------------------------------

    def _type(self, data: bytes) -> None:
        with self._lock:
            if self._terminal is None:
                raise ValueError("no run is active")
            self._terminal.write(data)

    def forward_keys(self, data: bytes) -> bool:
        """Keys typed in the serving terminal; False when no episode takes them."""
        with self._lock:
            if self._terminal is None:
                return False
            self._terminal.write(data)
            return True

    def _output(self, data: bytes) -> None:
        if self._mirror is not None:
            self._mirror(data)
        self._screen.feed(data)

    def _run_event(self, run: Run, event: dict[str, Any]) -> None:
        """One event of the run (``robot/events.py``) into this server's state and its
        clients' events."""
        kind = event.get("type")
        if kind == "hello":
            run.events = True
        elif kind == "log_dir" and isinstance(event.get("path"), str):
            run.log_dir = Path(event["path"])
            self.events.emit({"type": "run_log_dir", "id": run.id, "logDir": event["path"]})
        elif kind == "dialog":
            self._open_dialog(run, event)
        elif kind == "dialog_resolved":
            with self._lock:
                dialog = self._dialogs.pop(f"{run.id}:{event.get('id')}", None)
            if dialog is not None:
                by = dialog.answered_by or "terminal"
                answer = event.get("answer")
                self.events.emit(
                    {"type": "extension_ui_resolved", "id": dialog.id, "by": by, "answer": answer}
                )
        elif kind == "status":
            with self._lock:
                self.status = clip(event.get("text") or "")
            self.events.emit({"type": "status", "text": self.status})
        elif kind == "progress" and isinstance(event.get("step"), int):
            run.step = event["step"]
            self._send_progress(run)
        elif kind == "frame" and isinstance(event.get("path"), str):
            run.frames.add(str(event.get("camera")), event["path"])
        elif kind == "transcript":
            self.events.emit(
                {"type": "transcript", "id": run.id, "message": clip(event.get("message"))}
            )
        elif kind == "trial_end":
            self._send_progress(run, force=True)
            fields = ("status", "termination_reason", "verdict")
            self.events.emit(
                {"type": "trial_end", "id": run.id, **{k: event.get(k) for k in fields}}
            )
        elif kind == "episode_end":
            with self._lock:
                if self.phase == "running":
                    self.phase = "finishing"  # parked; summary and upload follow
            self.events.emit(
                {"type": "episode_end", "id": run.id, "exitCode": event.get("exit_code")}
            )
        elif kind == "summary" and isinstance(event.get("summary"), dict):
            run.summary = event["summary"]
        elif kind == "upload" and isinstance(event.get("upload"), dict):
            run.upload.append({"upload": event["upload"]})
        elif kind == "error":
            run.error = clip(str(event.get("message")))
        elif kind == "dropped":
            self.events.emit({"type": "events_dropped", "id": run.id, "count": event.get("count")})

    def _open_dialog(self, run: Run, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        if kind not in DIALOGS or not isinstance(event.get("id"), str):
            return
        request, cancel = DIALOGS[kind]
        request = {**request, "title": clip(str(event.get("title") or ""))}
        if kind == "verdict":
            request["options"] = [str(option) for option in event.get("options") or VERDICTS]
        # The run numbers its dialogs from 1: the run's id keeps a stale answer from
        # reaching the next run's gate.
        dialog = Dialog(f"{run.id}:{event['id']}", kind, request, cancel)
        with self._lock:
            self._dialogs[dialog.id] = dialog
        self.events.emit({"type": "extension_ui_request", "id": dialog.id, **request})

    def _send_progress(self, run: Run, *, force: bool = False) -> None:
        """A progress event at most every ``PROGRESS_INTERVAL_S``, and the last step when
        the trial ends."""
        sent_at, sent_step = run.step_sent
        if run.step is None or run.step == sent_step:
            return
        if force or time.monotonic() - sent_at >= PROGRESS_INTERVAL_S:
            run.step_sent = (time.monotonic(), run.step)
            self.events.emit({"type": "progress", "id": run.id, "step": run.step})

    def _episode(self, argv: list[str], run: Run) -> None:
        end: dict[str, Any] = {"type": "run_end", "id": run.id}
        reader: EventReader | None = None
        try:
            with self.preview.released():
                read_fd, write_fd = os.pipe()
                try:
                    reader = EventReader(read_fd, lambda event: self._run_event(run, event))
                    split = argv.index("--")
                    argv = [*argv[:split], "--events-fd", str(write_fd), *argv[split:]]
                    with self._lock:
                        terminal = self._terminal = Terminal(
                            argv, dict(os.environ), self._output, pass_fds=(write_fd,)
                        )
                        run.pid = terminal.process.pid
                finally:
                    os.close(write_fd)  # the run holds the only write end: EOF when it exits
                self.events.emit({"type": "run_start", **run.describe()})
                end["exitCode"] = terminal.wait()
                reader.wait(timeout=5)
        except Exception as exc:
            end["error"] = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._terminal = None
            left, self._dialogs = list(self._dialogs.values()), {}
            self.status = ""
        for dialog in left:
            self.events.emit({"type": "extension_ui_resolved", "id": dialog.id, "by": "exit"})
        end.update(
            summary=run.summary,
            upload=run.upload,
            logDir=run.describe()["logDir"],
            events=run.events,
        )
        if run.error is not None:
            end["runError"] = run.error
        if run.summary is None:  # refused or failed before the episode: say why
            end["tail"] = list(self._screen.lines)[-12:]
        # Queued before the phase turns idle, so a shutdown waiting for idle sees it.
        if self.config.store is not None and run.log_dir is not None and run.summary is not None:
            self.uploads.add(run.log_dir.name)
        with self._lock:
            self.last_run = {**run.describe(), **{k: v for k, v in end.items() if k != "type"}}
            # The run has exited: the arms are parked and the devices released, so the next
            # run can start now, while this one uploads.
            self.run, self.phase = None, "idle"
        self.events.emit(end)

    def _upload_command(self, run: str) -> list[str]:
        return [*abstract_command(), "robot", "upload", "--config", str(self.config_path), run]

    def _check(self) -> None:
        self.events.emit({"type": "check_start"})
        argv = [*abstract_command(), "robot", "check", "--config", str(self.config_path)]
        if self.env_file is not None:
            argv += ["--env-file", str(self.env_file)]
        ready, code = False, None
        try:
            with self.preview.released():
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    try:
                        result = json.loads(line)
                    except ValueError:
                        self.events.emit({"type": "output", "text": clip(line.rstrip())})
                        continue
                    if isinstance(result, dict) and "ready" in result:
                        ready = result["ready"] is True
                    else:
                        self.events.emit({"type": "check_result", "result": clip(result)})
                code = process.wait()
        finally:
            with self._lock:
                self.phase = "idle"
            self.events.emit({"type": "check_end", "ready": ready, "exitCode": code})

    # -- feed and shutdown -------------------------------------------------------------

    def camera_names(self) -> list[str]:
        """The config's RealSense cameras, plus any other the running episode stores."""
        run = self.run
        stored = run.frames.cameras() if run is not None else []
        return [*self.cameras, *(name for name in stored if name not in self.cameras)]

    def frame(self, camera: str) -> Frame | None:
        run = self.run
        if run is not None:
            return run.frames.frame(camera)
        return self.preview.frame(camera)

    def shutdown(self, timeout: float = 120.0, upload_timeout: float = 900.0) -> list[str]:
        """Request episode cancellation and await process completion, then pending uploads.

        Expiry raises: it must never be mistaken for verified physical shutdown.
        Returns runs left on disk for ``abstract robot upload``.
        """
        with self._lock:
            if self.phase == "running" and self._terminal is not None:
                self._terminal.write(CTRL_C)
        deadline = time.monotonic() + timeout
        while self.phase not in ("idle", "checking") and time.monotonic() < deadline:
            time.sleep(0.2)
        timed_out = self.phase not in ("idle", "checking")
        self.preview.close()
        remaining = self.uploads.close(upload_timeout)
        if timed_out:
            raise TimeoutError(
                "episode shutdown deadline exceeded; physical state is unverified; "
                "use the hardware stop"
            )
        return remaining


PAGE_PATH = Path(__file__).with_name("rpc_page.html")


class Handler(BaseHTTPRequestHandler):
    controller: Controller
    token: str
    server_version = "abstract-robot-rpc"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Quiet: the serving terminal mirrors the episode."""

    def _query(self) -> dict[str, list[str]]:
        return parse_qs(urlparse(self.path).query)

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        given = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
        given = given or (self._query().get("token") or [""])[0]
        return hmac.compare_digest(given.encode(), self.token.encode())

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: Any) -> None:
        self._send(status, json.dumps(value).encode(), "application/json")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            return self._send(200, PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        if path == "/health":
            return self._json(200, {"ok": True, "phase": self.controller.phase})
        if not self._authorized():
            return self._json(401, {"error": "missing or wrong token"})
        if path == "/events":
            return self._events()
        if path.startswith("/feed/"):
            camera, _, kind = path.removeprefix("/feed/").partition(".")
            if camera not in self.controller.camera_names():
                return self._json(404, {"error": f"no camera {camera!r}"})
            return self._feed(camera, still=kind in ("png", "jpg"))
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/rpc":
            return self._json(404, {"error": "not found"})
        if not self._authorized():
            return self._json(401, {"error": "missing or wrong token"})
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY_BYTES:
            return self._json(413, {"error": f"body must be 1 to {MAX_BODY_BYTES} bytes"})
        try:
            command = json.loads(self.rfile.read(length))
        except ValueError as exc:
            return self._json(
                400,
                {
                    "type": "response",
                    "command": "parse",
                    "success": False,
                    "error": f"Failed to parse command: {exc}",
                },
            )
        self._json(200, self.controller.handle(command))

    def _events(self) -> None:
        """Server-sent events after ``?since=`` or Last-Event-ID (default: from now)."""
        events = self.controller.events
        # A reconnecting EventSource sends the last id it saw; that, not the ?since= it
        # first asked for, is where it resumes.
        start = self.headers.get("Last-Event-ID") or (self._query().get("since") or [""])[0]
        seq = int(start) if start.isdigit() else events.seq
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                batch = events.after(seq, timeout=15)
                if not batch:
                    self.wfile.write(b": keepalive\n\n")
                for event in batch:
                    seq = event["seq"]
                    self.wfile.write(f"id: {seq}\ndata: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _feed(self, camera: str, *, still: bool) -> None:
        """One frame, or a multipart stream of them (an <img> plays it as video). A stream
        takes ``fps`` (1 to 15, default 5) and ``width`` (a JPEG wider than it is shrunk to
        it, for viewers on slower links; default: as captured)."""
        query = self._query()
        fps = query_int(query, "fps", 5, 1, 15)
        width = query_int(query, "width", 0, 160, 1920) if "width" in query else 0
        with self.controller.preview.watching():
            if still:
                deadline = time.monotonic() + 8
                while (frame := self.controller.frame(camera)) is None:
                    if time.monotonic() > deadline:
                        return self._json(503, {"error": f"no frame from {camera} yet"})
                    time.sleep(0.2)
                return self._send(200, frame.data, frame.content_type)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            shown = None
            try:
                while True:
                    frame = self.controller.frame(camera)
                    if frame is not None and frame.key != shown:
                        shown = frame.key
                        data = frame.data
                        if width and frame.content_type == "image/jpeg":
                            data = shrunk(data, width)
                        self.wfile.write(
                            f"--frame\r\nContent-Type: {frame.content_type}\r\n"
                            f"Content-Length: {len(data)}\r\n\r\n".encode()
                            + data
                            + b"\r\n"
                        )
                        self.wfile.flush()
                    time.sleep(1 / fps)
            except (BrokenPipeError, ConnectionResetError):
                return


@contextmanager
def keyboard(controller: Controller, stop: threading.Event) -> Iterator[None]:
    """Keys typed in the serving terminal go to the episode unbuffered, Ctrl-C included,
    so Esc, Enter and Ctrl-C work there as in ``robot run``; with no episode, Ctrl-C
    stops the server."""
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    attrs = termios.tcgetattr(fd)
    attrs[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
    attrs[6][termios.VMIN], attrs[6][termios.VTIME] = 1, 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)

    def pump() -> None:
        while not stop.is_set():
            try:
                if not select.select([fd], [], [], 0.2)[0]:
                    continue
                data = os.read(fd, 1024)
            except OSError:  # the terminal hung up; SIGHUP stops the server
                return
            if not data:
                return
            if not controller.forward_keys(data) and CTRL_C in data:
                stop.set()

    threading.Thread(target=pump, daemon=True).start()
    try:
        yield
    finally:
        with suppress(termios.error, OSError):
            termios.tcsetattr(fd, termios.TCSANOW, saved)


class Mirror:
    """Copies the episode's output to the serving terminal until that terminal fails.

    A write error must never reach the pty reader: an undrained pty blocks the run
    mid-episode."""

    def __init__(self) -> None:
        self.ok = True

    def __call__(self, data: bytes) -> None:
        if not self.ok:
            return
        try:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        except (OSError, ValueError):
            self.ok = False


def serve(args: argparse.Namespace) -> int:
    if not os.environ.get("TMUX") and not args.allow_no_tmux:
        raise ValueError(
            "serve inside tmux: a dropped SSH session kills the server and its episode "
            "before the arms park (or pass --allow-no-tmux)"
        )
    token = load_token(args.token_file)
    controller = Controller(
        args.config,
        allow_paid=args.allow_paid,
        env_file=args.env_file,
        allow_no_tmux=args.allow_no_tmux,
        no_verdict=args.no_verdict,
        auto_start=args.auto_start,
        mirror=Mirror() if sys.stdout.isatty() else None,
    )
    handler = type("BoundHandler", (Handler,), {"controller": controller, "token": token})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    stop = threading.Event()

    def on_signal(signum: int, frame: object) -> None:
        # The first signal parks a running episode and shuts down. A second one with
        # nothing running exits at once, in case the shutdown itself is stuck.
        if stop.is_set() and controller.phase in ("idle", "checking"):
            os._exit(128 + signum)
        stop.set()

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, on_signal)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = f"[{args.host}]" if ":" in args.host else args.host
    print(
        f"abstract robot serve: http://{host}:{args.port}/ (token in "
        f"{args.token_file}); {controller.cmd_get_state({})['policy']}, cameras "
        f"{', '.join(controller.cameras) or 'none'}"
        f"{'; verdicts off (--no-verdict)' if controller.no_verdict else ''}"
        f"{'; gates off (--auto-start)' if controller.auto_start else ''}"
        ". Ctrl-C here "
        "stops the server when no run is active; during a run it reaches the episode, which "
        "parks the arms.",
        file=sys.stderr,
        flush=True,
    )
    with keyboard(controller, stop):
        try:
            while not stop.wait(0.5):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            try:
                left = controller.shutdown()
                if left:
                    print(
                        "abstract robot serve: not uploaded, still in "
                        f"{controller.config.rig.log_root}: {', '.join(left)}; send them with "
                        f"abstract robot upload --config {controller.config_path}",
                        file=sys.stderr,
                        flush=True,
                    )
            finally:
                server.shutdown()
                server.server_close()
    return 0
