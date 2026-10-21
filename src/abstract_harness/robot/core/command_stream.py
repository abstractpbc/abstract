"""Continuous commands: live inputs streamed into a running policy.

A ``CommandStream`` holds named *axes*, continuous values such as ``vx``, ``yaw_rate`` or
``speed``, and queues discrete *events*, button presses such as ``go`` or ``stop``. Any
thread or source may write it while a trial runs: a gamepad, a UDP bridge from another
program, ``abstract robot serve``'s ``command`` RPC, or Python code. The rollout reads it
once per step and injects it into the policy-facing observation::

    observation.extra["commands"]        # {"vx": 0.4, "yaw_rate": 0.0}: latest values
    observation.extra["command_events"]  # [{"t": 12, "name": "go", "source": "pad"}]

``command_events`` holds the events since the previous ``act()`` call, as ``approvals`` and
``operator_messages`` do; ``commands`` is always the newest value of every live axis.

An axis declared with ``AxisSpec`` clamps to its range and falls back to its default when
no update arrives within ``timeout_s``: a deadman, so a dropped joystick or network link
stops a velocity command instead of repeating it forever. A stream with declared axes
refuses undeclared names; a stream without any accepts any name and drops a value that
goes ``default_timeout_s`` without an update. Such a free-form stream joins observations
only once it has carried a command (``active``), so a run that offers a command channel
nobody uses (every ``abstract robot serve`` run) observes and logs exactly as without one.
Each trial starts from the defaults with no queued events.

The event ``end_episode`` is reserved: it ends the trial as the console's Esc does, also
while inference is pending on physical runs. Commands are inputs to the policy and never
reach the robot except through the policy's actions, which still pass the approver chain.
A button is not an emergency stop.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import struct
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from abstract_harness.robot.core.errors import ConfigError

#: The reserved event that ends the trial (``OPERATOR_END``, no verdict).
END_EPISODE = "end_episode"
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
DEFAULT_TIMEOUT_S = 0.5
MAX_QUEUED_EVENTS = 1000


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not NAME.match(value):
        raise ValueError(f"{what} must match {NAME.pattern}, got {value!r}")
    return value


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{what} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return number


@dataclass(frozen=True)
class AxisSpec:
    """One continuous command: its range, resting value and deadman timeout.

    ``timeout_s=None`` keeps the last value until the trial ends (a setting such as a
    gait or a speed level, not a velocity).
    """

    name: str
    low: float = -1.0
    high: float = 1.0
    default: float = 0.0
    timeout_s: float | None = DEFAULT_TIMEOUT_S
    description: str = ""

    def __post_init__(self) -> None:
        try:
            _name(self.name, "axis name")
            for label in ("low", "high", "default"):
                _number(getattr(self, label), f"axis {self.name} {label}")
        except ValueError as exc:
            raise ConfigError(str(exc)) from None
        if self.low > self.high:
            raise ConfigError(f"axis {self.name}: low must be <= high")
        if not self.low <= self.default <= self.high:
            raise ConfigError(f"axis {self.name}: default must lie within [low, high]")
        if self.timeout_s is not None and (
            isinstance(self.timeout_s, bool)
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ConfigError(f"axis {self.name}: timeout_s must be positive or None")


@dataclass(frozen=True)
class CommandEvent:
    """One discrete command, as queued: its name, who sent it, and when (monotonic)."""

    name: str
    source: str
    time: float


@dataclass(frozen=True)
class CommandSnapshot:
    """What one step reads: every live axis value and the events drained since the
    previous snapshot."""

    values: Mapping[str, float] = field(default_factory=dict)
    events: tuple[CommandEvent, ...] = ()


class CommandSource(Protocol):
    """A producer of commands: ``start`` begins writing into the stream (usually on a
    daemon thread); ``close`` stops it and releases its device or socket."""

    def start(self, stream: CommandStream) -> None:
        """Begin delivering commands into ``stream``."""
        ...

    def close(self) -> None:
        """Stop delivering commands; idempotent."""
        ...


class CommandStream:
    """Thread-safe live command state shared by command sources and the rollout."""

    def __init__(
        self,
        axes: Sequence[AxisSpec] = (),
        *,
        events: Iterable[str] | None = None,
        default_timeout_s: float = DEFAULT_TIMEOUT_S,
        sources: Sequence[CommandSource] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        names = [axis.name for axis in axes]
        if len(set(names)) != len(names):
            raise ConfigError("command axes must have distinct names")
        self.axes: dict[str, AxisSpec] = {axis.name: axis for axis in axes}
        self.event_names = (
            frozenset(_name(e, "event name") for e in events) if events is not None else None
        )
        self.default_timeout_s = default_timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._values: dict[str, tuple[float, float]] = {}
        self._events: list[CommandEvent] = []
        self._end = False
        self._sources: list[CommandSource] = list(sources)
        self._started = False
        self._carried = False
        self.dropped_events = 0
        self.rejected = 0

    # -- writers -----------------------------------------------------------------

    def set(self, name: str, value: float, *, source: str = "api") -> None:
        """Set one axis (clamped to its declared range); raises ``ValueError`` when the
        name is unknown or the value is not a finite number."""
        self.update({name: value}, source=source)

    def update(self, values: Mapping[str, float], *, source: str = "api") -> None:
        """Set several axes at once; all are validated before any is applied."""
        now = self._clock()
        checked: dict[str, float] = {}
        for name, value in values.items():
            _name(name, "command name")
            if self.axes and name not in self.axes:
                raise ValueError(f"unknown command {name!r}; declared: {', '.join(self.axes)}")
            number = _number(value, f"command {name}")
            axis = self.axes.get(name)
            if axis is not None:
                number = min(max(number, axis.low), axis.high)
            checked[name] = number
        with self._lock:
            for name, number in checked.items():
                self._values[name] = (number, now)
            self._carried = self._carried or bool(checked)

    def press(self, name: str, *, source: str = "api") -> None:
        """Queue one discrete event (``end_episode`` ends the trial)."""
        _name(name, "event name")
        _name(source, "source")
        if self.event_names is not None and name not in self.event_names and name != END_EPISODE:
            declared = ", ".join(sorted(self.event_names))
            raise ValueError(f"unknown event {name!r}; declared: {declared}")
        with self._lock:
            self._carried = True
            if name == END_EPISODE:
                self._end = True
            if len(self._events) >= MAX_QUEUED_EVENTS:
                self.dropped_events += 1
                return
            self._events.append(CommandEvent(name, source, self._clock()))

    def apply(self, message: Mapping[str, Any], *, source: str) -> None:
        """One message from a wire source: ``{"values": {...}, "events": [...]}``, or a
        flat ``{"vx": 0.3}`` of values. Invalid messages are counted and skipped."""
        try:
            if "values" in message or "events" in message:
                values = message.get("values") or {}
                events = message.get("events") or []
                if not isinstance(values, Mapping) or not isinstance(events, list):
                    raise ValueError("values must be an object and events a list")
            else:
                values, events = message, []
            self.update(values, source=source)
            for event in events:
                self.press(event, source=source)
        except ValueError:
            self._reject()

    def _reject(self) -> None:
        with self._lock:
            self.rejected += 1

    # -- readers -----------------------------------------------------------------

    def snapshot(self) -> CommandSnapshot:
        """The live values (timeouts applied) and the events queued since the last call."""
        now = self._clock()
        with self._lock:
            events, self._events = tuple(self._events), []
            values = self._live_values(now)
        return CommandSnapshot(values=values, events=events)

    def peek(self) -> dict[str, float]:
        """The live values without draining events."""
        with self._lock:
            return self._live_values(self._clock())

    @property
    def active(self) -> bool:
        """Whether observations carry this stream: it declares axes or events, or it has
        carried a command."""
        with self._lock:
            return bool(self.axes) or self.event_names is not None or self._carried

    def end_requested(self) -> bool:
        """Whether ``end_episode`` arrived during this trial (not consumed by reading)."""
        with self._lock:
            return self._end

    def _live_values(self, now: float) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, axis in self.axes.items():
            value, at = self._values.get(name, (axis.default, -math.inf))
            if axis.timeout_s is not None and now - at > axis.timeout_s:
                value = axis.default
            values[name] = value
        if not self.axes:
            for name, (value, at) in list(self._values.items()):
                if now - at > self.default_timeout_s:
                    del self._values[name]
                else:
                    values[name] = value
        return values

    # -- lifecycle ---------------------------------------------------------------

    def add_source(self, source: CommandSource) -> None:
        """Attach a source; it starts at once when the stream already has."""
        self._sources.append(source)
        if self._started:
            source.start(self)

    def start(self) -> None:
        """Start every attached source (idempotent)."""
        if self._started:
            return
        self._started = True
        for source in self._sources:
            source.start(self)

    def begin_trial(self) -> None:
        """Reset to the defaults and drop queued events before a trial starts."""
        with self._lock:
            self._values.clear()
            self._events.clear()
            self._end = False

    def close(self) -> None:
        """Stop and release every source."""
        for source in self._sources:
            try:
                source.close()
            except Exception as exc:  # one bad source must not keep the others open
                warnings.warn(
                    f"command source {type(source).__name__} failed to close: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self._started = False

    def describe(self) -> str:
        """A short description for embodiment notes and run headers."""
        parts = [
            f"{axis.name} in [{axis.low:g}, {axis.high:g}] (rest {axis.default:g})"
            + (f": {axis.description}" if axis.description else "")
            for axis in self.axes.values()
        ]
        if self.event_names:
            parts.append("events: " + ", ".join(sorted(self.event_names)))
        return "; ".join(parts) if parts else "free-form commands"


class _ThreadSource:
    """A source that reads on one daemon thread until closed."""

    name = "source"

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, stream: CommandStream) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(stream,), name=f"commands-{self.name}", daemon=True
        )
        self._thread.start()

    def _run(self, stream: CommandStream) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def close(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2)


class UdpCommandSource(_ThreadSource):
    """JSON datagrams on a UDP port, one message each (``CommandStream.apply``).

    Binds loopback by default. UDP carries no authentication, so binding another
    address needs ``allow_remote=True`` and a trusted network.
    """

    name = "udp"

    def __init__(self, host: str = "127.0.0.1", port: int = 9870, *, allow_remote: bool = False):
        super().__init__()
        if host not in ("127.0.0.1", "localhost", "::1") and not allow_remote:
            raise ConfigError(
                f"UDP commands on {host!r} would accept unauthenticated motion inputs from "
                "the network; bind 127.0.0.1, or pass allow_remote=True on a trusted network"
            )
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        self._socket = socket.socket(family, socket.SOCK_DGRAM)
        self._socket.bind((host, port))
        self._socket.settimeout(0.1)
        self.address = self._socket.getsockname()

    def _run(self, stream: CommandStream) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._socket.recvfrom(65536)
            except TimeoutError:
                continue
            except OSError:
                return
            try:
                message = json.loads(data)
            except ValueError:
                stream._reject()
                continue
            if isinstance(message, dict):
                source = message.get("source", "udp")
                stream.apply(message, source=source if isinstance(source, str) else "udp")

    def close(self) -> None:
        super().close()
        self._socket.close()


class FdCommandSource(_ThreadSource):
    """JSON lines from an inherited pipe: how ``abstract robot serve`` forwards its
    ``command`` RPC into the run it supervises (``robot run --commands-fd``)."""

    name = "fd"

    def __init__(self, fd: int, *, source: str = "rpc") -> None:
        super().__init__()
        self._fd = fd
        self._source = source

    def _run(self, stream: CommandStream) -> None:
        try:
            with os.fdopen(self._fd, "rb", closefd=True) as pipe:
                for line in pipe:
                    if self._stop.is_set():
                        return
                    try:
                        message = json.loads(line)
                    except ValueError:
                        stream._reject()
                        continue
                    if isinstance(message, dict):
                        stream.apply(message, source=self._source)
        except OSError:
            return

    def close(self) -> None:
        # The reader blocks in readline until the writer closes its end; the thread is a
        # daemon, so it never holds the process open.
        self._stop.set()


# Linux joystick API (linux/joystick.h): struct js_event { __u32 time; __s16 value;
# __u8 type; __u8 number; }.
_JS_EVENT = struct.Struct("IhBB")
_JS_BUTTON, _JS_AXIS, _JS_INIT = 0x01, 0x02, 0x80
_JS_MAX = 32767.0

#: Button and axis indices of an Xbox-style pad under the Linux xpad/xone drivers.
XBOX_LAYOUT: dict[str, dict[str, int]] = {
    "buttons": {
        "a": 0,
        "b": 1,
        "x": 2,
        "y": 3,
        "lb": 4,
        "rb": 5,
        "back": 6,
        "start": 7,
        "guide": 8,
        "ls": 9,
        "rs": 10,
    },  # fmt: skip
    "axes": {"lx": 0, "ly": 1, "lt": 2, "rx": 3, "ry": 4, "rt": 5, "dx": 6, "dy": 7},
}
LAYOUTS = {"xbox": XBOX_LAYOUT}


@dataclass(frozen=True)
class AxisBinding:
    """A stick axis driving a command: ``value = scale * stick`` after the deadzone.

    The stick reads -1..1; ``scale`` maps it to command units (a negative scale inverts
    it: Linux reports stick-up as negative).
    """

    command: str
    axis: int
    scale: float = 1.0
    deadzone: float = 0.05

    def value(self, raw: float) -> float:
        if abs(raw) <= self.deadzone:
            return 0.0
        # Rescale past the deadzone so the command is continuous at its edge.
        span = (abs(raw) - self.deadzone) / (1.0 - self.deadzone)
        return math.copysign(span, raw) * self.scale


@dataclass(frozen=True)
class ButtonBinding:
    """A button and what pressing it does: ``event`` queues that event; ``set`` (a
    ``{"command": value}`` mapping) sets axes, e.g. a speed level."""

    button: int
    event: str | None = None
    set: Mapping[str, float] = field(default_factory=dict)


class JoystickSource(_ThreadSource):
    """A gamepad through the Linux joystick API (``/dev/input/js*``), no dependencies.

    Sticks drive axes (``AxisBinding``) and buttons queue events or set axes
    (``ButtonBinding``). The pad reports only changes, so held stick values are
    republished at ``refresh_hz`` to stay inside their axes' deadman timeouts; a pad
    that disconnects stops republishing and its axes fall back to rest. A wireless pad
    can also go silent with its device still open (out of range, flat battery): after
    ``max_silence_s`` without any event, the sticks publish rest until the pad speaks
    again. With a ``deadman`` button, sticks count only while it is held.
    """

    name = "joystick"

    def __init__(
        self,
        device: str = "/dev/input/js0",
        *,
        axes: Sequence[AxisBinding] = (),
        buttons: Sequence[ButtonBinding] = (),
        deadman: int | None = None,
        refresh_hz: float = 50.0,
        max_silence_s: float = 1.0,
        source: str = "joystick",
        opener: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        if isinstance(max_silence_s, bool) or not max_silence_s > 0:
            raise ConfigError("max_silence_s must be positive")
        self.max_silence_s = float(max_silence_s)
        self._clock = clock
        self._last_event: float | None = None
        self.device = device
        self.axes = tuple(axes)
        self.buttons = {binding.button: binding for binding in buttons}
        self.deadman = deadman
        self.refresh_s = 1.0 / refresh_hz
        self.source = source
        self._opener = opener or (lambda path: open(path, "rb", buffering=0))  # noqa: SIM115
        self._raw_axes: dict[int, float] = {}
        self._held: set[int] = set()
        self.connected = False

    def _publish(self, stream: CommandStream) -> None:
        silent = self._last_event is None or self._clock() - self._last_event > self.max_silence_s
        enabled = not silent and (self.deadman is None or self.deadman in self._held)
        values = {
            binding.command: binding.value(self._raw_axes.get(binding.axis, 0.0))
            if enabled
            else 0.0
            for binding in self.axes
        }
        if values:
            try:
                stream.update(values, source=self.source)
            except ValueError as exc:
                warnings.warn(f"joystick binding refused: {exc}", RuntimeWarning, stacklevel=2)

    def handle(self, stream: CommandStream, value: int, kind: int, number: int) -> None:
        """Apply one decoded ``js_event`` (public so tests and bridges can feed events)."""
        initial = bool(kind & _JS_INIT)
        kind &= ~_JS_INIT
        self._last_event = self._clock()
        if kind == _JS_AXIS:
            self._raw_axes[number] = max(-1.0, min(1.0, value / _JS_MAX))
        elif kind == _JS_BUTTON:
            if value:
                self._held.add(number)
            else:
                self._held.discard(number)
            binding = self.buttons.get(number)
            if binding is not None and value and not initial:
                try:
                    if binding.set:
                        stream.update(binding.set, source=self.source)
                    if binding.event is not None:
                        stream.press(binding.event, source=self.source)
                except ValueError as exc:
                    warnings.warn(f"joystick binding refused: {exc}", RuntimeWarning, stacklevel=2)
        self._publish(stream)

    def _run(self, stream: CommandStream) -> None:
        import select

        try:
            device = self._opener(self.device)
        except OSError as exc:
            warnings.warn(
                f"joystick {self.device} unavailable ({exc}); its commands stay at rest",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        self.connected = True
        try:
            while not self._stop.is_set():
                ready, _, _ = select.select([device], [], [], self.refresh_s)
                if not ready:
                    self._publish(stream)
                    continue
                data = device.read(_JS_EVENT.size)
                if not data or len(data) < _JS_EVENT.size:
                    return
                _, value, kind, number = _JS_EVENT.unpack(data)
                self.handle(stream, value, kind, number)
        except OSError:
            return
        finally:
            self.connected = False
            device.close()


def parse_source(spec: str) -> CommandSource:
    """A command source from its CLI form: ``udp:HOST:PORT`` or ``udp:PORT``."""
    kind, _, rest = spec.partition(":")
    if kind == "udp" and rest:
        host, _, port = rest.rpartition(":")
        try:
            number = int(port)
        except ValueError:
            raise ConfigError(f"--commands {spec!r}: expected udp:HOST:PORT") from None
        return UdpCommandSource(host or "127.0.0.1", number)
    raise ConfigError(f"--commands {spec!r}: expected udp:HOST:PORT (or udp:PORT)")
