"""Scoped CLI termination handlers; libraries never change a caller's signal policy."""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType


def _interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt(f"terminated by signal {signum}")


@contextmanager
def termination_signals() -> Iterator[None]:
    """Give SIGTERM/SIGHUP the same cleanup path as Ctrl-C, restoring prior handlers."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    signals = [signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        signals.append(signal.SIGHUP)
    previous = {sig: signal.getsignal(sig) for sig in signals}
    try:
        for sig in signals:
            signal.signal(sig, _interrupt)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
