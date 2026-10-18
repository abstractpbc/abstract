"""Logging sinks for robot runs.

``LogSink`` is the protocol; ``JsonLogSink`` is the canonical, always-on sink
that writes the immutable ``EvalLog`` to disk.
``LiveLogSink`` maintains the transient running snapshot. The optional
``RerunSink`` is lazily imported and no-ops if ``rerun-sdk`` is absent.
"""

from __future__ import annotations

from abstract_harness.robot.core.logging.json_log import JsonLogSink
from abstract_harness.robot.core.logging.live_log import LiveLogSink
from abstract_harness.robot.core.logging.rerun_sink import RerunSink
from abstract_harness.robot.core.logging.sink import LogSink, NullSink

__all__ = ["JsonLogSink", "LiveLogSink", "LogSink", "NullSink", "RerunSink"]
