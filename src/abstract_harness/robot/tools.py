"""``abstract robot holdcheck|health|pose``: the YAM operator tools, set up from the robot TOML.

Each tool reads the rig from ``--config`` as a run does: ``[rig.embodiment]`` holds the
yam_arms arguments (CAN channels, gripper type, cameras and their capture sizes, joint
limits, rest pose, ``pose_dir``), so a tool works on the same rig as the runs it prepares.
Each keeps its own flags; ``-E key=value`` on health and pose overrides one embodiment
argument for that invocation.

All but ``pose list|show|delete|rename`` connect to the arms. Run them at the rig, e-stop
in hand::

    abstract robot health --config rigs/yam-vla.toml          # connects, reads, drops torque
    abstract robot holdcheck --config rigs/yam-vla.toml left --zero-gravity false
    abstract robot pose --config rigs/yam-vla.toml capture ready   # hand-posed snapshot
    abstract robot pose --config rigs/yam-vla.toml goto ready      # moves the arms
    abstract robot pose --config rigs/yam-vla.toml list

holdcheck takes one arm: ``left`` or ``right`` (the TOML's ``left_channel`` or
``right_channel``), or a CAN interface name such as ``can0``.

The tools are in ``robot/yam`` and need the robot extra; without it, each command only
says so.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from abstract_harness.robot.config import RobotConfig

COMMANDS = ("holdcheck", "health", "pose")
# Top-level modules of the robot extra: without one of them the tools cannot load.
EXTRA_MODULES = frozenset(
    {"numpy", "cv2", "json_numpy", "requests", "rerun", "PIL", "mujoco", "i2rt", "websockets"}
)
MISSING_EXTRA = "needs the robot extra: uv sync --extra robot"


def _modules() -> dict[str, Any]:
    from abstract_harness.robot.yam import health, hold_check, pose_cli

    return {"holdcheck": hold_check, "health": health, "pose": pose_cli}


def configure_parser(commands: Any) -> None:
    """Add holdcheck, health and pose to ``abstract robot``'s subcommands."""
    try:
        modules = _modules()
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] not in EXTRA_MODULES:
            raise
        for name in COMMANDS:
            stub = commands.add_parser(name, help=MISSING_EXTRA)
            stub.add_argument("--config", type=Path)
            stub.add_argument("rest", nargs=argparse.REMAINDER)
            stub.set_defaults(tool=name, tool_parser=None)
        return
    for name in COMMANDS:
        module = modules[name]
        parser = commands.add_parser(name, help=module.HELP, description=module.DESCRIPTION)
        parser.add_argument(
            "--config", type=Path, required=True, help="robot TOML; the rig is its [rig.embodiment]"
        )
        module.configure(parser)
        parser.set_defaults(tool=name, tool_parser=parser)


def main(args: argparse.Namespace, **seams: Any) -> int:
    """Run the tool ``args`` selects with the yam_arms arguments a run gets. ``seams``
    (fake hardware, clocks, consoles) go to it."""
    parser: argparse.ArgumentParser | None = args.tool_parser
    if parser is None:
        raise ValueError(f"abstract robot {args.tool} {MISSING_EXTRA}")
    try:
        config = RobotConfig.load(args.config)
    except (OSError, ValueError) as exc:
        parser.error(f"{args.config}: {exc}")
    return _modules()[args.tool].run_command(parser, args, config.embodiment_args(), **seams)
