"""Suite-wide fixtures keeping tests hermetic against the developer's machine."""

import shutil
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_LOGS_DIR = _REPO_ROOT / "logs"
_REPO_LOGS_EXISTED_AT_SESSION_START = False


def pytest_sessionstart(session: pytest.Session) -> None:
    """Snapshot normal developer logs before tests can create checkout litter."""
    del session
    global _REPO_LOGS_EXISTED_AT_SESSION_START
    _REPO_LOGS_EXISTED_AT_SESSION_START = _REPO_LOGS_DIR.exists()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail and clean up only when the suite created the repo-root logs directory."""
    del exitstatus
    if not _REPO_LOGS_EXISTED_AT_SESSION_START and _REPO_LOGS_DIR.exists():
        print(f"test suite created repo-root log litter: {_REPO_LOGS_DIR}")
        session.exitstatus = 1
        shutil.rmtree(_REPO_LOGS_DIR)


def robot_args(argv: list[str]) -> Any:
    """``abstract robot <argv>`` parsed as the command line parses it."""
    from abstract_harness.cli import parser

    return parser().parse_args(["robot", *argv])


def robot(argv: list[str]) -> int:
    """``abstract robot <argv>`` in this process."""
    from abstract_harness.robot.cli import main

    return main(robot_args(argv))


def run_command(argv: list[str]) -> int:
    """The flags of the removed generic ``run`` command that ``run_task`` covers, onto
    ``commands.components`` and ``commands.run_task``: a registered ``--task`` (with
    ``-T``) or an ``--instruction``, a ``--policy`` and an ``--embodiment`` (with ``-P``
    and ``-E``; ``scripted`` on ``cubepick`` by default), the log directory,
    ``--no-prompt``, stored frames, a Rerun recording, the seed, and an instruction's
    ``--max-steps`` and ``--scorer``."""
    import argparse

    from abstract_harness.robot.core import commands

    parser = argparse.ArgumentParser(prog="run")
    parser.add_argument("--task")
    parser.add_argument("-T", dest="task_args", action="append")
    parser.add_argument("--instruction")
    parser.add_argument("--policy", default="scripted")
    parser.add_argument("-P", dest="policy_args", action="append")
    parser.add_argument("--embodiment", default="cubepick")
    parser.add_argument("-E", dest="embodiment_args", action="append")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--store-frames", action="store_true")
    parser.add_argument("--rerun-save", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--scorer", default="operator")
    args = parser.parse_args(argv)
    if (args.task is None) == (args.instruction is None):
        raise SystemExit("pass exactly one of --task or --instruction")
    if args.task is not None:
        task = commands._resolve_or_exit("task", args.task, **commands._parse_kvs(args.task_args))
    else:
        task = commands.adhoc_task(args.instruction, args.max_steps, args.scorer)
    parts = commands.components(
        args.policy,
        commands._parse_kvs(args.policy_args),
        args.embodiment,
        commands._parse_kvs(args.embodiment_args),
        "flags",
    )
    return commands.run_task(
        task,
        parts,
        log_dir=args.log_dir,
        prompt=not args.no_prompt,
        store_frames=args.store_frames,
        rerun_save=args.rerun_save,
        seed=args.seed,
    )


def main(argv: list[str]) -> int:
    """The removed generic command line's ``run``, ``inspect`` (now ``show``), ``view``
    and ``video``, as the robot commands run them."""
    if argv and argv[0] == "run":
        return run_command(argv[1:])
    if argv and " " in argv[0].strip():  # the old bare-instruction shorthand
        return run_command(["--instruction", *argv])
    if argv and argv[0] == "inspect":
        return robot(["show", *argv[1:]])
    return robot(argv)
