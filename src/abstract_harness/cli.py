"""Command-line entry point for `abstract-harness`."""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal, InvalidOperation

from pydantic import ValidationError

from abstract_harness._signals import termination_signals


def dollar_amount(value: str) -> Decimal:
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("expected a dollar amount") from exc
    if not amount.is_finite() or amount <= 0:
        raise argparse.ArgumentTypeError("expected a positive finite dollar amount")
    return amount


def _configure_eval_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    ev = sub.add_parser(
        "eval",
        help="evaluate any policy on any embodiment (default: scripted on cubepick)",
    )
    ev.add_argument("--task", help="registered task name (e.g. cubepick-reach)")
    ev.add_argument("-T", dest="task_args", action="append", help="task KEY=VALUE override")
    ev.add_argument("--instruction", help="ad-hoc task instruction string")
    ev.add_argument(
        "--policy",
        default="scripted",
        help="registered policy (scripted, random, noop, abstract, molmoact2, gr00t, agent)",
    )
    ev.add_argument("-P", dest="policy_args", action="append", help="policy KEY=VALUE override")
    ev.add_argument(
        "--embodiment",
        default="cubepick",
        help="registered embodiment (cubepick, yam_arms, ...)",
    )
    ev.add_argument(
        "-E", dest="embodiment_args", action="append", help="embodiment KEY=VALUE override"
    )
    ev.add_argument("--log-dir", default="runs/eval", help="directory for evaluation logs")
    ev.add_argument("--no-prompt", action="store_true", help="skip interactive operator prompts")
    ev.add_argument("--store-frames", action="store_true", help="persist camera frames on disk")
    ev.add_argument("--rerun-save", action="store_true", help="write a Rerun .rrd recording")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--max-steps", type=int, default=300)
    ev.add_argument("--scorer", default="success_at_end")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="abstract-harness — run and evaluate robotics policies across simulated "
        "and physical embodiments"
    )
    sub = root.add_subparsers(dest="command", required=True)
    _configure_eval_parser(sub)
    from abstract_harness.robot.cli import configure_parser as configure_robot_parser

    configure_robot_parser(sub)
    return root


def _run_eval(args: argparse.Namespace) -> int:
    from abstract_harness.robot.core import commands

    if (args.task is None) and (args.instruction is None):
        task = commands._resolve_or_exit(
            "task", "cubepick-reach", **commands._parse_kvs(args.task_args)
        )
    elif (args.task is not None) and (args.instruction is not None):
        raise ValueError("pass either --task or --instruction, not both")
    elif args.task is not None:
        task = commands._resolve_or_exit("task", args.task, **commands._parse_kvs(args.task_args))
    else:
        task = commands.adhoc_task(args.instruction, args.max_steps, args.scorer)

    parts = commands.components(
        args.policy,
        commands._parse_kvs(args.policy_args),
        args.embodiment,
        commands._parse_kvs(args.embodiment_args),
        "cli",
    )
    return commands.run_task(
        task,
        parts,
        log_dir=args.log_dir,
        prompt=not args.no_prompt,
        store_frames=args.store_frames,
        rerun_save=args.rerun_save,
        seed=args.seed,
        grade=not args.no_prompt,
    )


def main(argv: list[str] | None = None) -> None:
    with termination_signals():
        _main(argv)


def _main(argv: list[str] | None = None) -> None:
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "eval":
            code = _run_eval(args)
        elif args.command == "robot":
            from abstract_harness.robot.cli import main as robot_main

            code = robot_main(args)
        else:
            root.error(f"unknown command {args.command!r}")
            return
    except ValidationError as exc:
        messages = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        root.error("; ".join(messages))
    except (ValueError, OSError) as exc:
        root.error(str(exc))
    except KeyboardInterrupt:
        code = 130
    raise SystemExit(code)


if __name__ == "__main__":
    main(sys.argv[1:])
