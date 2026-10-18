"""``abstract robot``: a model controls the physical YAM rig.

The policy is either an OpenAI model through the agent policy (``[agent]``) or a VLA
behind an /act server through the YAM /act client (``[vla]``, MolmoAct2 or GR00T). Both
run in this process on ``yam_arms`` (``stack.py``; the code is in ``robot/core``,
``robot/agent`` and ``robot/yam``, the hardware libraries in the ``robot`` extra). The
episode owns the hardware: arm bring-up, the bounds clamp and per-step delta limit on
every action, the operator gates, and parking on every exit path, including Ctrl-C. For a
paid model, the loopback spend meter (``meter.py``) holds a per-run budget. ``check`` is
motion-free readiness; ``run`` moves the arms and needs an operator at the rig, and
``--allow-paid`` for an ``[agent]`` config. With ``[store]``, each finished run goes, with
its episode labels, to the Abstract API under the API key's account or to an R2 bucket,
then leaves the rig's disk (``store.py``, ``cloud.py``, ``episode.py``); ``label`` edits
the labels of runs already there. ``serve`` starts and
steers ``run`` over HTTP and streams its terminal, progress and cameras (``rpc.py``),
following the events ``run --events-fd`` writes (``events.py``).
``show``, ``view`` and ``video`` read run logs (``core/commands.py``); ``holdcheck``,
``health`` and ``pose`` are the YAM operator tools (``tools.py``), which energize or move
the arms like a run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

from abstract_harness._security import redact, sensitive_argument
from abstract_harness.robot import cloud, episode, events, formats, stack, store, tools
from abstract_harness.robot.config import (
    PROTECTED_AGENT_ARGS,
    PROTECTED_VLA_ARGS,
    RigConfig,
    RobotConfig,
    StoreConfig,
    parse_value,
)
from abstract_harness.robot.meter import MeterServer, SpendMeter

DEFAULT_ENV_FILE = Path("~/.config/abstract/openai.env")
CAN_CHANNELS = ("can0", "can1")
REHEARSAL_KEY_ENV = "ABSTRACT_REHEARSAL_KEY"
# Every argument the run gave its policy and yam_arms, written into the run directory.
SETUP = "setup.json"
LOG_COMMANDS = ("show", "view", "video")
# view --serve: the inline frame payload of each running page.
LIVE_FRAMES_BUDGET_MB = 8.0


def configure_parser(sub: Any) -> None:
    robot = sub.add_parser(
        "robot",
        help="an OpenAI model (paid) or a VLA server controls the physical YAM rig (moves arms)",
    )
    commands = robot.add_subparsers(dest="robot_command", required=True)
    check = commands.add_parser("check", help="motion-free readiness check of rig and policy")
    run = commands.add_parser("run", help="run one instruction on the real arms")
    upload = commands.add_parser(
        "upload", help="send finished runs to the [store] and delete them locally"
    )
    upload.add_argument("--config", type=Path, required=True, help="robot TOML")
    upload.add_argument(
        "runs",
        nargs="*",
        metavar="RUN",
        help="run directory names in rig.log_root (default: every finished run)",
    )
    label = commands.add_parser(
        "label", help="rebuild or edit the episode labels of runs already in the [store]"
    )
    label.add_argument("--config", type=Path, required=True, help="robot TOML")
    label.add_argument("runs", nargs="*", metavar="RUN", help="run names (YYYYMMDD-HHMMSS)")
    label.add_argument(
        "--all",
        action="store_true",
        help="every run of this rig in the store and in rig.log_root/runs.jsonl",
    )
    label.add_argument("--tag", action="append", default=[], help="add a tag (repeatable)")
    label.add_argument("--untag", action="append", default=[], help="remove a tag (repeatable)")
    label.add_argument("--policy", help="correct the name of the policy that acted")
    label.add_argument(
        "--outcome", choices=episode.OUTCOMES, help="replace the operator's verdict after review"
    )
    serve = commands.add_parser(
        "serve", help="start and steer runs over HTTP (Pi-style RPC); stream terminal and cameras"
    )
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to bind (default: %(default)s; bind a non-loopback address only on a "
        "trusted network)",
    )
    serve.add_argument("--port", type=int, default=8780, help="default: %(default)s")
    serve.add_argument(
        "--token-file",
        type=Path,
        default=Path("~/.config/abstract/rpc.env"),
        help="KEY=VALUE file with ABSTRACT_RPC_TOKEN, created with a random token if missing "
        "(default: %(default)s)",
    )
    serve.add_argument(
        "--allow-paid", action="store_true", help="let clients start runs of an [agent] config"
    )
    serve.add_argument(
        "--allow-no-tmux",
        action="store_true",
        help="serve outside tmux (a dropped SSH session then skips the park)",
    )
    serve.add_argument(
        "--no-verdict",
        action="store_true",
        help="demo mode: every run started here gets --no-verdict (no verdict or grader "
        "notes after the episode; the start gates stay)",
    )
    serve.add_argument(
        "--auto-start",
        action="store_true",
        help="demo mode: every run started here gets -E auto_start=true and skips the home "
        "and scene-ready gates (Esc, End episode and Ctrl-C still stop it). Keep someone at "
        "the rig with the e-stop",
    )
    for command in (check, run, serve):
        command.add_argument("--config", type=Path, required=True, help="robot TOML")
        command.add_argument(
            "--env-file",
            type=Path,
            default=DEFAULT_ENV_FILE,
            help="KEY=VALUE file holding the API key (default: %(default)s)",
        )
    run.add_argument("instruction")
    run.add_argument("--allow-paid", action="store_true", help="required for an [agent] config")
    run.add_argument(
        "--no-upload",
        action="store_true",
        help="keep the finished run on disk instead of sending it to the [store]; "
        "abstract robot upload sends it later (serve uploads its runs in the background)",
    )
    run.add_argument(
        "--no-verdict",
        action="store_true",
        help="do not ask for the verdict or grader notes after the episode: the run is "
        "unscored unless /y /n /p was typed at the console (label sets it later); the home "
        "and start gates still ask",
    )
    run.add_argument(
        "--allow-no-tmux",
        action="store_true",
        help="run outside tmux (a dropped SSH session then skips the park)",
    )
    run.add_argument(
        "-E",
        "--embodiment-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra yam_arms argument for this run",
    )
    run.add_argument(
        "-P",
        "--policy-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra policy argument for this run (keys set by the robot config are refused)",
    )
    run.add_argument(
        "--tag",
        action="append",
        default=[],
        help="label the run in the store's catalog, e.g. --tag cloth --tag demo (repeatable)",
    )
    run.add_argument(
        "--events-fd",
        type=int,
        metavar="FD",
        help="also write the run's events as JSON lines to this inherited pipe "
        "(how abstract robot serve follows a run; robot/events.py)",
    )
    configure_log_parsers(commands)
    tools.configure_parser(commands)


def configure_log_parsers(commands: Any) -> None:
    """``show``, ``view`` and ``video``: read the run logs (``core/commands.py``)."""
    p_show = commands.add_parser("show", help="print a saved run log")
    p_show.add_argument("log", help="path to an EvalLog JSON file")
    p_show.add_argument(
        "--transcript",
        action="store_true",
        help="append recorded policy transcripts",
    )
    p_show.add_argument(
        "--wire",
        nargs="?",
        const=True,
        default=None,
        type=int,
        metavar="CALL",
        help="append the wire-call table, or dump every attempt for CALL",
    )
    p_show.add_argument(
        "--trial",
        default=None,
        metavar="SCENE-eEPOCH",
        help="select a captured trial for --wire CALL",
    )
    p_view = commands.add_parser(
        "view",
        help="render saved eval logs as self-contained HTML reports",
        description=(
            "Render one EvalLog JSON file, or every top-level *.json in a logs "
            "directory with a browsable index. A directory log named index.json "
            "uses index_log.html (or the next collision-free suffix)."
        ),
    )
    p_view.add_argument(
        "log",
        help="path to an EvalLog JSON file, or a logs directory to render a browsable index",
    )
    p_view.add_argument(
        "-o",
        "--out",
        default=None,
        metavar="PATH",
        help=(
            "output HTML file for one log (default: LOG.html; - writes to stdout), "
            "or output directory for a logs directory (default: LOG_DIR/html)"
        ),
    )
    p_view.add_argument(
        "--open",
        action="store_true",
        help="open the written report in the default web browser",
    )
    p_view.add_argument(
        "--no-frames",
        action="store_true",
        help="render placeholders instead of embedding stored camera frames",
    )
    p_view.add_argument(
        "--no-video",
        action="store_true",
        help="skip embedded MP4 encoding while keeping the frame flipbook",
    )
    p_view.add_argument(
        "--frames-budget",
        type=float,
        default=50,
        metavar="MB",
        help=(
            "maximum inline frame payload in decimal MB, applied to each rendered "
            "page (default: 50; 0 is unlimited)"
        ),
    )
    p_view.add_argument(
        "--live-frames-budget",
        type=float,
        default=LIVE_FRAMES_BUDGET_MB,
        metavar="MB",
        help=(
            "maximum inline frame payload for each running page while serving "
            f"(default: {LIVE_FRAMES_BUDGET_MB:g}; 0 is unlimited and is not "
            "recommended while serving)"
        ),
    )
    p_view.add_argument(
        "--force",
        action="store_true",
        help=(
            "re-render existing pages in directory mode; use after changing "
            "--no-frames, --no-video, --frames-budget, or --live-frames-budget "
            "(ignored for one file)"
        ),
    )
    p_view.add_argument(
        "--serve",
        action="store_true",
        help="serve a rendered logs directory over HTTP until stopped",
    )
    p_view.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="N",
        help="listen port for --serve (default: 8300)",
    )
    p_view.add_argument(
        "--host",
        default=None,
        metavar="HOST",
        help="bind address for --serve (default: 127.0.0.1)",
    )

    p_video = commands.add_parser(
        "video",
        help="render a log's stored camera frames to one MP4 per camera stream",
    )
    p_video.add_argument("log", help="path to an EvalLog JSON file from a --store-frames run")
    p_video.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="output directory (default: the frames directory itself)",
    )
    p_video.add_argument(
        "--fps",
        type=float,
        default=None,
        help="playback rate (default: the log's control_hz, else 10)",
    )
    p_video.add_argument(
        "--ffmpeg",
        default=None,
        metavar="PATH",
        help="ffmpeg executable to use (default: found on PATH)",
    )


def main(args: argparse.Namespace) -> int:
    if args.robot_command == "check":
        return check(args)
    if args.robot_command == "upload":
        return upload(args)
    if args.robot_command == "label":
        return label(args)
    if args.robot_command == "serve":
        from abstract_harness.robot import rpc

        return rpc.serve(args)
    if args.robot_command in LOG_COMMANDS:
        from abstract_harness.robot.core import commands

        return commands.main(args)
    if args.robot_command in tools.COMMANDS:
        return tools.main(args)
    return run(args)


def env_value(name: str, env_file: Path | None) -> str | None:
    """A secret from the environment, else from a KEY=VALUE file (chmod 600)."""
    if value := os.environ.get(name):
        return value
    path = env_file.expanduser() if env_file else None
    if path is None or not path.is_file():
        return None
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if sep and not key.startswith("#") and key.strip() == name:
            return value.strip().strip("'\"") or None
    return None


def overrides(
    values: list[str], flag: str, protected: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """The operator's ``-E``/``-P`` ``KEY=VALUE`` arguments for one run; keys in
    ``protected`` are the robot config's and are refused."""
    out: dict[str, Any] = {}
    for value in values:
        key, sep, raw = value.partition("=")
        if sensitive_argument(key):
            raise ValueError(f"{flag}: credentials must use an environment-variable reference")
        if not sep or not key:
            raise ValueError(f"{flag} expects KEY=VALUE, got {value!r}")
        if key in protected:
            raise ValueError(f"{flag} {key} is set by the robot config and cannot be overridden")
        out[key] = parse_value(raw)
    return out


def write_setup(
    config: RobotConfig,
    path: Path,
    base_url: str | None = None,
    policy_overrides: dict[str, Any] | None = None,
    embodiment_overrides: dict[str, Any] | None = None,
) -> Path:
    """The run's components and every argument they get, with the operator's overrides."""
    setup = config.setup(base_url)
    setup["overrides"] = {
        "policy_args": policy_overrides or {},
        "embodiment_args": embodiment_overrides or {},
    }
    path.write_text(json.dumps(redact(setup), indent=2) + "\n")
    return path


def can_is_up(channel: str, sys_net: Path = Path("/sys/class/net")) -> bool:
    try:
        flags = int((sys_net / channel / "flags").read_text().strip(), 16)
    except (OSError, ValueError):
        return False
    return bool(flags & 0x1)  # IFF_UP


def _rehearsal_upstream(requests: list[dict[str, Any]]) -> ThreadingHTTPServer:
    """A stand-in for the Responses API that answers every call with ``done``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Silent."""

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "body": body})
            done = {"summary": "rehearsal", "hindsight": "none"}
            reply = {
                "id": "resp_rehearsal",
                "object": "response",
                "status": "completed",
                "model": body.get("model"),
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_rehearsal",
                        "call_id": "call_rehearsal",
                        "name": "done",
                        "arguments": json.dumps(done),
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                },
            }
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler)


def _rehearsal_episode(
    config: RobotConfig, work: Path
) -> tuple[int, str, SpendMeter, list[dict[str, Any]]]:
    """One simulated ``cubepick`` episode of the agent policy through a meter, with the
    same policy arguments as a real run: exit code, terminal output, meter and ledger."""
    assert config.agent is not None
    work.mkdir()
    meter = SpendMeter(config.agent, work / "meter.jsonl")
    with MeterServer(meter) as server:
        code, output = stack.captured(
            stack.episode,
            config,
            "rehearsal: call done without moving",
            work / "logs",
            source="rehearsal",
            base_url=server.base_url,
            env={REHEARSAL_KEY_ENV: "rehearsal"},
            embodiment="cubepick",
            max_steps=20,
            prompt=False,
            store_frames=False,
            rerun_save=False,
        )
    ledger = work / "meter.jsonl"
    lines = ledger.read_text().splitlines() if ledger.is_file() else []
    return code, output, meter, [json.loads(line) for line in lines]


def rehearse(config: RobotConfig) -> tuple[bool, str]:
    """Free and motion-free: the agent policy goes through the meter twice.

    The simulated ``cubepick`` embodiment (no arms) runs against a local stand-in for the
    Responses API, so the policy's real request and response handling is exercised end to
    end with a stand-in key, two calls at most and no spend. The
    first episode calls ``done``. The second has a budget too small for one call: the
    meter's 402 must end it with a non-zero exit, no upstream call and a ``blocked``
    ledger entry.
    """
    assert config.agent is not None
    requests: list[dict[str, Any]] = []
    upstream = _rehearsal_upstream(requests)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="abstract-rehearsal-") as tmp:
            port = upstream.server_address[1]
            agent = config.agent.model_copy(
                update={
                    "upstream": f"http://127.0.0.1:{port}/v1",
                    "api_key_env": REHEARSAL_KEY_ENV,
                    "max_calls": 2,
                }
            )
            done, done_output, meter, _ = _rehearsal_episode(
                config.model_copy(update={"agent": agent}), Path(tmp) / "done"
            )
            served = len(requests)
            # Far below the output reservation of any call (max_output_tokens >= 256 at a
            # nonzero price), so the meter refuses the first one.
            broke = agent.model_copy(update={"budget_usd": 1e-6})
            refused, refused_output, broke_meter, ledger = _rehearsal_episode(
                config.model_copy(update={"agent": broke}), Path(tmp) / "broke"
            )
    finally:
        upstream.shutdown()
        upstream.server_close()
    body = requests[0]["body"] if requests else {}
    done_ok = (
        done == 0
        and meter.calls == 1
        and served == 1
        and requests[0]["path"] == "/v1/responses"
        and body.get("model") == agent.model
        and body.get("max_output_tokens") == agent.max_output_tokens
    )
    leaked = len(requests) - served
    blocked = sum(entry.get("event") == "blocked" for entry in ledger)
    refused_ok = refused != 0 and leaked == 0 and broke_meter.calls == 0 and blocked > 0
    detail = (
        f"done: exit {done}, {served} call(s) through the meter; "
        f"budget refusal: exit {refused}, {leaked} upstream call(s), {blocked} blocked"
    )
    outputs = (("done", done_output, done_ok), ("budget refusal", refused_output, refused_ok))
    for name, output, ok in outputs:
        if not ok:
            tail = output.strip().splitlines()[-8:]
            detail += f"\n{name} output:\n" + "\n".join(tail)
    return done_ok and refused_ok, detail


def failsafe_armed(values: dict[str, int | str]) -> bool:
    expected = {str(motor_id) for motor_id in stack.YAM_MOTOR_IDS}
    return set(values) == expected and all(
        type(value) is int and value > 0 for value in values.values()
    )


def policy_probe_result(out: dict[str, Any]) -> tuple[bool, str, str | None]:
    """(ok, detail, warning) for ``policy_probe``'s output."""
    if out.get("error"):
        return False, str(out["error"]), None
    if out.get("compat_errors"):
        return False, "incompatible with yam_arms: " + "; ".join(out["compat_errors"]), None
    shape = out.get("chunk_shape") or []
    expected_dim = int(out.get("expected_dim", 14))
    if len(shape) != 2 or shape[0] < 1 or shape[1] != expected_dim:
        return False, f"unexpected action chunk shape {shape}", None
    detail = (
        f"{shape[0]}x{shape[1]} chunk in {out.get('latency_s')}s; first step moves at most "
        f"{out.get('first_step_rad')} rad from the home pose (synthetic black frames)"
    )
    warnings = list(out.get("compat_warnings") or [])
    if out.get("outside_bounds"):
        warnings.append(
            f"{out['outside_bounds']} predicted values outside the config's joint limits "
            "(the clamp holds them at the limit)"
        )
    return True, detail, "; ".join(warnings) or None


vla_probe_result = policy_probe_result


def depth_serials(rig: RigConfig) -> dict[str, str]:
    """RealSense serial per camera slot, from ``<slot>_depth_serial`` in [rig.embodiment];
    an empty or ``none`` value means the slot has no RealSense camera."""
    return {
        key.removesuffix("_depth_serial"): str(value)
        for key, value in rig.embodiment.items()
        if key.endswith("_depth_serial") and str(value).lower() not in ("", "none", "null")
    }


def camera_checks(
    serials: dict[str, str], devices: list[dict[str, Any]] | str
) -> list[tuple[str, bool, str, str | None]]:
    """(check, ok, detail, warning) per configured camera. A missing serial fails; a USB
    2.x link only warns, since yam_arms runs on it at reduced depth_fps."""
    results = []
    found = {} if isinstance(devices, str) else {d.get("serial"): d for d in devices}
    for slot, serial in serials.items():
        name = f"camera:{slot}"
        if isinstance(devices, str):
            results.append((name, False, f"{serial}: RealSense list failed: {devices}", None))
            continue
        device = found.get(serial)
        if device is None:
            seen = sorted(str(s) for s in found)
            detail = f"{serial} not connected; librealsense sees {seen}"
            results.append((name, False, detail, None))
            continue
        usb = device.get("usb") or "unknown"
        warning = None
        if str(usb).startswith("2"):
            warning = f"USB {usb} link: below USB 3 bandwidth; move it to a USB 3 port"
        results.append((name, True, f"{serial}: {device.get('name')}, USB {usb}", warning))
    return results


def connect_bucket(config: StoreConfig) -> cloud.Bucket:
    """The [store] bucket, with R2 credentials from the environment or its env_file."""
    credentials = {name: env_value(name, config.env_file) for name in cloud.CREDENTIAL_ENVS}
    return cloud.Bucket.connect(config, credentials)


def store_check(config: RobotConfig) -> tuple[bool, str]:
    """Reach the [store] with its credentials: the API answers the key with a listing, or
    the bucket takes a small object and gives it back. No run is touched or named."""
    try:
        with store.opened(config, connect_bucket) as target:
            return target.check()
    except (cloud.UploadError, httpx.HTTPError, ValueError) as exc:
        return False, str(exc)


def report_upload(record: dict[str, Any], *, flush: bool = False) -> None:
    """One ``{"upload": ...}`` stdout line, and the same record as an ``upload`` event."""
    print(json.dumps({"upload": record}), flush=flush)
    events.emit("upload", upload=record)


def offload(config: RobotConfig, run_dirs: list[Path]) -> int:
    """Upload each run and delete it locally, printing one ``{"upload": ...}`` line per
    run: ``ok`` with the run's record, or not ok with the error and the run kept."""
    log_root, failed = config.rig.log_root, 0

    def kept(run_dir: Path, exc: Exception) -> None:
        report_upload({"ok": False, "run": run_dir.name, "error": str(exc)})

    try:
        with cloud.upload_lock(log_root), store.opened(config, connect_bucket) as target:
            for run_dir in run_dirs:
                print(f"abstract robot: uploading {run_dir.name}", file=sys.stderr, flush=True)
                try:
                    record = store.offload(target, log_root, run_dir)
                except (cloud.UploadError, httpx.HTTPError, OSError) as exc:
                    failed += 1
                    kept(run_dir, exc)
                    continue
                report_upload({"ok": True, **record}, flush=True)
    except (cloud.UploadError, ValueError) as exc:  # lock held, credentials missing or bad
        for run_dir in run_dirs:
            kept(run_dir, exc)
        return 1
    return 1 if failed else 0


def upload(args: argparse.Namespace) -> int:
    config = RobotConfig.load(args.config)
    if config.store is None:
        raise ValueError("the robot config has no [store] section")
    log_root = config.rig.log_root
    if args.runs:
        for name in args.runs:
            if not cloud.RUN_NAME.match(name) or not (log_root / name).is_dir():
                raise ValueError(f"no run directory {name!r} in {log_root}")
        run_dirs = [log_root / name for name in args.runs]
    else:
        run_dirs = cloud.finished_runs(log_root)
    if not run_dirs:
        print(json.dumps({"upload": {"ok": True, "runs": 0, "log_root": str(log_root)}}))
        return 0
    return offload(config, run_dirs)


def logged_runs(config: RobotConfig, target: store.Store) -> list[str]:
    """Runs of this rig the store holds or the rig's own runs.jsonl says it sent there."""
    runs = set(target.runs())
    index = config.rig.log_root / cloud.INDEX
    lines = index.read_text().splitlines() if index.is_file() else []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and target.owns(row):
            runs.add(str(row.get("run")))
    return sorted(run for run in runs if cloud.RUN_NAME.match(run))


def label(args: argparse.Namespace) -> int:
    """Rebuild each run's episode card from its uploaded files, apply the edits, and update
    the catalog; one ``{"label": ...}`` line per run."""
    config = RobotConfig.load(args.config)
    if config.store is None:
        raise ValueError("the robot config has no [store] section")
    if bool(args.runs) == args.all:
        raise ValueError("name the runs to label, or pass --all")
    for name in args.runs:
        if not cloud.RUN_NAME.match(name):
            raise ValueError(f"{name!r} is not a run name (YYYYMMDD-HHMMSS)")
    add = [episode.tag(value) for value in args.tag]
    remove = [episode.tag(value) for value in args.untag]
    edits = (("policy", args.policy), ("outcome", args.outcome))
    overrides = {key: value for key, value in edits if value}
    failed, records = 0, []
    with cloud.upload_lock(config.rig.log_root), store.opened(config, connect_bucket) as target:
        for run in args.runs or logged_runs(config, target):
            try:
                record = target.relabel(run, add, remove, overrides)
            except (cloud.UploadError, httpx.HTTPError, ValueError) as exc:
                failed += 1
                print(json.dumps({"label": {"ok": False, "run": run, "error": str(exc)}}))
                continue
            records.append(record)
            shown = ("run", "policy", "outcome", "task", "tags", "markers")
            print(json.dumps({"label": {"ok": True, **{key: record[key] for key in shown}}}))
        if records:
            target.labelled(records)
    return 1 if failed else 0


def check(args: argparse.Namespace) -> int:
    config = RobotConfig.load(args.config)
    agent, vla, rig = config.agent, config.vla, config.rig
    results: list[dict[str, Any]] = []

    def report(name: str, ok: bool, detail: str, warning: str | None = None) -> None:
        results.append({"check": name, "ok": ok, "detail": detail})
        if warning:
            results[-1]["warning"] = warning
        print(json.dumps(results[-1]), flush=True)

    if rig.type == "yam_arms":
        left_ch = rig.embodiment["left_channel"]
        right_ch = rig.embodiment["right_channel"]
        channels = f"left arm {left_ch}, right arm {right_ch}"
        report("config", True, f"{config.policy_name} on yam_arms; {channels}")
        installed, versions = stack.installed()
        report("robot_stack", installed, versions)
        configured_channels = tuple(dict.fromkeys((str(left_ch), str(right_ch))))
        up = {channel: can_is_up(channel) for channel in configured_channels}
        for channel in configured_channels:
            report(f"can:{channel}", up[channel], "interface up" if up[channel] else "down")
        if installed:
            for channel in (channel for channel in configured_channels if up[channel]):
                values = stack.motor_timeouts(channel)
                armed = failsafe_armed(values)
                detail = f"TIMEOUT by motor id: {values}"
                if not armed:
                    detail += (
                        "; 0 or no answer means no loss-of-comms failsafe. Restore with "
                        f"i2rt/motor_config_tool/set_timeout.py --channel {channel} --timeout"
                    )
                report(f"failsafe:{channel}", armed, detail)
    else:
        report("config", True, f"{config.policy_name} on {rig.type}")
        installed, versions = stack.installed_for(rig.type)
        report("robot_stack", installed, versions)
    if installed and (serials := depth_serials(rig)):
        for check_result in camera_checks(serials, stack.realsense_devices()):
            report(*check_result)
    if agent is not None:
        key = env_value(agent.api_key_env, args.env_file)
        report("api_key", key is not None, f"${agent.api_key_env} (value not shown)")
        if key is not None:
            try:
                response = httpx.get(
                    f"{agent.upstream.rstrip('/')}/models/{agent.model}",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=20,
                )
                status = response.status_code
                report("model", status == 200, f"{agent.model}: HTTP {status}")
            except httpx.HTTPError as exc:
                report("model", False, f"{agent.model}: {type(exc).__name__}")
    if installed:
        # doctor builds the embodiment without resetting it: no driver connect, no motion.
        code, output = stack.captured(stack.doctor, config, args.config.name)
        tail = output.strip().splitlines()[-12:]
        report("doctor", code == 0, "\n".join(tail))
        if vla is not None:
            ok, detail, warning = vla_probe_result(stack.vla_probe(config))
            report(f"vla:{vla.policy}", ok, f"{vla.server_url}: {detail}", warning)
    if installed and agent is not None:
        report("rehearsal", *rehearse(config))
    if config.store is not None:
        report("store", *store_check(config))
    ok = all(result["ok"] for result in results)
    print(json.dumps({"ready": ok}))
    return 0 if ok else 1


def interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def new_log_dir(rig: RigConfig) -> Path:
    """This run's directory, announced as the first stdout line for anything driving the
    command (``{"log_dir": ...}``)."""
    log_dir = rig.log_root / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True)
    print(json.dumps({"log_dir": str(log_dir)}), flush=True)
    events.emit("log_dir", path=str(log_dir))
    return log_dir


def run(args: argparse.Namespace) -> int:
    """One episode; with ``--events-fd``, also its events for the supervisor (``serve``)."""
    with events.channel(args.events_fd):
        return _run(args)


def _run(args: argparse.Namespace) -> int:
    config = RobotConfig.load(args.config)
    agent, vla, rig = config.agent, config.vla, config.rig
    if agent is not None and not args.allow_paid:
        raise ValueError("robot run calls a paid model and moves real arms; pass --allow-paid")
    instruction = args.instruction.strip()
    if not 0 < len(instruction) <= 2000:
        raise ValueError("instruction must be 1 to 2000 characters")
    protected = PROTECTED_AGENT_ARGS if agent is not None else PROTECTED_VLA_ARGS
    policy_overrides = overrides(args.policy_arg, "-P", protected)
    embodiment_overrides = overrides(args.embodiment_arg, "-E")
    tags = sorted({episode.tag(value) for value in args.tag})
    installed, versions = stack.installed_for(rig.type)
    if not installed:
        raise ValueError(versions)
    if not interactive_terminal():
        raise ValueError("run from an interactive terminal: the episode gates motion on it")
    if not os.environ.get("TMUX") and not args.allow_no_tmux:
        raise ValueError(
            "run inside tmux: a dropped SSH session kills the episode before it parks "
            "the arms (or pass --allow-no-tmux)"
        )
    key = None
    if agent is not None:
        key = env_value(agent.api_key_env, args.env_file)
        if key is None:
            raise ValueError(f"${agent.api_key_env} is not set in the environment or --env-file")

    log_dir = new_log_dir(rig)

    def announce(policy: str) -> None:
        print(
            f"abstract robot: {policy}; logs in {log_dir}\n"
            "Ctrl-C stops the episode and parks the arms.",
            file=sys.stderr,
            flush=True,
        )

    episode_args: dict[str, Any] = {
        "source": args.config.name,
        "policy_args": policy_overrides,
        "embodiment_args": embodiment_overrides,
        "grade": not args.no_verdict,
    }
    if vla is not None:
        write_setup(config, log_dir / SETUP, None, policy_overrides, embodiment_overrides)
        announce(f"{vla.policy} at {vla.server_url}")
        code = stack.episode(config, instruction, log_dir, **episode_args)
        summary: dict[str, Any] = {"policy": vla.policy, "server_url": vla.server_url}
        summary_file, failed = "run-summary.json", code
    else:
        assert agent is not None and key is not None  # exactly one policy; key checked
        meter = SpendMeter(agent, log_dir / "meter.jsonl")
        with MeterServer(meter) as server:
            base_url = server.base_url
            write_setup(config, log_dir / SETUP, base_url, policy_overrides, embodiment_overrides)
            budget = f"budget ${agent.budget_usd:.2f}, at most {agent.max_calls} calls"
            announce(f"{agent.model}, {budget}")
            code = stack.episode(
                config,
                instruction,
                log_dir,
                base_url=base_url,
                env={agent.api_key_env: key},
                **episode_args,
            )
        summary = meter.summary()
        # Unknown billing fails the run even when the episode itself ended cleanly.
        summary_file, failed = "meter-summary.json", code or int(meter.uncertain)
    summary.update(config=args.config.name, tags=tags, exit_code=code, log_dir=str(log_dir))
    # Available artifacts may still upload after a failed conversion, but unverified
    # exports cannot authorize deleting raw frames. Failure remains visible to callers.
    if config.data.formats != ["rrd"]:
        print(f"abstract robot: writing {', '.join(config.data.formats)}", file=sys.stderr)
    summary["formats"] = formats.write(log_dir, config.data.formats)
    if any(not result.get("ok") for result in summary["formats"].values()):
        failed = failed or 1
    summary["exit_code"] = failed
    (log_dir / summary_file).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": summary}), flush=True)
    events.emit("summary", summary=summary)
    after_run(config, log_dir, upload=not args.no_upload)
    return failed


def after_run(config: RobotConfig, log_dir: Path, *, upload: bool = True) -> None:
    """With [store], the arms are parked by now: upload the run and free the disk. A failed
    upload keeps the run and does not change the run's exit code. Without ``upload``
    (``--no-upload``) the run stays on disk and its upload line says it was skipped."""
    if config.store is None:
        return
    if not upload:
        report_upload({"ok": True, "run": log_dir.name, "skipped": "--no-upload"}, flush=True)
        print(
            f"abstract robot: {log_dir} kept on disk (--no-upload); send it with "
            f"abstract robot upload --config <robot TOML> {log_dir.name}",
            file=sys.stderr,
            flush=True,
        )
        return
    try:
        failed = offload(config, [log_dir])
    except KeyboardInterrupt:  # the run stays; objects already sent are skipped on retry
        report_upload({"ok": False, "run": log_dir.name, "error": "interrupted"})
        failed = 1
    if failed:
        print(
            f"abstract robot: {log_dir} kept on disk; retry with "
            f"abstract robot upload --config <robot TOML> {log_dir.name}",
            file=sys.stderr,
            flush=True,
        )
