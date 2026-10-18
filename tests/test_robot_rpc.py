import contextlib
import json
import shutil
import struct
import sys
import threading
import time
import zlib
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from abstract_harness.robot import rpc

pytestmark = pytest.mark.skipif(
    shutil.which("setsid") is None, reason="serve needs util-linux setsid (Linux)"
)

TOKEN = "test-token"
# Stands in for `abstract robot run` (the server's child), with the real event channel on
# --events-fd: the {"log_dir"} line, the YAM gates, two stored frames, a step, a policy
# message, the operator console until /stop, the verdict and grader-notes prompts, then
# {"summary"} and {"upload"} (skipped under --no-upload). Ctrl-C "parks": exit 130.
# Called as `robot upload`, it uploads for $FAKE_UPLOAD_SECONDS, or fails with
# $FAKE_UPLOAD_FAIL set, and appends the time it took to <record>.uploads.
FAKE_ROBOT_RUN = """
import json, os, sys, time
from abstract_harness.robot import events
record, log_root, args = sys.argv[1], sys.argv[2], sys.argv[3:]
with open(record, "a") as f:
    f.write(json.dumps(args) + "\\n")
if args[:2] == ["robot", "upload"]:
    run, started = args[-1], time.time()
    time.sleep(float(os.environ.get("FAKE_UPLOAD_SECONDS", "0")))
    with open(record + ".uploads", "a") as f:
        f.write(json.dumps([run, started, time.time()]) + "\\n")
    if os.environ.get("FAKE_UPLOAD_FAIL"):
        print(json.dumps({"upload": {"ok": False, "run": run, "error": "HTTP 503"}}))
        sys.exit(1)
    print(json.dumps({"upload": {"ok": True, "run": run, "bytes": 1234}}))
    sys.exit(0)

def ask(kind, prompt, options=(), answer=lambda text: "confirmed"):
    with events.dialog(kind, prompt, options) as dialog:
        text = input(prompt)
        dialog.answer = answer(text)
    return text

with events.channel(int(args[args.index("--events-fd") + 1])):
    log_dir = os.path.join(log_root, time.strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid()))
    frames = os.path.join(log_dir, "frames", "20261012_000000_abcd")
    os.makedirs(frames, exist_ok=True)  # two runs can start within one second here
    print(json.dumps({"log_dir": log_dir}), flush=True)
    events.emit("log_dir", path=log_dir)
    code = 0
    try:
        gated = "auto_start=true" not in args  # yam_arms' auto_start drops both gates
        if gated:
            ask("gate", "Arms will move to the home pose - stand clear, then press Enter...")
        events.emit("status", text="homing: ramping arms to start pose")
        if gated:
            ask("gate", "Position the scene, then press Enter to start...")
        header = "{'descr': '|u1', 'fortran_order': False, 'shape': (4, 6, 3), }".ljust(117)
        for camera, step in (("top_cam", 0), ("side", 3)):
            path = os.path.join(frames, f"scene-0-e0_{camera}_{step:06d}.npy")
            with open(path, "wb") as f:
                f.write(b"\\x93NUMPY\\x01\\x00" + (len(header) + 1).to_bytes(2, "little"))
                f.write(header.encode() + b"\\n" + bytes(range(72)))
            events.emit("frame", trial="scene-0-e0", camera=camera, step=step, path=path)
        events.emit("progress", trial="scene-0-e0", step=3)
        events.emit("transcript", trial="scene-0-e0", step=2,
                    message={"role": "assistant", "content": "moving"})
        events.emit("status", text="Running. Max ~120s.")
        print("Running.", flush=True)
        while (line := sys.stdin.readline().strip()) != "/stop":
            print("policy got: " + line, flush=True)
        events.emit("status", text=None)
        verdict = ask("verdict", "did the robot succeed? [y/n/partial/skip] ",
                      ("y", "n", "partial", "skip"), answer=lambda text: text.strip())
        note = ask("notes", "grader notes (Enter for none): ",
                   answer=lambda text: "text" if text.strip() else "empty")
        print(json.dumps({"verdict": verdict, "note": note}), flush=True)
    except KeyboardInterrupt:
        print("parked", flush=True)
        code = 130
    events.emit("episode_end", exit_code=code)
    summary = {"exit_code": code, "log_dir": log_dir}
    print(json.dumps({"summary": summary}), flush=True)
    events.emit("summary", summary=summary)
    upload = {"ok": True, "prefix": "yam-1/x/"}
    if "--no-upload" in args:
        upload = {"ok": True, "run": os.path.basename(log_dir), "skipped": "--no-upload"}
    print(json.dumps({"upload": upload}), flush=True)
    events.emit("upload", upload=upload)
sys.exit(code)
"""
# The same run from a harness without run events: the prompts only on its terminal.
SILENT_ROBOT_RUN = """
import sys
print("abstract robot: logs in /tmp/x", flush=True)
input("Arms will move to the home pose - stand clear, then press Enter...")
sys.exit(0)
"""


RIG = (
    '[rig]\nlog_root = "logs"\n'
    '[rig.embodiment]\nleft_channel = "can1"\nright_channel = "can0"\ntop_depth_serial = "123"\n'
)


def write_config(tmp_path, store=False):
    path = tmp_path / ("robot-store.toml" if store else "robot.toml")
    text = '[vla]\npolicy = "molmoact2"\nserver_url = "http://127.0.0.1:8202"\n' + RIG
    path.write_text(
        text + ('[store]\nroute = "bucket"\nbucket = "my-robot-runs"\n' if store else "")
    )
    return path


@pytest.fixture
def controller(tmp_path, monkeypatch):
    record = tmp_path / "argv.jsonl"
    fake = [sys.executable, "-c", FAKE_ROBOT_RUN, str(record), str(tmp_path / "logs")]
    monkeypatch.setattr(rpc, "abstract_command", lambda: fake)
    controller = rpc.Controller(
        write_config(tmp_path), allow_paid=False, env_file=None, allow_no_tmux=True
    )
    controller.record = record
    yield controller
    controller.shutdown(timeout=10)


@pytest.fixture
def stored(tmp_path, monkeypatch):
    """A controller whose config has a [store]: its runs end unuploaded and it uploads
    them afterwards, through the fake's `robot upload`."""
    record = tmp_path / "argv.jsonl"
    fake = [sys.executable, "-c", FAKE_ROBOT_RUN, str(record), str(tmp_path / "logs")]
    monkeypatch.setattr(rpc, "abstract_command", lambda: fake)
    controller = rpc.Controller(
        write_config(tmp_path, store=True), allow_paid=False, env_file=None, allow_no_tmux=True
    )
    controller.record = record
    yield controller
    controller.shutdown(timeout=10, upload_timeout=10)


def run_through(controller, events, message):
    """One episode driven like the console: both gates, /stop, a y verdict, no note."""
    ok(controller, "prompt", message=message)
    for _ in range(2):
        ok(controller, "extension_ui_response", id=events.dialog("confirm")["id"], confirmed=True)
    events.until(lambda e: e["type"] == "progress")
    ok(controller, "stop")
    ok(controller, "extension_ui_response", id=events.dialog("select")["id"], value="y")
    ok(controller, "extension_ui_response", id=events.dialog("input")["id"], value="")
    return events.until(lambda e: e["type"] == "run_end")


class Waiter:
    def __init__(self, controller):
        self.controller, self.seq = controller, 0

    def until(self, predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for event in self.controller.events.after(self.seq, timeout=0.2):
                self.seq = event["seq"]
                if predicate(event):
                    return event
        raise AssertionError("event not seen")

    def dialog(self, method):
        return self.until(lambda e: e["type"] == "extension_ui_request" and e["method"] == method)

    def output(self, text):
        return self.until(lambda e: e["type"] == "output" and text in e["text"])


def ok(controller, type_, **fields):
    response = controller.handle({"id": "t", "type": type_, **fields})
    assert response["success"], response
    return response.get("data")


def test_episode_is_driven_like_the_console(controller):
    # Run events and terminal lines come through different pipes: one cursor each.
    events, terminal = Waiter(controller), Waiter(controller)
    started = ok(controller, "prompt", message="fold the cloth", tags=["Demo", "cloth", "demo"])
    assert started["disposition"] == "started"
    assert controller.cmd_get_state({})["run"]["tags"] == ["cloth", "demo"]
    refused = controller.handle({"type": "prompt", "message": "again"})
    assert not refused["success"] and "running" in refused["error"]

    stand_clear = events.dialog("confirm")
    assert "stand clear" in stand_clear["title"]
    assert controller.cmd_get_state({})["pendingDialogs"][0]["id"] == stand_clear["id"]
    ok(controller, "extension_ui_response", id=stand_clear["id"], confirmed=True)
    twice = controller.handle(
        {"type": "extension_ui_response", "id": stand_clear["id"], "confirmed": True}
    )
    assert not twice["success"]  # one Enter per gate, however often a client clicks
    resolved = events.until(lambda e: e["type"] == "extension_ui_resolved")
    assert (resolved["id"], resolved["by"], resolved["answer"]) == (
        stand_clear["id"],
        "rpc",
        "confirmed",
    )
    assert events.until(lambda e: e["type"] == "status")["text"].startswith("homing")
    position = events.dialog("confirm")
    assert "Position the scene" in position["title"] and position["id"] != stand_clear["id"]
    ok(controller, "extension_ui_response", id=position["id"], confirmed=True)

    assert events.until(lambda e: e["type"] == "progress")["step"] == 3
    transcript = events.until(lambda e: e["type"] == "transcript")
    assert transcript["message"] == {"role": "assistant", "content": "moving"}
    assert events.until(lambda e: e["type"] == "status")["text"] == "Running. Max ~120s."
    terminal.output("Running.")
    ok(controller, "steer", message="use the left arm")
    terminal.output("policy got: use the left arm")
    # The config's camera first, then one only the episode stores.
    assert controller.cmd_get_state({})["cameras"] == ["top_cam", "side"]
    frame = controller.frame("side")
    assert frame.content_type == "image/png" and frame.data.startswith(b"\x89PNG")
    assert struct.unpack(">II", frame.data[16:24]) == (6, 4)  # small frames keep their size

    ok(controller, "stop")
    verdict = events.dialog("select")
    assert verdict["options"] == ["y", "n", "partial", "skip"]
    bad = controller.handle(
        {"type": "extension_ui_response", "id": verdict["id"], "value": "maybe"}
    )
    assert not bad["success"]
    ok(controller, "extension_ui_response", id=verdict["id"], value="y")
    notes = events.dialog("input")
    ok(controller, "extension_ui_response", id=notes["id"], value="clean fold")
    terminal.output('{"verdict": "y", "note": "clean fold"}')
    assert events.until(lambda e: e["type"] == "episode_end")["exitCode"] == 0

    end = events.until(lambda e: e["type"] == "run_end")
    assert end["exitCode"] == 0 and end["summary"]["exit_code"] == 0 and end["events"]
    assert end["logDir"] == end["summary"]["log_dir"]
    assert end["upload"] == [{"upload": {"ok": True, "prefix": "yam-1/x/"}}]
    # robot run itself, with no -E/-P; its events pipe, the tags as labels, the
    # instruction after "--".
    (argv,) = [json.loads(line) for line in controller.record.read_text().splitlines()]
    fd = argv.index("--events-fd")
    assert argv[fd + 1].isdigit()
    del argv[fd : fd + 2]
    assert argv == [
        "robot",
        "run",
        "--config",
        str(controller.config_path),
        "--allow-no-tmux",
        "--tag",
        "cloth",
        "--tag",
        "demo",
        "--",
        "fold the cloth",
    ]
    state = controller.cmd_get_state({})
    assert state["phase"] == "idle" and state["run"] is None
    assert state["lastRun"]["instruction"] == "fold the cloth"


def test_abort_is_ctrl_c_and_parks(controller):
    events, terminal = Waiter(controller), Waiter(controller)
    ok(controller, "prompt", message="pick the jar")
    gate = events.dialog("confirm")
    ok(controller, "abort")
    resolved = events.until(lambda e: e["type"] == "extension_ui_resolved")
    assert resolved["id"] == gate["id"] and resolved["answer"] == "interrupted"
    terminal.output("parked")
    end = events.until(lambda e: e["type"] == "run_end")
    assert end["summary"]["exit_code"] == 130 and end["exitCode"] == 130
    assert controller.cmd_get_state({})["pendingDialogs"] == []
    stale = controller.handle(
        {"type": "extension_ui_response", "id": gate["id"], "confirmed": True}
    )
    assert not stale["success"]


def test_demo_mode_runs_every_episode_with_no_verdict(controller):
    # serve --no-verdict: every run it starts skips the verdict and grader notes. The
    # gates are the run's own and still ask, so the first dialog is still the home gate.
    controller.no_verdict = True
    assert controller.cmd_get_state({})["noVerdict"] is True
    events = Waiter(controller)
    ok(controller, "prompt", message="pick the jar")
    assert "stand clear" in events.dialog("confirm")["title"]
    ok(controller, "abort")
    events.until(lambda e: e["type"] == "run_end")
    (argv,) = [json.loads(line) for line in controller.record.read_text().splitlines()]
    assert argv.index("--no-verdict") < argv.index("--")


def test_runs_ask_for_the_verdict_by_default(controller):
    assert controller.cmd_get_state({})["noVerdict"] is False
    events = Waiter(controller)
    ok(controller, "prompt", message="pick the jar")
    events.dialog("confirm")
    ok(controller, "abort")
    events.until(lambda e: e["type"] == "run_end")
    (argv,) = [json.loads(line) for line in controller.record.read_text().splitlines()]
    assert "--no-verdict" not in argv


def test_auto_start_runs_skip_both_gates_and_still_stop(controller):
    # serve --auto-start (demo mode): every run gets -E auto_start=true, so yam_arms skips
    # both readiness gates; End episode still ends the run.
    controller.auto_start = True
    assert controller.cmd_get_state({})["autoStart"] is True
    events = Waiter(controller)
    ok(controller, "prompt", message="pick the jar")
    events.until(lambda e: e["type"] == "status" and e["text"] == "Running. Max ~120s.")
    ok(controller, "stop")
    ok(controller, "extension_ui_response", id=events.dialog("select")["id"], value="skip")
    ok(controller, "extension_ui_response", id=events.dialog("input")["id"], value="")
    assert events.until(lambda e: e["type"] == "run_end")["exitCode"] == 0
    asked = [e["title"] for e in controller.events.after(0) if e["type"] == "extension_ui_request"]
    assert not [title for title in asked if "stand clear" in title or "Position" in title]
    (argv,) = [json.loads(line) for line in controller.record.read_text().splitlines()]
    flag = argv.index("-E")
    assert argv[flag + 1] == "auto_start=true" and flag < argv.index("--")


def test_hold_pauses_new_runs_and_lets_a_running_one_finish(controller):
    assert controller.cmd_get_state({})["held"] is False
    assert ok(controller, "hold", on=True) == {"held": True}
    refused = controller.handle({"type": "prompt", "message": "pick the jar"})
    assert not refused["success"] and refused["error"].startswith("held")
    assert not controller.record.exists()  # nothing was started
    assert not controller.handle({"type": "hold", "on": "yes"})["success"]
    ok(controller, "hold", on=False)
    events = Waiter(controller)
    ok(controller, "prompt", message="pick the jar")
    ok(controller, "hold", on=True)
    events.dialog("confirm")  # the running episode goes on
    ok(controller, "abort")
    events.until(lambda e: e["type"] == "run_end")
    assert controller.cmd_get_state({})["held"] is True


def test_with_a_store_a_run_ends_idle_and_uploads_while_the_next_runs(stored, monkeypatch):
    monkeypatch.setenv("FAKE_UPLOAD_SECONDS", "1.5")
    events, uploads = Waiter(stored), Waiter(stored)
    first = run_through(stored, events, "fold the cloth")
    name = Path(first["logDir"]).name
    # The run has exited: idle at once, its upload under way, the next prompt accepted.
    state = stored.cmd_get_state({})
    assert state["phase"] == "idle"
    assert name in [state["uploads"]["current"], *state["uploads"]["queued"]]
    assert first["upload"] == [{"upload": {"ok": True, "run": name, "skipped": "--no-upload"}}]
    second = run_through(stored, events, "fold it again")
    ends = [uploads.until(lambda e: e["type"] == "upload_end") for _ in range(2)]
    names = [name, Path(second["logDir"]).name]
    assert [(e["run"], e["ok"], e["bytes"]) for e in ends] == [(n, True, 1234) for n in names]
    lines = [json.loads(line) for line in stored.record.read_text().splitlines()]
    runs = [argv for argv in lines if argv[:2] == ["robot", "run"]]
    assert all(argv.index("--no-upload") < argv.index("--") for argv in runs)
    config = str(stored.config_path)
    assert [argv for argv in lines if argv[:2] == ["robot", "upload"]] == [
        ["robot", "upload", "--config", config, n] for n in names
    ]
    # One at a time: the second upload began after the first had ended.
    spans = [json.loads(line) for line in Path(f"{stored.record}.uploads").read_text().splitlines()]
    assert spans[1][1] >= spans[0][2]
    assert stored.cmd_get_state({})["uploads"] == {
        "queued": [],
        "current": None,
        "last": {"run": names[1], "ok": True, "error": None, "bytes": 1234},
    }


def test_a_failed_upload_leaves_the_run_on_disk_and_says_so(stored, monkeypatch):
    monkeypatch.setenv("FAKE_UPLOAD_FAIL", "1")
    events, uploads = Waiter(stored), Waiter(stored)
    end = run_through(stored, events, "fold the cloth")
    done = uploads.until(lambda e: e["type"] == "upload_end")
    assert (done["ok"], done["error"]) == (False, "HTTP 503")
    assert stored.cmd_get_state({})["uploads"]["last"]["ok"] is False
    assert Path(end["logDir"]).is_dir()


def test_shutdown_waits_for_the_upload_under_way_and_names_the_rest(stored, monkeypatch):
    monkeypatch.setenv("FAKE_UPLOAD_SECONDS", "1")
    uploads = Waiter(stored)
    stored.uploads.add("20261012-000001")
    stored.uploads.add("20261012-000002")
    uploads.until(lambda e: e["type"] == "upload_start")
    assert stored.shutdown(timeout=1, upload_timeout=10) == ["20261012-000002"]
    done = uploads.until(lambda e: e["type"] == "upload_end")
    assert (done["run"], done["ok"]) == ("20261012-000001", True)


def test_a_dialog_of_an_earlier_run_never_answers_the_next(controller):
    events = Waiter(controller)
    ok(controller, "prompt", message="pick the jar")
    first = events.dialog("confirm")
    ok(controller, "abort")
    events.until(lambda e: e["type"] == "run_end")
    ok(controller, "prompt", message="pick the jar again")
    second = events.dialog("confirm")
    # The run numbers its dialogs from 1 each time; the ids here differ by run.
    assert first["id"] != second["id"] and first["id"].endswith(second["id"].split(":")[1])
    stale = controller.handle(
        {"type": "extension_ui_response", "id": first["id"], "confirmed": True}
    )
    assert not stale["success"] and controller.cmd_get_state({})["pendingDialogs"]
    ok(controller, "abort")
    events.until(lambda e: e["type"] == "run_end")


def test_a_run_without_events_still_runs_and_says_so(controller, monkeypatch):
    silent = [sys.executable, "-c", SILENT_ROBOT_RUN]
    monkeypatch.setattr(rpc, "abstract_command", lambda: silent)
    events = Waiter(controller)
    ok(controller, "prompt", message="x")
    events.output("logs in")
    assert controller.cmd_get_state({})["pendingDialogs"] == []  # nothing scraped
    controller.forward_keys(b"\r")  # the operator at the rig presses Enter
    end = events.until(lambda e: e["type"] == "run_end")
    assert end["exitCode"] == 0 and end["events"] is False


def test_refusals_before_and_without_a_run(tmp_path, controller):
    for command, error in [
        ({"type": "steer", "message": "hi"}, "no run is active"),
        ({"type": "abort"}, "no run is active"),
        ({"type": "prompt", "message": "two\nlines"}, "one line"),
        ({"type": "prompt", "message": ""}, "non-empty"),
        ({"type": "prompt", "message": "x", "tags": ["two words"]}, "no spaces"),
        ({"type": "prompt", "message": "x", "tags": "cloth"}, "list of strings"),
        ({"type": "stop", "verdict": "maybe"}, "verdict"),
        ({"type": "nope"}, "unknown command"),
    ]:
        response = controller.handle({"id": "x", **command})
        assert response["id"] == "x" and not response["success"]
        assert error in response["error"]
    assert controller.handle([1])["command"] == "parse"
    agent = tmp_path / "agent.toml"
    agent.write_text(
        '[agent]\nmodel = "gpt-6-sol"\n[agent.pricing]\ninput_per_million = 2.0\n'
        "cached_input_per_million = 0.2\noutput_per_million = 10.0\n" + RIG
    )
    paid = rpc.Controller(agent, allow_paid=False, env_file=None)
    response = paid.handle({"type": "prompt", "message": "move"})
    assert "--allow-paid" in response["error"] and paid.phase == "idle"
    paid.shutdown(timeout=1)


def test_a_refused_run_reports_its_output(controller, monkeypatch):
    refuse = [
        sys.executable,
        "-c",
        "import sys; from abstract_harness.robot import events\n"
        "args = sys.argv[1:]\n"
        "with events.channel(int(args[args.index('--events-fd') + 1])):\n"
        "    raise SystemExit(print('robot run calls a paid model; pass --allow-paid') or 2)",
    ]
    monkeypatch.setattr(rpc, "abstract_command", lambda: refuse)
    events = Waiter(controller)
    ok(controller, "prompt", message="x")
    end = events.until(lambda e: e["type"] == "run_end")
    assert end["exitCode"] == 2 and end["summary"] is None and end["events"]
    assert "pass --allow-paid" in " ".join(end["tail"])
    assert controller.phase == "idle"


def test_a_run_error_event_reaches_the_run_end(controller, monkeypatch):
    refuse = [
        sys.executable,
        "-c",
        "import sys; from abstract_harness.robot import events\n"
        "args = sys.argv[1:]\n"
        "with events.channel(int(args[args.index('--events-fd') + 1])):\n"
        "    raise ValueError('run inside tmux')",
    ]
    monkeypatch.setattr(rpc, "abstract_command", lambda: refuse)
    events = Waiter(controller)
    ok(controller, "prompt", message="x")
    end = events.until(lambda e: e["type"] == "run_end")
    assert end["exitCode"] == 1 and end["runError"] == "run inside tmux"


def test_screen_turns_terminal_output_into_lines():
    events = []
    screen = rpc.Screen(events.append)
    screen.feed(b"\x1b[1mhoming\x1b[0m: ramping\rRunning.\r\n")
    screen.feed(b"did the robot succeed? [y/n/partial/skip] ")
    assert [e["text"] for e in events] == ["Running."]  # prompts come as run events
    screen.feed(b"y\r\n")
    assert [e["text"] for e in events] == [
        "Running.",
        "did the robot succeed? [y/n/partial/skip] y",
    ]
    assert list(screen.lines) == [e["text"] for e in events]


def npy(height, width, pixels):
    header = f"{{'descr': '|u1', 'fortran_order': False, 'shape': ({height}, {width}, 3), }}"
    header = header.ljust(117) + "\n"
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header.encode() + pixels


def test_frames_decode_to_half_size_png(tmp_path):
    pixels = bytes(i % 251 for i in range(4 * 6 * 3))
    path = tmp_path / "f.npy"
    path.write_bytes(npy(4, 6, pixels))
    assert rpc.read_npy_rgb(path) == (6, 4, pixels)
    path.write_bytes(npy(4, 6, pixels[:-1]))  # still being written
    assert rpc.read_npy_rgb(path) is None
    data = rpc.png(6, 4, pixels)
    assert struct.unpack(">II", data[16:24]) == (3, 2)
    idat = data.index(b"IDAT")
    length = struct.unpack(">I", data[idat - 4 : idat])[0]
    raw = zlib.decompress(data[idat + 4 : idat + 4 + length])
    # Rows 0 and 2, pixels 0, 2 and 4 of each, after a filter byte per row.
    expected = b"".join(
        b"\x00" + b"".join(pixels[(y * 6 + x) * 3 : (y * 6 + x) * 3 + 3] for x in (0, 2, 4))
        for y in (0, 2)
    )
    assert raw == expected


def test_camera_frames_go_out_as_full_size_jpeg(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    # A 640x360 RGB frame, as the run stores the top camera: red left half, blue right.
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:, :320, 0] = 255
    image[:, 320:, 2] = 255
    path = tmp_path / "scene-0-e0_top_cam_000001.npy"
    np.save(path, image)
    frames = rpc.RunFrames()
    frames.add("top_cam", str(path))
    frame = frames.frame("top_cam")
    assert frame is not None and frame.content_type == "image/jpeg"
    decoded = cv2.imdecode(np.frombuffer(frame.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (360, 640, 3)  # full size, not halved
    left, right = decoded[180, 100], decoded[180, 540]  # BGR
    assert left[2] > 200 and left[0] < 60  # red stays red
    assert right[0] > 200 and right[2] < 60  # blue stays blue


def test_a_stream_takes_fps_and_shrinks_wide_frames(server, controller, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    ok, encoded = cv2.imencode(".jpg", np.full((720, 1280, 3), 128, dtype=np.uint8))
    assert ok
    frames = iter(range(1_000_000))
    # A new 1280x720 frame on every read; never the real cameras.
    monkeypatch.setattr(
        controller,
        "frame",
        lambda camera: rpc.Frame(str(next(frames)), "image/jpeg", encoded.tobytes()),
    )
    monkeypatch.setattr(controller.preview, "watching", contextlib.nullcontext)
    query = {"token": TOKEN, "fps": "15", "width": "320"}
    started = time.monotonic()
    with httpx.stream("GET", f"{server}/feed/top_cam", params=query, timeout=10) as response:
        assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
        parts, buffer = [], b""
        for chunk in response.iter_bytes():
            buffer += chunk
            while (head := buffer.find(b"\r\n\r\n")) != -1 and b"Content-Length" in buffer[:head]:
                length = int(buffer[:head].split(b"Content-Length: ")[1].split(b"\r\n")[0])
                if len(buffer) < head + 4 + length + 2:
                    break
                parts.append(buffer[head + 4 : head + 4 + length])
                buffer = buffer[head + 4 + length + 2 :]
            if len(parts) >= 8:
                break
    elapsed = time.monotonic() - started
    image = cv2.imdecode(np.frombuffer(parts[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (180, 320, 3)  # shrunk to the width, aspect kept
    assert elapsed < 8 / 5  # faster than the default 5 fps
    assert rpc.query_int({"fps": ["99"]}, "fps", 5, 1, 15) == 15
    assert rpc.query_int({"fps": ["x"]}, "fps", 5, 1, 15) == 5
    assert rpc.shrunk(parts[0], 640) == parts[0]  # never enlarged


def test_token_file_is_created_private_and_reused(tmp_path):
    path = tmp_path / "abstract" / "rpc.env"
    token = rpc.load_token(path)
    assert len(token) >= 32 and path.stat().st_mode & 0o777 == 0o600
    assert rpc.load_token(path) == token
    path.write_text("OTHER=1\n")
    with pytest.raises(ValueError, match="ABSTRACT_RPC_TOKEN"):
        rpc.load_token(path)


@pytest.fixture
def server(controller):
    handler = type("H", (rpc.Handler,), {"controller": controller, "token": TOKEN})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_http_needs_the_token_and_speaks_pi_rpc(server, controller):
    assert httpx.get(f"{server}/health").json()["phase"] == "idle"
    assert b"Rig control" in httpx.get(f"{server}/").content
    assert httpx.post(f"{server}/rpc", json={"type": "get_state"}).status_code == 401
    assert httpx.get(f"{server}/events", params={"token": "wrong"}).status_code == 401
    auth = {"Authorization": f"Bearer {TOKEN}"}
    state = httpx.post(f"{server}/rpc", json={"id": "1", "type": "get_state"}, headers=auth)
    body = state.json()
    assert body["id"] == "1" and body["type"] == "response" and body["command"] == "get_state"
    assert body["data"]["cameras"] == ["top_cam"] and body["data"]["policy"] == "molmoact2"
    garbled = httpx.post(f"{server}/rpc", content=b"{", headers=auth)
    assert garbled.status_code == 400 and garbled.json()["command"] == "parse"
    assert httpx.get(f"{server}/feed/side_cam", headers=auth).status_code == 404

    controller.events.emit({"type": "output", "text": "hello"})
    with httpx.stream(
        "GET", f"{server}/events", params={"since": 0, "token": TOKEN}, timeout=10
    ) as stream:
        for line in stream.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                assert event["type"] == "output" and event["text"] == "hello"
                break


def test_a_reconnecting_event_stream_resumes_after_its_last_event(server, controller):
    for text in ("one", "two", "three"):
        controller.events.emit({"type": "output", "text": text})
    # A browser's EventSource reconnects to the same URL, ?since= and all, and sends the
    # last id it saw: that is where it resumes, so nothing is replayed or skipped.
    headers = {"Last-Event-ID": "2"}
    with httpx.stream(
        "GET", f"{server}/events", params={"since": 0, "token": TOKEN}, headers=headers, timeout=10
    ) as stream:
        for line in stream.iter_lines():
            if line.startswith("data: "):
                assert json.loads(line[6:])["text"] == "three"
                break


def test_a_failing_output_handler_never_stops_the_pty_draining():
    def broken(data):
        raise RuntimeError("handler bug")

    # 1 MB is far beyond the pty buffer: if the reader stopped, the child would block.
    terminal = rpc.Terminal(
        [sys.executable, "-c", "print('x' * 1_000_000)"], {"PATH": "/usr/bin:/bin"}, broken
    )
    done = []
    waiter = threading.Thread(target=lambda: done.append(terminal.wait()), daemon=True)
    waiter.start()
    waiter.join(timeout=30)
    assert done == [0]


def test_mirror_gives_up_on_a_dead_terminal(monkeypatch):
    class Dead:
        def write(self, data):
            raise OSError(5, "Input/output error")

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", type("Out", (), {"buffer": Dead()})())
    mirror = rpc.Mirror()
    mirror(b"x")
    mirror(b"y")
    assert not mirror.ok
