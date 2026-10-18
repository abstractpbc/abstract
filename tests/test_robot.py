import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from abstract_harness.cli import parser
from abstract_harness.robot import cli as robot_cli
from abstract_harness.robot import events, stack
from abstract_harness.robot.config import RobotConfig
from abstract_harness.robot.meter import (
    IMAGE_TOKEN_BOUND,
    MeterServer,
    SpendMeter,
    estimate_input_tokens,
    response_usage,
)

MODEL = "gpt-6-sol"
PRICES = {"input_per_million": 2.0, "cached_input_per_million": 0.2, "output_per_million": 10.0}
LONG_PRICES = {
    "input_per_million": 4.0,
    "cached_input_per_million": 0.2,
    "output_per_million": 15.0,
    "cache_write_per_million": 5.0,
}


RIG = '[rig]\nlog_root = "logs"\n[rig.embodiment]\nleft_channel = "can1"\nright_channel = "can0"\n'


def write_config(tmp_path, *, pricing=None, long_context_pricing=None, rig=RIG, **agent):
    values = {"model": MODEL, "budget_usd": 1.0, "max_calls": 5, "max_output_tokens": 1000}
    values.update(agent)
    lines = ["[agent]", *(f"{k} = {json.dumps(v)}" for k, v in values.items())]
    lines += ["[agent.pricing]", *(f"{k} = {v}" for k, v in (pricing or PRICES).items())]
    if long_context_pricing:
        lines += ["[agent.long_context_pricing]"]
        lines += [f"{k} = {v}" for k, v in long_context_pricing.items()]
    path = tmp_path / "robot.toml"
    path.write_text("\n".join(lines) + "\n" + rig)
    return path


def completed(input_tokens=1000, cached=0, output_tokens=100, **fields):
    usage = {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output_tokens,
    }
    return {"id": "resp_1", "status": "completed", "output": [], "usage": usage, **fields}


class Upstream:
    """Records forwarded requests and answers with scripted responses or errors."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def metered(tmp_path):
    def make(*responses, **config):
        loaded = RobotConfig.load(write_config(tmp_path, **config))
        meter = SpendMeter(loaded.agent, tmp_path / "meter.jsonl")
        upstream = Upstream(*responses)
        return meter, upstream, MeterServer(meter, transport=httpx.MockTransport(upstream))

    return make


def post(server, body, path="/responses"):
    return httpx.post(
        f"{server.base_url}{path}",
        json=body,
        headers={"Authorization": "Bearer sk-test-secret"},
        timeout=10,
    )


def ledger(tmp_path):
    text = (tmp_path / "meter.jsonl").read_text()
    assert "sk-test-secret" not in text
    return [json.loads(line) for line in text.splitlines()]


def test_forwards_caps_output_and_settles_from_usage(metered, tmp_path):
    meter, upstream, server = metered(
        httpx.Response(200, json=completed(input_tokens=1000, cached=200, output_tokens=100))
    )
    with server:
        response = post(server, {"model": MODEL, "input": "pick up the cup"})
    assert response.status_code == 200 and response.json()["id"] == "resp_1"
    (request,) = upstream.requests
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["Authorization"] == "Bearer sk-test-secret"
    assert json.loads(request.content)["max_output_tokens"] == 1000
    # 800 uncached * $2 + 200 cached * $0.20 + 100 output * $10, per million tokens.
    assert meter.spent == Decimal("0.00264")
    (entry,) = ledger(tmp_path)
    assert entry["charged_usd"] == "0.002640" and not entry["billing_uncertain"]


@pytest.mark.parametrize(("requested", "sent"), [(200, 200), (5000, 1000), (True, 1000)])
def test_output_cap_keeps_only_lower_valid_requests(metered, requested, sent):
    _, upstream, server = metered(httpx.Response(200, json=completed()))
    with server:
        post(server, {"model": MODEL, "input": "x", "max_output_tokens": requested})
    assert json.loads(upstream.requests[0].content)["max_output_tokens"] == sent


def test_refuses_before_upstream_when_budget_cannot_cover_worst_case(metered, tmp_path):
    meter, upstream, server = metered(budget_usd=0.001)
    with server:
        response = post(server, {"model": MODEL, "input": "x"})
    assert response.status_code == 402
    assert "abstract spend meter" in response.json()["error"]["message"]
    assert upstream.requests == [] and meter.calls == 0 and meter.blocked == 1
    assert ledger(tmp_path)[0]["event"] == "blocked"


def test_call_limit_refuses_the_next_call(metered):
    meter, upstream, server = metered(httpx.Response(200, json=completed()), max_calls=1)
    with server:
        assert post(server, {"model": MODEL, "input": "x"}).status_code == 200
        assert post(server, {"model": MODEL, "input": "x"}).status_code == 402
    assert len(upstream.requests) == 1 and meter.calls == 1


def test_rate_limit_is_not_billed_and_keeps_retry_after(metered):
    meter, _, server = metered(
        httpx.Response(429, json={"error": {"message": "slow down"}}, headers={"Retry-After": "2"}),
        httpx.Response(200, json=completed()),
    )
    with server:
        limited = post(server, {"model": MODEL, "input": "x"})
        assert limited.status_code == 429 and limited.headers["Retry-After"] == "2"
        assert meter.calls == 0 and meter.spent == 0 and not meter.uncertain
        assert post(server, {"model": MODEL, "input": "x"}).status_code == 200
    assert meter.calls == 1


@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(500, json={"error": {"message": "boom"}}),
        httpx.ReadTimeout("slow"),
        httpx.Response(200, json={"id": "resp_1", "status": "completed", "output": []}),
    ],
    ids=["server-error", "timeout", "missing-usage"],
)
def test_unknown_billing_keeps_reservation_and_blocks(metered, failure):
    meter, upstream, server = metered(failure)
    with server:
        first = post(server, {"model": MODEL, "input": "x"})
        second = post(server, {"model": MODEL, "input": "x"})
    assert first.status_code == (504 if isinstance(failure, Exception) else failure.status_code)
    assert second.status_code == 402
    assert meter.uncertain and meter.spent > 0 and len(upstream.requests) == 1


def test_rejects_other_models_paths_and_streaming(metered):
    _, upstream, server = metered()
    with server:
        assert post(server, {"model": "gpt-6-astra", "input": "x"}).status_code == 400
        assert post(server, {"model": MODEL, "input": "x", "stream": True}).status_code == 400
        assert post(server, {"model": MODEL}, path="/chat/completions").status_code == 404
        assert httpx.get(f"{server.base_url}/models", timeout=10).status_code == 404
    assert upstream.requests == []


def test_counts_refusals_and_incomplete_responses(metered):
    refusal = {"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}
    meter, _, server = metered(
        httpx.Response(200, json=completed(status="incomplete", output=[refusal]))
    )
    with server:
        post(server, {"model": MODEL, "input": "x"})
    assert meter.refusals == 1 and meter.incomplete == 1


def test_cache_write_rate_bills_uncached_input_at_the_higher_rate(metered):
    meter, _, server = metered(
        httpx.Response(200, json=completed(input_tokens=1000, cached=0, output_tokens=0)),
        pricing={**PRICES, "cache_write_per_million": 2.5},
    )
    with server:
        post(server, {"model": MODEL, "input": "x"})
    assert meter.spent == Decimal("0.0025")


@pytest.mark.parametrize(("extra", "status"), [(0, 200), (2, 402)])
def test_refuses_before_upstream_above_the_long_context_threshold(metered, tmp_path, extra, status):
    meter, upstream, server = metered(httpx.Response(200, json=completed()))
    body = {"model": MODEL, "input": "x" * (2 * 271_000 - len(MODEL) + extra)}
    assert estimate_input_tokens(body) == 272_000 + extra // 2
    with server:
        assert post(server, body).status_code == status
    assert len(upstream.requests) == meter.calls == int(status == 200)
    if status == 402:
        (entry,) = ledger(tmp_path)
        assert entry["event"] == "blocked" and "long-context threshold" in entry["reason"]


@pytest.mark.parametrize(
    ("input_tokens", "spent"),
    [
        # 222,000 uncached * $2 + 50,000 cached * $0.20 + 1,000 output * $10.
        (272_000, "0.464"),
        # Above the threshold: 222,001 uncached at the $5 cache-write rate, cached
        # $0.20, output $15.
        (272_001, "1.135005"),
    ],
)
def test_settles_above_the_threshold_at_long_context_rates(metered, tmp_path, input_tokens, spent):
    usage = completed(input_tokens=input_tokens, cached=50_000, output_tokens=1_000)
    meter, _, server = metered(httpx.Response(200, json=usage), long_context_pricing=LONG_PRICES)
    with server:
        assert post(server, {"model": MODEL, "input": "x"}).status_code == 200
    assert meter.spent == Decimal(spent) and not meter.uncertain
    (entry,) = ledger(tmp_path)
    assert entry["long_context"] is (input_tokens > 272_000)


def test_long_context_usage_without_long_context_rates_blocks(metered, tmp_path):
    meter, upstream, server = metered(httpx.Response(200, json=completed(input_tokens=300_000)))
    with server:
        assert post(server, {"model": MODEL, "input": "x"}).status_code == 200
        assert post(server, {"model": MODEL, "input": "x"}).status_code == 402
    assert meter.uncertain and len(upstream.requests) == 1
    call, blocked = ledger(tmp_path)
    assert call["long_context"] and call["billing_uncertain"] and blocked["event"] == "blocked"


def test_input_estimate_counts_images_not_their_base64():
    image = "data:image/png;base64," + "A" * 100_000
    body = {"input": [{"content": [{"type": "input_image", "image_url": image}]}], "x": "y" * 1000}
    text = len("input_image") + 1000
    assert estimate_input_tokens(body) == text // 2 + IMAGE_TOKEN_BOUND + 1000


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": 10},
        {"input_tokens": 10, "output_tokens": "5"},
        {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 11}, "output_tokens": 1},
    ],
)
def test_malformed_usage_is_unknown(usage):
    assert response_usage({"usage": usage}) is None


def test_config_resolves_paths_and_refuses_unsafe_values(tmp_path):
    loaded = RobotConfig.load(write_config(tmp_path))
    assert loaded.rig.log_root == tmp_path / "logs"
    for bad in (
        {"budget_usd": 5.01},
        {"model": ""},
        {"upstream": "http://api.openai.com/v1"},
        {"pricing": {**PRICES, "output_per_million": 0}},
        {"long_context_pricing": {**LONG_PRICES, "input_per_million": 0}},
        {"long_context_threshold": 0},
    ):
        with pytest.raises(ValidationError):
            RobotConfig.load(write_config(tmp_path, **bad))


def test_meter_settings_are_pinned_and_operator_overrides_are_parsed(tmp_path):
    loaded = RobotConfig.load(write_config(tmp_path))
    assert loaded.policy_args("http://127.0.0.1:1234/v1") == {
        "model": MODEL,
        "wire": "responses",
        "api_key_env": "OPENAI_API_KEY",
        # A string: None would leave thinking on.
        "effort": "low",
        "max_llm_calls": 5,
        # The policy sends the cap the meter enforces, and waits longer than the meter.
        "max_output_tokens": 1000,
        "timeout_s": 120.0,
        "base_url": "http://127.0.0.1:1234/v1",
    }
    assert robot_cli.overrides(["cam_width=448"], "-E") == {"cam_width": 448}
    protected = robot_cli.PROTECTED_AGENT_ARGS
    assert robot_cli.overrides(["image_horizon=3", "note='3'"], "-P", protected) == {
        "image_horizon": 3,
        "note": "3",
    }


@pytest.mark.parametrize(
    ("flag", "values"),
    [("-P", ["base_url=http://evil/v1"]), ("-P", ["model=other"]), ("-E", ["no-equals"])],
)
def test_overrides_refuse_meter_owned_or_malformed_keys(flag, values):
    with pytest.raises(ValueError):
        robot_cli.overrides(values, flag, robot_cli.PROTECTED_AGENT_ARGS)


def test_env_value_prefers_environment_then_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / "openai.env"
    env_file.write_text("# OPENAI_API_KEY=commented\nexport OPENAI_API_KEY='sk-file'\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert robot_cli.env_value("OPENAI_API_KEY", env_file) == "sk-file"
    assert robot_cli.env_value("OTHER_KEY", env_file) is None
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert robot_cli.env_value("OPENAI_API_KEY", env_file) == "sk-env"


def test_can_is_up_reads_interface_flags(tmp_path):
    for name, flags in (("can0", "0x40c1"), ("can1", "0x40c0")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "flags").write_text(flags + "\n")
    assert robot_cli.can_is_up("can0", tmp_path)
    assert not robot_cli.can_is_up("can1", tmp_path)
    assert not robot_cli.can_is_up("can2", tmp_path)


def run_args(config, *extra):
    return parser().parse_args(["robot", "run", "--config", str(config), *extra, "move the cup"])


@pytest.fixture
def interactive(monkeypatch):
    monkeypatch.setattr(stack, "installed", lambda: (True, "robot extra (fake)"))
    monkeypatch.setattr(robot_cli, "interactive_terminal", lambda: True)
    monkeypatch.setenv("TMUX", "/tmp/tmux-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")


def test_run_refuses_unattended_or_unapproved_starts(tmp_path, interactive, monkeypatch):
    config = write_config(tmp_path)
    with pytest.raises(ValueError, match="--allow-paid"):
        robot_cli.run(run_args(config))
    monkeypatch.delenv("TMUX")
    with pytest.raises(ValueError, match="tmux"):
        robot_cli.run(run_args(config, "--allow-paid"))
    monkeypatch.setattr(robot_cli, "interactive_terminal", lambda: False)
    with pytest.raises(ValueError, match="interactive terminal"):
        robot_cli.run(run_args(config, "--allow-paid", "--allow-no-tmux"))
    assert not (tmp_path / "logs").exists()


def fake_rig(monkeypatch, *, doctor="doctor ok", doctor_exit=0, run_exit=None, survive_402=False):
    """Stands in for the episode and the doctor in this process: ``doctor`` prints its
    report; an episode sends one Responses call to the agent's base URL with the key the
    harness set, as the agent policy would, with the arguments the run gives it. Returns
    every call: ``{"doctor": source}``, or an episode's arguments."""
    calls = []

    def fake_doctor(config, source):
        calls.append({"doctor": source})
        print(doctor)
        return doctor_exit

    def fake_episode(config, instruction, log_dir, **kwargs):
        calls.append({"instruction": instruction, "log_dir": log_dir, **kwargs})
        if run_exit is not None:
            if run_exit == 0:
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "standin.rrd").write_bytes(b"RRF2")
            return run_exit
        args = {**config.policy_args(kwargs.get("base_url")), **(kwargs.get("policy_args") or {})}
        if "base_url" not in args:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "standin.rrd").write_bytes(b"RRF2")
            return 0
        response = httpx.post(
            args["base_url"] + "/responses",
            json={"model": args["model"], "input": "x"},
            headers={"Authorization": "Bearer " + kwargs["env"][args["api_key_env"]]},
            timeout=10,
        )
        print(response.status_code)
        code = 0 if response.status_code == 200 or survive_402 else 1
        if code == 0:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "standin.rrd").write_bytes(b"RRF2")
        return code

    monkeypatch.setattr(stack, "episode", fake_episode)
    monkeypatch.setattr(stack, "doctor", fake_doctor)
    monkeypatch.setattr(stack, "installed", lambda: (True, "robot extra (fake)"))
    return calls


def test_run_drives_the_episode_through_the_meter(tmp_path, interactive, monkeypatch, capsys):
    record = fake_rig(monkeypatch)
    upstream = Upstream(httpx.Response(200, json=completed(input_tokens=1000, output_tokens=100)))
    monkeypatch.setattr(
        robot_cli,
        "MeterServer",
        lambda meter: MeterServer(meter, transport=httpx.MockTransport(upstream)),
    )
    config = write_config(tmp_path)
    assert robot_cli.run(run_args(config, "--allow-paid", "-E", "cam_width=448")) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    summary = lines[-1]["summary"]
    assert lines[0] == {"log_dir": summary["log_dir"]}
    assert summary["model_calls"] == 1 and summary["spent_usd"] == "0.003000"
    assert summary["exit_code"] == 0 and not summary["billing_uncertain"]
    log_dir = tmp_path / "logs" / summary["log_dir"].rsplit("/", 1)[-1]
    assert json.loads((log_dir / "meter-summary.json").read_text()) == summary
    assert upstream.requests[0].headers["Authorization"] == "Bearer sk-test-secret"
    (call,) = record
    assert call["log_dir"] == log_dir and call["source"] == "robot.toml"
    assert call["embodiment_args"] == {"cam_width": 448} and call["policy_args"] == {}
    assert call["base_url"].startswith("http://127.0.0.1:")
    # Every argument the components got stays with the run, and goes to R2 with it.
    setup = json.loads((log_dir / robot_cli.SETUP).read_text())
    assert (setup["policy"], setup["embodiment"]) == ("agent", "yam_arms")
    assert setup["policy_args"]["base_url"] == call["base_url"]
    assert setup["overrides"] == {"policy_args": {}, "embodiment_args": {"cam_width": 448}}


def event_pipe():
    """A pipe whose events a thread collects as serve would: (write end, events, reader)."""
    read_fd, write_fd = os.pipe()
    got = []

    def read():
        with os.fdopen(read_fd, "rb") as pipe:
            got.extend(json.loads(line) for line in pipe)

    reader = threading.Thread(target=read)
    reader.start()
    return write_fd, got, reader


def run_with_events(args):
    """``robot_cli.run(args)`` with ``--events-fd`` on a real pipe: (exit code or the
    exception it raised, the events)."""
    args.events_fd, got, reader = event_pipe()
    try:
        outcome = robot_cli.run(args)
    except ValueError as exc:
        outcome = exc
    reader.join(10)
    return outcome, got


@contextmanager
def run_events():
    write_fd, got, reader = event_pipe()
    try:
        with events.channel(write_fd):
            yield got
    finally:
        reader.join(10)


def test_no_verdict_reaches_the_episode_and_the_default_still_grades(
    tmp_path, interactive, monkeypatch
):
    calls = fake_rig(monkeypatch, run_exit=0)
    # One log root each: both runs may start within the second their directories are named by.
    for name, extra in (("demo", ["--no-verdict"]), ("graded", [])):
        (tmp_path / name).mkdir()
        assert robot_cli.run(run_args(write_config(tmp_path / name), "--allow-paid", *extra)) == 0
    assert [call["grade"] for call in calls] == [False, True]
    serve = parser().parse_args(["robot", "serve", "--config", "robot.toml", "--no-verdict"])
    assert serve.no_verdict is True


def test_auto_start_reaches_the_embodiment_and_serve_takes_the_flag(
    tmp_path, interactive, monkeypatch
):
    calls = fake_rig(monkeypatch, run_exit=0)
    config = write_config(tmp_path)
    assert robot_cli.run(run_args(config, "--allow-paid", "-E", "auto_start=true")) == 0
    assert calls[-1]["embodiment_args"] == {"auto_start": True}
    serve = parser().parse_args(["robot", "serve", "--config", "robot.toml", "--auto-start"])
    assert serve.auto_start is True
    assert parser().parse_args(["robot", "serve", "--config", "robot.toml"]).auto_start is False


def test_run_events_follow_the_run_and_leave_stdout_alone(
    tmp_path, interactive, monkeypatch, capsys
):
    calls = fake_rig(monkeypatch, run_exit=0)
    active = []
    episode = stack.episode
    monkeypatch.setattr(
        stack, "episode", lambda *a, **k: active.append(events.active()) or episode(*a, **k)
    )
    # One log root each: both runs may start within the second their directories are named by.
    plain_root, events_root = tmp_path / "plain", tmp_path / "events"
    plain_root.mkdir()
    events_root.mkdir()
    assert robot_cli.run(run_args(write_config(plain_root), "--allow-paid")) == 0
    plain = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]

    code, got = run_with_events(run_args(write_config(events_root), "--allow-paid"))
    assert code == 0 and active == [False, True] and len(calls) == 2
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [list(line) for line in lines] == [list(line) for line in plain]  # same lines
    assert [event["type"] for event in got] == ["hello", "log_dir", "summary"]
    assert got[1]["path"] == lines[0]["log_dir"]
    assert got[2]["summary"] == lines[-1]["summary"]


def test_a_refused_run_tells_its_supervisor_why(tmp_path, interactive):
    refusal, got = run_with_events(run_args(write_config(tmp_path)))
    assert isinstance(refusal, ValueError)
    assert [event["type"] for event in got] == ["hello", "error"]
    assert "--allow-paid" in got[1]["message"]


def test_uploads_are_events_too(capsys):
    record = {"ok": True, "run": "20261012-000000", "prefix": "yam-1/2026/10/12/x/"}
    with run_events() as got:
        robot_cli.report_upload(record, flush=True)
    assert json.loads(capsys.readouterr().out) == {"upload": record}
    assert got[1] == {"t": got[1]["t"], "type": "upload", "upload": record}


@pytest.fixture
def checkable(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    monkeypatch.setattr(robot_cli, "can_is_up", lambda channel: True)
    armed = {str(motor_id): 8000 for motor_id in stack.YAM_MOTOR_IDS}
    monkeypatch.setattr(stack, "motor_timeouts", lambda channel: dict(armed))
    monkeypatch.setattr(stack, "realsense_devices", lambda: [])
    seen = []

    def fake_get(url, headers, timeout):
        seen.append((url, headers["Authorization"]))
        return httpx.Response(200, json={"id": MODEL})

    monkeypatch.setattr(robot_cli.httpx, "get", fake_get)
    return seen


def check_args(config):
    return parser().parse_args(["robot", "check", "--config", str(config)])


def test_check_is_motion_free_and_rehearses_the_agent(tmp_path, checkable, monkeypatch, capsys):
    record = fake_rig(monkeypatch)
    assert robot_cli.check(check_args(write_config(tmp_path))) == 0
    doctor, *rehearsals = record
    assert doctor == {"doctor": "robot.toml"}
    assert len(rehearsals) == 2
    for rehearsal in rehearsals:
        assert rehearsal["embodiment"] == "cubepick" and rehearsal["prompt"] is False
        assert rehearsal["base_url"].startswith("http://127.0.0.1:")
        assert "embodiment_args" not in rehearsal
    assert checkable == [(f"https://api.openai.com/v1/models/{MODEL}", "Bearer sk-test-secret")]
    output = capsys.readouterr().out
    assert '"ready": true' in output and "sk-test-secret" not in output


def test_check_fails_when_rehearsal_or_doctor_fails(tmp_path, checkable, monkeypatch, capsys):
    fake_rig(monkeypatch, doctor="  [error] bounds: missing", doctor_exit=1, run_exit=3)
    assert robot_cli.check(check_args(write_config(tmp_path))) == 1
    results = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    failed = {r["check"] for r in results if "check" in r and not r["ok"]}
    assert failed == {"doctor", "rehearsal"}


def rehearse(path):
    return robot_cli.rehearse(RobotConfig.load(path))


def test_rehearsal_proves_the_budget_refusal(tmp_path, monkeypatch):
    record = fake_rig(monkeypatch)
    ok, detail = rehearse(write_config(tmp_path))
    assert ok, detail
    assert detail == (
        "done: exit 0, 1 call(s) through the meter; "
        "budget refusal: exit 1, 0 upstream call(s), 1 blocked"
    )
    # The stand-in policy found base_url, model and the stand-in key's name in the
    # arguments the run gave it, or it would not have reached the meter.
    assert len(record) == 2


def test_rehearsal_fails_when_the_agent_survives_a_402(tmp_path, monkeypatch):
    fake_rig(monkeypatch, survive_402=True)
    ok, detail = rehearse(write_config(tmp_path))
    assert not ok
    assert "budget refusal: exit 0, 0 upstream call(s), 1 blocked" in detail
    assert "budget refusal output" in detail and "done output" not in detail


def test_the_committed_agent_rig_passes_the_doctor():
    # yam_arms in eef_pos mode declares a 14-D joint_pos and a 14-D eef_state; the
    # doctor resolves them as the agent policy does, so check need not tolerate a
    # state_alignment error. Built only, never connected. The doctor
    # also checks the arm's runtime requirements, so this needs the robot extra.
    for module in ("numpy", "cv2", "i2rt"):
        pytest.importorskip(module)
    config = RobotConfig.load(Path(__file__).resolve().parents[1] / "rigs" / "yam-agent.toml")
    code, output = stack.captured(stack.doctor, config, "yam-agent.toml")
    assert code == 0, output
    assert "[error]" not in output


def test_cli_refuses_paid_run_without_flag(tmp_path):
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "abstract_harness.cli",
            "robot",
            "run",
            "--config",
            str(write_config(tmp_path)),
            "move the cup",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert run.returncode == 2 and "--allow-paid" in run.stderr


def test_check_fails_when_a_motor_has_no_loss_of_comms_failsafe(
    tmp_path, checkable, monkeypatch, capsys
):
    fake_rig(monkeypatch)

    def timeouts(channel):
        values = {str(motor_id): 8000 for motor_id in stack.YAM_MOTOR_IDS}
        if channel == "can0":
            values["3"] = 0
        return values

    monkeypatch.setattr(stack, "motor_timeouts", timeouts)
    assert robot_cli.check(check_args(write_config(tmp_path))) == 1
    results = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    failed = {r["check"]: r["detail"] for r in results if "check" in r and not r["ok"]}
    assert list(failed) == ["failsafe:can0"]
    assert "set_timeout.py --channel can0 --timeout" in failed["failsafe:can0"]


@pytest.mark.parametrize(
    ("values", "armed"),
    [
        ({str(i): 8000 for i in range(1, 8)}, True),
        ({**{str(i): 8000 for i in range(1, 8)}, "7": 0}, False),
        ({**{str(i): 8000 for i in range(1, 8)}, "2": "CanError"}, False),
        ({str(i): 8000 for i in range(1, 7)}, False),
        ({"read": "ModuleNotFoundError"}, False),
    ],
)
def test_failsafe_needs_a_positive_timeout_on_all_seven_motors(values, armed):
    assert robot_cli.failsafe_armed(values) is armed


def test_motor_timeouts_reports_an_unreadable_bus():
    # No such CAN interface (or no i2rt without the robot extra): nothing is read.
    values = stack.motor_timeouts("abstract-test-can9")
    assert not robot_cli.failsafe_armed(values)
    assert all(isinstance(value, str) for value in values.values())


CAMERA_RIG = (
    RIG
    + """top_depth_serial = "000000000001"
left_depth_serial = "000000000002"
right_depth_serial = "none"
"""
)


def realsense(serial, usb="3.2"):
    return {"serial": serial, "name": "Intel RealSense D405", "usb": usb}


@pytest.mark.parametrize(
    ("devices", "failed", "warned"),
    [
        ([realsense("000000000001"), realsense("000000000002")], set(), set()),
        # USB descriptor serials are not librealsense serials.
        (
            [realsense("000000000001", usb="2.1"), realsense("0123456789AB")],
            {"camera:left"},
            {"camera:top"},
        ),
        (
            "ModuleNotFoundError: No module named 'pyrealsense2'",
            {"camera:top", "camera:left"},
            set(),
        ),
    ],
    ids=["all-present", "missing-and-usb2", "list-failed"],
)
def test_check_verifies_the_profiles_realsense_serials(
    tmp_path, checkable, monkeypatch, capsys, devices, failed, warned
):
    fake_rig(monkeypatch)
    config = write_config(tmp_path, rig=CAMERA_RIG)
    listed = []
    monkeypatch.setattr(stack, "realsense_devices", lambda: listed.append(1) or devices)
    code = robot_cli.check(check_args(config))
    results = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    checks = {r["check"]: r for r in results if "check" in r}
    assert listed == [1]
    assert {name for name, r in checks.items() if not r["ok"]} == failed
    assert {name for name, r in checks.items() if "warning" in r} == warned
    assert [name for name in checks if name.startswith("camera:")] == ["camera:top", "camera:left"]
    assert code == int(bool(failed))


class FakeDevice:
    def __init__(self, **info):
        self.info = info

    def supports(self, key):
        return key in self.info

    def get_info(self, key):
        return self.info[key]


def fake_librealsense(devices):
    def context():
        if isinstance(devices, Exception):
            raise devices
        return SimpleNamespace(query_devices=lambda: devices)

    keys = SimpleNamespace(serial_number="serial", name="name", usb_type_descriptor="usb")
    return SimpleNamespace(camera_info=keys, context=context)


@pytest.mark.parametrize(
    ("librealsense", "expected"),
    [
        (
            fake_librealsense([FakeDevice(serial="1", usb="3.2")]),
            [{"serial": "1", "name": None, "usb": "3.2"}],
        ),
        (fake_librealsense(RuntimeError("no device")), "RuntimeError: no device"),
        (None, "ModuleNotFoundError"),
    ],
    ids=["listed", "failed", "not-installed"],
)
def test_realsense_devices_lists_without_streaming(monkeypatch, librealsense, expected):
    # A stand-in librealsense: no camera is touched.
    monkeypatch.setitem(sys.modules, "pyrealsense2", librealsense)
    devices = stack.realsense_devices()
    if isinstance(expected, str):
        assert isinstance(devices, str) and devices.startswith(expected)
    else:
        assert devices == expected


def test_effort_none_reaches_the_wire_as_a_string(tmp_path):
    # None would leave thinking on.
    loaded = RobotConfig.load(write_config(tmp_path, effort="none"))
    assert loaded.policy_args()["effort"] == "none"


def write_vla_config(tmp_path, *, server_url="http://policy-host:8202"):
    path = tmp_path / "robot-vla.toml"
    path.write_text(f'[vla]\npolicy = "molmoact2"\nserver_url = "{server_url}"\n' + RIG)
    return path


def test_config_needs_exactly_one_policy(tmp_path):
    vla = RobotConfig.load(write_vla_config(tmp_path))
    assert vla.agent is None and vla.vla.server_url == "http://policy-host:8202"
    both = write_config(tmp_path)
    both.write_text(both.read_text() + '[vla]\nserver_url = "http://127.0.0.1:8202"\n')
    neither = tmp_path / "neither.toml"
    neither.write_text(RIG)
    for path in (both, neither):
        with pytest.raises(ValidationError, match="exactly one policy"):
            RobotConfig.load(path)


@pytest.mark.parametrize(
    "url",
    [
        "policy-host:8202",
        "ftp://h:8202",
        "http://h:8202/act",
        "http://u:p@h:8202",
        "http://h?x=1",
    ],
)
def test_vla_server_url_must_be_a_plain_base_url(tmp_path, url):
    with pytest.raises(ValidationError, match="server_url"):
        RobotConfig.load(write_vla_config(tmp_path, server_url=url))


def test_vla_config_drives_the_yam_act_client(tmp_path):
    config = RobotConfig.load(write_vla_config(tmp_path))
    assert config.policy_args() == {"server_url": "http://policy-host:8202"}
    setup = config.setup()
    assert (setup["policy"], setup["embodiment"], setup["max_steps"]) == (
        "molmoact2",
        "yam_arms",
        1200,
    )
    with pytest.raises(ValueError, match="server_url"):
        robot_cli.overrides(["server_url=http://other:1"], "-P", robot_cli.PROTECTED_VLA_ARGS)


def test_vla_run_needs_no_paid_flag_key_or_meter(tmp_path, interactive, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    record = fake_rig(monkeypatch)
    config = write_vla_config(tmp_path)
    args = parser().parse_args(["robot", "run", "--config", str(config), "-P", "num_steps=5", "x"])
    assert robot_cli.run(args) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["summary"]
    assert summary["policy"] == "molmoact2" and summary["exit_code"] == 0
    log_dir = tmp_path / "logs" / summary["log_dir"].rsplit("/", 1)[-1]
    assert json.loads((log_dir / "run-summary.json").read_text()) == summary
    assert not (log_dir / "meter.jsonl").exists()
    (call,) = record
    assert call["policy_args"] == {"num_steps": 5} and call["source"] == config.name
    assert "env" not in call and "base_url" not in call
    # Every argument the components got stays with the run, and goes to R2 with it.
    assert json.loads((log_dir / robot_cli.SETUP).read_text()) == {
        **RobotConfig.load(config).setup(),
        "overrides": {"policy_args": {"num_steps": 5}, "embodiment_args": {}},
    }


def test_vla_check_probes_the_server_instead_of_rehearsing(
    tmp_path, checkable, monkeypatch, capsys
):
    record = fake_rig(monkeypatch)
    probes = []

    def probe(config):
        probes.append(config)
        return {
            "compat_errors": [],
            "compat_warnings": [],
            "chunk_shape": [30, 14],
            "latency_s": 0.4,
            "first_step_rad": 0.02,
            "outside_bounds": 0,
        }

    monkeypatch.setattr(stack, "vla_probe", probe)
    config = write_vla_config(tmp_path)
    assert robot_cli.check(check_args(config)) == 0
    results = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    names = [r["check"] for r in results if "check" in r]
    assert "vla:molmoact2" in names
    assert not {"api_key", "model", "rehearsal"} & set(names)
    (probed,) = probes
    assert probed.vla.server_url == "http://policy-host:8202"
    assert probed.embodiment_args()["left_channel"] == "can1"
    assert [list(call) for call in record] == [["doctor"]]


@pytest.mark.parametrize(
    ("out", "ok", "warns"),
    [
        ({"chunk_shape": [30, 14], "latency_s": 0.4, "first_step_rad": 0.02}, True, False),
        ({"chunk_shape": [30, 14], "outside_bounds": 3}, True, True),
        ({"chunk_shape": [30, 14], "compat_warnings": ["rate: 15 Hz vs 10 Hz"]}, True, True),
        ({"chunk_shape": [30, 7]}, False, False),
        ({"chunk_shape": []}, False, False),
        ({"compat_errors": ["action_dim: 7 vs 14"], "chunk_shape": [30, 14]}, False, False),
        ({"error": "ConnectError: refused"}, False, False),
    ],
)
def test_vla_probe_result(out, ok, warns):
    passed, detail, warning = robot_cli.vla_probe_result(out)
    assert passed is ok and bool(warning) is warns and detail


def test_vla_probe_reports_a_missing_yam_client_before_any_network_call(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "abstract_harness.robot.yam.config", None)
    config = RobotConfig.load(write_vla_config(tmp_path))
    out = stack.vla_probe(config)
    assert "error" in out and not robot_cli.vla_probe_result(out)[0]


RIGS = Path(__file__).resolve().parents[1] / "rigs"


@pytest.mark.parametrize("name", ["yam-agent.toml", "yam-vla.toml"])
def test_committed_rig_configs_build_their_components(name, monkeypatch):
    config = RobotConfig.load(RIGS / name)
    # The examples keep runs local: their [store] sections are commented out.
    assert config.rig.name == "yam-1" and config.store is None
    assert robot_cli.depth_serials(config.rig) == {
        "top": "000000000001",
        "left": "000000000002",
        "right": "000000000003",
    }
    embodiment = config.embodiment_args()
    # Serials stay strings, numbers and booleans keep their types, the CAN wiring holds.
    assert embodiment["top_depth_serial"] == "000000000001"
    assert (embodiment["left_channel"], embodiment["right_channel"]) == ("can1", "can0")
    assert embodiment["zero_gravity_mode"] is False and embodiment["rest_secs"] == 6.0
    assert embodiment["motor_temp_limit"] == 70
    if "joint_low" in config.rig.embodiment:
        # A list reaches the embodiment as one comma-separated string, as -E would pass it.
        assert embodiment["joint_low"].split(", ")[:2] == ["-2.618", "0.0"]
    # The registry builds both from these arguments; neither touches hardware or network
    # until reset.
    registry = pytest.importorskip("abstract_harness.robot.core.registry")
    yam = pytest.importorskip("abstract_harness.robot.yam.config")
    cfg = yam.YamConfig.from_kwargs(**embodiment)
    assert (cfg.left_channel, cfg.right_channel) == ("can1", "can0")
    assert cfg.top_depth_serial == "000000000001" and cfg.zero_gravity_mode is False
    if "joint_low" in config.rig.embodiment:
        assert tuple(cfg.joint_low[:2]) == (-2.618, 0.0)
    registry.resolve("embodiment", "yam_arms", **embodiment).close()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    policy = registry.resolve("policy", config.policy_name, **config.policy_args("http://x/v1"))
    assert policy.info.name == config.policy_name


@pytest.mark.parametrize(
    ("rig", "section", "match"),
    [
        ('[rig.embodiment]\nleft_channel = "can1"\n', "", "left_channel and right_channel"),
        (RIG + 'serials = ["a,b"]\n', "", "cannot contain a comma"),
        (RIG, '[agent.args]\neffort = "high"\n', "harness sets them"),
        (RIG, "[agent.args]\nNested = 1\n", "pattern"),
    ],
)
def test_config_refuses_args_a_component_would_misread(tmp_path, rig, section, match):
    path = write_config(tmp_path, rig=rig)
    path.write_text(path.read_text() + section)
    with pytest.raises(ValidationError, match=match):
        RobotConfig.load(path)


def test_yam_tools_say_when_the_robot_extra_is_missing(tmp_path, monkeypatch):
    from abstract_harness.robot import tools

    def missing(name):
        def modules():
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

        return modules

    monkeypatch.setattr(tools, "_modules", missing("numpy"))
    config = str(write_vla_config(tmp_path))
    args = parser().parse_args(["robot", "holdcheck", "--config", config, "left", "-x", "1"])
    with pytest.raises(ValueError, match="uv sync --extra robot"):
        robot_cli.main(args)
    # Anything else missing is a bug, not a missing extra.
    monkeypatch.setattr(tools, "_modules", missing("abstract_harness.robot.yam.oops"))
    with pytest.raises(ModuleNotFoundError):
        parser()
