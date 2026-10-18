import base64
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from abstract_harness.cli import parser
from abstract_harness.robot import cli as robot_cli
from abstract_harness.robot import cloud, stack, store
from abstract_harness.robot.config import RobotConfig, StoreConfig
from abstract_harness.robot.episode import CATALOG, EPISODE, card

BUCKET = "my-robot-runs"
CATALOG_KEY = f"yam-1/{CATALOG}"
JOINTS = [
    f"{arm}_{axis}"
    for arm in ("left", "right")
    for axis in (*(f"j{i}" for i in range(6)), "gripper")
]

ACCOUNT = "0123456789abcdef0123456789abcdef"
RUN = "20261011-231136"
PREFIX = f"yam-1/2026/10/11/{RUN}/"


class FakeR2:
    """An R2 bucket over its S3 API: checks what a real one checks and keeps the objects."""

    def __init__(self, failures=(), corrupt=(), stale=()):
        self.objects = {}
        self.requests = []
        self.failures = list(failures)  # (method, key suffix, status), consumed in order
        self.corrupt = set(corrupt)  # key suffixes stored with a flipped byte
        self.stale = set(stale)  # key suffixes whose GET body no longer matches the ETag

    def key(self, request):
        prefix = f"/{BUCKET}/"
        assert request.url.host == f"{ACCOUNT}.r2.cloudflarestorage.com"
        assert request.url.raw_path.decode().startswith(prefix)
        return request.url.raw_path.decode().removeprefix(prefix)

    def __call__(self, request):
        key = self.key(request)
        self.requests.append((request.method, key))
        assert "/auto/s3/aws4_request" in request.headers["authorization"]
        for failure in self.failures:
            method, suffix, status = failure
            if request.method == method and key.endswith(suffix):
                self.failures.remove(failure)
                return httpx.Response(status)
        if request.method == "PUT":
            body = request.read()
            assert request.headers["x-amz-content-sha256"] == hashlib.sha256(body).hexdigest()
            md5 = hashlib.md5(body).digest()
            assert base64.b64decode(request.headers["content-md5"]) == md5
            assert int(request.headers["content-length"]) == len(body)
            if any(key.endswith(suffix) for suffix in self.corrupt):
                body = bytes([body[0] ^ 1]) + body[1:]
            self.objects[key] = body
            return httpx.Response(200, headers={"etag": f'"{hashlib.md5(body).hexdigest()}"'})
        if request.method == "HEAD":
            body = self.objects.get(key)
            if body is None:
                return httpx.Response(404)
            etag = f'"{hashlib.md5(body).hexdigest()}"'
            if request.headers.get("accept-encoding") != "identity":
                # What R2 does for compressible content and a client that accepts gzip.
                return httpx.Response(200, headers={"etag": f"W/{etag}"})
            return httpx.Response(200, headers={"etag": etag, "content-length": str(len(body))})
        if request.method == "GET":
            body = self.objects.get(key)
            if body is None:
                return httpx.Response(404)
            etag = f'"{hashlib.md5(body).hexdigest()}"'
            if any(key.endswith(suffix) for suffix in self.stale):
                body = body + b" "
            return httpx.Response(200, content=body, headers={"etag": etag})
        return httpx.Response(405)

    def puts(self):
        return [key for method, key in self.requests if method == "PUT"]


def bucket(fake, sleeps=None):
    return cloud.Bucket(
        ACCOUNT,
        "AKIDEXAMPLE",
        "runs-secret",
        BUCKET,
        client=httpx.Client(transport=httpx.MockTransport(fake)),
        clock=lambda: datetime(2026, 10, 12, tzinfo=UTC),
        sleep=(sleeps.append if sleeps is not None else lambda seconds: None),
    )


def write_actions(path, labels, actions):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"kind": "header", "labels": labels}]
    rows += [{"t": t, "action": action} for t, action in enumerate(actions)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_run(log_root, name=RUN, summary=True, tags=()):
    """A VLA run as an episode leaves it: three steps in which the right arm turns and
    its gripper closes then opens again, one operator message, a "y" verdict."""
    run = log_root / name
    trial = "20261011_231136_0cdd0718"
    (run / "frames" / trial).mkdir(parents=True)
    still = [0.0] * 6 + [1.0]
    write_actions(
        run / "actions" / trial / "scene-0-e0.jsonl",
        JOINTS,
        [still + still, still + [0.3, *[0.0] * 5, 0.1], still + [0.3, *[0.0] * 5, 0.9]],
    )
    for camera in ("left_cam", "top_cam"):
        (run / "frames" / trial / f"scene-0-e0_{camera}_000000.npy").write_bytes(b"\0" * 4096)
    (run / "adhoc_9c3f3481.rrd").write_bytes(b"RRF2" + b"\1" * 2000)
    eval_log = {
        "eval": {
            "policy": "isaac-0.5",
            "embodiment_info": {"control_hz": 30},
            "git_commit": "947d462",
            "harness_version": "0.1.0",
        },
        "results": {"metrics": {"operator": 1.0}},
        "samples": [
            {
                "instruction": "Fold the cloth",
                "termination_reasons": ["operator_end"],
                "operator_judgements": ["y"],
                "operator_notes": ["folded once"],
                "operator_messages": [[{"t": 1, "text": "slower", "source": "console"}]],
                "reduced": {"operator": 1.0},
                "trial_metadata": [{"actions": f"actions/{trial}/scene-0-e0.jsonl"}],
            }
        ],
        "stats": {"total_steps": 3, "duration_s": 0.123, "started_at": "2026-10-11T23:11:36Z"},
        "status": "success",
    }
    (run / "adhoc_9c3f3481.json").write_text(json.dumps(eval_log))
    if summary:
        (run / "run-summary.json").write_text(
            json.dumps(
                {
                    "policy": "molmoact2",
                    "server_url": "http://127.0.0.1:8204",
                    "config": "robot.isaac.local.toml",
                    "tags": list(tags),
                    "exit_code": 0,
                }
            )
        )
    return run


def cloud_config(prefix="yam-1"):
    return StoreConfig(route="bucket", bucket=BUCKET, prefix=prefix)


def bucket_store(r2):
    """The bucket route of [store] for the yam-1 rig, on a fake R2."""
    return store.BucketStore(cloud_config(), "yam-1", r2)


def offload(r2, log_root, run):
    return store.offload(bucket_store(r2), log_root, run)


def test_signs_the_aws_documented_example():
    # "Example: GET Object" in AWS's Signature Version 4 documentation for S3.
    headers = cloud.sigv4_headers(
        "GET",
        "https://examplebucket.s3.amazonaws.com/test.txt",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        region="us-east-1",
        service="s3",
        now=datetime(2013, 5, 24, tzinfo=UTC),
        headers={"Range": "bytes=0-9"},
    )
    assert headers["authorization"] == (
        "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, "
        "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
    )


def test_uploads_a_run_with_packed_frames_then_deletes_it(tmp_path):
    log_root = tmp_path / "logs"
    run = make_run(log_root)
    fake = FakeR2()
    with bucket(fake) as r2:
        record = offload(r2, log_root, run)
    assert run.is_dir() and record["local_retained"] is True
    assert set(fake.objects) == {
        PREFIX + "run-summary.json",
        PREFIX + "adhoc_9c3f3481.json",
        PREFIX + "adhoc_9c3f3481.rrd",
        PREFIX + "actions/20261011_231136_0cdd0718/scene-0-e0.jsonl",
        PREFIX + EPISODE,
        PREFIX + "manifest.json",
        CATALOG_KEY,
    }
    # The manifest goes after the run's other objects (its presence marks a complete run),
    # then the run's line in the catalog.
    assert fake.puts()[-2:] == [PREFIX + "manifest.json", CATALOG_KEY]
    manifest = json.loads(fake.objects[PREFIX + "manifest.json"])
    for item in manifest["files"]:
        body = fake.objects[PREFIX + item["path"]]
        assert item["bytes"] == len(body) and item["sha256"] == hashlib.sha256(body).hexdigest()
    # The raw frames stay out of the bucket; the recording carries the cameras.
    assert not any("frames" in key for key in fake.objects)
    (line,) = (log_root / cloud.INDEX).read_text().splitlines()
    kept = json.loads(line)
    assert kept == record and kept["prefix"] == PREFIX and kept["objects"] == 6
    assert kept["frames"] == {"npy": 2} and kept["sent"] == 6
    # The episode labels: what was asked, which policy acted and how it ended.
    assert kept["instruction"] == "Fold the cloth" and kept["task"] == "fold-the-cloth"
    assert (kept["policy"], kept["kind"], kept["client"]) == ("isaac-0.5", "vla", "molmoact2")
    assert (kept["outcome"], kept["score"], kept["verdict"]) == ("success", 1.0, "y")
    assert kept["note"] == "folded once" and kept["config"] == "robot.isaac.local.toml"
    assert kept["arms_moved"] == ["right"] and kept["gripper_closes"] == {"left": 0, "right": 1}
    assert kept["displacement"] == {"left": 0.0, "right": 0.3} and kept["duration_s"] == 0.1
    (row,) = [json.loads(line) for line in fake.objects[CATALOG_KEY].decode().splitlines()]
    assert row == {
        key: value
        for key, value in record.items()
        if key not in ("sent", "local_retained", "retention_reason")
    }
    stored = json.loads(fake.objects[PREFIX + EPISODE])
    assert [(m["t"], m["kind"]) for m in stored["markers"]] == [
        (1, "gripper_close"),
        (1, "operator_message"),
        (2, "gripper_open"),
        (2, "end"),
    ]
    assert row["markers"] == 4


def test_a_failed_upload_keeps_the_run_and_the_retry_sends_only_what_is_missing(tmp_path):
    log_root = tmp_path / "logs"
    run = make_run(log_root)
    fake = FakeR2(failures=[("PUT", ".rrd", 403)])
    with bucket(fake) as r2, pytest.raises(cloud.UploadError, match="HTTP 403"):
        record = offload(r2, log_root, run)
    assert run.is_dir() and len(list((run / "frames").rglob("*.npy"))) == 2
    assert PREFIX + "manifest.json" not in fake.objects
    assert not (log_root / cloud.INDEX).exists()
    already = set(fake.objects)
    fake.requests.clear()
    with bucket(fake) as r2:
        record = offload(r2, log_root, run)
    assert run.is_dir() and record["local_retained"] is True and record["sent"] == 6 - len(already)
    assert record["frames"] == {"npy": 2}
    assert not already & set(fake.puts())


def test_transient_errors_are_retried_with_backoff(tmp_path):
    log_root = tmp_path / "logs"
    run = make_run(log_root)
    fake = FakeR2(failures=[("PUT", ".rrd", 503), ("HEAD", "manifest.json", 500)])
    sleeps = []
    with bucket(fake, sleeps) as r2:
        record = offload(r2, log_root, run)
    assert run.is_dir() and record["local_retained"] is True and sleeps == [2, 2]


def test_retries_give_up_and_keep_the_run(tmp_path):
    log_root = tmp_path / "logs"
    run = make_run(log_root)
    fake = FakeR2(failures=[("PUT", ".rrd", 503)] * cloud.ATTEMPTS)
    with bucket(fake) as r2, pytest.raises(cloud.UploadError, match="503"):
        offload(r2, log_root, run)
    assert run.is_dir()


def test_an_object_the_bucket_holds_differently_is_not_trusted(tmp_path):
    log_root = tmp_path / "logs"
    run = make_run(log_root)
    fake = FakeR2(corrupt=[".rrd"])
    with bucket(fake) as r2, pytest.raises(cloud.UploadError, match="bucket holds"):
        offload(r2, log_root, run)
    assert run.is_dir() and PREFIX + "manifest.json" not in fake.objects


def test_a_run_still_being_written_is_left_alone(tmp_path):
    log_root = tmp_path / "logs"
    fresh = make_run(log_root, "20261011-231500", summary=False)
    stale = make_run(log_root, "20261011-200000", summary=False)
    done = make_run(log_root, RUN)
    old = time.time() - cloud.QUIET_SECONDS - 60
    for path in [stale, *stale.rglob("*")]:
        os.utime(path, (old, old))
    (log_root / "notes").mkdir()
    assert cloud.finished_runs(log_root) == [stale, done]
    with bucket(FakeR2()) as r2, pytest.raises(cloud.UploadError, match="still being written"):
        offload(r2, log_root, fresh)
    assert fresh.is_dir()


def test_deletes_only_run_directories_in_the_log_root(tmp_path):
    log_root = tmp_path / "logs"
    log_root.mkdir()
    elsewhere = make_run(tmp_path / "other")
    with pytest.raises(cloud.UploadError, match="refusing to delete"):
        cloud.remove_run(log_root, elsewhere, {})
    link = log_root / "20261011-000000"
    link.symlink_to(elsewhere)
    with pytest.raises(cloud.UploadError, match="refusing to delete"):
        cloud.remove_run(log_root, link, {})
    assert elsewhere.is_dir()


def test_one_uploader_per_log_root(tmp_path):
    with cloud.upload_lock(tmp_path):
        with pytest.raises(cloud.UploadError, match="another upload"):
            with cloud.upload_lock(tmp_path):
                pass


def write_cloud_config(tmp_path, extra="", bucket=BUCKET):
    path = tmp_path / "robot-vla.toml"
    bucket_line = f'bucket = "{bucket}"\n' if bucket else ""
    path.write_text(
        '[vla]\npolicy = "molmoact2"\nserver_url = "http://policy-host:8202"\n'
        '[rig]\nname = "yam-1"\nlog_root = "logs"\n[rig.embodiment]\nleft_channel = "can1"\n'
        'right_channel = "can0"\n'
        f'[store]\nroute = "bucket"\n{bucket_line}env_file = "r2.env"\n{extra}'
    )
    return path


@pytest.fixture
def episode(monkeypatch):
    """The episode, replaced by a stand-in that leaves in its log directory what a real
    episode does; no robot extra or arms needed."""
    calls = []

    def fake(config, instruction, log_dir, **kwargs):
        calls.append(kwargs)
        log = Path(log_dir)
        (log / "frames" / "trial").mkdir(parents=True)
        (log / "frames" / "trial" / "scene-0-e0_top_cam_000000.npy").write_bytes(b"x" * 64)
        (log / "adhoc_1.rrd").write_bytes(b"RRF2")
        return 0

    monkeypatch.setattr(stack, "episode", fake)
    monkeypatch.setattr(stack, "installed", lambda: (True, "robot extra (fake)"))
    monkeypatch.setattr(robot_cli, "interactive_terminal", lambda: True)
    monkeypatch.setenv("TMUX", "/tmp/tmux-test")
    return calls


def test_config_resolves_the_env_file_and_needs_a_bucket(tmp_path):
    loaded = RobotConfig.load(write_cloud_config(tmp_path))
    assert loaded.store.bucket == BUCKET and loaded.store.prefix is None
    assert loaded.store.env_file == tmp_path / "r2.env"
    # The bucket route has no default bucket: the user names one of their own.
    with pytest.raises(ValidationError, match='route = "bucket" needs bucket'):
        RobotConfig.load(write_cloud_config(tmp_path, bucket=None))
    with pytest.raises(ValidationError, match='route = "bucket" needs bucket'):
        StoreConfig(route="bucket")
    with pytest.raises(ValidationError, match="prefix"):
        StoreConfig(route="bucket", bucket=BUCKET, prefix="yam-1/../x")


def test_credentials_come_from_the_env_file_and_are_never_printed(tmp_path, monkeypatch):
    for name in cloud.CREDENTIAL_ENVS:
        monkeypatch.delenv(name, raising=False)
    config = RobotConfig.load(write_cloud_config(tmp_path))
    with pytest.raises(cloud.UploadError, match="R2_ACCOUNT_ID, R2_RUNS_ACCESS_KEY_ID"):
        robot_cli.connect_bucket(config.store)
    (tmp_path / "r2.env").write_text(
        f"R2_ACCOUNT_ID={ACCOUNT}\nR2_RUNS_ACCESS_KEY_ID=key-id\n"
        "R2_RUNS_SECRET_ACCESS_KEY=runs-secret\n"
    )
    with robot_cli.connect_bucket(config.store) as r2:
        assert r2.name == BUCKET
    (tmp_path / "r2.env").write_text(
        "R2_ACCOUNT_ID=evil.example/x\nR2_RUNS_ACCESS_KEY_ID=a\nR2_RUNS_SECRET_ACCESS_KEY=b\n"
    )
    with pytest.raises(ValueError, match="32-character"):
        robot_cli.connect_bucket(config.store)


@pytest.fixture
def r2(monkeypatch):
    fake = FakeR2()
    monkeypatch.setattr(robot_cli, "connect_bucket", lambda config: bucket(fake))
    return fake


def test_upload_command_backfills_every_finished_run(tmp_path, r2, capsys):
    config = write_cloud_config(tmp_path)
    log_root = tmp_path / "logs"
    runs = [make_run(log_root, name) for name in ("20261011-205353", RUN)]
    args = parser().parse_args(["robot", "upload", "--config", str(config)])
    assert robot_cli.main(args) == 0
    lines = [json.loads(line)["upload"] for line in capsys.readouterr().out.splitlines()]
    assert [line["run"] for line in lines] == [run.name for run in runs]
    assert all(line["ok"] for line in lines)
    assert all(run.exists() for run in runs)
    assert len((log_root / cloud.INDEX).read_text().splitlines()) == 2
    named = parser().parse_args(["robot", "upload", "--config", str(config), "../etc"])
    with pytest.raises(ValueError, match="no run directory"):
        robot_cli.main(named)


def test_vla_run_uploads_after_the_episode(tmp_path, r2, episode, capsys):
    config = write_cloud_config(tmp_path)
    assert robot_cli.main(parser().parse_args(["robot", "run", "--config", str(config), "x"])) == 0
    (call,) = episode
    assert call["source"] == "robot-vla.toml"
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    # Matched by key by `abstract robot serve`: log_dir first, summary after the park,
    # then the upload, after which the run directory is gone.
    assert [next(iter(line)) for line in out] == ["log_dir", "summary", "upload"]
    log_dir, summary, upload = out[0]["log_dir"], out[1]["summary"], out[2]["upload"]
    assert summary["log_dir"] == log_dir and upload["ok"]
    assert upload["run"] == log_dir.rsplit("/", 1)[-1]
    assert upload["local_retained"] and list((tmp_path / "logs").glob("2*"))
    assert {key.rsplit("/", 1)[-1] for key in r2.objects} == {
        "setup.json",
        "run-summary.json",
        "adhoc_1.rrd",
        EPISODE,
        "manifest.json",
        CATALOG,
    }


def test_a_run_writes_its_formats_before_the_upload(tmp_path, r2, episode, capsys):
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()

    def run(config, expected_code=0):
        argv = ["robot", "run", "--config", str(config), "x"]
        assert robot_cli.main(parser().parse_args(argv)) == expected_code
        out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        return out[1]["summary"], out[2]["upload"]

    # The default keeps the live recording.
    summary, first = run(write_cloud_config(tmp_path / "a"))
    assert (
        summary["formats"] == {"rrd": {"ok": True, "files": ["adhoc_1.rrd"], "verified": False}}
        and first["ok"]
    )
    # A format that cannot be written (this stand-in leaves no eval log) is reported; the
    # recording stays and the run still uploads.
    summary, upload = run(
        write_cloud_config(tmp_path / "b", '[data]\nformats = ["lerobot"]\n'), expected_code=1
    )
    assert not summary["formats"]["lerobot"]["ok"] and "rrd" not in summary["formats"]
    assert "no embodiment spaces" in summary["formats"]["lerobot"]["error"] and upload["ok"]
    # The same objects as the first run, the recording among them.
    assert upload["objects"] == first["objects"]


def test_data_formats_are_known_and_distinct(tmp_path):
    for formats in ('["hdf5"]', '["rrd", "rrd"]', "[]"):
        with pytest.raises(ValidationError):
            RobotConfig.load(write_cloud_config(tmp_path, f"[data]\nformats = {formats}\n"))
    assert RobotConfig.load(write_cloud_config(tmp_path)).data.formats == ["rrd"]
    both = write_cloud_config(tmp_path, '[data]\nformats = ["rrd", "lerobot"]\n')
    assert RobotConfig.load(both).data.formats == ["rrd", "lerobot"]


def test_no_upload_keeps_the_run_and_says_so(tmp_path, r2, episode, capsys):
    config = write_cloud_config(tmp_path)
    argv = ["robot", "run", "--config", str(config), "--no-upload", "x"]
    assert robot_cli.main(parser().parse_args(argv)) == 0
    captured = capsys.readouterr()
    out = [json.loads(line) for line in captured.out.splitlines()]
    assert [next(iter(line)) for line in out] == ["log_dir", "summary", "upload"]
    name = out[0]["log_dir"].rsplit("/", 1)[-1]
    assert out[2]["upload"] == {"ok": True, "run": name, "skipped": "--no-upload"}
    assert (tmp_path / "logs" / name / "run-summary.json").is_file() and r2.objects == {}
    assert "kept on disk (--no-upload)" in captured.err


def test_run_tags_and_config_name_reach_the_catalog(tmp_path, r2, episode, capsys):
    config = write_cloud_config(tmp_path)
    argv = ["robot", "run", "--config", str(config), "--tag", "Cloth", "--tag", "demo", "x"]
    assert robot_cli.main(parser().parse_args(argv)) == 0
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert out[1]["summary"]["tags"] == ["cloth", "demo"]
    assert out[1]["summary"]["config"] == "robot-vla.toml"
    (row,) = [json.loads(line) for line in r2.objects[CATALOG_KEY].decode().splitlines()]
    assert row["tags"] == ["cloth", "demo"] and row["config"] == "robot-vla.toml"
    bad = ["robot", "run", "--config", str(config), "--tag", "two words", "x"]
    with pytest.raises(ValueError, match="tag 'two words'"):
        robot_cli.main(parser().parse_args(bad))
    assert len(episode) == 1


def test_an_agent_card_marks_each_tool_call_until_the_next(tmp_path):
    run = tmp_path / "20261011-214257"
    trial = "20261011_214258_965e63ea"
    labels = [
        f"{arm}_{axis}"
        for arm in ("left", "right")
        for axis in [*"xyz", "yaw", "pitch", "roll", "gripper"]
    ]
    still = [0.3, 0.0, 0.18, 0.0, 0.0, 0.0, 1.0]
    write_actions(
        run / "actions" / trial / "scene-0-e0.jsonl",
        labels,
        [still + still, still + [0.35, 0.0, 0.13, 0.0, 0.0, 0.0, 1.0]] * 2,
    )
    observation = {
        "role": "user",
        "content": [{"type": "text", "text": "Step budget: step {}/2400"}],
    }
    transcript = [
        {"role": "system", "content": "You are controlling a real robot"},
        json.loads(json.dumps(observation).replace("{}", "0")),
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "move_to",
                        "arguments": json.dumps({"targets": {"right_z": 0.13}, "note": "approach"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "executing move_to over 30 steps (3.0s)"},
        json.loads(json.dumps(observation).replace("{}", "3")),
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "done", "arguments": '{"summary": "placed"}'}}],
        },
    ]
    path = run / "transcripts" / trial / "scene-0-e0.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(message) + "\n" for message in transcript))
    (run / "meter-summary.json").write_text(
        json.dumps({"model": "gpt-6.1-sol", "spent_usd": "0.189979", "model_calls": 2})
    )
    (run / "adhoc_1.json").write_text(
        json.dumps(
            {
                "eval": {"policy": "agent", "policy_config": {"effort": "low"}},
                "samples": [
                    {
                        "instruction": "Put the jar by the mug",
                        "operator_judgements": ["partial"],
                        "termination_reasons": ["done"],
                        "trial_metadata": [
                            {
                                "actions": f"actions/{trial}/scene-0-e0.jsonl",
                                "transcript": f"transcripts/{trial}/scene-0-e0.jsonl",
                                "hindsight": "z lags",
                            }
                        ],
                    }
                ],
                "stats": {"total_steps": 4},
            }
        )
    )
    labelled = card(run, "yam-1")
    assert (labelled["policy"], labelled["kind"], labelled["outcome"]) == (
        "gpt-6.1-sol",
        "agent",
        "partial",
    )
    assert labelled["spent_usd"] == 0.189979 and labelled["hindsight"] == "z lags"
    assert labelled["control_interface"] == "eef_pos" and labelled["arms_moved"] == ["right"]
    calls = [m for m in labelled["markers"] if m["kind"] == "tool_call"]
    assert calls == [
        {
            "t": 0,
            "kind": "tool_call",
            "tool": "move_to",
            "note": "approach",
            "args": {"targets": {"right_z": 0.13}},
            "until": 3,
        },
        {"t": 3, "kind": "tool_call", "tool": "done", "args": {"summary": "placed"}, "until": 4},
    ]


def test_an_errored_run_is_labelled_error_and_still_uploads(tmp_path):
    log_root = tmp_path / "logs"
    run = make_run(log_root)
    log = json.loads((run / "adhoc_9c3f3481.json").read_text())
    log["samples"][0].update(
        error="PolicyError: Read timed out.", operator_judgements=[None], status="error"
    )
    (run / "adhoc_9c3f3481.json").write_text(json.dumps(log))
    labelled = card(run, "yam-1")
    assert labelled["outcome"] == "error" and labelled["markers"][-1]["error"].startswith("Policy")
    # A card that cannot be built never holds the data back.
    log["stats"]["duration_s"] = "soon"
    (run / "adhoc_9c3f3481.json").write_text(json.dumps(log))
    fake = FakeR2()
    with bucket(fake) as r2:
        record = offload(r2, log_root, run)
    assert "label_error" in record and run.is_dir()


def test_label_backfills_old_runs_and_edits_keep_through_rebuilds(tmp_path, r2, capsys):
    config = write_cloud_config(tmp_path)
    log_root = tmp_path / "logs"
    with bucket(r2) as r2_bucket:
        offload(r2_bucket, log_root, make_run(log_root))
    # A run uploaded before labels existed: no episode.json, an old manifest, no catalog.
    del r2.objects[PREFIX + EPISODE], r2.objects[CATALOG_KEY]
    manifest = json.loads(r2.objects[PREFIX + "manifest.json"])
    manifest["files"] = [item for item in manifest["files"] if item["path"] != EPISODE]
    r2.objects[PREFIX + "manifest.json"] = json.dumps(manifest).encode()

    def label(*extra):
        args = parser().parse_args(["robot", "label", "--config", str(config), *extra])
        assert robot_cli.main(args) == 0
        return [json.loads(line)["label"] for line in capsys.readouterr().out.splitlines()]

    (done,) = label("--all")
    assert done["ok"] and done["outcome"] == "success" and done["markers"] == 4
    manifest = json.loads(r2.objects[PREFIX + "manifest.json"])
    assert EPISODE in [item["path"] for item in manifest["files"]] and manifest["objects"] == 6
    assert json.loads(r2.objects[CATALOG_KEY])["run"] == RUN

    (edited,) = label(RUN, "--tag", "demo", "--policy", "isaac-0.5-yam", "--outcome", "partial")
    assert edited["tags"] == ["demo"] and edited["outcome"] == "partial"
    (rebuilt,) = label("--all")
    assert rebuilt["tags"] == ["demo"] and rebuilt["policy"] == "isaac-0.5-yam"
    stored = json.loads(r2.objects[PREFIX + EPISODE])
    assert stored["overrides"] == {"outcome": "partial", "policy": "isaac-0.5-yam"}
    (untagged,) = label(RUN, "--untag", "demo")
    assert untagged["tags"] == []
    with pytest.raises(ValueError, match="name the runs"):
        robot_cli.main(parser().parse_args(["robot", "label", "--config", str(config)]))


def test_a_get_whose_body_does_not_match_its_etag_is_refused(tmp_path):
    fake = FakeR2(stale=["episodes.jsonl"])
    with bucket(fake) as r2:
        r2.store(CATALOG_KEY, b'{"run": "x"}\n', "application/x-ndjson")
        assert r2.get("yam-1/missing.json") is None
        with pytest.raises(cloud.UploadError, match="does not match its ETag"):
            r2.get(CATALOG_KEY)


def test_a_run_without_credentials_keeps_its_data_and_exit_code(
    tmp_path, episode, monkeypatch, capsys
):
    for name in cloud.CREDENTIAL_ENVS:
        monkeypatch.delenv(name, raising=False)
    config = write_cloud_config(tmp_path)
    assert robot_cli.main(parser().parse_args(["robot", "run", "--config", str(config), "x"])) == 0
    captured = capsys.readouterr()
    upload = json.loads(captured.out.splitlines()[-1])["upload"]
    assert not upload["ok"] and "R2 credentials missing" in upload["error"]
    assert "kept on disk" in captured.err
    assert len(list((tmp_path / "logs").glob("2*/run-summary.json"))) == 1


def test_store_check_writes_and_reads_back(tmp_path, r2):
    config = RobotConfig.load(write_cloud_config(tmp_path))
    ok, detail = robot_cli.store_check(config)
    assert ok and "write and read-back OK" in detail
    assert list(r2.objects) == ["yam-1/.check.json"]
    r2.objects.clear()
    r2.failures.append(("PUT", ".check.json", 403))
    ok, detail = robot_cli.store_check(config)
    assert not ok and "403" in detail
