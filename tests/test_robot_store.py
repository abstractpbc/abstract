"""[store]: the Abstract runs API route against a fake of its v1 contract, the config's
key lookup and defaults, and the CLI on either route."""

import hashlib
import json
import re
from pathlib import Path
from threading import Barrier, Lock
from urllib.parse import unquote

import httpx
import pytest
from pydantic import ValidationError
from test_robot_cloud import RUN, make_run

from abstract_harness.cli import parser
from abstract_harness.robot import cli as robot_cli
from abstract_harness.robot import cloud, store
from abstract_harness.robot.config import (
    RobotConfig,
    StoreConfig,
    default_log_root,
    default_rig_name,
)
from abstract_harness.robot.episode import EPISODE

BASE = "https://api.abstract.test/v1"
KEY = "abs_live_" + "0123abcd" * 4 + "_" + "feedbeef" * 3 + "_" + "k" * 43
SAFE = "abs_live_0123abcd…edbeef"
ROBOT = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
RUN_NAME = re.compile(r"^\d{8}-\d{6}$")
SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MiB = 1024**2


def error(status, code, message=""):
    body = {"error": {"message": message or code, "type": "invalid_request_error", "code": code}}
    return httpx.Response(status, json=body)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class FakeApi:
    """The robot runs API, v1, for one account: what the Worker checks, kept in memory."""

    def __init__(self, failures=(), corrupt=(), page_size=1000, min_part=5 * MiB):
        self.files = {}  # (robot, run, path) -> bytes
        self.manifests = {}  # (robot, run) -> manifest
        self.uploads = {}  # upload_id -> {"key", "sha256", "parts"}
        self.completed_uploads = {}
        self.next_upload = 1
        self.requests = []
        self.failures = list(failures)  # (method, path suffix, status), consumed in order
        self.corrupt = set(corrupt)  # path suffixes whose PUT body arrives altered
        self.page_size = page_size
        self.min_part = min_part
        self.miscount = False

    def __call__(self, request):
        assert str(request.url).startswith(BASE + "/robot/runs")
        route = request.url.raw_path.decode().split("?")[0].removeprefix("/v1/robot/runs")
        self.requests.append((request.method, unquote(route), dict(request.url.params)))
        if request.headers.get("authorization") != f"Bearer {KEY}":
            return error(401, "invalid_api_key", "the API key is not valid")
        for failure in self.failures:
            method, suffix, status = failure
            if request.method == method and unquote(route).endswith(suffix):
                self.failures.remove(failure)
                return error(status, "storage_unavailable" if status >= 500 else "refused")
        if route in ("", "/"):
            return self.listing(request)
        parts = [unquote(part) for part in route.strip("/").split("/")]
        robot, run, *rest = parts + ["", ""][: max(0, 2 - len(parts))]
        if not ROBOT.match(robot) or not RUN_NAME.match(run):
            return error(422, "invalid_request", "robot or run name")
        if not rest:
            if request.method == "GET":
                manifest = self.manifests.get((robot, run))
                return (
                    httpx.Response(200, json=manifest) if manifest else error(404, "run_not_found")
                )
            return error(405, "method")
        kind, path = rest[0], "/".join(rest[1:])
        if kind == "complete" and not path:
            return self.complete(robot, run, json.loads(request.read()))
        segments = path.split("/")
        if not path or len(segments) > 8 or not all(SEGMENT.match(s) for s in segments):
            return error(422, "invalid_request", f"path {path!r}")
        if path == "manifest.json":
            return error(422, "invalid_request", "manifest.json is the server's")
        if kind == "files":
            return self.file(request, (robot, run, path))
        if kind == "uploads":
            return self.upload(request, (robot, run, path))
        return error(404, "not_found")

    def incomplete(self, robot, run):
        self.manifests.pop((robot, run), None)

    def file(self, request, key):
        data = self.files.get(key)
        if request.method == "PUT":
            if "content-length" not in request.headers:
                return error(411, "length_required")
            body = request.read()
            if len(body) > 64 * MiB:
                return error(413, "request_too_large")
            if any(key[2].endswith(suffix) for suffix in self.corrupt):
                body = bytes([body[0] ^ 1]) + body[1:]
            if sha(body) != request.headers.get("x-content-sha256"):
                return error(422, "checksum_mismatch", f"{key[2]}: the body is not that sha256")
            self.incomplete(*key[:2])
            self.files[key] = body
            return httpx.Response(
                201, json={"path": key[2], "size": len(body), "sha256": sha(body)}
            )
        if request.method in ("GET", "HEAD"):
            if data is None:
                return (
                    httpx.Response(404)
                    if request.method == "HEAD"
                    else error(404, "file_not_found")
                )
            headers = {"x-content-sha256": sha(data), "etag": f'"{sha(data)[:32]}"'}
            if request.method == "HEAD":
                return httpx.Response(200, headers={**headers, "content-length": str(len(data))})
            return httpx.Response(200, content=data, headers=headers)
        if request.method == "DELETE":
            if data is None:
                return error(404, "file_not_found")
            del self.files[key]
            self.incomplete(*key[:2])
            return httpx.Response(204)
        return error(405, "method")

    def upload(self, request, key):
        upload_id = request.url.params.get("upload_id")
        if request.method == "POST" and upload_id is None:
            given = request.headers.get("x-content-sha256", "")
            if not re.fullmatch(r"[0-9a-f]{64}", given):
                return error(422, "invalid_request", "X-Content-SHA256 required")
            size = int(request.headers["x-content-length"])
            upload_id = f"up{self.next_upload}"
            self.next_upload += 1
            self.uploads[upload_id] = {"key": key, "sha256": given, "size": size, "parts": {}}
            return httpx.Response(201, json={"upload_id": upload_id})
        upload = self.uploads.get(upload_id)
        if upload_id in self.completed_uploads:
            completed = self.completed_uploads[upload_id]
            assert completed["key"] == key
            return httpx.Response(200, json={**completed["result"], "state": "complete"})
        if upload is None or upload["key"] != key:
            return error(404, "upload_not_found")
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "state": "uploading",
                    "size": upload["size"],
                    "sha256": upload["sha256"],
                    "parts": [
                        {"part": n, "etag": f"etag-{n}", "size": len(data)}
                        for n, data in upload["parts"].items()
                    ],
                },
            )
        if request.method == "PUT":
            body = request.read()
            if len(body) > 64 * MiB:
                return error(413, "request_too_large")
            number = int(request.url.params["part"])
            upload["parts"][number] = body
            return httpx.Response(200, json={"part": number, "etag": f"etag-{number}"})
        if request.method == "POST":
            given = json.loads(request.read())
            numbers = [part["part"] for part in given["parts"]]
            assert "sha256" not in given and numbers == sorted(upload["parts"])
            assert [part["etag"] for part in given["parts"]] == [f"etag-{n}" for n in numbers]
            assert all(len(upload["parts"][n]) >= self.min_part for n in numbers[:-1])
            data = b"".join(upload["parts"][n] for n in numbers)
            del self.uploads[upload_id]
            assert len(data) == upload["size"]
            if len(data) != given["size"] or sha(data) != upload["sha256"]:
                return error(422, "checksum_mismatch", f"{key[2]}: assembled object differs")
            self.incomplete(*key[:2])
            self.files[key] = data
            result = {"path": key[2], "size": len(data), "sha256": sha(data)}
            self.completed_uploads[upload_id] = {"key": key, "result": result}
            return httpx.Response(201, json=result)
        if request.method == "DELETE":
            del self.uploads[upload_id]
            return httpx.Response(204)
        return error(405, "method")

    def complete(self, robot, run, body):
        listed = {item["path"]: item for item in body["files"]}
        stored = {path: data for (r, n, path), data in self.files.items() if (r, n) == (robot, run)}
        problems = [
            path
            for path, item in listed.items()
            if len(stored.get(path, b"?" * -1)) != item["size"] or path not in stored
        ] + [f"unlisted {path}" for path in stored if path not in listed]
        if problems:
            return error(409, "run_incomplete", ", ".join(sorted(problems)))
        assert len(json.dumps(body["episode"]).encode()) <= 64 * 1024
        assert "markers" not in body["episode"] or isinstance(body["episode"]["markers"], int)
        total = sum(len(data) for data in stored.values())
        done = {"robot": robot, "run": run, "files": len(listed), "bytes": total}
        done["completed_at"] = "2026-10-12T20:00:00Z"
        self.manifests[(robot, run)] = {**done, "files": body["files"], "episode": body["episode"]}
        if self.miscount:
            done["bytes"] += 1
        return httpx.Response(200, json=done)

    def listing(self, request):
        robot = request.url.params.get("robot")
        limit = min(int(request.url.params.get("limit", 100)), self.page_size)
        start = int(request.url.params.get("cursor", 0))
        rows = [
            {k: m[k] for k in ("robot", "run", "bytes", "completed_at", "episode")}
            for key, m in sorted(self.manifests.items())
            if robot in (None, key[0])
        ]
        page = rows[start : start + limit]
        cursor = str(start + limit) if start + limit < len(rows) else None
        return httpx.Response(200, json={"runs": page, "cursor": cursor})

    def puts(self):
        return [route for method, route, _ in self.requests if method == "PUT"]

    def stored(self, robot="yam-1", run=RUN):
        return {path: data for (r, n, path), data in self.files.items() if (r, n) == (robot, run)}


def api_store(fake, sleeps=None, key=KEY, robot="yam-1"):
    return store.ApiStore(
        robot,
        BASE,
        key,
        client=httpx.Client(transport=httpx.MockTransport(fake)),
        sleep=(sleeps.append if sleeps is not None else lambda seconds: None),
    )


def test_a_run_goes_up_file_by_file_completes_then_leaves_the_disk(tmp_path):
    log_root = tmp_path / "logs"
    run = make_run(log_root)
    fake = FakeApi()
    record = store.offload(api_store(fake), log_root, run)
    assert run.is_dir() and record["local_retained"] is True
    files = fake.stored()
    assert set(files) == {
        "run-summary.json",
        "adhoc_9c3f3481.json",
        "adhoc_9c3f3481.rrd",
        "actions/20261011_231136_0cdd0718/scene-0-e0.jsonl",
        EPISODE,
    }
    assert not any(path.startswith("frames/") for path in files)  # the raw frames stay
    manifest = fake.manifests[("yam-1", RUN)]
    assert {item["path"]: (item["size"], item["sha256"]) for item in manifest["files"]} == {
        path: (len(data), sha(data)) for path, data in files.items()
    }
    # The catalog row: the card without markers, the count instead.
    assert manifest["episode"]["markers"] == 4 and manifest["episode"]["rig"] == "yam-1"
    assert manifest["episode"]["frames"] == {"npy": 2}
    assert manifest["episode"]["outcome"] == "success"
    # complete comes last, after every file.
    assert [method for method, _, _ in fake.requests][-1] == "POST"
    (line,) = (log_root / cloud.INDEX).read_text().splitlines()
    kept = json.loads(line)
    assert kept == record and KEY not in line
    assert (kept["store"], kept["robot"], kept["api_url"]) == ("abstract", "yam-1", BASE)
    assert (kept["objects"], kept["sent"]) == (5, 5)
    assert kept["bytes"] == sum(len(data) for data in files.values())
    assert kept["completed_at"] == "2026-10-12T20:00:00Z" and kept["frames"] == {"npy": 2}


def test_a_failed_upload_keeps_the_run_and_the_retry_skips_what_is_stored(tmp_path):
    log_root = tmp_path / "logs"
    run = make_run(log_root)
    fake = FakeApi(failures=[("PUT", ".rrd", 403)])
    with pytest.raises(cloud.UploadError, match="HTTP 403"):
        record = store.offload(api_store(fake), log_root, run)
    assert run.is_dir() and not fake.manifests and not (log_root / cloud.INDEX).exists()
    already = set(fake.stored())
    fake.requests.clear()
    record = store.offload(api_store(fake), log_root, run)
    # The card comes out the same, so even episode.json is found stored and skipped.
    assert (
        run.is_dir()
        and record["local_retained"] is True
        and EPISODE in already
        and record["sent"] == 5 - len(already) == 1
    )
    sent = {route.rsplit("/files/", 1)[-1] for route in fake.puts()}
    assert sent == {"adhoc_9c3f3481.rrd"}


def test_server_errors_and_rate_limits_are_retried_with_backoff(tmp_path):
    log_root = tmp_path / "logs"
    run = make_run(log_root)
    fake = FakeApi(failures=[("PUT", ".rrd", 503), ("POST", "/complete", 429)])
    sleeps = []
    record = store.offload(api_store(fake, sleeps), log_root, run)
    assert run.is_dir() and record["local_retained"] is True and sleeps == [2, 2]


def test_retries_give_up_and_keep_the_run(tmp_path):
    log_root = tmp_path / "logs"
    run = make_run(log_root)
    fake = FakeApi(failures=[("PUT", ".rrd", 503)] * cloud.ATTEMPTS)
    with pytest.raises(cloud.UploadError, match="503 storage_unavailable"):
        store.offload(api_store(fake), log_root, run)
    assert run.is_dir()


def test_a_body_that_is_not_its_checksum_is_refused_and_the_run_kept(tmp_path):
    log_root = tmp_path / "logs"
    run = make_run(log_root)
    fake = FakeApi(corrupt=[".rrd"])
    with pytest.raises(cloud.UploadError, match="422 checksum_mismatch"):
        store.offload(api_store(fake), log_root, run)
    assert run.is_dir() and not fake.manifests


def test_large_files_go_up_in_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SINGLE_PUT_BYTES", 1024)
    monkeypatch.setattr(store, "PART_BYTES", 512)
    log_root = tmp_path / "logs"
    run = make_run(log_root)  # its recording is 2004 bytes
    fake = FakeApi(min_part=512)
    store.offload(api_store(fake), log_root, run)
    rrd = fake.stored()["adhoc_9c3f3481.rrd"]
    assert rrd == b"RRF2" + b"\1" * 2000
    parts = [
        params
        for method, route, params in fake.requests
        if method == "PUT" and route.endswith("/uploads/adhoc_9c3f3481.rrd")
    ]
    assert sorted(params["part"] for params in parts) == ["1", "2", "3", "4"]
    assert not fake.uploads  # completed, none left open


def test_a_failed_part_keeps_progress_and_a_new_client_resumes_it(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SINGLE_PUT_BYTES", 1024)
    monkeypatch.setattr(store, "PART_BYTES", 512)
    log_root = tmp_path / "logs"
    run = make_run(log_root)
    fake = FakeApi(min_part=512)
    upload = fake.upload

    def fail_part(request, key):
        if (
            key[2].endswith(".rrd")
            and request.method == "PUT"
            and request.url.params.get("part") == "2"
        ):
            return error(500, "storage_unavailable")
        return upload(request, key)

    fake.upload = fail_part
    with pytest.raises(cloud.UploadError, match="500"):
        store.offload(api_store(fake), log_root, run)
    assert run.is_dir() and len(fake.uploads) == 1
    assert not any(method == "DELETE" for method, _, _ in fake.requests)
    (progress,) = (log_root / store.UPLOAD_STATE).glob("*.json")
    saved = json.loads(progress.read_text())
    assert saved["size"] == 2004 and saved["part_bytes"] == 512
    assert saved["path"] == "adhoc_9c3f3481.rrd" and KEY not in progress.read_text()
    assert progress.stat().st_mode & 0o777 == 0o600
    accepted = set(fake.uploads[saved["upload_id"]]["parts"])
    assert accepted and 2 not in accepted

    fake.upload = upload
    fake.requests.clear()
    store.offload(api_store(fake), log_root, run)
    sent = {
        int(params["part"])
        for method, route, params in fake.requests
        if method == "PUT" and "/uploads/" in route
    }
    assert sent == {1, 2, 3, 4} - accepted
    assert not progress.exists() and run.is_dir() and not fake.uploads
    assert all(store.UPLOAD_STATE not in path for path in fake.stored())


def test_parts_are_uploaded_in_parallel_with_at_most_four_bodies(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SINGLE_PUT_BYTES", 1024)
    monkeypatch.setattr(store, "PART_BYTES", 512)
    source = tmp_path / "large.rrd"
    source.write_bytes(b"r" * 4096)
    fake = FakeApi(min_part=512)
    upload = fake.upload
    barrier, lock = Barrier(4), Lock()
    active = maximum = 0
    sizes = []

    def concurrent_part(request, key):
        nonlocal active, maximum
        if request.method != "PUT":
            return upload(request, key)
        with lock:
            active += 1
            maximum = max(maximum, active)
            sizes.append(len(request.read()))
        try:
            barrier.wait(timeout=5)
            return upload(request, key)
        finally:
            with lock:
                active -= 1

    fake.upload = concurrent_part
    api_store(fake).put_file(RUN, source.name, source, resume_dir=tmp_path / "progress")
    assert maximum == 4 and sizes == [512] * 8
    assert fake.stored()[source.name] == source.read_bytes()


def test_failed_finalization_is_resumed_without_sending_parts_again(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SINGLE_PUT_BYTES", 1024)
    monkeypatch.setattr(store, "PART_BYTES", 512)
    run = make_run(tmp_path)
    fake = FakeApi(min_part=512)
    upload = fake.upload

    def fail_finish(request, key):
        if (
            key[2].endswith(".rrd")
            and request.method == "POST"
            and "upload_id" in request.url.params
        ):
            return error(503, "storage_unavailable")
        return upload(request, key)

    fake.upload = fail_finish
    with pytest.raises(cloud.UploadError, match="503"):
        record = store.offload(api_store(fake), tmp_path, run)
    assert run.exists() and len(fake.uploads) == 1
    assert not any(method == "DELETE" for method, _, _ in fake.requests)
    fake.upload = upload
    fake.requests.clear()
    record = store.offload(api_store(fake), tmp_path, run)
    assert not fake.puts()
    assert (
        run.is_dir()
        and record["local_retained"] is True
        and not list((tmp_path / store.UPLOAD_STATE).glob("*.json"))
    )


def test_lost_finalization_response_retries_the_same_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SINGLE_PUT_BYTES", 1024)
    monkeypatch.setattr(store, "PART_BYTES", 512)
    run = make_run(tmp_path)
    fake = FakeApi(min_part=512)
    upload = fake.upload
    finished = []

    def lose_response(request, key):
        result = upload(request, key)
        if (
            key[2].endswith(".rrd")
            and request.method == "POST"
            and "upload_id" in request.url.params
        ):
            finished.append(request.url.params["upload_id"])
            if len(finished) == 1:
                raise httpx.ReadError("connection closed after completion", request=request)
        return result

    fake.upload = lose_response
    record = store.offload(api_store(fake), tmp_path, run)
    assert len(finished) == 2 and finished[0] == finished[1]
    assert run.is_dir() and record["local_retained"] is True and not fake.uploads
    assert not any(method == "DELETE" for method, _, _ in fake.requests)


@pytest.mark.parametrize("change", ["expired", "changed", "smaller"])
def test_expired_or_changed_upload_starts_over_safely(tmp_path, monkeypatch, change):
    monkeypatch.setattr(store, "SINGLE_PUT_BYTES", 1024)
    monkeypatch.setattr(store, "PART_BYTES", 512)
    source = tmp_path / "large.rrd"
    source.write_bytes(b"r" * 2004)
    fake = FakeApi(min_part=512)
    upload = fake.upload

    def fail_finish(request, key):
        if request.method == "POST" and "upload_id" in request.url.params:
            return error(503, "storage_unavailable")
        return upload(request, key)

    fake.upload = fail_finish
    progress = tmp_path / "progress"
    with pytest.raises(cloud.UploadError, match="503"):
        api_store(fake).put_file(RUN, source.name, source, resume_dir=progress)
    if change != "expired":
        source.write_bytes(b"s" * (2004 if change == "changed" else 10))
    else:
        fake.uploads.clear()  # The server expired this upload between CLI invocations.
    fake.upload = upload
    fake.requests.clear()
    api_store(fake).put_file(RUN, source.name, source, resume_dir=progress)
    assert fake.stored()[source.name] == source.read_bytes()
    assert any(method == "DELETE" for method, _, _ in fake.requests) == (change != "expired")
    assert not fake.uploads and not list(progress.glob("*.json"))


@pytest.mark.parametrize("state", ["assembling", "publishing", "complete"])
def test_finalizing_upload_resumes_without_reopening_parts(tmp_path, monkeypatch, state):
    monkeypatch.setattr(store, "SINGLE_PUT_BYTES", 1024)
    monkeypatch.setattr(store, "PART_BYTES", 512)
    source = tmp_path / "large.rrd"
    source.write_bytes(b"r" * 2004)
    digest = cloud.Digest.of(source)
    target = api_store(FakeApi())
    progress = tmp_path / "progress"
    name = hashlib.sha256(target._url(RUN, "uploads", source.name).encode()).hexdigest()
    store._save_upload(
        progress / f"{name}.json",
        {
            "upload_id": "in-progress",
            "size": digest.size,
            "sha256": digest.sha256,
            "part_bytes": 512,
        },
    )
    methods = []

    def finalizing(request):
        methods.append(request.method)
        if request.method == "HEAD":
            if state == "complete" and methods.count("HEAD") == 2:
                return httpx.Response(
                    200,
                    headers={"content-length": str(digest.size), "x-content-sha256": digest.sha256},
                )
            return httpx.Response(404)
        assert request.method in {"GET", "POST"}
        assert request.url.params["upload_id"] == "in-progress"
        # Once assembly has begun, only the server can resume it. Even an empty
        # parts list is not a reason for the client to send part data again.
        return httpx.Response(
            200,
            json={"state": state, "parts": [], "size": digest.size, "sha256": digest.sha256},
        )

    target._client = httpx.Client(transport=httpx.MockTransport(finalizing))
    target.put_file(RUN, source.name, source, resume_dir=progress)
    assert methods == (["HEAD", "GET", "HEAD"] if state == "complete" else ["HEAD", "GET", "POST"])
    assert not list(progress.glob("*.json"))


def test_a_completed_receipt_does_not_skip_a_later_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SINGLE_PUT_BYTES", 1024)
    monkeypatch.setattr(store, "PART_BYTES", 512)
    source = tmp_path / "large.rrd"
    source.write_bytes(b"r" * 2004)
    digest = cloud.Digest.of(source)
    fake = FakeApi(min_part=512)
    target = api_store(fake)
    progress = tmp_path / "progress"
    name = hashlib.sha256(target._url(RUN, "uploads", source.name).encode()).hexdigest()
    saved = {
        "upload_id": "completed-before-interruption",
        "size": digest.size,
        "sha256": digest.sha256,
        "part_bytes": 512,
    }
    store._save_upload(progress / f"{name}.json", saved)
    key = ("yam-1", RUN, source.name)
    fake.completed_uploads[saved["upload_id"]] = {"key": key, "result": saved}
    fake.files[key] = b"a subsequent replacement"
    target.put_file(RUN, source.name, source, resume_dir=progress)
    assert fake.stored()[source.name] == source.read_bytes()
    assert len(fake.puts()) == 4
    assert not list(progress.glob("*.json"))


def test_returning_to_stored_bytes_releases_an_interrupted_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SINGLE_PUT_BYTES", 1024)
    monkeypatch.setattr(store, "PART_BYTES", 512)
    source = tmp_path / "large.rrd"
    original = b"a" * 2004
    source.write_bytes(b"b" * 2004)
    fake = FakeApi(min_part=512)
    fake.files[("yam-1", RUN, source.name)] = original
    upload = fake.upload

    def fail_parts(request, key):
        if request.method == "PUT":
            return error(503, "storage_unavailable")
        return upload(request, key)

    fake.upload = fail_parts
    progress = tmp_path / "progress"
    with pytest.raises(cloud.UploadError, match="503"):
        api_store(fake).put_file(RUN, source.name, source, resume_dir=progress)
    assert len(fake.uploads) == 1
    source.write_bytes(original)
    digest, sent = api_store(fake).put_file(RUN, source.name, source, resume_dir=progress)
    assert not sent and digest.sha256 == sha(original)
    assert not fake.uploads and not list(progress.glob("*.json"))
    assert fake.stored()[source.name] == original


def test_an_incomplete_or_miscounted_run_is_not_deleted(tmp_path):
    log_root = tmp_path / "logs"
    run = make_run(log_root)
    fake = FakeApi()
    fake.files[("yam-1", RUN, "stale.json")] = b"{}"  # left by something else
    with pytest.raises(cloud.UploadError, match="409 run_incomplete: unlisted stale.json"):
        store.offload(api_store(fake), log_root, run)
    assert run.is_dir()
    del fake.files[("yam-1", RUN, "stale.json")]
    fake.miscount = True
    with pytest.raises(cloud.UploadError, match="the API counted 5 files"):
        store.offload(api_store(fake), log_root, run)
    assert run.is_dir() and not (log_root / cloud.INDEX).exists()


def test_labels_are_rebuilt_from_the_stored_files_and_completed_again(tmp_path):
    log_root = tmp_path / "logs"
    fake = FakeApi()
    store.offload(api_store(fake), log_root, make_run(log_root))
    target = api_store(fake)
    edited = target.relabel(RUN, ["demo"], [], {"outcome": "partial", "policy": "isaac-0.5-yam"})
    assert edited["tags"] == ["demo"] and edited["outcome"] == "partial"
    manifest = fake.manifests[("yam-1", RUN)]
    assert manifest["episode"]["frames"] == {"npy": 2}  # kept through a relabel
    assert manifest["episode"]["outcome"] == "partial" and manifest["episode"]["tags"] == ["demo"]
    stored = json.loads(fake.stored()[EPISODE])
    assert stored["overrides"] == {"outcome": "partial", "policy": "isaac-0.5-yam"}
    rebuilt = target.relabel(RUN)  # edits survive a rebuild
    assert rebuilt["tags"] == ["demo"] and rebuilt["policy"] == "isaac-0.5-yam"
    assert target.relabel(RUN, remove=["demo"])["tags"] == []
    with pytest.raises(cloud.UploadError, match="not a completed run"):
        target.relabel("20261011-000000")


def test_a_stored_card_source_that_differs_from_its_manifest_is_refused(tmp_path):
    log_root = tmp_path / "logs"
    fake = FakeApi()
    store.offload(api_store(fake), log_root, make_run(log_root))
    fake.files[("yam-1", RUN, "run-summary.json")] += b" "
    with pytest.raises(cloud.UploadError, match="not the file its manifest lists"):
        api_store(fake).relabel(RUN)


def test_the_listing_pages_through_every_run_of_the_robot(tmp_path):
    fake = FakeApi(page_size=1)
    for name in ("20261011-000001", "20261011-000002", "20261011-000003"):
        fake.manifests[("yam-1", name)] = {
            "robot": "yam-1",
            "run": name,
            "bytes": 1,
            "completed_at": "x",
            "episode": {},
        }
    fake.manifests[("other", "20261011-000004")] = {
        "robot": "other",
        "run": "20261011-000004",
        "bytes": 1,
        "completed_at": "x",
        "episode": {},
    }
    assert api_store(fake).runs() == ["20261011-000001", "20261011-000002", "20261011-000003"]


def test_check_names_the_api_and_the_keys_safe_prefix_never_the_key():
    ok, detail = api_store(FakeApi()).check()
    assert ok and BASE in detail and SAFE in detail and KEY not in detail
    wrong = KEY[:-1] + "x"
    ok, detail = api_store(FakeApi(), key=wrong).check()
    assert not ok and "401 invalid_api_key" in detail and wrong not in detail
    assert store.safe_prefix("sk-not-ours") == "abs_live_…"


# -- config: the key, the base URL, the rig's name and the log root -------------------


def write_credentials(directory, **fields):
    directory.mkdir(parents=True, exist_ok=True)
    values = {
        "version": 1,
        "apiKey": KEY,
        "keyId": "key_1",
        "baseUrl": BASE,
        "createdAt": "2026-10-12T00:00:00Z",
        **fields,
    }
    (directory / "credentials.json").write_text(json.dumps(values))


def test_the_key_comes_from_the_environment_else_abstract_logins_credentials(tmp_path):
    abstract = StoreConfig(route="abstract")
    home = tmp_path / "home"
    write_credentials(home / ".config" / "abstract")
    assert store.api_credentials(abstract, {}, home) == (BASE, KEY)
    env_key = KEY.replace("k" * 43, "e" * 43)
    assert store.api_credentials(abstract, {"ABSTRACT_API_KEY": env_key}, home) == (BASE, env_key)
    # $XDG_CONFIG_HOME/abstract, and $ABSTRACT_CONFIG_DIR before it, as the CLI looks.
    write_credentials(tmp_path / "xdg" / "abstract", baseUrl="https://xdg.abstract.test/v1")
    write_credentials(tmp_path / "dir", baseUrl="https://dir.abstract.test/v1")
    xdg = {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}
    assert store.api_credentials(abstract, xdg, home)[0] == "https://xdg.abstract.test/v1"
    both = {**xdg, "ABSTRACT_CONFIG_DIR": str(tmp_path / "dir")}
    assert store.api_credentials(abstract, both, home)[0] == "https://dir.abstract.test/v1"


def test_the_base_url_is_the_override_else_the_saved_one_else_the_default(tmp_path):
    home = tmp_path / "home"
    env = {"ABSTRACT_API_KEY": KEY}
    assert store.api_credentials(StoreConfig(route="abstract"), env, home) == (
        "https://api.abstract.inc/v1",
        KEY,
    )
    local = StoreConfig(route="abstract", api_url="http://127.0.0.1:8787/v1/")
    write_credentials(home / ".config" / "abstract")
    assert store.api_credentials(local, {}, home) == ("http://127.0.0.1:8787/v1", KEY)


def test_no_key_or_a_bad_credentials_file_is_a_clear_refusal(tmp_path):
    abstract = StoreConfig(route="abstract")
    with pytest.raises(cloud.UploadError, match="set ABSTRACT_API_KEY, or run abstract login"):
        store.api_credentials(abstract, {}, tmp_path)
    (tmp_path / ".config" / "abstract").mkdir(parents=True)
    (tmp_path / ".config" / "abstract" / "credentials.json").write_text('{"version": 2}')
    with pytest.raises(cloud.UploadError, match="not a credentials file"):
        store.api_credentials(abstract, {}, tmp_path)
    # The environment's key needs no file, readable or not.
    assert store.api_credentials(abstract, {"ABSTRACT_API_KEY": KEY}, tmp_path)[1] == KEY
    write_credentials(tmp_path / ".config" / "abstract", baseUrl="http://api.abstract.test/v1")
    with pytest.raises(cloud.UploadError, match="HTTPS required"):
        store.api_credentials(abstract, {}, tmp_path)


def test_each_route_takes_only_its_own_fields():
    with pytest.raises(ValidationError, match='for route = "bucket"'):
        StoreConfig(route="abstract", bucket="my-robot-runs")
    with pytest.raises(ValidationError, match="api_url is for route"):
        StoreConfig(route="bucket", api_url="https://api.abstract.inc/v1")
    with pytest.raises(ValidationError, match="plain API base URL"):
        StoreConfig(route="abstract", api_url="https://user:pass@api.abstract.inc/v1")
    with pytest.raises(ValidationError):
        StoreConfig(route="somewhere")


def test_a_cloud_section_names_its_replacement(tmp_path):
    path = tmp_path / "robot.toml"
    path.write_text(
        '[vla]\nserver_url = "http://127.0.0.1:8202"\n'
        '[rig.embodiment]\nleft_channel = "can1"\nright_channel = "can0"\n'
        '[cloud]\nprefix = "yam-1"\n'
    )
    with pytest.raises(ValidationError, match=r"\[cloud\] is now \[store\]"):
        RobotConfig.load(path)


def test_the_rig_is_named_after_the_host_and_writes_under_the_users_data_dir(monkeypatch):
    assert default_rig_name("YAM-1") == "yam-1"
    assert default_rig_name("Lab Rig #2.local") == "lab-rig-2.local"
    assert default_rig_name("_x") == "x" and default_rig_name("---") == "robot"
    assert len(default_rig_name("a" * 100)) == 64
    assert default_log_root({"XDG_DATA_HOME": "/data"}) == Path("/data/abstract/robot/runs")
    monkeypatch.setenv("HOME", "/home/someone")
    assert default_log_root({}) == Path("/home/someone/.local/share/abstract/robot/runs")


# -- the CLI on the API route -----------------------------------------------------------


def write_api_config(tmp_path):
    path = tmp_path / "robot.toml"
    path.write_text(
        '[vla]\npolicy = "molmoact2"\nserver_url = "http://policy-host:8202"\n'
        '[rig]\nname = "yam-1"\nlog_root = "logs"\n[rig.embodiment]\nleft_channel = "can1"\n'
        'right_channel = "can0"\n[store]\nroute = "abstract"\n'
    )
    return path


@pytest.fixture
def api(monkeypatch):
    fake = FakeApi()
    real = store.ApiStore
    monkeypatch.setattr(store, "api_credentials", lambda config: (BASE, KEY))
    monkeypatch.setattr(
        store,
        "ApiStore",
        lambda robot, base_url, key: real(
            robot, base_url, key, client=httpx.Client(transport=httpx.MockTransport(fake))
        ),
    )
    return fake


def test_upload_and_label_commands_go_through_the_api(tmp_path, api, capsys):
    config = write_api_config(tmp_path)
    log_root = tmp_path / "logs"
    make_run(log_root)
    assert robot_cli.main(parser().parse_args(["robot", "upload", "--config", str(config)])) == 0
    (line,) = capsys.readouterr().out.splitlines()
    upload = json.loads(line)["upload"]
    assert upload["ok"] and upload["run"] == RUN and upload["store"] == "abstract"
    assert ("yam-1", RUN) in api.manifests and KEY not in line

    args = ["robot", "label", "--config", str(config), "--all", "--tag", "demo"]
    assert robot_cli.main(parser().parse_args(args)) == 0
    (labelled,) = [json.loads(line)["label"] for line in capsys.readouterr().out.splitlines()]
    assert labelled["ok"] and labelled["tags"] == ["demo"]
    assert api.manifests[("yam-1", RUN)]["episode"]["tags"] == ["demo"]


def test_check_reports_the_store(tmp_path, api):
    ok, detail = robot_cli.store_check(RobotConfig.load(write_api_config(tmp_path)))
    assert ok and SAFE in detail and KEY not in detail
