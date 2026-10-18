"""Where finished runs go, behind one interface: ``[store]`` in the robot TOML.

``route = "abstract"`` (``ApiStore``): the Abstract runs API, under the account of the API
key, the user's own; nothing else names the account. ``route = "bucket"``
(``BucketStore``): an R2 bucket of the user's that the rig writes over its S3 API
(``cloud.py``). Either way a run is prepared the same way (its episode card written as
``episode.json``, every file but the raw frames listed; ``cloud.prepare_run``), every file
is sent and checked, and only then is the run complete: the API's ``complete``, or the
bucket's ``manifest.json``. ``offload`` rechecks export receipts before deleting local raw
frames and keeps one line for the upload in ``<log_root>/runs.jsonl``; any failure keeps the run for
``abstract robot upload`` to retry.

The API contract (v1, https://abstract.inc/docs):
``PUT .../robot/runs/{robot}/{run}/files/{path}`` with
``X-Content-SHA256`` (64 MiB at most; larger files go up in parts through
``.../uploads/{path}``), ``HEAD`` of the same path for what is stored, then
``POST .../{run}/complete`` with every file and the catalog row; ``GET`` reads a run's
manifest, files and the account's runs. The key goes in the Authorization header and in no
message, log or record.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from abstract_harness.robot import cloud, episode
from abstract_harness.robot.cloud import UploadError
from abstract_harness.robot.config import (
    DEFAULT_API_URL,
    RobotConfig,
    StoreConfig,
    plain_api_url,
)

API_KEY_ENV = "ABSTRACT_API_KEY"
CREDENTIALS = "credentials.json"
API_KEY = re.compile(r"^abs_live_([a-f0-9]{32})_([a-f0-9]{24})_[A-Za-z0-9_-]{43}$")
# One PUT carries at most this much; a larger file goes up in parts of PART_BYTES.
SINGLE_PUT_BYTES = 64 * 1024**2
PART_BYTES = 32 * 1024**2
PART_WORKERS = 4
UPLOAD_STATE = ".abstract-uploads"
# The catalog row the API keeps for a run: its episode card without markers.
MAX_ROW_BYTES = 64 * 1024


class Store(Protocol):
    """A place finished runs go: what upload, label and check need of one."""

    description: str

    def upload_run(self, run_dir: Path) -> dict[str, Any]:
        """Send a finished run and complete it; the record for runs.jsonl. Deletes nothing."""
        ...

    def relabel(
        self, run: str, add: Iterable[str], remove: Iterable[str], overrides: Mapping[str, str]
    ) -> dict[str, Any]:
        """Rebuild a stored run's episode card, apply the edits, store it again."""
        ...

    def labelled(self, records: list[dict[str, Any]]) -> None:
        """After relabelling: a catalog the store keeps beside the runs, updated."""
        ...

    def runs(self) -> list[str]:
        """Every run of this robot the store holds."""
        ...

    def owns(self, record: Mapping[str, Any]) -> bool:
        """Whether a runs.jsonl line records a run sent to this store."""
        ...

    def check(self) -> tuple[bool, str]:
        """Reach the store with its credentials, without touching any run."""
        ...

    def close(self) -> None: ...


def safe_prefix(key: str) -> str:
    """The part of an API key the account lists it by, as the ``abstract`` CLI shows it; it
    reveals no secret."""
    match = API_KEY.match(key)
    return f"abs_live_{match[1][:8]}…{match[2][-6:]}" if match else "abs_live_…"


def credentials_path(env: Mapping[str, str], home: Path) -> Path:
    """Where ``abstract login`` keeps the key: $ABSTRACT_CONFIG_DIR, else
    $XDG_CONFIG_HOME/abstract, else ~/.config/abstract, as the ``abstract`` CLI saves it."""
    if directory := env.get("ABSTRACT_CONFIG_DIR"):
        return Path(directory) / CREDENTIALS
    if config_home := env.get("XDG_CONFIG_HOME"):
        return Path(config_home) / "abstract" / CREDENTIALS
    return home / ".config" / "abstract" / CREDENTIALS


def read_credentials(path: Path) -> dict[str, str] | None:
    """The saved credentials, or None when this machine is not signed in."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise UploadError(f"{path}: {exc.strerror}") from None
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    fields = ("apiKey", "keyId", "baseUrl", "createdAt")
    if (
        not isinstance(value, dict)
        or value.get("version") != 1
        or not all(isinstance(value.get(name), str) for name in fields)
    ):
        raise UploadError(
            f"{path} is not a credentials file this version reads; run abstract login"
        )
    return {name: value[name] for name in fields}


def api_credentials(
    config: StoreConfig, env: Mapping[str, str] | None = None, home: Path | None = None
) -> tuple[str, str]:
    """(API base URL, key) for ``route = "abstract"``: the key from $ABSTRACT_API_KEY, else
    from ``abstract login``'s credentials.json; the URL from [store] api_url, else the one
    saved with the key, else https://api.abstract.inc/v1."""
    env = os.environ if env is None else env
    path = credentials_path(env, Path.home() if home is None else home)
    key = env.get(API_KEY_ENV)
    try:
        saved = read_credentials(path)
    except UploadError:
        if not key:
            raise
        saved = None  # the environment's key needs no file; an unreadable one is not used
    key = key or (saved or {}).get("apiKey")
    if not key:
        raise UploadError(f"no Abstract API key: set {API_KEY_ENV}, or run abstract login")
    url = config.api_url or (saved or {}).get("baseUrl") or DEFAULT_API_URL
    try:
        return plain_api_url(url), key
    except ValueError as exc:
        raise UploadError(str(exc)) from None


def _row(card: Mapping[str, Any]) -> dict[str, Any]:
    """The catalog row for a run: the card without markers, long text cut so it fits."""
    row = episode.row(card)
    if len(json.dumps(row).encode()) <= MAX_ROW_BYTES:
        return row
    return {
        key: value[:2000] + "…" if isinstance(value, str) and len(value) > 2000 else value
        for key, value in row.items()
    }


class ApiStore:
    """The Abstract runs API, under the key's account (``route = "abstract"``)."""

    def __init__(
        self,
        robot: str,
        base_url: str,
        key: str,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.robot = robot
        self.base_url = base_url
        self.description = f"{base_url} (robot {robot}, key {safe_prefix(key)})"
        self._runs = f"{base_url}/robot/runs"
        self._auth = {"authorization": f"Bearer {key}"}
        self._client = client or httpx.Client(timeout=httpx.Timeout(120.0, connect=20.0))
        self._sleep = sleep

    def close(self) -> None:
        self._client.close()

    def _url(self, run: str, kind: str = "", path: str = "") -> str:
        url = f"{self._runs}/{self.robot}/{run}"
        if kind:
            url += f"/{kind}"
        if path:
            url += "/" + quote(path, safe="/-_.~")
        return url

    def _send(
        self,
        method: str,
        url: str,
        *,
        ok: Iterable[int] = (200, 201, 204),
        missing: bool = False,
        body: Callable[[], Iterable[bytes] | bytes] | None = None,
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response | None:
        """One API call, retried on 5xx, 429 and transport errors. 404 is None when
        ``missing`` allows it; any other status outside ``ok`` is an UploadError naming the
        API's error code. ``body`` makes the content anew for each attempt."""
        what = f"{method} {url.removeprefix(self._runs)}"
        sent_headers = {**self._auth, **(headers or {})}
        for attempt in range(cloud.ATTEMPTS):
            try:
                content = body() if body is not None else None
                response = self._client.request(
                    method, url, headers=sent_headers, content=content, **kwargs
                )
                if response.status_code >= 500 or response.status_code == 429:
                    raise cloud._Retryable(f"{what}: {_error(response)}")
            except (httpx.TransportError, cloud._Retryable) as exc:
                if attempt == cloud.ATTEMPTS - 1:
                    raise UploadError(f"{what}: {type(exc).__name__}: {exc}") from exc
                self._sleep(2 * 4**attempt)
                continue
            if response.status_code == 404 and missing:
                return None
            if response.status_code not in ok:
                raise UploadError(f"{what}: {_error(response)}")
            return response
        raise AssertionError("unreachable")

    def _json(self, response: httpx.Response | None, what: str) -> dict[str, Any]:
        try:
            value = response.json() if response is not None else None
        except ValueError:
            value = None
        if not isinstance(value, dict):
            raise UploadError(f"{what}: the API answered with no JSON object")
        return value

    def stored(self, run: str, path: str) -> tuple[int, str] | None:
        """(size, sha256) of a stored file, or None when the run does not hold it."""
        response = self._send("HEAD", self._url(run, "files", path), ok=(200,), missing=True)
        if response is None:
            return None
        size = int(response.headers.get("content-length", -1))
        return size, response.headers.get("x-content-sha256", "")

    def put_file(
        self, run: str, path: str, source: Path | bytes, *, resume_dir: Path | None = None
    ) -> tuple[cloud.Digest, bool]:
        """Store one file unless the API already holds these exact bytes. True when sent."""
        digest = cloud.Digest.of(source)
        state_path = None
        if resume_dir is not None:
            name = hashlib.sha256(self._url(run, "uploads", path).encode()).hexdigest()
            state_path = resume_dir / f"{name}.json"
        if self.stored(run, path) == (digest.size, digest.sha256):
            # A prior attempt may be replacing these bytes with a different version.
            # Do not forget its id while it still holds quota and a path reservation.
            if saved := _load_upload(state_path):
                url = self._url(run, "uploads", path)
                response = self._send(
                    "GET", url, params={"upload_id": saved["upload_id"]}, missing=True, ok=(200,)
                )
                if response is not None:
                    status = self._json(response, f"GET uploads/{path}")
                    if status.get("state") != "complete":
                        self._send(
                            "DELETE", url, params={"upload_id": saved["upload_id"]}, missing=True
                        )
            _clear_upload(state_path)
            return digest, False
        if digest.size > SINGLE_PUT_BYTES:
            stored = self._put_parts(run, path, source, digest, state_path)
        else:
            # The file may have shrunk below the multipart threshold since a failed
            # attempt. Release that upload's reservation before replacing it.
            if saved := _load_upload(state_path):
                self._send(
                    "DELETE",
                    self._url(run, "uploads", path),
                    params={"upload_id": saved["upload_id"]},
                    missing=True,
                )
                _clear_upload(state_path)
            headers = {"content-length": str(digest.size), "x-content-sha256": digest.sha256}
            response = self._send(
                "PUT",
                self._url(run, "files", path),
                body=lambda: cloud._chunks(source),
                headers=headers,
                ok=(200, 201),
            )
            stored = self._json(response, f"PUT {path}")
        if (stored.get("size"), stored.get("sha256")) != (digest.size, digest.sha256):
            raise UploadError(f"{path}: the API stored {stored}, sent {digest.size} bytes")
        _clear_upload(state_path)
        return digest, True

    def _put_parts(
        self,
        run: str,
        path: str,
        source: Path | bytes,
        digest: cloud.Digest,
        state_path: Path | None,
    ) -> dict[str, Any]:
        url = self._url(run, "uploads", path)
        saved = _load_upload(state_path)
        upload_id, status = None, None
        if saved is not None:
            upload_id = saved["upload_id"]
            if (saved.get("size"), saved.get("sha256"), saved.get("part_bytes")) != (
                digest.size,
                digest.sha256,
                PART_BYTES,
            ):
                # A changed local file cannot reuse the old upload's reserved size or parts.
                self._send("DELETE", url, params={"upload_id": upload_id}, missing=True)
                _clear_upload(state_path)
                upload_id = None
            else:
                response = self._send(
                    "GET", url, params={"upload_id": upload_id}, missing=True, ok=(200,)
                )
                if response is None:
                    _clear_upload(state_path)
                    upload_id = None
                else:
                    status = self._json(response, f"GET uploads/{path}")
                    if (status.get("size"), status.get("sha256")) != (
                        digest.size,
                        digest.sha256,
                    ):
                        raise UploadError(f"{path}: the saved upload describes another file")
                    if status.get("state") not in {
                        "uploading",
                        "assembling",
                        "publishing",
                        "complete",
                    }:
                        raise UploadError(f"{path}: the saved upload has an unknown state")
                    if status.get("state") == "complete":
                        # A completion receipt is historical: another upload may have
                        # replaced or deleted this path since the receipt was written.
                        if self.stored(run, path) == (digest.size, digest.sha256):
                            return status
                        _clear_upload(state_path)
                        upload_id, status = None, None
        if upload_id is None:
            response = self._send(
                "POST",
                url,
                headers={"x-content-sha256": digest.sha256, "x-content-length": str(digest.size)},
                ok=(200, 201),
            )
            upload_id = self._json(response, f"POST uploads/{path}").get("upload_id")
            if not isinstance(upload_id, str) or not upload_id:
                raise UploadError(f"POST uploads/{path}: no upload_id")
            _save_upload(
                state_path,
                {
                    "upload_id": upload_id,
                    "path": path,
                    "size": digest.size,
                    "sha256": digest.sha256,
                    "part_bytes": PART_BYTES,
                },
            )

        parts = _uploaded_parts(status, digest.size, path)

        def send_part(number: int) -> dict[str, Any]:
            offset = (number - 1) * PART_BYTES
            length = min(PART_BYTES, digest.size - offset)
            data = _read(source, offset, length)
            if len(data) != length:
                raise UploadError(f"{path}: the local file changed while uploading")
            response = self._send(
                "PUT",
                url,
                params={"upload_id": upload_id, "part": number},
                body=lambda: data,
                headers={"content-length": str(length)},
                ok=(200,),
            )
            etag = self._json(response, f"PUT {path} part {number}").get("etag")
            if not isinstance(etag, str) or not etag:
                raise UploadError(f"PUT {path} part {number}: no etag")
            return {"part": number, "etag": etag}

        finalizing = status is not None and status.get("state") in {"assembling", "publishing"}
        missing = iter(
            number
            for number in range(1, (digest.size + PART_BYTES - 1) // PART_BYTES + 1)
            if number not in parts and not finalizing
        )
        # Read inside each worker, and queue only one task per worker. Part bodies and
        # retries together hold at most PART_WORKERS * PART_BYTES (128 MiB by default).
        with ThreadPoolExecutor(max_workers=PART_WORKERS) as pool:
            pending = {pool.submit(send_part, number) for number in islice(missing, PART_WORKERS)}
            try:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        part = future.result()
                        parts[part["part"]] = part
                    pending.update(
                        pool.submit(send_part, number) for number in islice(missing, len(done))
                    )
            finally:
                for future in pending:
                    future.cancel()
        # Completion is idempotent. Keep the upload id even after a failed/ambiguous
        # response: a retry reconciles server status and finishes the same upload.
        response = self._send(
            "POST",
            url,
            params={"upload_id": upload_id},
            json={"parts": [parts[number] for number in sorted(parts)], "size": digest.size},
            ok=(200, 201),
        )
        return self._json(response, f"POST uploads/{path} (complete)")

    def complete(
        self, run: str, files: list[dict[str, Any]], row: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Close the run: the API checks every listed file is there with its size and
        nothing else is, and only then writes its manifest. The counts must match."""
        response = self._send(
            "POST", self._url(run, "complete"), json={"files": files, "episode": row}, ok=(200,)
        )
        done = self._json(response, f"complete {run}")
        total = sum(item["size"] for item in files)
        if (done.get("files"), done.get("bytes")) != (len(files), total):
            raise UploadError(
                f"complete {run}: the API counted {done.get('files')} files and "
                f"{done.get('bytes')} bytes, sent {len(files)} and {total}"
            )
        return done

    def upload_run(self, run_dir: Path) -> dict[str, Any]:
        run = run_dir.name
        card, stored, paths = cloud.prepare_run(run_dir, self.robot)
        files, sent = [], 0
        for path in paths:
            relative = path.relative_to(run_dir).as_posix()
            digest, was_sent = self.put_file(
                run, relative, path, resume_dir=run_dir.parent / UPLOAD_STATE
            )
            sent += was_sent
            files.append({"path": relative, "size": digest.size, "sha256": digest.sha256})
        # The raw frames' count goes into the manifest's catalog row with the card.
        done = self.complete(run, files, {**_row(card), "frames": stored})
        record = {
            **episode.row(card),
            "store": "abstract",
            "api_url": self.base_url,
            "robot": self.robot,
            "completed_at": done.get("completed_at"),
            "objects": len(files),
            "bytes": done["bytes"],
            "frames": stored,
        }
        return {**record, "sent": sent}

    def relabel(
        self,
        run: str,
        add: Iterable[str] = (),
        remove: Iterable[str] = (),
        overrides: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        response = self._send("GET", self._url(run), ok=(200,), missing=True)
        if response is None:
            raise UploadError(f"{run}: not a completed run of {self.robot} in the API")
        manifest = self._json(response, f"GET {run}")
        listed = [item for item in manifest.get("files", []) if isinstance(item, dict)]
        files = [item for item in listed if item.get("path") != episode.EPISODE]
        with tempfile.TemporaryDirectory(prefix="abstract-relabel-") as tmp:
            work = Path(tmp) / run
            for item in listed:
                if not cloud.CARD_SOURCES.fullmatch(str(item["path"])):
                    continue
                got = self._send(
                    "GET", self._url(run, "files", item["path"]), ok=(200,), missing=True
                )
                data = got.content if got is not None else None
                digest = cloud.Digest.of(data) if data is not None else None
                if digest is None or (digest.size, digest.sha256) != (
                    item.get("size"),
                    item.get("sha256"),
                ):
                    raise UploadError(f"{run}: {item['path']} is not the file its manifest lists")
                target = work / item["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            card = cloud.rebuild_card(work, self.robot, add, remove, overrides)
        digest, _ = self.put_file(run, episode.EPISODE, episode.dumps(card))
        files.append({"path": episode.EPISODE, "size": digest.size, "sha256": digest.sha256})
        frames = (manifest.get("episode") or {}).get("frames")
        done = self.complete(run, files, {**_row(card), "frames": frames})
        return {
            **episode.row(card),
            "store": "abstract",
            "api_url": self.base_url,
            "robot": self.robot,
            "completed_at": done.get("completed_at"),
            "objects": len(files),
            "bytes": done["bytes"],
        }

    def labelled(self, records: list[dict[str, Any]]) -> None:
        """The API's listing is the catalog: each ``complete`` updated it."""

    def runs(self) -> list[str]:
        runs, cursor = [], None
        while True:
            params = {"robot": self.robot, "limit": 1000, **({"cursor": cursor} if cursor else {})}
            response = self._send("GET", self._runs, params=params, ok=(200,))
            page = self._json(response, "GET runs")
            runs += [str(row.get("run")) for row in page.get("runs", []) if isinstance(row, dict)]
            cursor = page.get("cursor")
            if not cursor:
                return runs

    def owns(self, record: Mapping[str, Any]) -> bool:
        return record.get("store") == "abstract" and record.get("robot") == self.robot

    def check(self) -> tuple[bool, str]:
        try:
            self._send("GET", self._runs, params={"limit": 1}, ok=(200,))
        except UploadError as exc:
            return False, f"{self.description}: {exc}"
        return True, f"{self.description}: the runs API answers"


class BucketStore:
    """An R2 bucket the rig writes over its S3 API (``route = "bucket"``, ``cloud.py``)."""

    def __init__(self, config: StoreConfig, robot: str, bucket: cloud.Bucket) -> None:
        self.robot = robot
        self.prefix = config.prefix or robot
        self.bucket = bucket
        self.description = f"{bucket.name}/{self.prefix}"

    def close(self) -> None:
        self.bucket.__exit__()

    def upload_run(self, run_dir: Path) -> dict[str, Any]:
        return cloud.upload_run(self.bucket, self.prefix, self.robot, run_dir)

    def relabel(
        self,
        run: str,
        add: Iterable[str] = (),
        remove: Iterable[str] = (),
        overrides: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return cloud.relabel(self.bucket, self.prefix, self.robot, run, add, remove, overrides)

    def labelled(self, records: list[dict[str, Any]]) -> None:
        cloud.update_catalog(self.bucket, self.prefix, records)

    def runs(self) -> list[str]:
        return cloud.catalog_runs(self.bucket, self.prefix)

    def owns(self, record: Mapping[str, Any]) -> bool:
        return str(record.get("prefix", "")).startswith(self.prefix + "/")

    def check(self) -> tuple[bool, str]:
        """Write one small object and read its size and ETag back: the token, the bucket
        and the clock (SigV4 refuses a skewed one) all work. Overwritten on every check."""
        key = f"{self.prefix}/.check.json"
        data = json.dumps({"checked_at": datetime.now(UTC).isoformat(timespec="seconds")})
        try:
            self.bucket.store(key, data.encode(), "application/json")
        except (UploadError, httpx.HTTPError, ValueError) as exc:
            return False, f"{self.bucket.name}/{key}: {exc}"
        return True, f"{self.bucket.name}/{key}: write and read-back OK"


def _error(response: httpx.Response) -> str:
    """HTTP status and the API's error code and message, if it sent them."""
    try:
        error = response.json().get("error") or {}
    except (ValueError, AttributeError):
        error = {}
    if isinstance(error, dict) and (error.get("code") or error.get("message")):
        return f"HTTP {response.status_code} {error.get('code')}: {error.get('message')}"
    return f"HTTP {response.status_code} {response.text[:300]}".rstrip()


def _read(source: Path | bytes, offset: int, size: int) -> bytes:
    if isinstance(source, bytes):
        return source[offset : offset + size]
    with source.open("rb") as f:
        f.seek(offset)
        return f.read(size)


def _load_upload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        saved = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise UploadError(f"cannot read upload progress {path}: {exc}") from exc
    if not isinstance(saved, dict) or not isinstance(saved.get("upload_id"), str):
        raise UploadError(f"invalid upload progress in {path}")
    return saved


def _save_upload(path: Path | None, saved: dict[str, Any]) -> None:
    if path is None:
        return
    temporary = None
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(saved, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        with_directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(with_directory)
        finally:
            os.close(with_directory)
    except OSError as exc:
        raise UploadError(f"cannot save upload progress {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _clear_upload(path: Path | None) -> None:
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise UploadError(f"cannot clear upload progress {path}: {exc}") from exc


def _uploaded_parts(
    status: dict[str, Any] | None, size: int, path: str
) -> dict[int, dict[str, Any]]:
    if status is None:
        return {}
    result = {}
    parts = status.get("parts")
    if not isinstance(parts, list):
        raise UploadError(f"GET uploads/{path}: no parts list")
    count = (size + PART_BYTES - 1) // PART_BYTES
    for item in parts:
        if not isinstance(item, dict):
            raise UploadError(f"GET uploads/{path}: invalid part")
        number, etag = item.get("part"), item.get("etag")
        if (
            type(number) is not int
            or not 1 <= number <= count
            or number in result
            or not isinstance(etag, str)
            or not etag
            or item.get("size") != min(PART_BYTES, size - (number - 1) * PART_BYTES)
        ):
            raise UploadError(f"GET uploads/{path}: invalid part")
        result[number] = {"part": number, "etag": etag}
    return result


def connect(config: RobotConfig, bucket: Callable[[StoreConfig], cloud.Bucket]) -> Store:
    """The store [store] names, connected (``bucket`` opens an R2 bucket with its
    credentials); UploadError or ValueError when it cannot be."""
    store, robot = config.store, config.rig.name
    if store is None:
        raise ValueError("the robot config has no [store] section")
    if store.route == "abstract":
        base_url, key = api_credentials(store)
        return ApiStore(robot, base_url, key)
    return BucketStore(store, robot, bucket(store))


@contextmanager
def opened(config: RobotConfig, bucket: Callable[[StoreConfig], cloud.Bucket]) -> Iterator[Store]:
    target = connect(config, bucket)
    try:
        yield target
    finally:
        target.close()


def offload(target: Store, log_root: Path, run_dir: Path) -> dict[str, Any]:
    """Upload a finished run; delete raw frames only after rechecking export receipts."""
    from abstract_harness.robot.formats import verified_exports

    if not cloud.run_finished(run_dir):
        raise UploadError(f"{run_dir.name} is still being written")
    record = target.upload_run(run_dir)
    # Upload success and safe local deletion are different decisions. Legacy,
    # interrupted and degraded recordings have no completion evidence, so retain
    # their raw frames even after the selected artifacts reach the remote store.
    verified = False
    for name in cloud.SUMMARY_FILES:
        path = run_dir / name
        if path.is_file():
            try:
                formats = json.loads(path.read_text()).get("formats", {})
                verified = verified_exports(run_dir, formats)
            except (ValueError, OSError, AttributeError):
                verified = False
            break
    raw = any((run_dir / cloud.FRAMES).rglob("*.npy"))
    retained = raw and not verified
    if retained:
        record = {
            **record,
            "local_retained": True,
            "retention_reason": "exports not verified; raw frames retained",
        }
    cloud.remove_run(log_root, run_dir, record, delete=not retained)
    return record
