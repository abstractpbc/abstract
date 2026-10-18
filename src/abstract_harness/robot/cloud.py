"""Finished robot runs on the rig's disk, and the R2 bucket route of ``[store]``.

A run directory holds the episode's eval log, its per-step action log, a Rerun recording
(cameras, joint state, actions), the camera frames stored as raw ``.npy`` (about 99% of
the bytes) and, for an agent run, transcripts, wire captures and the spend ledger.
``prepare_run``, the one step both routes take before sending anything, writes the run's
episode card (``episode.py``) into the run as ``episode.json`` and lists every file but the
raw frames; the Rerun recording carries the cameras, one JPEG image per camera per step.
The bucket route's ``upload_run`` then puts those files under
``<prefix>/<YYYY>/<MM>/<DD>/<run>/`` and checks each object's size and MD5 ETag in the
bucket. ``manifest.json`` goes after every other object and lists them, so its presence
marks a complete run; then the card, without its markers, replaces the run's line in the
bucket's catalog ``<prefix>/episodes.jsonl``. The caller deletes the run directory and
keeps one line per run in ``<log_root>/runs.jsonl``. A failure anywhere leaves the run on
disk for ``abstract robot upload`` to retry; objects already in the bucket are not sent
again. ``relabel`` rebuilds the card of a run already in the bucket from its uploaded files
(``rebuild_card``, shared too), with tags and overrides from ``abstract robot label``.
``store.py`` puts both routes behind one interface.

Requests are signed with AWS Signature Version 4 here, so no AWS SDK is needed.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import httpx

from abstract_harness.robot import episode
from abstract_harness.robot.config import StoreConfig

ACCOUNT_ENV = "R2_ACCOUNT_ID"
ACCESS_KEY_ENV = "R2_RUNS_ACCESS_KEY_ID"
SECRET_KEY_ENV = "R2_RUNS_SECRET_ACCESS_KEY"
CREDENTIAL_ENVS = (ACCOUNT_ENV, ACCESS_KEY_ENV, SECRET_KEY_ENV)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
CHUNK_BYTES = 1 << 20
# R2 takes at most 5 GiB in one PUT; a packed run stays far below it.
MAX_OBJECT_BYTES = 5 * 1024**3
ATTEMPTS = 3
SUMMARY_FILES = ("run-summary.json", "meter-summary.json")
# A run without a summary (abstract robot was killed, or the run predates it) counts as
# finished once nothing in it has changed for this long.
QUIET_SECONDS = 600
FRAMES = "frames"
MANIFEST = "manifest.json"
INDEX = "runs.jsonl"
LOCK = ".upload.lock"
RUN_NAME = re.compile(r"^(\d{4})(\d{2})(\d{2})-\d{6}$")
# What the manifest says about where a run is; the rest of it is the episode card's row.
LOCATION = ("bucket", "prefix", "uploaded_at", "objects", "bytes", "frames")
# The uploaded files an episode card is built from: top-level JSON (the eval log and the
# run or meter summary), the action log and an agent's transcript.
CARD_SOURCES = re.compile(r"[^/]+\.json|(?:actions|transcripts)/.+\.jsonl")
CONTENT_TYPES = {
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}


class UploadError(RuntimeError):
    """The run stays on disk; nothing was deleted."""


class _Retryable(UploadError):
    pass


@dataclass(frozen=True)
class Digest:
    size: int
    sha256: str
    md5: str

    @classmethod
    def of(cls, source: Path | bytes) -> Digest:
        sha256, md5, size = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
        for chunk in _chunks(source):
            sha256.update(chunk)
            md5.update(chunk)
            size += len(chunk)
        return cls(size, sha256.hexdigest(), md5.hexdigest())


def _chunks(source: Path | bytes) -> Iterator[bytes]:
    if isinstance(source, bytes):
        yield source
        return
    with source.open("rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            yield chunk


def sigv4_headers(
    method: str,
    url: str,
    *,
    access_key_id: str,
    secret_access_key: str,
    region: str,
    service: str,
    now: datetime,
    headers: Mapping[str, str] | None = None,
    payload_sha256: str = EMPTY_SHA256,
) -> dict[str, str]:
    """The headers that sign one request with AWS Signature Version 4, for S3 and R2.

    `url` is sent as given, so its path must already
    be percent-encoded; the canonical request uses it unchanged, which is what S3 expects.
    """
    parts = urlsplit(url)
    amz_date = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    signed = {key.lower(): value.strip() for key, value in (headers or {}).items()}
    signed.update(
        {"host": parts.netloc, "x-amz-content-sha256": payload_sha256, "x-amz-date": amz_date}
    )
    names = sorted(signed)
    pairs = sorted(
        (unquote(key), unquote(value))
        for key, _, value in (pair.partition("=") for pair in parts.query.split("&") if pair)
    )
    query = "&".join(
        f"{quote(key, safe='-_.~')}={quote(value, safe='-_.~')}" for key, value in pairs
    )
    canonical = "\n".join(
        [
            method,
            parts.path or "/",
            query,
            "".join(f"{name}:{signed[name]}\n" for name in names),
            ";".join(names),
            payload_sha256,
        ]
    )
    scope = f"{amz_date[:8]}/{region}/{service}/aws4_request"
    to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()]
    )
    key = f"AWS4{secret_access_key}".encode()
    for part in (amz_date[:8], region, service, "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    authorization = (
        f"AWS4-HMAC-SHA256 Credential={access_key_id}/{scope}, "
        f"SignedHeaders={';'.join(names)}, Signature={signature}"
    )
    return {**signed, "authorization": authorization}


class Bucket:
    """One R2 bucket over its S3 API, with a token for object read and write on it."""

    def __init__(
        self,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        name: str,
        client: httpx.Client | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{32}", account_id):
            raise ValueError(f"{ACCOUNT_ENV} must be the 32-character Cloudflare account ID")
        self.name = name
        self._origin = f"https://{account_id}.r2.cloudflarestorage.com"
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=20.0))
        self._clock = clock
        self._sleep = sleep

    @classmethod
    def connect(cls, config: StoreConfig, credentials: Mapping[str, str | None]) -> Bucket:
        missing = [name for name in CREDENTIAL_ENVS if not credentials.get(name)]
        if missing:
            raise UploadError(
                f"R2 credentials missing: {', '.join(missing)} (environment or {config.env_file})"
            )
        return cls(
            str(credentials[ACCOUNT_ENV]),
            str(credentials[ACCESS_KEY_ENV]),
            str(credentials[SECRET_KEY_ENV]),
            config.bucket,
        )

    def __enter__(self) -> Bucket:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.close()

    def _url(self, key: str) -> str:
        return f"{self._origin}/{self.name}/{quote(key, safe='/-_.~')}"

    def _signed(
        self, method: str, url: str, headers: Mapping[str, str] | None = None, sha256: str = ""
    ) -> dict[str, str]:
        signed = sigv4_headers(
            method,
            url,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            region="auto",
            service="s3",
            now=self._clock(),
            headers=headers,
            payload_sha256=sha256 or EMPTY_SHA256,
        )
        signed.pop("host")
        return signed

    def head(self, key: str) -> tuple[int, str] | None:
        """(size, ETag) of an object, or None when the bucket does not hold it."""
        url = self._url(key)
        # Cloudflare compresses JSON for a client that accepts gzip, and then answers with a
        # weak ETag and no length, which say nothing about the stored bytes.
        headers = {**self._signed("HEAD", url), "accept-encoding": "identity"}
        response = self._client.head(url, headers=headers)
        if response.status_code == 404:
            return None
        if response.status_code >= 500 or response.status_code == 429:
            raise _Retryable(f"HEAD {key}: HTTP {response.status_code}")
        if response.status_code != 200:
            raise UploadError(f"HEAD {key}: HTTP {response.status_code}")
        etag = response.headers.get("etag", "").strip('"')
        return int(response.headers.get("content-length", -1)), etag

    def _put(self, key: str, source: Path | bytes, digest: Digest, content_type: str) -> None:
        url = self._url(key)
        md5 = base64.b64encode(bytes.fromhex(digest.md5)).decode()
        headers = self._signed(
            "PUT", url, {"content-type": content_type, "content-md5": md5}, digest.sha256
        )
        response = self._client.put(
            url, content=_chunks(source), headers={**headers, "content-length": str(digest.size)}
        )
        if response.status_code >= 500 or response.status_code == 429:
            raise _Retryable(f"PUT {key}: HTTP {response.status_code}")
        if response.status_code != 200:
            raise UploadError(f"PUT {key}: HTTP {response.status_code} {response.text[:300]}")

    def store(self, key: str, source: Path | bytes, content_type: str) -> tuple[Digest, bool]:
        """Put one object unless the bucket already holds these exact bytes, then confirm
        its size and MD5 ETag there. True when it was sent."""
        digest = Digest.of(source)
        if digest.size > MAX_OBJECT_BYTES:
            raise UploadError(f"{key}: {digest.size} bytes is over R2's single-PUT limit")
        expected = (digest.size, digest.md5)
        sent = False
        for attempt in range(ATTEMPTS):
            try:
                if self.head(key) == expected:
                    return digest, sent
                self._put(key, source, digest, content_type)
                sent = True
                found = self.head(key)
            except (httpx.TransportError, _Retryable) as exc:
                if attempt == ATTEMPTS - 1:
                    raise UploadError(f"{key}: {type(exc).__name__}: {exc}") from exc
                self._sleep(2 * 4**attempt)
                continue
            if found != expected:
                raise UploadError(f"{key}: bucket holds {found}, sent {expected}")
            return digest, sent
        raise AssertionError("unreachable")

    def get(self, key: str) -> bytes | None:
        """An object's bytes, checked against its MD5 ETag, or None when it is absent."""
        url = self._url(key)
        for attempt in range(ATTEMPTS):
            try:
                headers = {**self._signed("GET", url), "accept-encoding": "identity"}
                response = self._client.get(url, headers=headers)
                if response.status_code >= 500 or response.status_code == 429:
                    raise _Retryable(f"GET {key}: HTTP {response.status_code}")
            except (httpx.TransportError, _Retryable) as exc:
                if attempt == ATTEMPTS - 1:
                    raise UploadError(f"{key}: {type(exc).__name__}: {exc}") from exc
                self._sleep(2 * 4**attempt)
                continue
            if response.status_code == 404:
                return None
            if response.status_code != 200:
                raise UploadError(f"GET {key}: HTTP {response.status_code}")
            body = response.content
            etag = response.headers.get("etag", "").strip('"')
            md5 = hashlib.md5(body, usedforsecurity=False).hexdigest()
            if etag != md5:
                raise UploadError(f"GET {key}: body does not match its ETag")
            return body
        raise AssertionError("unreachable")


def run_finished(run_dir: Path, now: float | None = None) -> bool:
    """A summary exists, or nothing in the run has changed for QUIET_SECONDS."""
    if any((run_dir / name).is_file() for name in SUMMARY_FILES):
        return True
    newest = max((path.stat().st_mtime for path in run_dir.rglob("*")), default=0.0)
    return (time.time() if now is None else now) - newest > QUIET_SECONDS


def finished_runs(log_root: Path) -> list[Path]:
    if not log_root.is_dir():
        return []
    return [
        path
        for path in sorted(log_root.iterdir())
        if path.is_dir() and not path.is_symlink() and RUN_NAME.match(path.name)
        if run_finished(path)
        if not (path / ".uploaded.json").is_file()
    ]


def key_prefix(prefix: str, run: str) -> str:
    match = RUN_NAME.match(run)
    if match is None:
        raise UploadError(f"{run} is not a run directory name (YYYYMMDD-HHMMSS)")
    year, month, day = match.groups()
    return f"{prefix}/{year}/{month}/{day}/{run}/"


def prepare_run(
    run_dir: Path, rig: str
) -> tuple[dict[str, Any], dict[str, int] | None, list[Path]]:
    """The one step before either route sends anything (packing the frames would go here):
    the run's episode card, written as ``episode.json``; how many raw frames it stored and
    leaves behind (``{"npy": N}``, so the recording's images can be checked against it);
    and the files to send, smallest first: everything but the frames and partial files."""
    frames = run_dir / FRAMES
    # Old versions allowed literal credentials in setup/overrides. Refuse to upload
    # those artifacts until the operator has repaired them; never print their values.
    setup = run_dir / "setup.json"
    if setup.is_file():
        from abstract_harness._security import redact

        try:
            original = json.loads(setup.read_text())
        except (ValueError, OSError) as exc:
            raise UploadError("cannot validate setup.json before upload") from exc
        if redact(original) != original:
            raise UploadError("setup.json contains credential fields; redact them before upload")
    stored = {"npy": sum(1 for _ in frames.rglob("*.npy"))} if frames.is_dir() else None
    try:
        card = episode.card(run_dir, rig=rig)
    except (TypeError, ValueError, KeyError, AttributeError, IndexError) as exc:
        # Labels never hold data back: the run goes up with what is known, and
        # `abstract robot label` rebuilds the card once the cause is fixed.
        card = {"run": run_dir.name, "rig": rig, "label_error": repr(exc)}
    (run_dir / episode.EPISODE).write_bytes(episode.dumps(card))
    files = [
        path
        for path in run_dir.rglob("*")
        if path.is_file()
        and frames not in path.parents
        and not path.name.endswith(".part")
        and path.relative_to(run_dir).as_posix() != MANIFEST
        and path.name != ".uploaded.json"
    ]
    return card, stored, sorted(files, key=lambda path: path.stat().st_size)


def rebuild_card(
    work: Path,
    rig: str,
    add: Iterable[str] = (),
    remove: Iterable[str] = (),
    overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The episode card of a run already stored, from its card sources fetched into
    ``work``: the tags and overrides it had are kept, then these edits applied."""
    stored = work / episode.EPISODE
    previous = json.loads(stored.read_text()) if stored.is_file() else {}
    stored.unlink(missing_ok=True)
    card = episode.card(work, rig=rig)
    tags = set(previous.get("tags", card["tags"])) | {episode.tag(value) for value in add}
    tags -= {episode.tag(value) for value in remove}
    episode.edit(card, tags, {**previous.get("overrides", {}), **(overrides or {})})
    return card


def upload_run(bucket: Bucket, prefix: str, rig: str, run_dir: Path) -> dict[str, Any]:
    """Every file of a finished run but its raw frames, then the manifest, then its
    catalog line. Deletes nothing."""
    base = key_prefix(prefix, run_dir.name)
    card, stored, files = prepare_run(run_dir, rig)
    objects, sent = [], 0
    for path in files:
        relative = path.relative_to(run_dir).as_posix()
        content_type = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        digest, was_sent = bucket.store(base + relative, path, content_type)
        sent += was_sent
        objects.append({"path": relative, "bytes": digest.size, "sha256": digest.sha256})
    record = {
        **episode.row(card),
        "bucket": bucket.name,
        "prefix": base,
        "uploaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "objects": len(objects) + 1,
        "bytes": sum(item["bytes"] for item in objects),
        "frames": stored,
    }
    store_manifest(bucket, base, record, objects)
    update_catalog(bucket, prefix, [record])
    return {**record, "sent": sent + 1}


def store_manifest(
    bucket: Bucket, base: str, record: Mapping[str, Any], objects: list[dict[str, Any]]
) -> None:
    data = (json.dumps({**record, "files": objects}, indent=2) + "\n").encode()
    bucket.store(base + MANIFEST, data, CONTENT_TYPES[".json"])


def update_catalog(bucket: Bucket, prefix: str, records: list[Mapping[str, Any]]) -> None:
    """Put each record's line into ``<prefix>/episodes.jsonl`` (one line per run, sorted by
    run), keeping every other line. Callers hold ``upload_lock``: one writer per rig."""
    key = f"{prefix}/{episode.CATALOG}"
    rows: dict[str, Any] = {}
    for line in (bucket.get(key) or b"").decode().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and isinstance(row.get("run"), str):
            rows[row["run"]] = row
    for record in records:
        rows[str(record["run"])] = {k: v for k, v in record.items() if k != "sent"}
    data = "".join(json.dumps(rows[run]) + "\n" for run in sorted(rows)).encode()
    bucket.store(key, data, CONTENT_TYPES[".jsonl"])


def catalog_runs(bucket: Bucket, prefix: str) -> list[str]:
    """Every run named in the bucket's catalog."""
    data = bucket.get(f"{prefix}/{episode.CATALOG}") or b""
    runs = []
    for line in data.decode().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and isinstance(row.get("run"), str):
            runs.append(row["run"])
    return runs


def relabel(
    bucket: Bucket,
    prefix: str,
    rig: str,
    run: str,
    add: Iterable[str] = (),
    remove: Iterable[str] = (),
    overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Rebuild the episode card of a run already in the bucket from its uploaded files,
    keep the tags and overrides it had, apply these edits, then store ``episode.json`` and
    the manifest again. Returns the run's record; the caller updates the catalog."""
    base = key_prefix(prefix, run)
    raw = bucket.get(base + MANIFEST)
    if raw is None:
        raise UploadError(f"{run}: no {MANIFEST} in {bucket.name}/{base}")
    manifest = json.loads(raw)
    files = [item for item in manifest["files"] if item["path"] != episode.EPISODE]
    with tempfile.TemporaryDirectory(prefix="abstract-relabel-") as tmp:
        work = Path(tmp) / run
        for item in manifest["files"]:
            if not CARD_SOURCES.fullmatch(item["path"]):
                continue
            data = bucket.get(base + item["path"])
            if data is None:
                raise UploadError(f"{run}: {item['path']} is listed but not in the bucket")
            target = work / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        card = rebuild_card(work, rig, add, remove, overrides)
    data = episode.dumps(card)
    digest, _ = bucket.store(base + episode.EPISODE, data, CONTENT_TYPES[".json"])
    files.append({"path": episode.EPISODE, "bytes": digest.size, "sha256": digest.sha256})
    record = {
        **episode.row(card),
        **{key: manifest.get(key) for key in LOCATION},
        "objects": len(files) + 1,
        "bytes": sum(item["bytes"] for item in files),
    }
    store_manifest(bucket, base, record, files)
    return record


def remove_run(
    log_root: Path, run_dir: Path, record: Mapping[str, Any], *, delete: bool = True
) -> None:
    """Delete an uploaded run, keeping its record in ``<log_root>/runs.jsonl``."""
    root = log_root.resolve()
    target = run_dir.resolve()
    if target.parent != root or run_dir.is_symlink() or not RUN_NAME.match(target.name):
        raise UploadError(f"refusing to delete {run_dir}: not a run directory in {log_root}")
    with (root / INDEX).open("a") as index:
        index.write(json.dumps(dict(record)) + "\n")
        index.flush()
        os.fsync(index.fileno())
    if delete:
        shutil.rmtree(target)
    else:
        receipt = target / ".uploaded.json"
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(record)) + "\n")
        temporary.replace(receipt)


@contextmanager
def upload_lock(log_root: Path) -> Iterator[None]:
    """One uploader per log root, so a run is never uploaded and deleted twice at once."""
    log_root.mkdir(parents=True, exist_ok=True)
    with (log_root / LOCK).open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise UploadError("another upload is running for this log root") from None
        yield
