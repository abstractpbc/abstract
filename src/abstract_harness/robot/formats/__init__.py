"""The formats a finished run is kept in, chosen per robot TOML (``[data] formats``).

``write`` runs after the episode and before any upload. Raw-frame deletion requires
matching export completion receipts as well as successful transport. Each format puts its
output inside the run directory, so every transport uploads it as ordinary files:

- ``rrd``: the Rerun recording the episode wrote live (cameras, joint state, actions,
  markers). Nothing to convert; the format checks it is there.
- ``lerobot``: a LeRobot v3.0 dataset in ``lerobot/`` (``lerobot.py``), built from the
  raw frames, the per-step state and actions, and the embodiment's own description.

Formats know no robot: every name, size and rate comes from the run (``run.py``), so the
same code serves a YAM, a Franka or a UR. When ``rrd`` is not chosen, the live recording
is removed from the run once every chosen format has succeeded, so it is not uploaded;
if one fails, the recording stays as the run's data. ``write`` never raises: a format
that fails is recorded with its error and the upload goes on.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from abstract_harness.robot.formats import lerobot, run

FORMATS = ("rrd", "lerobot")
LEROBOT_DIR = "lerobot"


def _recordings(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*.rrd"))


def _digest(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha}


def _verified_recordings(run_dir: Path) -> bool:
    found = _recordings(run_dir)
    covered: dict[str, str] = {}
    for path in found:
        receipt = json.loads(path.with_suffix(".complete.json").read_text())
        if not isinstance(receipt, dict):
            return False
        frames = receipt.pop("frames", {})
        if receipt != _digest(path) or not isinstance(frames, dict):
            return False
        covered.update(frames)
    raw = {}
    for frame in sorted((run_dir / "frames").rglob("*.npy")):
        with frame.open("rb") as stream:
            raw[frame.relative_to(run_dir).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return bool(found) and raw == covered


def _dataset_manifest(directory: Path) -> dict[str, Any]:
    return {
        path.relative_to(directory).as_posix(): _digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != directory / ".complete.json"
    }


def verified_exports(run_dir: Path, results: Any) -> bool:
    """Recheck completion evidence against the actual files before deleting raw data.

    Summary flags alone are never evidence: files may have changed since export.
    Older runs without receipts remain recoverable locally.
    """
    if not isinstance(results, dict):
        return False
    selected = {
        name: value
        for name, value in results.items()
        if not (isinstance(value, dict) and "removed" in value)
    }
    if not selected:
        return False
    try:
        for name, value in selected.items():
            if not isinstance(value, dict) or value.get("ok") is not True:
                return False
            if name == "rrd":
                if not _verified_recordings(run_dir):
                    return False
            elif name == "lerobot":
                directory = run_dir / LEROBOT_DIR
                manifest = _dataset_manifest(directory)
                if (
                    not manifest
                    or json.loads((directory / ".complete.json").read_text()) != manifest
                ):
                    return False
            else:
                return False
    except (OSError, ValueError, TypeError):
        return False
    return True


def write(run_dir: Path, formats: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Each chosen format's result: ``{"ok": True, ...}`` or ``{"ok": False, "error"}``."""
    results: dict[str, dict[str, Any]] = {}
    for name in formats:
        try:
            if name == "rrd":
                found = _recordings(run_dir)
                if not found:
                    raise run.RunError(
                        "the run has no .rrd (Rerun recording off or rerun-sdk missing)"
                    )
                try:
                    verified = _verified_recordings(run_dir)
                except (OSError, ValueError, TypeError):
                    verified = False
                results[name] = {
                    "ok": True,
                    "files": [path.name for path in found],
                    "verified": verified,
                }
            elif name == "lerobot":
                written = lerobot.write(run.read(run_dir), run_dir / LEROBOT_DIR)
                directory = run_dir / LEROBOT_DIR
                receipt = directory / ".complete.json"
                temporary = receipt.with_suffix(".part")
                temporary.unlink(missing_ok=True)
                manifest = _dataset_manifest(directory)
                temporary.write_text(json.dumps(manifest, sort_keys=True))
                temporary.replace(receipt)
                results[name] = {"ok": True, "verified": True, **written}
            else:
                raise run.RunError(f"unknown format {name!r}; known: {', '.join(FORMATS)}")
        except Exception as exc:  # a format never holds the run's upload back
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if "rrd" not in formats and results and all(result["ok"] for result in results.values()):
        removed = []
        for path in _recordings(run_dir):
            path.unlink()
            path.with_suffix(".complete.json").unlink(missing_ok=True)
            removed.append(path.name)
        if removed:
            results["rrd"] = {"ok": True, "removed": removed}
    return results
