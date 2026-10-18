"""Validate release artifacts without importing the source checkout or dependencies."""

from __future__ import annotations

import email
import sys
import tarfile
import zipfile
from pathlib import Path


def check(directory: Path) -> None:
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("expected exactly one wheel and one source distribution")
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        metadata_path = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = email.message_from_bytes(wheel.read(metadata_path))
        license_root = metadata_path.removesuffix("METADATA") + "licenses/"
        required = {
            "abstract_harness/py.typed",
            "abstract_harness/robot/yam/assets/yam_collision.xml",
            "abstract_harness/robot/yam/assets/MENAGERIE_LICENSE",
            *(license_root + name for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")),
        }
        if missing := required - names:
            raise SystemExit(f"wheel is missing {sorted(missing)}")
        notices = wheel.read(license_root + "THIRD_PARTY_NOTICES.md").decode()
        if "Permission is hereby granted" not in notices or "Copyright" not in notices:
            raise SystemExit("third-party license grant and attribution must ship in the wheel")
        for requirement in metadata.get_all("Requires-Dist", []):
            if requirement.lower().startswith("i2rt"):
                raise SystemExit("wheel must not resolve a bare or mutable i2rt dependency")
        if metadata["License-Expression"] != "Apache-2.0 AND MIT":
            raise SystemExit("unexpected SPDX license expression")
        if any("/__pycache__/" in name or name.endswith(".pyc") for name in names):
            raise SystemExit("wheel contains generated Python caches")
    with tarfile.open(sdists[0], "r:gz") as source:
        names = {name.partition("/")[2] for name in source.getnames()}
        required = {
            "pyproject.toml",
            "uv.lock",
            "LICENSE",
            "NOTICE",
            "THIRD_PARTY_NOTICES.md",
            "README.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "docs/SAFETY.md",
            "scripts/check_dist.py",
            "scripts/smoke_installed.py",
            "tests/test_release_safety.py",
        }
        if missing := required - names:
            raise SystemExit(f"source distribution is missing {sorted(missing)}")
    print("wheel and source distribution: metadata, assets, typing and license checks passed")


if __name__ == "__main__":
    check(Path(sys.argv[1] if len(sys.argv) > 1 else "dist"))
