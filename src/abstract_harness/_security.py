"""Credential-safe configuration snapshots. Values never belong in run metadata."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_SENSITIVE = re.compile(
    r"(?:^|_)(?:api_key|access_key|secret_key|secret|token|password|authorization|credentials)$"
)


def sensitive_argument(name: str) -> bool:
    """Environment-variable references are public configuration; their values are not."""
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower().replace("-", "_")
    return bool(_SENSITIVE.search(normalized))


def redact(value: Any) -> Any:
    """Recursively remove credential fields from a snapshot, including overrides."""
    if isinstance(value, Mapping):
        return {
            str(key): "[redacted]" if sensitive_argument(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value
