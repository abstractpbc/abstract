"""Smoke tests for the package: it imports and has a version."""

from __future__ import annotations

import abstract_harness.robot.core


def test_has_version() -> None:
    assert isinstance(abstract_harness.robot.core.__version__, str)
    assert abstract_harness.robot.core.__version__


def test_public_api_is_fenced() -> None:
    # Everything reachable as public must be declared in __all__ (no accidental
    # surface growth). This guard tightens as the API grows.
    assert "__version__" in abstract_harness.robot.core.__all__
