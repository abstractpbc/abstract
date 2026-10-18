"""Registry and decorators for robot components.

Components register by name via decorators and are resolved from strings (so
``eval(policy="scripted")`` and the CLI work). The core's own components load on the
first lookup, then the bundled ones: the agent and YAM components in the harness's other
robot packages (``BUNDLED``), whose optional dependencies may be missing. A bundled
component that fails to import is skipped with a warning.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar

Kind = str  # "task" | "policy" | "embodiment" | "scorer" | "grader" | "sink" | "operator_input"
KINDS: tuple[Kind, ...] = (
    "task",
    "policy",
    "embodiment",
    "scorer",
    "grader",
    "sink",
    "operator_input",
)

# Components of the harness's agent and YAM packages, as "module:attribute".
BUNDLED: dict[Kind, dict[str, str]] = {
    "policy": {
        "agent": "abstract_harness.robot.agent.policy:agent_policy",
        "policy": "abstract_harness.robot.vla:vla_policy",
    },
    "embodiment": {"yam_arms": "abstract_harness.robot.yam.embodiment:YAMEmbodiment"},
}


class Bundled(NamedTuple):
    """One ``BUNDLED`` entry, imported by ``load()``."""

    name: str
    value: str

    def load(self) -> Callable[..., Any]:
        """Import the module and return the factory."""
        module, _, attribute = self.value.partition(":")
        return getattr(importlib.import_module(module), attribute)  # type: ignore[no-any-return]


def bundled(kind: Kind) -> list[Bundled]:
    """The bundled components of ``kind``."""
    return [Bundled(name, value) for name, value in BUNDLED.get(kind, {}).items()]


_FACTORIES: dict[Kind, dict[str, Callable[..., Any]]] = {k: {} for k in KINDS}
_loaded_bundled = False
_loaded_builtins = False

F = TypeVar("F", bound=Callable[..., Any])


def register(kind: Kind, name: str | None = None) -> Callable[[F], F]:
    """Register a factory under ``kind``/``name`` (defaults to its ``__name__``)."""
    if kind not in _FACTORIES:
        raise ValueError(f"unknown registry kind {kind!r}; valid: {KINDS}")

    def decorator(factory: F) -> F:
        key = name or getattr(factory, "__name__", None)
        if key is None:
            raise ValueError("cannot determine a registry name for the factory")
        _FACTORIES[kind][key] = factory
        return factory

    return decorator


def task(name: str | None = None) -> Callable[[F], F]:
    """Decorator: register a task factory under ``name``."""
    return register("task", name)


def policy(name: str | None = None) -> Callable[[F], F]:
    """Decorator: register a policy factory under ``name``."""
    return register("policy", name)


def embodiment(name: str | None = None) -> Callable[[F], F]:
    """Decorator: register an embodiment factory under ``name``."""
    return register("embodiment", name)


def scorer(name: str | None = None) -> Callable[[F], F]:
    """Decorator: register a scorer factory under ``name``."""
    return register("scorer", name)


def grader(name: str | None = None) -> Callable[[F], F]:
    """Decorator: register a grader factory under ``name``."""
    return register("grader", name)


def sink(name: str | None = None) -> Callable[[F], F]:
    """Decorator: register a log-sink factory under ``name``."""
    return register("sink", name)


def operator_input(name: str | None = None) -> Callable[[F], F]:
    """Decorator: register an operator-input factory under ``name``."""
    return register("operator_input", name)


def _ensure_loaded() -> None:
    global _loaded_builtins, _loaded_bundled
    if not _loaded_builtins:
        # Set the flag only once the import has succeeded: a raising import that
        # left it set would make every later call take the "already loaded" path
        # and serve an empty registry instead of re-raising.
        import abstract_harness.robot.core._builtins  # noqa: F401  (registers builtin components)

        _loaded_builtins = True
    if not _loaded_bundled:
        for kind in KINDS:
            for ep in bundled(kind):
                try:
                    factory = ep.load()
                except Exception as exc:
                    # A component whose dependencies are missing must not crash
                    # discovery, but it must not vanish silently either.
                    warnings.warn(
                        f"failed to load bundled {kind} {ep.name!r} from {ep.value!r}: {exc!r}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                _FACTORIES[kind].setdefault(ep.name, factory)
        _loaded_bundled = True


def registered(kind: Kind) -> dict[str, Callable[..., Any]]:
    """Return all registered factories for ``kind`` (core and bundled)."""
    if kind not in _FACTORIES:
        raise ValueError(f"unknown registry kind {kind!r}; valid: {KINDS}")
    _ensure_loaded()
    return dict(_FACTORIES[kind])


def resolve(kind: Kind, name: str, /, **kwargs: Any) -> Any:
    """Construct a registered component by name with the given keyword args."""
    factories = registered(kind)
    if name not in factories:
        raise KeyError(f"no {kind} named {name!r}; available: {sorted(factories)}")
    return factories[name](**kwargs)
