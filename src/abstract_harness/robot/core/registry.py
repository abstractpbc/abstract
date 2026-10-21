"""Registry and decorators for robot components.

Components register by name via decorators and are resolved from strings (so
``eval(policy="scripted")`` and the CLI work). The core's own components load on the
first lookup, then the bundled ones: the agent and YAM components in the harness's other
robot packages (``BUNDLED``), whose optional dependencies may be missing, then installed
plugins: every ``abstract_harness.plugins`` entry point is imported, and its decorators
register what it provides. A bundled component or plugin that fails to import is skipped
with a warning.

Wherever a component is named by a string (``--policy``, a rig's ``[vla] policy``, a
preset's ``driver``, a controller's ``kind``, an orchestrator extension), the string is a
registered name, or ``"package.module:attribute"`` or ``"path/to/file.py:attribute"`` for
an unregistered factory (``lookup``).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from collections.abc import Callable
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

from abstract_harness.robot.core.errors import ConfigError

# "task" | "policy" | "embodiment" | "scorer" | "grader" | "sink" | "operator_input" |
# "driver" (a robot preset's hardware driver) | "command_source" (a preset controller's
# live input) | "extension" (a function that registers orchestrator tools)
Kind = str
KINDS: tuple[Kind, ...] = (
    "task",
    "policy",
    "embodiment",
    "scorer",
    "grader",
    "sink",
    "operator_input",
    "driver",
    "command_source",
    "extension",
)
#: The entry-point group of installed plugins: each entry point names a module to import.
PLUGIN_GROUP = "abstract_harness.plugins"

# Components of the harness's agent and YAM packages, as "module:attribute".
BUNDLED: dict[Kind, dict[str, str]] = {
    "policy": {
        "agent": "abstract_harness.robot.agent.policy:agent_policy",
        "orchestrator": "abstract_harness.robot.orchestrator.policy:orchestrator_policy",
        "policy": "abstract_harness.robot.vla:vla_policy",
    },
    "embodiment": {
        "yam_arms": "abstract_harness.robot.yam.embodiment:YAMEmbodiment",
        "preset": "abstract_harness.robot.preset.embodiment:PresetEmbodiment",
    },
    "driver": {
        "sim": "abstract_harness.robot.preset.driver:SimDriver",
    },
}


class Bundled(NamedTuple):
    """One ``BUNDLED`` entry, imported by ``load()``."""

    name: str
    value: str

    def load(self) -> Callable[..., Any]:
        """Import the module and return the factory."""
        return load_object(self.value)  # type: ignore[no-any-return]


def bundled(kind: Kind) -> list[Bundled]:
    """The bundled components of ``kind``."""
    return [Bundled(name, value) for name, value in BUNDLED.get(kind, {}).items()]


_FACTORIES: dict[Kind, dict[str, Callable[..., Any]]] = {k: {} for k in KINDS}
_loaded_bundled = False
_loaded_builtins = False
_loaded_plugins = False

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


def driver(name: str | None = None) -> Callable[[F], F]:
    """Decorator: register a robot-preset driver factory under ``name``."""
    return register("driver", name)


def command_source(name: str | None = None) -> Callable[[F], F]:
    """Decorator: register a live-command source factory (a controller ``kind``)."""
    return register("command_source", name)


def extension(name: str | None = None) -> Callable[[F], F]:
    """Decorator: register an orchestrator extension (``register(harness)``) under ``name``."""
    return register("extension", name)


def _load_plugins() -> None:
    before = {kind: dict(factories) for kind, factories in _FACTORIES.items()}
    for ep in entry_points(group=PLUGIN_GROUP):
        try:
            ep.load()
        except Exception as exc:
            warnings.warn(
                f"failed to load plugin {ep.name!r} from {ep.value!r}: {exc!r}",
                RuntimeWarning,
                stacklevel=3,
            )
    for kind, factories in before.items():
        for name, factory in factories.items():
            if _FACTORIES[kind].get(name) is not factory:
                warnings.warn(
                    f"a plugin replaced the {kind} {name!r}", RuntimeWarning, stacklevel=3
                )


def _ensure_loaded() -> None:
    global _loaded_builtins, _loaded_bundled, _loaded_plugins
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
    if not _loaded_plugins:
        # Set first, so a plugin that looks a component up while importing does not
        # re-enter; cleared again if the scan itself raises, so the next call retries.
        _loaded_plugins = True
        try:
            _load_plugins()
        except BaseException:
            _loaded_plugins = False
            raise


def registered(kind: Kind) -> dict[str, Callable[..., Any]]:
    """Return all registered factories for ``kind`` (core and bundled)."""
    if kind not in _FACTORIES:
        raise ValueError(f"unknown registry kind {kind!r}; valid: {KINDS}")
    _ensure_loaded()
    return dict(_FACTORIES[kind])


def _split(spec: str, default: str | None) -> tuple[str, str]:
    target, sep, attribute = spec.rpartition(":")
    return (target, attribute) if sep else (spec, default or "")


def _file(target: str, base: Path | None) -> Path:
    path = Path(target).expanduser()
    return (path if path.is_absolute() else (base or Path.cwd()) / path).resolve()


def rebase(spec: str, base: Path) -> str:
    """``spec`` with a ``file.py`` reference made absolute against ``base`` (a config
    file's directory); registered names and module references are returned unchanged."""
    target, attribute = _split(spec, None)
    if not target.endswith(".py"):
        return spec
    path = str(_file(target, base))
    return f"{path}:{attribute}" if attribute else path


def load_object(spec: str, *, base: Path | None = None, default: str | None = None) -> Any:
    """Import ``"package.module:attribute"`` or ``"path/to/file.py:attribute"`` (a file
    relative to ``base``, default the working directory). Without ``:attribute``, the
    attribute is ``default``. Raises ``ConfigError``."""
    target, attribute = _split(spec, default)
    if not target or not attribute:
        raise ConfigError(f'{spec!r}: expected "module:attribute" or "file.py:attribute"')
    try:
        if target.endswith(".py"):
            path = _file(target, base)
            if not path.is_file():
                raise ConfigError(f"{spec!r}: no such file {path}")
            unique = f"_abstract_plugin_{abs(hash(str(path)))}"
            module = sys.modules.get(unique)
            if module is None:
                module_spec = importlib.util.spec_from_file_location(unique, path)
                if module_spec is None or module_spec.loader is None:
                    raise ConfigError(f"{spec!r}: cannot load {path}")
                module = importlib.util.module_from_spec(module_spec)
                sys.modules[unique] = module
                try:
                    module_spec.loader.exec_module(module)
                except BaseException:
                    del sys.modules[unique]
                    raise
        else:
            module = importlib.import_module(target)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"{spec!r}: import failed: {type(exc).__name__}: {exc}") from exc
    found = getattr(module, attribute, None)
    if found is None:
        raise ConfigError(f"{spec!r}: {target} has no {attribute}")
    return found


def lookup(
    kind: Kind, spec: str, /, *, base: Path | None = None, default: str | None = None
) -> Callable[..., Any]:
    """The factory a string names: a registered name, else ``module:attribute`` or
    ``file.py:attribute`` (relative to ``base``; a bare module or file names its
    ``default`` attribute, when given). Raises ``KeyError`` for an unknown name and
    ``ConfigError`` for a reference that does not load."""
    factories = registered(kind)
    if spec in factories:
        return factories[spec]
    if ":" not in spec and default is None:
        raise KeyError(f"no {kind} named {spec!r}; available: {sorted(factories)}")
    factory = load_object(spec, base=base, default=default)
    if not callable(factory):
        raise ConfigError(f"{spec!r} is not callable")
    return factory


def resolve(kind: Kind, name: str, /, **kwargs: Any) -> Any:
    """Construct a component by name (or ``module:attribute``) with the given keyword args."""
    return lookup(kind, name)(**kwargs)
