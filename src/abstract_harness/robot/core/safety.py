"""Shared safety construction for CLI and Python execution."""

from __future__ import annotations

from abstract_harness.robot.core.approver import Approver
from abstract_harness.robot.core.embodiment import Embodiment
from abstract_harness.robot.core.errors import ConfigError
from abstract_harness.robot.core.spaces import ABSOLUTE_CONTROL_MODES, Box


def build_guardrails(
    space: Box,
    max_action_delta: float | None,
    embodiment: Embodiment | None = None,
) -> tuple[Approver, list[str], list[str]]:
    """The default CLI safety chain for an action space (plan 0008 §3e).

    Returns ``(approver, active, warnings)``. Degrades per component instead
    of blocking: a component that cannot apply to this space is skipped with
    a warning naming the actual refusal reason (the constructor's message),
    so the CLI is never *less* protective than running without guardrails —
    and never silently unprotected either.
    """
    from abstract_harness.robot.core.approver import (
        AutoApprover,
        ChainApprover,
        ClampApprover,
        DeltaLimitApprover,
        GuardrailContribution,
    )

    parts: list[Approver] = []
    active: list[str] = []
    warnings: list[str] = []
    if (
        embodiment is not None
        and not embodiment.info.is_simulated
        and (space.semantics is None or space.low is None or space.high is None)
    ):
        raise ConfigError(
            "physical control requires declared action semantics and bounds; supply an "
            "explicit approver only when implementing your own safety controller"
        )
    if space.low is None and space.high is None:
        warnings.append("bounds clamp skipped: the action space declares no low/high bounds")
    else:
        parts.append(ClampApprover(space))
        active.append("clamp")
    try:
        # Catch the constructor's refusal generically — whatever the §3a
        # reason — rather than pre-checking an enumerated list.
        parts.append(DeltaLimitApprover(space, max_delta=max_action_delta))
        active.append("delta-limit")
    except ValueError as exc:
        warnings.append(f"delta limit skipped: {exc}")

    missing = object()
    hook = getattr(embodiment, "contribute_guardrails", missing)
    if hook is not missing:
        if not callable(hook):
            assert embodiment is not None
            raise ConfigError(
                f"embodiment {embodiment.info.name!r} contribute_guardrails is "
                f"not callable (got {type(hook).__name__})"
            )
        contribution = hook(space)
        if not isinstance(contribution, GuardrailContribution):
            assert embodiment is not None
            raise ConfigError(
                f"embodiment {embodiment.info.name!r} contribute_guardrails returned "
                f"{type(contribution).__name__}, expected GuardrailContribution"
            )
        for name, approver in contribution.approvers:
            parts.append(approver)
            active.append(name)
        warnings.extend(contribution.warnings)

    if not parts:
        warnings.append(
            "no guardrails are active for this action space; declare bounds/semantics "
            "on the embodiment or pass --max-action-delta"
        )
        return AutoApprover(), active, warnings
    physical_absolute = bool(
        embodiment is not None
        and not embodiment.info.is_simulated
        and space.semantics is not None
        and space.semantics.control_mode in ABSOLUTE_CONTROL_MODES
    )
    return ChainApprover(*parts, require_reference=physical_absolute), active, warnings
