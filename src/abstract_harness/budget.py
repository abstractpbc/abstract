"""Conservative local accounting, not a provider-side hard spending limit."""

from decimal import Decimal

from abstract_harness.config import Limits, Pricing
from abstract_harness.models import Usage


class BudgetExceeded(RuntimeError):
    pass


def usage_cost(usage: Usage, pricing: Pricing) -> Decimal:
    uncached = usage.input_tokens - usage.cached_input_tokens
    return (
        Decimal(uncached) * Decimal(str(pricing.input_per_million))
        + Decimal(usage.cached_input_tokens) * Decimal(str(pricing.cached_input_per_million))
        + Decimal(usage.output_tokens) * Decimal(str(pricing.output_per_million))
    ) / Decimal(1_000_000)


class Budget:
    def __init__(self, limits: Limits, pricing: Pricing):
        self.limits = limits
        self.pricing = pricing
        self.calls = 0
        self.spent = Decimal(0)
        self.uncertain = False
        self._reserved: Decimal | None = None

    def reserve(self) -> Decimal:
        amount = Decimal(str(self.limits.call_reservation_usd))
        if self.pricing.input_per_million == self.pricing.output_per_million == 0:
            amount = Decimal(0)
        if self._reserved is not None:
            raise RuntimeError("a model request is already in flight")
        if self.uncertain:
            raise BudgetExceeded("billing is uncertain; further model calls are blocked")
        if self.calls >= self.limits.max_model_calls:
            raise BudgetExceeded("model call limit exhausted")
        if self.spent + amount > Decimal(str(self.limits.budget_usd)):
            raise BudgetExceeded("insufficient budget for another model-call reservation")
        self.calls += 1
        self._reserved = amount
        return amount

    def settle(self, usage: Usage | None, *, attempted: bool = True) -> None:
        if self._reserved is None:
            raise RuntimeError("no model-call reservation")
        reserved, self._reserved = self._reserved, None
        if not attempted:
            self.calls -= 1
            return
        if usage is None:
            # Missing usage/timeouts are never logged as free calls. This is an
            # estimate, NOT an upper bound on what the remote provider charged.
            self.spent += reserved
            self.uncertain = True
            raise BudgetExceeded(
                "model billing unknown; reservation retained and execution blocked"
            )
        self.spent += usage_cost(usage, self.pricing)
        if self.spent > Decimal(str(self.limits.budget_usd)):
            raise BudgetExceeded("reported model cost exceeded budget; execution blocked")
