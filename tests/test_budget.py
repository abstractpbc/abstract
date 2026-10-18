from decimal import Decimal

import pytest

from abstract_harness.budget import Budget, BudgetExceeded, usage_cost
from abstract_harness.config import Limits, Pricing
from abstract_harness.models import Usage

PRICES = Pricing(input_per_million=2, cached_input_per_million=0.5, output_per_million=10)


def test_cached_input_discount_and_no_reasoning_double_count():
    usage = Usage(input_tokens=1000, cached_input_tokens=500, output_tokens=100)
    assert usage_cost(usage, PRICES) == Decimal("0.00225")


def test_reservation_prevents_call():
    budget = Budget(Limits(budget_usd=0.01, call_reservation_usd=0.02), PRICES)
    with pytest.raises(BudgetExceeded, match="reservation"):
        budget.reserve()
    assert budget.calls == 0


def test_call_limit():
    budget = Budget(Limits(max_model_calls=1), PRICES)
    budget.reserve()
    budget.settle(Usage(input_tokens=0, cached_input_tokens=0, output_tokens=0))
    with pytest.raises(BudgetExceeded, match="call limit"):
        budget.reserve()


def test_timeout_never_refunded_as_free():
    budget = Budget(Limits(), PRICES)
    budget.reserve()
    with pytest.raises(BudgetExceeded, match="unknown"):
        budget.settle(None)
    assert budget.spent == Decimal("0.1")
    assert budget.uncertain
    with pytest.raises(BudgetExceeded, match="uncertain"):
        budget.reserve()


def test_actual_overspend_blocks_after_accounting():
    budget = Budget(Limits(budget_usd=0.1, call_reservation_usd=0.01), PRICES)
    budget.reserve()
    with pytest.raises(BudgetExceeded, match="exceeded"):
        budget.settle(Usage(input_tokens=0, cached_input_tokens=0, output_tokens=100_000))
    assert budget.spent == Decimal("1")


def test_local_request_rejection_is_not_billed():
    budget = Budget(Limits(), PRICES)
    budget.reserve()
    budget.settle(None, attempted=False)
    assert budget.spent == 0
    assert budget.calls == 0
    assert not budget.uncertain
    budget.reserve()  # Local validation did not consume a remote-call slot.
