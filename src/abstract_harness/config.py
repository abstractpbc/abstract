from __future__ import annotations

from typing import Annotated

from pydantic import Field

from abstract_harness.models import Record

Positive = Annotated[float, Field(gt=0)]
Nonnegative = Annotated[float, Field(ge=0)]


class Pricing(Record):
    """Operator-supplied USD per million tokens; never guess a provider's prices."""

    input_per_million: Nonnegative
    cached_input_per_million: Nonnegative
    output_per_million: Nonnegative


class Limits(Record):
    episode_timeout_s: Positive = 90.0
    observation_max_age_s: Positive = 2.0
    command_timeout_s: Positive = 2.0
    status_timeout_s: Positive = 0.5
    stop_timeout_s: Positive = 0.5
    lease_ttl_s: Positive = 3.0
    max_model_calls: Annotated[int, Field(ge=1, le=10)] = 2
    max_steps: Annotated[int, Field(ge=1, le=16)] = 16
    budget_usd: Positive = 0.50
    # A reservation, NOT an enforced cap on a provider's bill.
    call_reservation_usd: Positive = 0.10
    cache_max_age_s: Positive = 86400.0
