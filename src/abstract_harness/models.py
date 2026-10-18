"""Closed, data-only Pydantic base record and token usage model."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class Usage(Record):
    input_tokens: Annotated[int, Field(ge=0, strict=True)]
    cached_input_tokens: Annotated[int, Field(ge=0, strict=True)]
    output_tokens: Annotated[int, Field(ge=0, strict=True)]

    @model_validator(mode="after")
    def cached_subset(self) -> Self:
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached tokens exceed input tokens")
        return self
