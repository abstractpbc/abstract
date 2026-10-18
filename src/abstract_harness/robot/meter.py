"""Spend meter between the agent policy and the OpenAI Responses API.

The agent policy (``robot/agent``) tracks no dollars and has no spend limit: this meter
is the only spend accounting. ``abstract robot run`` points the policy's ``base_url`` at
this loopback proxy, which:

- forwards only non-streaming ``POST /v1/responses`` for the configured model;
- caps ``max_output_tokens`` (the policy sends the same ``[agent]`` value);
- reserves a conservative worst case before each call and refuses with HTTP 402 when the
  remaining budget cannot cover it, or when the input estimate exceeds the long-context
  threshold (OpenAI bills such requests at higher rates);
- settles each call from the reported ``usage``, at long-context rates above the threshold.
  A call whose billing is unknown (timeout, 5xx, missing usage, or long-context usage
  without configured long-context rates) keeps its reservation and blocks every further
  call.

The policy does not retry a 402, so a refusal ends the trial and the episode parks the
arms. Dollar figures are local estimates, not a provider-side cap. The proxy forwards the
caller's Authorization header and never stores or logs it.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self

import httpx
from pydantic import ValidationError

from abstract_harness.budget import BudgetExceeded, usage_cost
from abstract_harness.models import Usage
from abstract_harness.robot.config import AgentConfig

ROUTE = "/v1/responses"
# Upper bound on the input tokens of one inline image; a 640x480 frame costs well under
# it with OpenAI's tile and patch formulas.
IMAGE_TOKEN_BOUND = 2_000
# Message framing and other tokens not visible as text.
FRAMING_TOKENS = 1_000
MILLION = Decimal(1_000_000)


def estimate_input_tokens(body: Any) -> int:
    """Conservative upper bound: two characters per token plus a bound per image."""
    chars, images, stack = 0, 0, [body]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, str):
            if value.startswith("data:image/"):
                images += 1
            else:
                chars += len(value)
    return chars // 2 + images * IMAGE_TOKEN_BOUND + FRAMING_TOKENS


def response_usage(payload: Any) -> Usage | None:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return None
    details = usage.get("input_tokens_details")
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    try:
        return Usage(
            input_tokens=usage["input_tokens"],
            cached_input_tokens=cached,
            output_tokens=usage["output_tokens"],
        )
    except (KeyError, ValidationError):
        return None


def refused(payload: Any) -> bool:
    output = payload.get("output") if isinstance(payload, dict) else None
    for item in output if isinstance(output, list) else []:
        content = item.get("content") if isinstance(item, dict) else None
        for part in content if isinstance(content, list) else []:
            if isinstance(part, dict) and part.get("type") == "refusal":
                return True
    return False


def money(value: Decimal) -> str:
    return f"{value:.6f}"


class SpendMeter:
    """Accounting for one run. The agent sends one request at a time; so does the meter."""

    def __init__(self, agent: AgentConfig, ledger: Path):
        self.agent = agent
        self.ledger = ledger
        self.budget = Decimal(str(agent.budget_usd))
        self.calls = 0
        self.spent = Decimal(0)
        self.uncertain = False
        self.refusals = 0
        self.incomplete = 0
        self.blocked = 0
        self._in_flight = False
        self._lock = threading.Lock()

    def long_context(self, usage: Usage) -> bool:
        return usage.input_tokens > self.agent.long_context_threshold

    def cost(self, usage: Usage) -> Decimal | None:
        """USD for one call; None when it is long-context and those rates are not set."""
        pricing = self.agent.pricing
        if self.long_context(usage):
            if self.agent.long_context_pricing is None:
                return None
            pricing = self.agent.long_context_pricing
        amount = usage_cost(usage, pricing)
        if pricing.cache_write_per_million is not None:
            uplift = Decimal(str(pricing.cache_write_per_million)) - Decimal(
                str(pricing.input_per_million)
            )
            if uplift > 0:
                uncached = usage.input_tokens - usage.cached_input_tokens
                amount += Decimal(uncached) * uplift / MILLION
        return amount

    def worst_case(self, input_tokens: int) -> Decimal:
        """Short-context rates: reserve() refuses any estimate above the threshold."""
        pricing = self.agent.pricing
        input_rate = max(
            Decimal(str(pricing.input_per_million)),
            Decimal(str(pricing.cache_write_per_million or 0)),
        )
        output = Decimal(self.agent.max_output_tokens) * Decimal(str(pricing.output_per_million))
        return (Decimal(input_tokens) * input_rate + output) / MILLION

    def reserve(self, body: dict[str, Any]) -> Decimal:
        with self._lock:
            try:
                if self._in_flight:
                    raise BudgetExceeded("a model request is already in flight")
                if self.uncertain:
                    raise BudgetExceeded("billing is uncertain; further model calls are blocked")
                if self.calls >= self.agent.max_calls:
                    raise BudgetExceeded(f"model call limit ({self.agent.max_calls}) exhausted")
                tokens = estimate_input_tokens(body)
                if tokens > self.agent.long_context_threshold:
                    raise BudgetExceeded(
                        f"input estimate of {tokens} tokens exceeds the long-context threshold "
                        f"({self.agent.long_context_threshold}); long-context calls are refused"
                    )
                amount = self.worst_case(tokens)
                if self.spent + amount > self.budget:
                    raise BudgetExceeded(
                        f"budget ${self.budget} cannot cover the next call: ${money(self.spent)} "
                        f"spent, up to ${money(amount)} more"
                    )
            except BudgetExceeded as exc:
                self.blocked += 1
                self._write({"event": "blocked", "reason": str(exc)})
                raise
            self.calls += 1
            self._in_flight = True
            return amount

    def settle(
        self,
        reserved: Decimal,
        *,
        status: int,
        payload: Any,
        elapsed_s: float,
        error: str | None = None,
    ) -> Decimal:
        """Record one finished request and return what it is charged.

        200 settles from usage. 5xx, transport errors, 200 without usage and long-context
        usage without long-context rates keep the reservation and block further calls. Other
        4xx (rate limit, rejected request) are not billed by the provider and give the call
        back.
        """
        billed = status == 200 or status >= 500
        usage = response_usage(payload) if status == 200 else None
        cost = self.cost(usage) if usage is not None else None
        with self._lock:
            self._in_flight = False
            if not billed:
                self.calls -= 1
                charged = Decimal(0)
            elif cost is None:
                charged = reserved
                self.uncertain = True
            else:
                charged = cost
            self.spent += charged
            refusal = status == 200 and refused(payload)
            incomplete = isinstance(payload, dict) and payload.get("status") == "incomplete"
            self.refusals += refusal
            self.incomplete += incomplete
            self._write(
                {
                    "event": "call",
                    "http_status": status,
                    "response_status": payload.get("status") if isinstance(payload, dict) else None,
                    "reserved_usd": money(reserved),
                    "charged_usd": money(charged),
                    "spent_usd": money(self.spent),
                    "usage": usage.model_dump() if usage else None,
                    "long_context": usage is not None and self.long_context(usage),
                    "billing_uncertain": billed and cost is None,
                    "refusal": refusal,
                    "incomplete": incomplete,
                    "elapsed_s": round(elapsed_s, 3),
                    "error": error,
                }
            )
            return charged

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "model": self.agent.model,
                "budget_usd": money(self.budget),
                "spent_usd": money(self.spent),
                "billing_uncertain": self.uncertain,
                "model_calls": self.calls,
                "blocked_calls": self.blocked,
                "refusals": self.refusals,
                "incomplete": self.incomplete,
            }

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps({"time": datetime.now(UTC).isoformat(), **record})
        with self.ledger.open("a") as f:
            f.write(line + "\n")


def _handler(meter: SpendMeter, client: httpx.Client) -> type[BaseHTTPRequestHandler]:
    agent = meter.agent

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Silent: the operator's terminal belongs to the episode."""

        def do_GET(self) -> None:
            self._error(404, f"only POST {ROUTE} is metered")

        def do_POST(self) -> None:
            if self.path != ROUTE:
                return self._error(404, f"only POST {ROUTE} is metered")
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= agent.max_request_bytes:
                self.close_connection = True
                return self._error(413, "request body missing or above max_request_bytes")
            try:
                body = json.loads(self.rfile.read(length))
            except ValueError:
                return self._error(400, "request body is not JSON")
            if not isinstance(body, dict) or body.get("model") != agent.model:
                return self._error(400, f"this meter only forwards model {agent.model!r}")
            if body.get("stream"):
                return self._error(400, "streaming responses are not metered")
            requested = body.get("max_output_tokens")
            if type(requested) is not int or not 0 < requested <= agent.max_output_tokens:
                body["max_output_tokens"] = agent.max_output_tokens
            try:
                reserved = meter.reserve(body)
            except BudgetExceeded as exc:
                return self._error(402, f"abstract spend meter: {exc}")
            headers = {"Content-Type": "application/json"}
            if auth := self.headers.get("Authorization"):
                headers["Authorization"] = auth
            started = time.monotonic()
            try:
                response = client.post("responses", json=body, headers=headers)
            except httpx.HTTPError as exc:
                meter.settle(
                    reserved,
                    status=504,
                    payload=None,
                    elapsed_s=time.monotonic() - started,
                    error=type(exc).__name__,
                )
                return self._error(
                    504,
                    f"upstream request failed ({type(exc).__name__}); "
                    "billing unknown, further calls blocked",
                )
            try:
                payload = response.json()
            except ValueError:
                payload = None
            meter.settle(
                reserved,
                status=response.status_code,
                payload=payload,
                elapsed_s=time.monotonic() - started,
            )
            self._send(
                response.status_code,
                response.content,
                response.headers.get("Content-Type", "application/json"),
                retry_after=response.headers.get("Retry-After"),
            )

        def _send(
            self, status: int, data: bytes, content_type: str, *, retry_after: str | None = None
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            if retry_after:
                self.send_header("Retry-After", retry_after)
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str) -> None:
            body = {"error": {"message": message, "type": "abstract_spend_meter"}}
            self._send(status, json.dumps(body).encode(), "application/json")

    return Handler


class MeterServer:
    """Loopback HTTP server for one run; its ``base_url`` goes to the agent's ``base_url``."""

    def __init__(self, meter: SpendMeter, *, transport: httpx.BaseTransport | None = None):
        self.meter = meter
        upstream = meter.agent.upstream.rstrip("/") + "/"
        self._client = httpx.Client(
            base_url=upstream, timeout=meter.agent.request_timeout_s, transport=transport
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(meter, self._client))
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="abstract-spend-meter", daemon=True
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._client.close()
