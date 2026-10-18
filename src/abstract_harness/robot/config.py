"""Operator configuration for live model control of the physical YAM rig.

One TOML holds everything: the policy, the rig (its name, yam_arms settings, where runs
are written), run limits and where finished runs go (``[store]``).
``RobotConfig.policy_args`` and ``embodiment_args`` are the constructor arguments the run
gives the policy and ``yam_arms``; ``setup`` records them, and the run directory keeps that
record as ``setup.json``.
"""

from __future__ import annotations

import os
import re
import socket
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlparse

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from abstract_harness._security import redact, sensitive_argument
from abstract_harness.config import Nonnegative, Pricing
from abstract_harness.models import Record

# A hard ceiling on one run's model budget, whatever the config says.
MAX_BUDGET_USD = 5.0
# Policy arguments the harness sets from the config: the spend meter depends on the
# agent's, and a VLA's server comes from [vla] only.
PROTECTED_AGENT_ARGS = frozenset(
    {
        "model",
        "wire",
        "base_url",
        "api_key_env",
        "effort",
        "max_llm_calls",
        "max_output_tokens",
        "timeout_s",
    }
)
# The agent policy waits this much longer than the meter's upstream timeout, so the meter
# answers first (and settles the call) rather than the policy timing out under it.
AGENT_TIMEOUT_MARGIN_S = 10.0
PROTECTED_VLA_ARGS = frozenset({"server_url"})

EMBODIMENT = "yam_arms"

# Constructor arguments for a policy or embodiment, as TOML values. A list reaches the
# component as one comma-separated string, the form of its -E/-P KEY=VALUE flags, which
# is how the YAM embodiment reads joint limits.
Scalar = StrictBool | StrictInt | StrictFloat | StrictStr
Args = dict[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")], Scalar | list[Scalar]]


def parse_value(text: str) -> Any:
    """A -E/-P ``KEY=VALUE`` value: true/false, none or null, an integer or a float
    becomes that type; anything else, or a value in matching quotes, stays a string
    (``-P effort="'none'"`` sends the string ``none`` instead of omitting the parameter).
    """
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            continue
    return text


def _list_item(key: str, value: Scalar) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if "," in value:
        raise ValueError(f"{key}: a list item cannot contain a comma: {value!r}")
    return value


def component_args(args: Mapping[str, Scalar | list[Scalar]]) -> dict[str, Scalar]:
    """Arguments as the component receives them (see ``Args``)."""
    return {
        key: ", ".join(_list_item(key, item) for item in value)
        if isinstance(value, list)
        else value
        for key, value in args.items()
    }


def _check_args(section: str, args: Mapping[str, Scalar | list[Scalar]]) -> None:
    if any(sensitive_argument(key) for key in args):
        raise ValueError(f"[{section}] credentials must use an environment-variable reference")
    try:
        component_args(args)
    except ValueError as exc:
        raise ValueError(f"[{section}] {exc}") from None


class AgentPricing(Pricing):
    """Current USD per million tokens, supplied by the operator; never guessed."""

    # Prompt-cache writes can cost more than plain input. When set, every uncached input
    # token is billed at the higher of the two rates, because usage does not say which
    # tokens were written to the cache.
    cache_write_per_million: Nonnegative | None = None


class AgentConfig(Record):
    model: Annotated[str, Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$")]
    pricing: AgentPricing
    # OpenAI bills a request with more input tokens than this at long-context rates. The
    # meter refuses any request whose input estimate exceeds it; a call reported above it
    # anyway is settled at long_context_pricing, or blocks further calls when that is unset.
    long_context_threshold: Annotated[int, Field(ge=1_000, le=10_000_000)] = 272_000
    long_context_pricing: AgentPricing | None = None
    upstream: str = "https://api.openai.com/v1"
    api_key_env: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*$")] = "OPENAI_API_KEY"
    effort: Literal["none", "minimal", "low", "medium", "high"] = "low"
    max_output_tokens: Annotated[int, Field(ge=256, le=32_768)] = 4096
    max_calls: Annotated[int, Field(ge=1, le=100)] = 40
    budget_usd: Annotated[float, Field(gt=0, le=MAX_BUDGET_USD)] = 2.0
    # How long the meter waits for the upstream; the agent policy waits
    # AGENT_TIMEOUT_MARGIN_S longer per attempt (120 s by default) before it retries.
    request_timeout_s: Annotated[float, Field(gt=0, le=600)] = 110.0
    max_request_bytes: Annotated[int, Field(ge=1024, le=32_000_000)] = 16_000_000
    # Agent policy arguments other than those the meter pins.
    args: Args = Field(default_factory=dict)

    @model_validator(mode="after")
    def explicit_and_secure(self) -> Self:
        if protected := sorted(PROTECTED_AGENT_ARGS & set(self.args)):
            raise ValueError(f"[agent.args] cannot set {protected}: the harness sets them")
        _check_args("agent.args", self.args)
        url = urlparse(self.upstream)
        if not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("upstream must be a plain API base URL without credentials or query")
        local = url.hostname in {"127.0.0.1", "localhost", "::1"}
        if url.scheme != "https" and not (url.scheme == "http" and local):
            raise ValueError("HTTPS required except for loopback endpoints")
        for pricing in (self.pricing, self.long_context_pricing):
            if pricing and (pricing.input_per_million == 0 or pricing.output_per_million == 0):
                raise ValueError(
                    "live runs need the model's current nonzero input and output prices"
                )
        return self

    def policy_args(self, base_url: str | None = None) -> dict[str, Scalar | list[Scalar]]:
        """The agent policy's arguments: [agent.args] plus what the meter pins. The
        meter's base_url exists only once it listens, so a run passes it in."""
        pinned: dict[str, Scalar | list[Scalar]] = {
            "model": self.model,
            "wire": "responses",
            "api_key_env": self.api_key_env,
            # A string, so "none" reaches the wire as a string instead of Python None,
            # which the policy would drop, leaving thinking on.
            "effort": self.effort,
            "max_llm_calls": self.max_calls,
            # The same cap the meter enforces and reserves for, now sent by the policy.
            "max_output_tokens": self.max_output_tokens,
            "timeout_s": self.request_timeout_s + AGENT_TIMEOUT_MARGIN_S,
        }
        if base_url is not None:
            pinned["base_url"] = base_url
        return {**self.args, **pinned}


class ActionPolicyConfig(Record):
    """An action policy (VLA or World-Action Model) on a cloud `/v1` gateway, a local
    `/v1` or `/act` server, or an in-process local model (`target` in `args`)."""

    policy: Literal["policy", "molmoact2", "gr00t"] = "policy"
    server_url: str = "https://api.abstract.inc/v1"
    # The policy client's arguments other than server_url, e.g. model, target, or camera size.
    args: Args = Field(default_factory=dict)

    @model_validator(mode="after")
    def plain_url(self) -> Self:
        if self.policy in ("molmoact2", "gr00t") and "server_url" not in self.model_fields_set:
            raise ValueError("server_url is required for [vla] molmoact2 and gr00t policies")
        if PROTECTED_VLA_ARGS & set(self.args):
            raise ValueError("[vla.args] cannot set server_url; set it in [vla]")
        _check_args("vla.args", self.args)
        if (
            self.policy == "policy"
            and "target" in self.args
            and "server_url" not in self.model_fields_set
        ):
            return self
        url = urlparse(self.server_url)
        allowed_paths = ("", "/", "/v1", "/v1/") if self.policy == "policy" else ("", "/")
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
            or url.path not in allowed_paths
            or url.query
            or url.fragment
        ):
            raise ValueError("server_url must be a plain http(s) base URL, e.g. http://host:8202")
        return self

    def policy_args(self) -> dict[str, Scalar | list[Scalar]]:
        if (
            self.policy == "policy"
            and "target" in self.args
            and "server_url" not in self.model_fields_set
        ):
            return dict(self.args)
        return {**self.args, "server_url": self.server_url}


VlaConfig = ActionPolicyConfig


# A robot's name, as the runs API takes it (`robot`): lower case, so one rig is one name.
ROBOT_NAME = r"^[a-z0-9][a-z0-9._-]{0,63}$"
DEFAULT_API_URL = "https://api.abstract.inc/v1"


def plain_api_url(url: str) -> str:
    """An API base URL with no credentials, query or fragment; HTTPS unless loopback."""
    parts = urlparse(url)
    if not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError(
            f"expected a plain API base URL without credentials, e.g. {DEFAULT_API_URL}"
        )
    local = parts.hostname in {"127.0.0.1", "localhost", "::1"}
    if parts.scheme != "https" and not (parts.scheme == "http" and local):
        raise ValueError("HTTPS required except for loopback endpoints")
    return url.rstrip("/")


class StoreConfig(Record):
    """Where finished runs go (``store.py``). ``route = "abstract"``: the Abstract API, under
    the account of the API key (``ABSTRACT_API_KEY``, else the key ``abstract login`` saved).
    ``route = "bucket"``: an R2 bucket the rig writes itself over its S3 API (``cloud.py``)."""

    route: Literal["abstract", "bucket"]
    # abstract: the API base, /v1 included. Default: the one saved with the key by
    # `abstract login`, else https://api.abstract.inc/v1.
    api_url: str | None = None
    # bucket: the name of an R2 bucket you own (required), the first key segment (default:
    # [rig] name), and a KEY=VALUE file with R2_ACCOUNT_ID, R2_RUNS_ACCESS_KEY_ID,
    # R2_RUNS_SECRET_ACCESS_KEY.
    bucket: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")] | None = None
    prefix: Annotated[str, Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")] | None = None
    env_file: Path = Path("~/.config/abstract/r2.env")

    @model_validator(mode="after")
    def route_fields(self) -> Self:
        bucket_fields = {"bucket", "prefix", "env_file"} & self.model_fields_set
        if self.route == "abstract" and bucket_fields:
            raise ValueError(f'[store] {sorted(bucket_fields)} are for route = "bucket"')
        if self.route == "bucket" and self.api_url is not None:
            raise ValueError('[store] api_url is for route = "abstract"')
        if self.route == "bucket" and self.bucket is None:
            raise ValueError(
                '[store] route = "bucket" needs bucket, the name of an R2 bucket you own'
            )
        if self.api_url is not None:
            plain_api_url(self.api_url)
        return self


class DataConfig(Record):
    """What a finished run is kept as (``formats/``), written into the run before any
    upload: ``rrd``, the Rerun recording the episode writes live; ``lerobot``, a LeRobot
    v3.0 dataset in ``lerobot/``. Without ``rrd``, the recording is dropped once every
    chosen format has succeeded."""

    formats: Annotated[list[Literal["rrd", "lerobot"]], Field(min_length=1)] = Field(
        default_factory=lambda: ["rrd"]
    )

    @model_validator(mode="after")
    def distinct(self) -> Self:
        if len(set(self.formats)) != len(self.formats):
            raise ValueError("[data] formats names a format twice")
        return self


def default_rig_name(hostname: str | None = None) -> str:
    """This machine's hostname as a robot name: lower case, other characters as "-"."""
    name = re.sub(r"[^a-z0-9._-]+", "-", (hostname or socket.gethostname()).lower())
    name = name.lstrip("._-")[:64]
    return name or "robot"


def default_log_root(env: Mapping[str, str] | None = None) -> Path:
    """$XDG_DATA_HOME/abstract/robot/runs, else ~/.local/share/abstract/robot/runs."""
    data = (os.environ if env is None else env).get("XDG_DATA_HOME")
    return (Path(data) if data else Path.home() / ".local" / "share") / "abstract/robot/runs"


class RigConfig(Record):
    # The robot's name in the runs store; default this machine's hostname.
    name: Annotated[str, Field(pattern=ROBOT_NAME)] = Field(default_factory=default_rig_name)
    # Registered embodiment name (default "yam_arms"; e.g. "franka", "so100", "cubepick").
    type: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")] = EMBODIMENT
    # Embodiment constructor arguments: CAN channels, cameras, control mode, limits.
    embodiment: Args = Field(default_factory=dict)
    # Relative paths resolve against the directory of the TOML file.
    log_root: Path = Field(default_factory=default_log_root)
    max_steps: Annotated[int, Field(ge=1, le=6000)] = 1200

    @model_validator(mode="after")
    def channels_explicit(self) -> Self:
        # Which CAN interface drives which arm depends on the rig's wiring, and yam_arms'
        # defaults (left can0, right can1) are only a guess; never rely on them.
        if self.type == EMBODIMENT and not {"left_channel", "right_channel"} <= set(
            self.embodiment
        ):
            raise ValueError("[rig.embodiment] must set left_channel and right_channel")
        _check_args("rig.embodiment", self.embodiment)
        return self


class RobotConfig(Record):
    agent: AgentConfig | None = None
    vla: ActionPolicyConfig | None = None
    rig: RigConfig
    # Without it, runs stay in rig.log_root.
    store: StoreConfig | None = None
    # What a run is kept as; the Rerun recording unless it says otherwise.
    data: DataConfig = Field(default_factory=DataConfig)

    @model_validator(mode="before")
    @classmethod
    def no_cloud(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "cloud" in data:
            raise ValueError(
                '[cloud] is now [store] with route = "bucket" (bucket, prefix and env_file '
                'move into it), or route = "abstract" for the Abstract API'
            )
        if isinstance(data, Mapping) and "policy" in data and "vla" not in data:
            copied = dict(data)
            copied["vla"] = copied.pop("policy")
            return copied
        return data

    @model_validator(mode="after")
    def one_policy(self) -> Self:
        if (self.agent is None) == (self.vla is None):
            raise ValueError(
                "configure exactly one policy: [agent] (an OpenAI model) or [vla] (an /act server)"
            )
        return self

    @property
    def policy_name(self) -> str:
        return "agent" if self.agent is not None else self.vla.policy  # type: ignore[union-attr]

    def policy_args(self, base_url: str | None = None) -> dict[str, Scalar]:
        """The policy's constructor arguments; an agent's base_url is the meter's."""
        if self.agent is not None:
            return component_args(self.agent.policy_args(base_url))
        return component_args(self.vla.policy_args())  # type: ignore[union-attr]

    def embodiment_args(self) -> dict[str, Scalar]:
        """yam_arms' constructor arguments."""
        return component_args(self.rig.embodiment)

    def setup(self, base_url: str | None = None) -> dict[str, Any]:
        """Every argument a run of this config gives its components."""
        return redact(
            {
                "policy": self.policy_name,
                "policy_args": self.policy_args(base_url),
                "embodiment": self.rig.type,
                "embodiment_args": self.embodiment_args(),
                "max_steps": self.rig.max_steps,
            }
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        with path.open("rb") as f:
            loaded = cls.model_validate(tomllib.load(f))
        base = path.resolve().parent

        def absolute(value: Path) -> Path:
            value = value.expanduser()
            return value if value.is_absolute() else base / value

        rig, store = loaded.rig, loaded.store
        return loaded.model_copy(
            update={
                "rig": rig.model_copy(update={"log_root": absolute(rig.log_root)}),
                "store": store and store.model_copy(update={"env_file": absolute(store.env_file)}),
            }
        )
