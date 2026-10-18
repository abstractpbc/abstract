"""The LLM agent policy: a frontier model drives the embodiment through tool calls.

An LLM behind a supported API (OpenRouter, OpenAI, local vLLM/Ollama,
Anthropic, Tinker, or compatible gateways) drives whatever embodiment it is
paired with: each tool call becomes one smooth, approver-checked action
chunk. ``wire=messages`` selects the Messages API; the permanent
``anthropic`` alias remains accepted. ``wire=interactions`` selects
Google's stateful Interactions HTTP API. Registered as the policy ``agent``;
programmatically::

    from abstract_harness.robot.core import eval
    from abstract_harness.robot.agent import LLMAgentPolicy

    eval("my-task", LLMAgentPolicy(model="openai/gpt-5.2"), "cubepick")
"""

from __future__ import annotations

from importlib.metadata import version

from abstract_harness.robot.agent._anthropic import AnthropicClient
from abstract_harness.robot.agent._gemini_live import GeminiLiveClient
from abstract_harness.robot.agent._interactions import InteractionsClient
from abstract_harness.robot.agent._llm import (
    ENV_MODEL,
    AssistantMessage,
    ChatClient,
    Provider,
    resolve_provider,
)
from abstract_harness.robot.agent._png import encode_png, png_data_url
from abstract_harness.robot.agent._responses import ResponsesClient
from abstract_harness.robot.agent.policy import AgentPolicyConfig, LLMAgentPolicy, agent_policy

__all__ = [
    "ENV_MODEL",
    "AgentPolicyConfig",
    "AnthropicClient",
    "AssistantMessage",
    "ChatClient",
    "GeminiLiveClient",
    "InteractionsClient",
    "LLMAgentPolicy",
    "Provider",
    "ResponsesClient",
    "agent_policy",
    "encode_png",
    "png_data_url",
    "resolve_provider",
]

# Derived from package metadata so it can never drift from pyproject again
# (0.2.0 and 0.2.1 shipped with a stale hardcoded "0.1.0").
__version__ = version("abstract-harness")
