from __future__ import annotations


def test_package_imports_and_exports() -> None:
    from importlib.metadata import version

    import abstract_harness.robot.agent

    # The harness's own version, from its metadata.
    assert abstract_harness.robot.agent.__version__ == version("abstract-harness")
    assert callable(abstract_harness.robot.agent.agent_policy)
    assert abstract_harness.robot.agent.__all__ == [
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
