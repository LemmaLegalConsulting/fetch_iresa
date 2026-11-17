"""Offline regression coverage for provider routing and telemetry SDK calls."""

from unittest.mock import Mock, patch

import pytest

from app.providers.mistral import MistralProvider
from app.providers.openai import GPT52Provider, OpenAIProvider
from app.telemetry import langfuse_client


@pytest.fixture
def provider_env(monkeypatch):
    for name in (
        "OPENAI_BASE_URL_GPT_5_2",
        "OPENAI_GPT_5_2_API_KEY",
        "OPENAI_GPT_5_2_API_VERSION",
        "OPENAI_GPT_5_2_MODEL",
        "AZURE_MISTRAL_BASE_URL",
        "AZURE_MISTRAL_API_KEY",
        "AZURE_MISTRAL_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-general-key")
    return monkeypatch


def test_mistral_reuses_azure_credentials(provider_env):
    provider_env.setenv("OPENAI_BASE_URL_GPT_5_2", "https://example.test/openai/v1/")
    provider_env.setenv("AZURE_MISTRAL_MODEL", "my-mistral-deployment")
    with patch("app.providers.mistral.AsyncOpenAI") as client:
        provider = MistralProvider()
    assert provider.model_name == "my-mistral-deployment"
    assert provider.instance_name == "mistral"
    client.assert_called_once_with(
        base_url="https://example.test/openai/v1/",
        api_key="test-general-key",
        max_retries=0,
    )


def test_mistral_uses_dedicated_azure_resource(provider_env):
    provider_env.setenv("AZURE_MISTRAL_BASE_URL", "https://mistral.test/openai/v1/")
    provider_env.setenv("AZURE_MISTRAL_API_KEY", "test-mistral-key")
    provider_env.setenv("AZURE_MISTRAL_MODEL", "my-mistral-deployment")
    with patch("app.providers.mistral.AsyncOpenAI") as client:
        MistralProvider()
    client.assert_called_once_with(
        base_url="https://mistral.test/openai/v1/",
        api_key="test-mistral-key",
        max_retries=0,
    )


def test_mistral_requires_deployment_name(provider_env):
    provider_env.setenv("OPENAI_BASE_URL_GPT_5_2", "https://example.test/openai/v1/")
    with patch("app.providers.mistral.AsyncOpenAI") as client:
        with pytest.raises(ValueError, match="AZURE_MISTRAL_MODEL"):
            MistralProvider()
    client.assert_not_called()


@pytest.mark.asyncio
async def test_mistral_sends_azure_deployment_name(provider_env):
    from unittest.mock import AsyncMock

    provider_env.setenv("OPENAI_BASE_URL_GPT_5_2", "https://example.test/openai/v1/")
    provider_env.setenv("AZURE_MISTRAL_MODEL", "my-mistral-deployment")
    response = Mock(usage=None)
    response.choices = [Mock(message=Mock(content='{"categories": ["Housing"]}'))]
    with patch("app.providers.mistral.AsyncOpenAI") as client:
        client.return_value.chat.completions.create = AsyncMock(return_value=response)
        provider = MistralProvider()
        result = await provider.classify("test", [], custom_prompt="Return JSON")
    assert "error" not in result
    assert result["labels"][0].label == "Housing"
    assert client.return_value.chat.completions.create.call_args.kwargs["model"] == (
        "my-mistral-deployment"
    )


def test_disabled_langfuse_does_not_create_client(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    with patch.object(langfuse_client, "Langfuse") as client:
        assert langfuse_client._get_langfuse_client() is None
    client.assert_not_called()


@pytest.mark.parametrize("dedicated_key", ["", "test-azure-key"])
@pytest.mark.parametrize("trailing_slash", ["", "/"])
def test_gpt52_azure_v1_uses_configured_endpoint(
    provider_env, dedicated_key, trailing_slash
):
    endpoint = "https://example.test/openai/v1" + trailing_slash
    provider_env.setenv("OPENAI_BASE_URL_GPT_5_2", endpoint)
    provider_env.setenv("OPENAI_GPT_5_2_API_KEY", dedicated_key)
    with patch("app.providers.openai.AsyncOpenAI") as client:
        GPT52Provider()
    client.assert_called_once_with(
        base_url=endpoint,
        api_key=dedicated_key or "test-general-key",
        max_retries=0,
    )


def test_gpt52_versioned_azure_configuration(provider_env):
    provider_env.setenv("OPENAI_BASE_URL_GPT_5_2", "https://example.test/")
    provider_env.setenv("OPENAI_GPT_5_2_API_KEY", "test-azure-key")
    provider_env.setenv("OPENAI_GPT_5_2_API_VERSION", "test-version")
    provider_env.setenv("OPENAI_GPT_5_2_MODEL", "custom-deployment")
    with patch("app.providers.openai.AsyncAzureOpenAI") as client:
        provider = GPT52Provider()
    assert provider.model_name == "custom-deployment"
    client.assert_called_once_with(
        azure_endpoint="https://example.test/",
        api_key="test-azure-key",
        api_version="test-version",
        max_retries=0,
    )


def test_gpt52_incomplete_azure_configuration_does_not_fall_back(provider_env):
    provider_env.setenv("OPENAI_BASE_URL_GPT_5_2", "https://example.test/")
    with patch("app.providers.openai.AsyncOpenAI") as client:
        with pytest.raises(ValueError, match="OPENAI_GPT_5_2_API_VERSION"):
            GPT52Provider()
    client.assert_not_called()


def test_gpt52_public_openai_without_azure_endpoint(provider_env):
    with patch("app.providers.openai.AsyncOpenAI") as client:
        GPT52Provider()
    client.assert_called_once_with(api_key="test-general-key", max_retries=0)


def test_all_openai_providers_use_azure_endpoint(provider_env):
    provider_env.setenv("OPENAI_BASE_URL_GPT_5_2", "https://example.test/openai/v1/")
    with patch("app.providers.openai.AsyncOpenAI") as client:
        OpenAIProvider()
    client.assert_called_once_with(
        base_url="https://example.test/openai/v1/",
        api_key="test-general-key",
        max_retries=0,
    )


def test_semantic_merge_uses_azure_endpoint(provider_env):
    from app.services.classification_service import ClassificationService

    provider_env.setenv("OPENAI_BASE_URL_GPT_5_2", "https://example.test/openai/v1/")
    provider_env.delenv("OFFLINE_MODE", raising=False)
    provider_env.delenv("DISABLE_EXTERNAL_PROVIDERS", raising=False)
    with (
        patch.object(ClassificationService, "_init_providers", return_value=[]),
        patch.object(ClassificationService, "_load_all_taxonomies", return_value={}),
        patch("app.providers.openai.AsyncOpenAI") as client,
    ):
        service = ClassificationService()
    assert service.openai_client is client.return_value
    client.assert_called_once_with(
        base_url="https://example.test/openai/v1/",
        api_key="test-general-key",
        max_retries=0,
    )


def test_request_trace_uses_observation_api():
    client = Mock(spec=langfuse_client.Langfuse)
    with patch.object(langfuse_client, "_get_langfuse_client", return_value=client):
        span = langfuse_client.start_request_trace("request", {"text": "test"})
    assert span is client.start_observation.return_value
    client.start_observation.assert_called_once_with(
        as_type="span", name="request", input={"text": "test"}, metadata=None
    )
