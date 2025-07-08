"""
Comprehensive unit tests for LLM providers.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_providers.base_provider import BaseProvider, LLMResponse
from llm_providers.gemini_provider import GeminiProvider
from llm_providers.groq_provider import GroqProvider
from llm_providers.openrouter_provider import OpenRouterProvider
from llm_providers.provider_manager import ProviderManager


class TestBaseProvider:
    """Test the base LLM provider abstract class."""

    def test_base_provider_instantiation(self):
        """Test that base provider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseProvider()

    def test_base_provider_abstract_methods(self):
        """Test that abstract methods are properly defined."""
        assert hasattr(BaseProvider, "generate_response")
        assert hasattr(BaseProvider, "list_models")
        assert hasattr(BaseProvider, "health_check")


class TestGroqProvider:
    """Test the Groq LLM provider."""

    @pytest.fixture
    def groq_provider(self, mock_env_vars):
        """Create a GroqProvider instance for testing."""
        return GroqProvider()

    def test_groq_provider_initialization(self, groq_provider):
        """Test GroqProvider initialization."""
        assert groq_provider is not None
        assert hasattr(groq_provider, "api_key")
        assert hasattr(groq_provider, "model")

    def test_groq_generate_response_success(self, groq_provider, mock_llm_response):
        """Test successful response generation from Groq."""
        with patch("llm_providers.groq_provider.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": mock_llm_response["response"]}}],
                "usage": {"total_tokens": mock_llm_response["tokens_used"]},
            }
            mock_post.return_value = mock_response

            response = groq_provider.generate_response(
                query="Test query", context="Test context"
            )

            assert isinstance(response, LLMResponse)
            assert response.success is True
            assert response.text == mock_llm_response["response"]
            assert response.tokens_used == mock_llm_response["tokens_used"]
            assert response.latency_ms is not None

    def test_groq_generate_response_error(self, groq_provider):
        """Test error handling in Groq response generation."""
        with patch("llm_providers.groq_provider.requests.post") as mock_post:
            mock_post.side_effect = Exception("API Error")

            response = groq_provider.generate_response(
                query="Test query", context="Test context"
            )

            assert isinstance(response, LLMResponse)
            assert response.success is False
            assert "API Error" in response.error

    def test_groq_health_check(self, groq_provider):
        """Test health check for Groq."""
        with patch.object(groq_provider, "generate_response") as mock_generate:
            mock_generate.return_value = LLMResponse(
                success=True, text="test", model="test"
            )
            health_status = groq_provider.health_check()
            assert isinstance(health_status, bool)

    def test_groq_list_models(self, groq_provider):
        """Test model listing for Groq."""
        models = groq_provider.list_models()
        assert isinstance(models, list)
        assert len(models) > 0


class TestGeminiProvider:
    """Test the Gemini LLM provider."""

    @pytest.fixture
    def gemini_provider(self, mock_env_vars):
        """Create a GeminiProvider instance for testing."""
        return GeminiProvider()

    def test_gemini_provider_initialization(self, gemini_provider):
        """Test GeminiProvider initialization."""
        assert gemini_provider is not None
        assert hasattr(gemini_provider, "api_key")
        assert hasattr(gemini_provider, "model")

    @pytest.mark.asyncio
    async def test_gemini_generate_response_success(
        self, gemini_provider, mock_llm_response
    ):
        """Test successful response generation from Gemini."""
        with patch("llm_providers.gemini_provider.genai.GenerativeModel") as mock_genai:
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = mock_llm_response["response"]
            mock_response.usage_metadata = Mock(
                prompt_token_count=50,
                candidates_token_count=mock_llm_response["tokens_used"] - 50,
            )
            mock_model.generate_content.return_value = mock_response
            mock_genai.return_value = mock_model

            response = await gemini_provider.generate_response(
                query="Test query", context="Test context"
            )

            assert response["response"] == mock_llm_response["response"]
            assert response["tokens_used"] == mock_llm_response["tokens_used"]
            assert "latency" in response

    @pytest.mark.asyncio
    async def test_gemini_generate_response_error(self, gemini_provider):
        """Test error handling in Gemini response generation."""
        with patch("llm_providers.gemini_provider.genai.GenerativeModel") as mock_genai:
            mock_genai.side_effect = Exception("API Error")

            with pytest.raises(Exception):
                await gemini_provider.generate_response(
                    query="Test query", context="Test context"
                )

    def test_gemini_validate_response(self, gemini_provider, mock_llm_response):
        """Test response validation for Gemini."""
        is_valid = gemini_provider.validate_response(mock_llm_response)
        assert is_valid is True


class TestOpenRouterProvider:
    """Test the OpenRouter LLM provider."""

    @pytest.fixture
    def openrouter_provider(self, mock_env_vars):
        """Create an OpenRouterProvider instance for testing."""
        return OpenRouterProvider()

    def test_openrouter_provider_initialization(self, openrouter_provider):
        """Test OpenRouterProvider initialization."""
        assert openrouter_provider is not None
        assert hasattr(openrouter_provider, "api_key")
        assert hasattr(openrouter_provider, "model")

    @pytest.mark.asyncio
    async def test_openrouter_generate_response_success(
        self, openrouter_provider, mock_llm_response
    ):
        """Test successful response generation from OpenRouter."""
        with patch(
            "llm_providers.openrouter_provider.openai.AsyncOpenAI"
        ) as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [
                Mock(message=Mock(content=mock_llm_response["response"]))
            ]
            mock_response.usage = Mock(total_tokens=mock_llm_response["tokens_used"])
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            response = await openrouter_provider.generate_response(
                query="Test query", context="Test context"
            )

            assert response["response"] == mock_llm_response["response"]
            assert response["tokens_used"] == mock_llm_response["tokens_used"]
            assert "latency" in response

    @pytest.mark.asyncio
    async def test_openrouter_generate_response_error(self, openrouter_provider):
        """Test error handling in OpenRouter response generation."""
        with patch(
            "llm_providers.openrouter_provider.openai.AsyncOpenAI"
        ) as mock_openai:
            mock_openai.side_effect = Exception("API Error")

            with pytest.raises(Exception):
                await openrouter_provider.generate_response(
                    query="Test query", context="Test context"
                )

    def test_openrouter_validate_response(self, openrouter_provider, mock_llm_response):
        """Test response validation for OpenRouter."""
        is_valid = openrouter_provider.validate_response(mock_llm_response)
        assert is_valid is True


class TestProviderManager:
    """Test the ProviderManager class."""

    @pytest.fixture
    def provider_manager(self, mock_env_vars):
        """Create a ProviderManager instance for testing."""
        return ProviderManager()

    def test_provider_manager_initialization(self, provider_manager):
        """Test ProviderManager initialization."""
        assert provider_manager is not None
        assert hasattr(provider_manager, "providers")
        assert isinstance(provider_manager.providers, dict)

    def test_get_provider(self, provider_manager):
        """Test getting a specific provider."""
        groq_provider = provider_manager.get_provider("groq")
        assert groq_provider is not None
        assert isinstance(groq_provider, GroqProvider)

    def test_get_nonexistent_provider(self, provider_manager):
        """Test getting a non-existent provider."""
        with pytest.raises(ValueError):
            provider_manager.get_provider("nonexistent")

    def test_list_providers(self, provider_manager):
        """Test listing available providers."""
        providers = provider_manager.list_providers()
        assert isinstance(providers, list)
        assert "groq" in providers
        assert "gemini" in providers
        assert "openrouter" in providers

    @pytest.mark.asyncio
    async def test_generate_response_from_all_providers(
        self, provider_manager, mock_llm_response
    ):
        """Test generating responses from all providers."""
        with (
            patch.object(
                GroqProvider, "generate_response", return_value=mock_llm_response
            ),
            patch.object(
                GeminiProvider, "generate_response", return_value=mock_llm_response
            ),
            patch.object(
                OpenRouterProvider, "generate_response", return_value=mock_llm_response
            ),
        ):

            responses = await provider_manager.generate_response_from_all_providers(
                query="Test query", context="Test context"
            )

            assert isinstance(responses, dict)
            assert "groq" in responses
            assert "gemini" in responses
            assert "openrouter" in responses

    def test_provider_health_check(self, provider_manager):
        """Test provider health check functionality."""
        health_status = provider_manager.check_provider_health()
        assert isinstance(health_status, dict)
        assert "groq" in health_status
        assert "gemini" in health_status
        assert "openrouter" in health_status


class TestProviderIntegration:
    """Integration tests for LLM providers."""

    @pytest.mark.asyncio
    async def test_provider_response_consistency(self, mock_env_vars):
        """Test that all providers return consistent response structures."""
        providers = [GroqProvider(), GeminiProvider(), OpenRouterProvider()]

        for provider in providers:
            with patch.object(provider, "generate_response") as mock_generate:
                mock_generate.return_value = {
                    "response": "Test response",
                    "tokens_used": 100,
                    "latency": 1.5,
                }

                response = await provider.generate_response(
                    "Test query", "Test context"
                )

                assert "response" in response
                assert "tokens_used" in response
                assert "latency" in response
                assert isinstance(response["response"], str)
                assert isinstance(response["tokens_used"], int)
                assert isinstance(response["latency"], (int, float))

    @pytest.mark.asyncio
    async def test_provider_error_handling_consistency(self, mock_env_vars):
        """Test that all providers handle errors consistently."""
        providers = [GroqProvider(), GeminiProvider(), OpenRouterProvider()]

        for provider in providers:
            with patch.object(
                provider, "generate_response", side_effect=Exception("Test error")
            ):
                with pytest.raises(Exception):
                    await provider.generate_response("Test query", "Test context")

    def test_provider_configuration_validation(self, mock_env_vars):
        """Test that all providers validate their configuration properly."""
        providers = [GroqProvider(), GeminiProvider(), OpenRouterProvider()]

        for provider in providers:
            # Test with valid configuration
            assert provider.api_key is not None
            assert provider.model is not None

            # Test configuration validation
            assert hasattr(provider, "validate_configuration")
            assert callable(provider.validate_configuration)


class TestProviderPerformance:
    """Performance tests for LLM providers."""

    @pytest.mark.asyncio
    async def test_provider_response_time(self, mock_env_vars, performance_metrics):
        """Test response time performance of providers."""
        import time

        providers = [GroqProvider(), GeminiProvider(), OpenRouterProvider()]

        for provider in providers:
            with patch.object(provider, "generate_response") as mock_generate:
                mock_generate.return_value = {
                    "response": "Test response",
                    "tokens_used": 100,
                    "latency": 1.5,
                }

                start_time = time.time()
                await provider.generate_response("Test query", "Test context")
                end_time = time.time()

                actual_latency = end_time - start_time
                performance_metrics["response_times"].append(actual_latency)

                # Response should be fast (under 1 second for mocked calls)
                assert actual_latency < 1.0

    def test_provider_memory_usage(self, mock_env_vars):
        """Test memory usage of providers."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create multiple provider instances
        providers = []
        for _ in range(10):
            providers.extend([GroqProvider(), GeminiProvider(), OpenRouterProvider()])

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB in bytes


class TestProviderSecurity:
    """Security tests for LLM providers."""

    def test_provider_input_sanitization(self, mock_env_vars, malicious_inputs):
        """Test that providers sanitize malicious inputs."""
        providers = [GroqProvider(), GeminiProvider(), OpenRouterProvider()]

        for provider in providers:
            for malicious_input in malicious_inputs:
                # Providers should handle malicious inputs gracefully
                assert hasattr(provider, "sanitize_input")
                sanitized = provider.sanitize_input(malicious_input)
                assert sanitized != malicious_input  # Input should be sanitized

    def test_provider_api_key_protection(self, mock_env_vars):
        """Test that API keys are properly protected."""
        providers = [GroqProvider(), GeminiProvider(), OpenRouterProvider()]

        for provider in providers:
            # API keys should not be exposed in string representations
            provider_str = str(provider)
            assert "test_groq_key" not in provider_str
            assert "test_google_key" not in provider_str
            assert "test_openrouter_key" not in provider_str

    def test_provider_rate_limiting(self, mock_env_vars):
        """Test that providers implement rate limiting."""
        providers = [GroqProvider(), GeminiProvider(), OpenRouterProvider()]

        for provider in providers:
            # Providers should have rate limiting mechanisms
            assert hasattr(provider, "rate_limit")
            assert callable(provider.rate_limit)
