"""
Pytest configuration and fixtures for comprehensive testing of the LLM evaluation system.
"""

import asyncio
import os
import shutil
import tempfile
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

import pytest


# Test data and configurations
@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "llm_providers": {
            "groq": {"model": "mixtral-8x7b-32768", "api_key": "test_key"},
            "gemini": {"model": "gemini-pro", "api_key": "test_key"},
            "openrouter": {"model": "openai/gpt-3.5-turbo", "api_key": "test_key"},
        },
        "evaluation": {
            "metrics": ["accuracy", "latency", "token_count", "relevance"],
            "sample_size": 10,
            "timeout": 30,
        },
        "rag": {"chunk_size": 1000, "chunk_overlap": 200, "top_k": 5},
    }


@pytest.fixture
def sample_queries():
    """Sample business queries for testing."""
    return [
        "What are the key performance indicators for retail sales?",
        "How can we improve customer retention in the finance sector?",
        "What are the compliance requirements for healthcare data?",
        "Analyze market trends in e-commerce",
        "What are the best practices for risk management?",
    ]


@pytest.fixture
def sample_contexts():
    """Sample industry contexts for testing."""
    return {
        "retail": "Retail industry focusing on customer experience and sales optimization",
        "finance": "Financial services with emphasis on risk management and compliance",
        "healthcare": "Healthcare sector with focus on patient care and regulatory compliance",
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "response": "This is a test response from the LLM provider.",
        "tokens_used": 150,
        "latency": 2.5,
        "model": "test-model",
        "provider": "test-provider",
    }


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_evaluation_data():
    """Sample evaluation data for testing."""
    return {
        "query": "What are the key performance indicators for retail sales?",
        "context": "retail",
        "responses": [
            {
                "provider": "groq",
                "response": "KPI response 1",
                "tokens_used": 100,
                "latency": 1.5,
            },
            {
                "provider": "gemini",
                "response": "KPI response 2",
                "tokens_used": 120,
                "latency": 2.0,
            },
        ],
        "ground_truth": "Expected KPI analysis for retail",
        "user_preference": "groq",
    }


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "GROQ_API_KEY": "test_groq_key",
        "GOOGLE_API_KEY": "test_google_key",
        "OPENROUTER_API_KEY": "test_openrouter_key",
        "SECRET_KEY": "test_secret_key",
        "DATA_ENCRYPTION_KEY": "test_encryption_key",
    }

    # Store original values
    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value

    yield env_vars

    # Restore original values
    for key, value in original_values.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


@pytest.fixture
def mock_api_client():
    """Mock API client for testing."""
    mock_client = Mock()
    mock_client.generate_response = AsyncMock(
        return_value={"response": "Test response", "tokens_used": 100, "latency": 1.5}
    )
    return mock_client


@pytest.fixture
def sample_documents():
    """Sample documents for RAG testing."""
    return [
        "Retail KPIs include sales per square foot, customer lifetime value, and inventory turnover.",
        "Financial risk management involves credit risk, market risk, and operational risk assessment.",
        "Healthcare compliance requires HIPAA adherence, patient data protection, and audit trails.",
        "E-commerce trends show increasing mobile usage, AI-powered recommendations, and omnichannel retail.",
        "Customer retention strategies include loyalty programs, personalized marketing, and excellent service.",
    ]


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = Mock()
    mock_store.add_documents = Mock()
    mock_store.search = Mock(
        return_value=[
            {"content": "Test document 1", "score": 0.9},
            {"content": "Test document 2", "score": 0.8},
        ]
    )
    mock_store.save = Mock()
    mock_store.load = Mock()
    return mock_store


# Async test support
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Performance testing fixtures
@pytest.fixture
def performance_metrics():
    """Track performance metrics during tests."""
    return {"response_times": [], "memory_usage": [], "error_counts": 0}


# Security testing fixtures
@pytest.fixture
def malicious_inputs():
    """Malicious inputs for security testing."""
    return [
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "../../../etc/passwd",
        "exec('rm -rf /')",
        "import os; os.system('rm -rf /')",
    ]


@pytest.fixture
def valid_inputs():
    """Valid inputs for testing."""
    return [
        "What are retail KPIs?",
        "How to manage financial risk?",
        "Healthcare compliance requirements",
        "E-commerce best practices",
        "Customer retention strategies",
    ]


# Integration testing fixtures
@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "test_all_providers": True,
        "test_rag_pipeline": True,
        "test_evaluation_metrics": True,
        "test_user_interface": True,
        "test_data_processing": True,
        "test_security": True,
    }


# Test utilities
class TestUtils:
    """Utility functions for testing."""

    @staticmethod
    def assert_response_structure(response: Dict[str, Any]) -> None:
        """Assert that response has the expected structure."""
        assert isinstance(response, dict)
        assert "response" in response
        assert "tokens_used" in response
        assert "latency" in response
        assert isinstance(response["response"], str)
        assert isinstance(response["tokens_used"], int)
        assert isinstance(response["latency"], (int, float))

    @staticmethod
    def assert_metrics_structure(metrics: Dict[str, Any]) -> None:
        """Assert that metrics have the expected structure."""
        assert isinstance(metrics, dict)
        required_fields = ["accuracy", "latency", "token_count", "relevance"]
        for field in required_fields:
            assert field in metrics
            assert isinstance(metrics[field], (int, float))

    @staticmethod
    def create_test_file(content: str, extension: str = ".txt") -> str:
        """Create a temporary test file."""
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=extension, delete=False
        )
        temp_file.write(content)
        temp_file.close()
        return temp_file.name


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils
