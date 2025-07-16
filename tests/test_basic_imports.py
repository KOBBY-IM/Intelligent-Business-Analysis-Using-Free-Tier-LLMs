"""
Basic import tests to verify the system can be imported correctly.
"""

import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestBasicImports:
    """Test that all modules can be imported correctly."""

    def test_config_imports(self):
        """Test configuration module imports."""
        try:
            from config.config_loader import ConfigLoader

            assert ConfigLoader is not None
        except ImportError as e:
            pytest.fail(f"Failed to import ConfigLoader: {e}")

    def test_llm_providers_imports(self):
        """Test LLM providers module imports."""
        try:
            from llm_providers.base_provider import BaseProvider, LLMResponse

            assert BaseProvider is not None
            assert LLMResponse is not None
        except ImportError as e:
            pytest.fail(f"Failed to import BaseProvider: {e}")

        try:
            from llm_providers.groq_provider import GroqProvider

            assert GroqProvider is not None
        except ImportError as e:
            pytest.fail(f"Failed to import GroqProvider: {e}")

        try:
            from llm_providers.gemini_provider import GeminiProvider

            assert GeminiProvider is not None
        except ImportError as e:
            pytest.fail(f"Failed to import GeminiProvider: {e}")

        try:
            from llm_providers.openrouter_provider import OpenRouterProvider

            assert OpenRouterProvider is not None
        except ImportError as e:
            pytest.fail(f"Failed to import OpenRouterProvider: {e}")

    def test_evaluation_imports(self):
        """Test evaluation module imports."""
        try:
            from evaluation.evaluator import LLMEvaluator

            assert LLMEvaluator is not None
        except ImportError as e:
            pytest.fail(f"Failed to import LLMEvaluator: {e}")

        try:
            from evaluation.metrics import EvaluationMetricsCalculator

            assert EvaluationMetricsCalculator is not None
        except ImportError as e:
            pytest.fail(f"Failed to import EvaluationMetricsCalculator: {e}")

    def test_rag_imports(self):
        """Test RAG module imports."""
        try:
            from rag.pipeline import RAGPipeline

            assert RAGPipeline is not None
        except ImportError as e:
            pytest.fail(f"Failed to import RAGPipeline: {e}")

        try:
            from rag.retrieval import DocumentRetriever

            assert DocumentRetriever is not None
        except ImportError as e:
            pytest.fail(f"Failed to import DocumentRetriever: {e}")

        try:
            from rag.vector_store import FAISSVectorStore

            assert FAISSVectorStore is not None
        except ImportError as e:
            pytest.fail(f"Failed to import FAISSVectorStore: {e}")


class TestBasicFunctionality:
    """Test basic functionality of core components."""

    def test_config_loader_creation(self):
        """Test that ConfigLoader can be instantiated."""
        try:
            from config.config_loader import ConfigLoader

            config_loader = ConfigLoader()
            assert config_loader is not None
        except Exception as e:
            pytest.fail(f"Failed to create ConfigLoader: {e}")

    def test_groq_provider_creation(self):
        """Test that GroqProvider can be instantiated."""
        try:
            from llm_providers.groq_provider import GroqProvider

            # Set a dummy API key for testing
            os.environ["GROQ_API_KEY"] = "test_key"
            provider = GroqProvider(model_list=["llama3-8b-8192"])
            assert provider is not None
            assert provider.provider_name == "Groq"
        except Exception as e:
            pytest.fail(f"Failed to create GroqProvider: {e}")

    def test_llm_response_creation(self):
        """Test that LLMResponse can be created."""
        try:
            from llm_providers.base_provider import LLMResponse

            response = LLMResponse(
                success=True,
                text="Test response",
                model="test-model",
                tokens_used=100,
                latency_ms=150.0,
            )
            assert response.success is True
            assert response.text == "Test response"
            assert response.model == "test-model"
            assert response.tokens_used == 100
            assert response.latency_ms == 150.0
        except Exception as e:
            pytest.fail(f"Failed to create LLMResponse: {e}")


class TestEnvironmentSetup:
    """Test environment setup and configuration."""

    def test_environment_variables(self):
        """Test that required environment variables can be set."""
        test_vars = {
            "GROQ_API_KEY": "test_groq_key",
            "GOOGLE_API_KEY": "test_google_key",
            "OPENROUTER_API_KEY": "test_openrouter_key",
        }

        for var_name, var_value in test_vars.items():
            os.environ[var_name] = var_value
            assert os.getenv(var_name) == var_value

    def test_project_structure(self):
        """Test that project structure is correct."""
        project_root = os.path.join(os.path.dirname(__file__), "..")

        required_dirs = [
            "src",
            "src/config",
            "src/llm_providers",
            "src/evaluation",
            "src/rag",
            "tests",
            "config",
            "data",
        ]

        for dir_path in required_dirs:
            full_path = os.path.join(project_root, dir_path)
            assert os.path.exists(full_path), f"Required directory missing: {dir_path}"

    def test_config_files_exist(self):
        """Test that configuration files exist."""
        project_root = os.path.join(os.path.dirname(__file__), "..")

        required_files = [
            "config/app_config.yaml",
            "config/llm_config.yaml",
            "config/evaluation_config.yaml",
            "config/logging_config.yaml",
            "requirements.txt",
            "requirements-dev.txt",
        ]

        for file_path in required_files:
            full_path = os.path.join(project_root, file_path)
            assert os.path.exists(full_path), f"Required file missing: {file_path}"


class TestDependencies:
    """Test that required dependencies are available."""

    def test_required_packages(self):
        """Test that required packages can be imported."""
        required_packages = [
            "pytest",
            "requests",
            "pandas",
            "numpy",
            "streamlit",
            "langchain",
            "faiss",
            "chromadb",
        ]

        for package in required_packages:
            try:
                __import__(package)
            except ImportError as e:
                pytest.fail(f"Required package {package} not available: {e}")

    def test_optional_packages(self):
        """Test that optional packages can be imported if available."""
        optional_packages = ["torch", "transformers", "sentence_transformers", "plotly"]

        for package in optional_packages:
            try:
                __import__(package)
            except ImportError:
                # Optional packages are not required to pass tests
                pass
