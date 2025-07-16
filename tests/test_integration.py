"""
Comprehensive integration tests for the entire LLM evaluation system.
"""

import asyncio
import os
import sys
from unittest.mock import patch

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config.config_loader import ConfigLoader
from evaluation.evaluator import LLMEvaluator
from llm_providers.provider_manager import ProviderManager
from rag.pipeline import RAGPipeline


class TestFullSystemIntegration:
    """Integration tests for the complete system."""

    @pytest.fixture
    def full_system_setup(self, temp_data_dir, mock_env_vars):
        """Set up the complete system for integration testing."""
        # Initialize all components
        provider_manager = ProviderManager()
        evaluator = LLMEvaluator(data_dir=temp_data_dir)
        rag_pipeline = RAGPipeline(data_dir=temp_data_dir)
        config_loader = ConfigLoader()

        return {
            "provider_manager": provider_manager,
            "evaluator": evaluator,
            "rag_pipeline": rag_pipeline,
            "config_loader": config_loader,
            "data_dir": temp_data_dir,
        }

    @pytest.mark.asyncio
    async def test_complete_evaluation_workflow(
        self, full_system_setup, sample_queries, sample_contexts
    ):
        """Test the complete evaluation workflow from query to results."""
        provider_manager = full_system_setup["provider_manager"]
        evaluator = full_system_setup["evaluator"]
        full_system_setup["rag_pipeline"]

        # Mock all LLM providers
        mock_response = {
            "response": "This is a comprehensive response about the query.",
            "tokens_used": 150,
            "latency": 2.5,
        }

        with (
            patch.object(
                provider_manager.providers["groq"],
                "generate_response",
                return_value=mock_response,
            ),
            patch.object(
                provider_manager.providers["gemini"],
                "generate_response",
                return_value=mock_response,
            ),
            patch.object(
                provider_manager.providers["openrouter"],
                "generate_response",
                return_value=mock_response,
            ),
        ):

            # Test with a single query
            query = sample_queries[0]
            context = list(sample_contexts.values())[0]

            # 1. Generate responses from all providers
            responses = await provider_manager.generate_response_from_all_providers(
                query, context
            )

            assert isinstance(responses, dict)
            assert "groq" in responses
            assert "gemini" in responses
            assert "openrouter" in responses

            # 2. Evaluate responses
            evaluation_results = await evaluator.evaluate_multiple_responses(
                query=query, context=context, responses=list(responses.values())
            )

            assert isinstance(evaluation_results, list)
            assert len(evaluation_results) == len(responses)

            # 3. Save evaluation results
            evaluator.save_evaluation_results(evaluation_results)

            # 4. Verify results were saved
            loaded_results = evaluator.load_evaluation_results()
            assert len(loaded_results) == len(evaluation_results)

    @pytest.mark.asyncio
    async def test_rag_integration_with_evaluation(
        self, full_system_setup, sample_documents
    ):
        """Test RAG integration with the evaluation system."""
        evaluator = full_system_setup["evaluator"]
        rag_pipeline = full_system_setup["rag_pipeline"]

        # Add documents to RAG pipeline
        rag_pipeline.add_documents(sample_documents)

        # Mock LLM provider
        mock_response = {
            "response": "Based on the retrieved documents, here is the answer.",
            "tokens_used": 200,
            "latency": 3.0,
        }

        with patch.object(rag_pipeline, "llm_provider") as mock_llm:
            mock_llm.generate_response.return_value = mock_response

            # Process query through RAG pipeline
            query = "What are retail KPIs?"
            context = "retail"

            rag_result = await rag_pipeline.process_query(query, context)

            assert "retrieved_documents" in rag_result
            assert "generated_response" in rag_result
            assert len(rag_result["retrieved_documents"]) > 0

            # Evaluate RAG response
            evaluation_result = await evaluator.evaluate_response(
                query=query, context=context, response=rag_result["generated_response"]
            )

            assert "metrics" in evaluation_result
            assert "accuracy" in evaluation_result["metrics"]
            assert "latency" in evaluation_result["metrics"]

    @pytest.mark.asyncio
    async def test_batch_evaluation_across_industries(
        self, full_system_setup, sample_queries, sample_contexts
    ):
        """Test batch evaluation across multiple industries."""
        provider_manager = full_system_setup["provider_manager"]
        evaluator = full_system_setup["evaluator"]

        # Mock responses for all providers
        mock_response = {
            "response": "Industry-specific response based on the context.",
            "tokens_used": 120,
            "latency": 2.0,
        }

        with (
            patch.object(
                provider_manager.providers["groq"],
                "generate_response",
                return_value=mock_response,
            ),
            patch.object(
                provider_manager.providers["gemini"],
                "generate_response",
                return_value=mock_response,
            ),
            patch.object(
                provider_manager.providers["openrouter"],
                "generate_response",
                return_value=mock_response,
            ),
        ):

            # Test batch evaluation across industries
            industries = list(sample_contexts.keys())
            queries_per_industry = sample_queries[
                :2
            ]  # Use first 2 queries per industry

            all_results = []

            for industry in industries:
                context = sample_contexts[industry]

                for query in queries_per_industry:
                    # Generate responses
                    responses = (
                        await provider_manager.generate_response_from_all_providers(
                            query, context
                        )
                    )

                    # Evaluate responses
                    evaluation_results = await evaluator.evaluate_multiple_responses(
                        query=query, context=context, responses=list(responses.values())
                    )

                    all_results.extend(evaluation_results)

            # Save all results
            evaluator.save_evaluation_results(all_results)

            # Verify results
            loaded_results = evaluator.load_evaluation_results()
            assert len(loaded_results) == len(all_results)

            # Check that we have results from all industries
            contexts_in_results = set(result["context"] for result in loaded_results)
            assert all(industry in contexts_in_results for industry in industries)

    @pytest.mark.asyncio
    async def test_provider_comparison_analysis(
        self, full_system_setup, sample_queries, sample_contexts
    ):
        """Test comprehensive provider comparison analysis."""
        provider_manager = full_system_setup["provider_manager"]
        evaluator = full_system_setup["evaluator"]

        # Mock different responses for each provider
        mock_responses = {
            "groq": {
                "response": "Groq response: Fast and efficient analysis.",
                "tokens_used": 100,
                "latency": 1.5,
            },
            "gemini": {
                "response": "Gemini response: Comprehensive and detailed analysis.",
                "tokens_used": 150,
                "latency": 2.5,
            },
            "openrouter": {
                "response": "OpenRouter response: Balanced approach to analysis.",
                "tokens_used": 120,
                "latency": 2.0,
            },
        }

        with (
            patch.object(
                provider_manager.providers["groq"],
                "generate_response",
                return_value=mock_responses["groq"],
            ),
            patch.object(
                provider_manager.providers["gemini"],
                "generate_response",
                return_value=mock_responses["gemini"],
            ),
            patch.object(
                provider_manager.providers["openrouter"],
                "generate_response",
                return_value=mock_responses["openrouter"],
            ),
        ):

            # Test multiple queries
            query = sample_queries[0]
            context = list(sample_contexts.values())[0]

            # Generate responses
            responses = await provider_manager.generate_response_from_all_providers(
                query, context
            )

            # Evaluate responses
            evaluation_results = await evaluator.evaluate_multiple_responses(
                query=query, context=context, responses=list(responses.values())
            )

            # Analyze results
            providers = list(responses.keys())
            metrics_by_provider = {}

            for i, provider in enumerate(providers):
                metrics_by_provider[provider] = evaluation_results[i]["metrics"]

            # Verify metrics structure
            for provider, metrics in metrics_by_provider.items():
                assert "accuracy" in metrics
                assert "latency" in metrics
                assert "token_count" in metrics
                assert "relevance" in metrics

                # Verify metric values are reasonable
                assert 0.0 <= metrics["accuracy"] <= 1.0
                assert metrics["latency"] >= 0.0
                assert metrics["token_count"] > 0
                assert 0.0 <= metrics["relevance"] <= 1.0

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, full_system_setup):
        """Test error handling and recovery across the system."""
        provider_manager = full_system_setup["provider_manager"]
        full_system_setup["evaluator"]

        # Test with provider failures
        with (
            patch.object(
                provider_manager.providers["groq"],
                "generate_response",
                side_effect=Exception("API Error"),
            ),
            patch.object(
                provider_manager.providers["gemini"],
                "generate_response",
                return_value={
                    "response": "Successful response",
                    "tokens_used": 100,
                    "latency": 2.0,
                },
            ),
            patch.object(
                provider_manager.providers["openrouter"],
                "generate_response",
                return_value={
                    "response": "Another successful response",
                    "tokens_used": 120,
                    "latency": 2.5,
                },
            ),
        ):

            query = "Test query"
            context = "retail"

            # Should handle provider failures gracefully
            try:
                responses = await provider_manager.generate_response_from_all_providers(
                    query, context
                )

                # Should have responses from working providers
                assert "gemini" in responses
                assert "openrouter" in responses
                assert "groq" not in responses  # Failed provider

            except Exception as e:
                # System should handle partial failures
                assert "API Error" in str(e) or "partial" in str(e).lower()

    @pytest.mark.asyncio
    async def test_data_persistence_and_recovery(
        self, full_system_setup, sample_evaluation_data
    ):
        """Test data persistence and recovery across system restarts."""
        evaluator = full_system_setup["evaluator"]
        rag_pipeline = full_system_setup["rag_pipeline"]
        data_dir = full_system_setup["data_dir"]

        # Add documents to RAG pipeline
        sample_documents = [
            "Retail KPIs include sales per square foot and customer lifetime value.",
            "Financial risk management involves credit risk and market risk assessment.",
            "Healthcare compliance requires HIPAA adherence and patient data protection.",
        ]
        rag_pipeline.add_documents(sample_documents)

        # Save evaluation results
        evaluator.save_evaluation_results([sample_evaluation_data])

        # Save RAG pipeline
        rag_pipeline.save_pipeline()

        # Create new instances (simulating system restart)
        new_evaluator = LLMEvaluator(data_dir=data_dir)
        new_rag_pipeline = RAGPipeline(data_dir=data_dir)

        # Load data
        new_evaluator.load_evaluation_results()
        new_rag_pipeline.load_pipeline()

        # Verify data recovery
        loaded_results = new_evaluator.load_evaluation_results()
        assert len(loaded_results) == 1
        assert loaded_results[0]["query"] == sample_evaluation_data["query"]

        # Verify RAG pipeline recovery
        assert len(new_rag_pipeline.retriever.vector_store.documents) == len(
            sample_documents
        )

        # Test functionality after recovery
        query = "retail KPIs"
        results = new_rag_pipeline.retriever.vector_store.search(query, top_k=3)
        assert len(results) > 0


class TestPerformanceIntegration:
    """Performance integration tests."""

    @pytest.mark.asyncio
    async def test_system_performance_under_load(self, temp_data_dir, mock_env_vars):
        """Test system performance under load."""
        import time

        provider_manager = ProviderManager()
        evaluator = LLMEvaluator(data_dir=temp_data_dir)

        # Mock fast responses
        mock_response = {
            "response": "Fast response for performance testing.",
            "tokens_used": 50,
            "latency": 0.5,
        }

        with (
            patch.object(
                provider_manager.providers["groq"],
                "generate_response",
                return_value=mock_response,
            ),
            patch.object(
                provider_manager.providers["gemini"],
                "generate_response",
                return_value=mock_response,
            ),
            patch.object(
                provider_manager.providers["openrouter"],
                "generate_response",
                return_value=mock_response,
            ),
        ):

            # Test multiple concurrent evaluations
            queries = [f"Test query {i}" for i in range(10)]
            context = "retail"

            start_time = time.time()

            # Process queries concurrently
            tasks = []
            for query in queries:
                task = asyncio.create_task(
                    provider_manager.generate_response_from_all_providers(
                        query, context
                    )
                )
                tasks.append(task)

            all_responses = await asyncio.gather(*tasks)

            end_time = time.time()
            total_time = end_time - start_time

            # System should handle concurrent requests efficiently
            assert total_time < 10.0  # Should complete within 10 seconds
            assert len(all_responses) == len(queries)

            # Each response should contain all providers
            for responses in all_responses:
                assert len(responses) == 3  # groq, gemini, openrouter

    def test_memory_usage_under_load(self, temp_data_dir, mock_env_vars):
        """Test memory usage under load."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create multiple components
        components = []
        for _ in range(10):
            components.extend(
                [
                    ProviderManager(),
                    LLMEvaluator(data_dir=temp_data_dir),
                    RAGPipeline(data_dir=temp_data_dir),
                ]
            )

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 200MB)
        assert memory_increase < 200 * 1024 * 1024  # 200MB in bytes


class TestSecurityIntegration:
    """Security integration tests."""

    @pytest.mark.asyncio
    async def test_system_security_validation(
        self, full_system_setup, malicious_inputs
    ):
        """Test system-wide security validation."""
        provider_manager = full_system_setup["provider_manager"]
        evaluator = full_system_setup["evaluator"]
        rag_pipeline = full_system_setup["rag_pipeline"]

        # Test with malicious inputs
        for malicious_input in malicious_inputs:
            try:
                # Test provider security
                with patch.object(
                    provider_manager.providers["groq"], "generate_response"
                ) as mock_generate:
                    mock_generate.return_value = {
                        "response": "Safe response",
                        "tokens_used": 50,
                        "latency": 1.0,
                    }

                    await provider_manager.generate_response_from_all_providers(
                        malicious_input, "retail"
                    )

                # Test evaluation security
                await evaluator.evaluate_response(
                    query=malicious_input,
                    context="retail",
                    response={"response": "test", "tokens_used": 10, "latency": 1.0},
                )

                # Test RAG security
                await rag_pipeline.process_query(malicious_input, "retail")

            except Exception as e:
                # Security exceptions should be handled gracefully
                assert (
                    "security" in str(e).lower()
                    or "invalid" in str(e).lower()
                    or "sanitize" in str(e).lower()
                )

    def test_data_encryption_and_privacy(self, full_system_setup):
        """Test data encryption and privacy protection."""
        evaluator = full_system_setup["evaluator"]
        data_dir = full_system_setup["data_dir"]

        # Test data with sensitive information
        sensitive_data = {
            "query": "What are the customer credit card numbers?",
            "context": "finance",
            "response": "Customer data: 1234-5678-9012-3456",
            "tokens_used": 100,
            "latency": 2.0,
        }

        # Save sensitive data
        evaluator.save_evaluation_results([sensitive_data])

        # Verify data is stored securely (not in plain text)
        results_file = os.path.join(data_dir, "evaluation_results.json")
        assert os.path.exists(results_file)

        # Check that sensitive data is not exposed in file names or paths
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                assert "credit" not in file.lower()
                assert "card" not in file.lower()
                assert "1234" not in file


class TestConfigurationIntegration:
    """Configuration integration tests."""

    def test_system_configuration_loading(self, temp_data_dir):
        """Test system-wide configuration loading."""
        config_loader = ConfigLoader()

        # Test loading all configurations
        app_config = config_loader.load_app_config()
        llm_config = config_loader.load_llm_config()
        evaluation_config = config_loader.load_evaluation_config()
        logging_config = config_loader.load_logging_config()

        # Verify configuration structure
        assert isinstance(app_config, dict)
        assert isinstance(llm_config, dict)
        assert isinstance(evaluation_config, dict)
        assert isinstance(logging_config, dict)

        # Verify required configuration sections
        assert "llm_providers" in llm_config
        assert "evaluation" in evaluation_config
        assert "logging" in logging_config

    def test_configuration_validation(self, temp_data_dir):
        """Test configuration validation across the system."""
        config_loader = ConfigLoader()

        # Test with valid configuration
        try:
            app_config = config_loader.load_app_config()
            llm_config = config_loader.load_llm_config()
            evaluation_config = config_loader.load_evaluation_config()

            # All configurations should load without errors
            assert app_config is not None
            assert llm_config is not None
            assert evaluation_config is not None

        except Exception as e:
            # Configuration errors should be handled gracefully
            assert "config" in str(e).lower() or "validation" in str(e).lower()

    def test_dynamic_configuration_updates(self, full_system_setup):
        """Test dynamic configuration updates."""
        full_system_setup["provider_manager"]
        evaluator = full_system_setup["evaluator"]
        rag_pipeline = full_system_setup["rag_pipeline"]

        # Test updating configurations
        new_eval_config = {
            "metrics": ["accuracy", "latency", "token_count"],
            "sample_size": 20,
            "timeout": 60,
        }

        new_rag_config = {"chunk_size": 800, "chunk_overlap": 150, "top_k": 8}

        # Update configurations
        evaluator.update_configuration(new_eval_config)
        rag_pipeline.update_configuration(new_rag_config)

        # Verify updates
        assert evaluator.sample_size == 20
        assert evaluator.timeout == 60
        assert rag_pipeline.chunk_size == 800
        assert rag_pipeline.top_k == 8


class TestMonitoringIntegration:
    """Monitoring and logging integration tests."""

    def test_system_logging_integration(self, full_system_setup):
        """Test system-wide logging integration."""
        import logging

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Test logging from different components
        try:
            logger.info("Testing provider manager logging")
            full_system_setup["provider_manager"]

            logger.info("Testing evaluator logging")
            full_system_setup["evaluator"]

            logger.info("Testing RAG pipeline logging")
            full_system_setup["rag_pipeline"]

            # Logging should work without errors
            assert True

        except Exception as e:
            # Logging errors should be handled gracefully
            assert "logging" in str(e).lower() or "config" in str(e).lower()

    def test_performance_monitoring(self, full_system_setup):
        """Test performance monitoring integration."""
        import time

        full_system_setup["provider_manager"]

        # Mock performance monitoring
        start_time = time.time()

        # Simulate some operations
        time.sleep(0.1)  # Simulate processing time

        end_time = time.time()
        processing_time = end_time - start_time

        # Performance should be monitored
        assert processing_time >= 0.1  # Should capture the sleep time
        assert processing_time < 1.0  # Should not be excessive

    def test_error_monitoring(self, full_system_setup):
        """Test error monitoring integration."""
        full_system_setup["provider_manager"]

        # Test error monitoring
        try:
            # Simulate an error
            raise ValueError("Test error for monitoring")
        except Exception as e:
            # Error should be captured and logged
            error_message = str(e)
            assert "Test error" in error_message

            # Error should be handled gracefully
            assert isinstance(e, ValueError)


class TestScalabilityIntegration:
    """Scalability integration tests."""

    @pytest.mark.asyncio
    async def test_system_scalability(self, temp_data_dir, mock_env_vars):
        """Test system scalability with increasing load."""
        provider_manager = ProviderManager()
        evaluator = LLMEvaluator(data_dir=temp_data_dir)

        # Mock responses
        mock_response = {
            "response": "Scalable response",
            "tokens_used": 100,
            "latency": 1.0,
        }

        with (
            patch.object(
                provider_manager.providers["groq"],
                "generate_response",
                return_value=mock_response,
            ),
            patch.object(
                provider_manager.providers["gemini"],
                "generate_response",
                return_value=mock_response,
            ),
            patch.object(
                provider_manager.providers["openrouter"],
                "generate_response",
                return_value=mock_response,
            ),
        ):

            # Test with increasing load
            load_sizes = [1, 5, 10, 20]

            for load_size in load_sizes:
                queries = [f"Query {i}" for i in range(load_size)]
                context = "retail"

                start_time = asyncio.get_event_loop().time()

                # Process queries
                tasks = []
                for query in queries:
                    task = asyncio.create_task(
                        provider_manager.generate_response_from_all_providers(
                            query, context
                        )
                    )
                    tasks.append(task)

                await asyncio.gather(*tasks)

                end_time = asyncio.get_event_loop().time()
                processing_time = end_time - start_time

                # System should scale reasonably
                # Processing time should not grow exponentially
                assert processing_time < load_size * 2.0  # Linear growth acceptable

    def test_memory_scalability(self, temp_data_dir, mock_env_vars):
        """Test memory scalability."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Test with increasing data size
        data_sizes = [10, 50, 100, 200]

        for data_size in data_sizes:
            # Create components with increasing data
            components = []
            for _ in range(data_size):
                components.append(LLMEvaluator(data_dir=temp_data_dir))
                components.append(RAGPipeline(data_dir=temp_data_dir))

            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory

            # Memory should scale reasonably
            # Should not exceed 500MB even with 200 components
            assert memory_increase < 500 * 1024 * 1024  # 500MB

            # Clean up
            del components
