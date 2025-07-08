"""
Comprehensive unit tests for the evaluation system.
"""

import json
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.evaluator import Evaluator
from evaluation.ground_truth import GroundTruthManager
from evaluation.metrics import MetricsCalculator
from evaluation.statistical_analysis import StatisticalAnalyzer


class TestMetricsCalculator:
    """Test the MetricsCalculator class."""

    @pytest.fixture
    def metrics_calculator(self):
        """Create a MetricsCalculator instance for testing."""
        return MetricsCalculator()

    def test_metrics_calculator_initialization(self, metrics_calculator):
        """Test MetricsCalculator initialization."""
        assert metrics_calculator is not None
        assert hasattr(metrics_calculator, "calculate_accuracy")
        assert hasattr(metrics_calculator, "calculate_latency")
        assert hasattr(metrics_calculator, "calculate_token_count")
        assert hasattr(metrics_calculator, "calculate_relevance")

    def test_calculate_accuracy(self, metrics_calculator):
        """Test accuracy calculation."""
        # Test with perfect match
        accuracy = metrics_calculator.calculate_accuracy(
            response="This is a test response", ground_truth="This is a test response"
        )
        assert accuracy == 1.0

        # Test with partial match
        accuracy = metrics_calculator.calculate_accuracy(
            response="This is a test response",
            ground_truth="This is a different response",
        )
        assert 0.0 <= accuracy <= 1.0

        # Test with empty response
        accuracy = metrics_calculator.calculate_accuracy(
            response="", ground_truth="This is a test response"
        )
        assert accuracy == 0.0

    def test_calculate_latency(self, metrics_calculator):
        """Test latency calculation."""
        start_time = 1000.0
        end_time = 1002.5

        latency = metrics_calculator.calculate_latency(start_time, end_time)
        assert latency == 2.5
        assert isinstance(latency, float)

    def test_calculate_token_count(self, metrics_calculator):
        """Test token count calculation."""
        response = "This is a test response with multiple words"

        token_count = metrics_calculator.calculate_token_count(response)
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_calculate_relevance(self, metrics_calculator):
        """Test relevance calculation."""
        query = "What are retail KPIs?"
        response = (
            "Retail KPIs include sales per square foot and customer lifetime value."
        )

        relevance = metrics_calculator.calculate_relevance(query, response)
        assert 0.0 <= relevance <= 1.0
        assert isinstance(relevance, float)

    def test_calculate_bleu_score(self, metrics_calculator):
        """Test BLEU score calculation."""
        reference = ["This is a test response"]
        candidate = "This is a test response"

        bleu_score = metrics_calculator.calculate_bleu_score(candidate, reference)
        assert 0.0 <= bleu_score <= 1.0
        assert isinstance(bleu_score, float)

    def test_calculate_rouge_score(self, metrics_calculator):
        """Test ROUGE score calculation."""
        reference = "This is a test response"
        candidate = "This is a test response"

        rouge_score = metrics_calculator.calculate_rouge_score(candidate, reference)
        assert 0.0 <= rouge_score <= 1.0
        assert isinstance(rouge_score, float)

    def test_calculate_all_metrics(self, metrics_calculator):
        """Test calculation of all metrics together."""
        query = "What are retail KPIs?"
        response = (
            "Retail KPIs include sales per square foot and customer lifetime value."
        )
        ground_truth = "Retail KPIs include sales per square foot, customer lifetime value, and inventory turnover."
        start_time = 1000.0
        end_time = 1002.5

        metrics = metrics_calculator.calculate_all_metrics(
            query=query,
            response=response,
            ground_truth=ground_truth,
            start_time=start_time,
            end_time=end_time,
        )

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "latency" in metrics
        assert "token_count" in metrics
        assert "relevance" in metrics
        assert "bleu_score" in metrics
        assert "rouge_score" in metrics

        for metric_name, metric_value in metrics.items():
            assert isinstance(metric_value, (int, float))
            if metric_name != "token_count":
                assert 0.0 <= metric_value <= 1.0


class TestGroundTruthManager:
    """Test the GroundTruthManager class."""

    @pytest.fixture
    def ground_truth_manager(self, temp_data_dir):
        """Create a GroundTruthManager instance for testing."""
        return GroundTruthManager(data_dir=temp_data_dir)

    def test_ground_truth_manager_initialization(self, ground_truth_manager):
        """Test GroundTruthManager initialization."""
        assert ground_truth_manager is not None
        assert hasattr(ground_truth_manager, "load_ground_truth")
        assert hasattr(ground_truth_manager, "save_ground_truth")

    def test_load_ground_truth(self, ground_truth_manager, temp_data_dir):
        """Test loading ground truth data."""
        # Create test ground truth file
        test_data = {
            "retail": {
                "What are retail KPIs?": "Retail KPIs include sales per square foot, customer lifetime value, and inventory turnover.",
                "How to improve customer retention?": "Improve customer retention through loyalty programs, personalized marketing, and excellent service.",
            },
            "finance": {
                "What is risk management?": "Risk management involves identifying, assessing, and mitigating financial risks."
            },
        }

        ground_truth_file = os.path.join(temp_data_dir, "ground_truth.json")
        with open(ground_truth_file, "w") as f:
            json.dump(test_data, f)

        # Test loading
        loaded_data = ground_truth_manager.load_ground_truth()
        assert isinstance(loaded_data, dict)
        assert "retail" in loaded_data
        assert "finance" in loaded_data

    def test_save_ground_truth(self, ground_truth_manager, temp_data_dir):
        """Test saving ground truth data."""
        test_data = {
            "healthcare": {
                "What is HIPAA compliance?": "HIPAA compliance ensures patient data protection and privacy."
            }
        }

        ground_truth_manager.save_ground_truth(test_data)

        # Verify file was created
        ground_truth_file = os.path.join(temp_data_dir, "ground_truth.json")
        assert os.path.exists(ground_truth_file)

        # Verify content
        with open(ground_truth_file, "r") as f:
            saved_data = json.load(f)

        assert saved_data == test_data

    def test_get_ground_truth_for_query(self, ground_truth_manager):
        """Test getting ground truth for a specific query."""
        query = "What are retail KPIs?"
        context = "retail"

        # Mock the load_ground_truth method
        with patch.object(ground_truth_manager, "load_ground_truth") as mock_load:
            mock_load.return_value = {
                "retail": {
                    "What are retail KPIs?": "Retail KPIs include sales per square foot, customer lifetime value, and inventory turnover."
                }
            }

            ground_truth = ground_truth_manager.get_ground_truth_for_query(
                query, context
            )
            assert (
                ground_truth
                == "Retail KPIs include sales per square foot, customer lifetime value, and inventory turnover."
            )

    def test_get_ground_truth_for_query_not_found(self, ground_truth_manager):
        """Test getting ground truth for a query that doesn't exist."""
        query = "Nonexistent query"
        context = "retail"

        with patch.object(ground_truth_manager, "load_ground_truth") as mock_load:
            mock_load.return_value = {
                "retail": {
                    "What are retail KPIs?": "Retail KPIs include sales per square foot, customer lifetime value, and inventory turnover."
                }
            }

            ground_truth = ground_truth_manager.get_ground_truth_for_query(
                query, context
            )
            assert ground_truth is None


class TestStatisticalAnalyzer:
    """Test the StatisticalAnalyzer class."""

    @pytest.fixture
    def statistical_analyzer(self):
        """Create a StatisticalAnalyzer instance for testing."""
        return StatisticalAnalyzer()

    def test_statistical_analyzer_initialization(self, statistical_analyzer):
        """Test StatisticalAnalyzer initialization."""
        assert statistical_analyzer is not None
        assert hasattr(statistical_analyzer, "calculate_descriptive_statistics")
        assert hasattr(statistical_analyzer, "perform_anova_test")
        assert hasattr(statistical_analyzer, "perform_t_test")

    def test_calculate_descriptive_statistics(self, statistical_analyzer):
        """Test descriptive statistics calculation."""
        data = {
            "groq": [0.8, 0.9, 0.7, 0.85, 0.9],
            "gemini": [0.75, 0.8, 0.85, 0.9, 0.8],
            "openrouter": [0.7, 0.75, 0.8, 0.85, 0.9],
        }

        stats = statistical_analyzer.calculate_descriptive_statistics(data)

        assert isinstance(stats, dict)
        assert "groq" in stats
        assert "gemini" in stats
        assert "openrouter" in stats

        for provider_stats in stats.values():
            assert "mean" in provider_stats
            assert "std" in provider_stats
            assert "min" in provider_stats
            assert "max" in provider_stats
            assert "median" in provider_stats

    def test_perform_anova_test(self, statistical_analyzer):
        """Test ANOVA test performance."""
        data = {
            "groq": [0.8, 0.9, 0.7, 0.85, 0.9],
            "gemini": [0.75, 0.8, 0.85, 0.9, 0.8],
            "openrouter": [0.7, 0.75, 0.8, 0.85, 0.9],
        }

        anova_result = statistical_analyzer.perform_anova_test(data)

        assert isinstance(anova_result, dict)
        assert "f_statistic" in anova_result
        assert "p_value" in anova_result
        assert "is_significant" in anova_result
        assert isinstance(anova_result["f_statistic"], float)
        assert isinstance(anova_result["p_value"], float)
        assert isinstance(anova_result["is_significant"], bool)

    def test_perform_t_test(self, statistical_analyzer):
        """Test t-test performance."""
        group1 = [0.8, 0.9, 0.7, 0.85, 0.9]
        group2 = [0.75, 0.8, 0.85, 0.9, 0.8]

        t_test_result = statistical_analyzer.perform_t_test(group1, group2)

        assert isinstance(t_test_result, dict)
        assert "t_statistic" in t_test_result
        assert "p_value" in t_test_result
        assert "is_significant" in t_test_result
        assert isinstance(t_test_result["t_statistic"], float)
        assert isinstance(t_test_result["p_value"], float)
        assert isinstance(t_test_result["is_significant"], bool)

    def test_calculate_confidence_intervals(self, statistical_analyzer):
        """Test confidence interval calculation."""
        data = [0.8, 0.9, 0.7, 0.85, 0.9, 0.75, 0.8, 0.85, 0.9, 0.8]

        ci = statistical_analyzer.calculate_confidence_intervals(
            data, confidence_level=0.95
        )

        assert isinstance(ci, dict)
        assert "lower_bound" in ci
        assert "upper_bound" in ci
        assert "confidence_level" in ci
        assert ci["lower_bound"] < ci["upper_bound"]
        assert ci["confidence_level"] == 0.95

    def test_calculate_effect_size(self, statistical_analyzer):
        """Test effect size calculation."""
        group1 = [0.8, 0.9, 0.7, 0.85, 0.9]
        group2 = [0.75, 0.8, 0.85, 0.9, 0.8]

        effect_size = statistical_analyzer.calculate_effect_size(group1, group2)

        assert isinstance(effect_size, float)
        assert effect_size >= 0.0

    def test_generate_statistical_report(self, statistical_analyzer):
        """Test statistical report generation."""
        data = {
            "groq": [0.8, 0.9, 0.7, 0.85, 0.9],
            "gemini": [0.75, 0.8, 0.85, 0.9, 0.8],
            "openrouter": [0.7, 0.75, 0.8, 0.85, 0.9],
        }

        report = statistical_analyzer.generate_statistical_report(data)

        assert isinstance(report, dict)
        assert "descriptive_statistics" in report
        assert "anova_test" in report
        assert "pairwise_comparisons" in report
        assert "effect_sizes" in report
        assert "confidence_intervals" in report


class TestEvaluator:
    """Test the Evaluator class."""

    @pytest.fixture
    def evaluator(self, temp_data_dir):
        """Create an Evaluator instance for testing."""
        return Evaluator(data_dir=temp_data_dir)

    def test_evaluator_initialization(self, evaluator):
        """Test Evaluator initialization."""
        assert evaluator is not None
        assert hasattr(evaluator, "evaluate_response")
        assert hasattr(evaluator, "evaluate_multiple_responses")
        assert hasattr(evaluator, "save_evaluation_results")

    @pytest.mark.asyncio
    async def test_evaluate_response(self, evaluator, mock_llm_response):
        """Test single response evaluation."""
        query = "What are retail KPIs?"
        context = "retail"
        response = mock_llm_response

        # Mock ground truth
        with patch.object(
            evaluator.ground_truth_manager, "get_ground_truth_for_query"
        ) as mock_gt:
            mock_gt.return_value = "Retail KPIs include sales per square foot, customer lifetime value, and inventory turnover."

            evaluation_result = await evaluator.evaluate_response(
                query=query, context=context, response=response
            )

            assert isinstance(evaluation_result, dict)
            assert "query" in evaluation_result
            assert "context" in evaluation_result
            assert "response" in evaluation_result
            assert "metrics" in evaluation_result
            assert "timestamp" in evaluation_result

    @pytest.mark.asyncio
    async def test_evaluate_multiple_responses(self, evaluator, sample_evaluation_data):
        """Test multiple response evaluation."""
        query = sample_evaluation_data["query"]
        context = sample_evaluation_data["context"]
        responses = sample_evaluation_data["responses"]

        # Mock ground truth
        with patch.object(
            evaluator.ground_truth_manager, "get_ground_truth_for_query"
        ) as mock_gt:
            mock_gt.return_value = sample_evaluation_data["ground_truth"]

            evaluation_results = await evaluator.evaluate_multiple_responses(
                query=query, context=context, responses=responses
            )

            assert isinstance(evaluation_results, list)
            assert len(evaluation_results) == len(responses)

            for result in evaluation_results:
                assert "provider" in result
                assert "metrics" in result
                assert "response" in result

    def test_save_evaluation_results(
        self, evaluator, temp_data_dir, sample_evaluation_data
    ):
        """Test saving evaluation results."""
        results = [sample_evaluation_data]

        evaluator.save_evaluation_results(results)

        # Verify file was created
        results_file = os.path.join(temp_data_dir, "evaluation_results.json")
        assert os.path.exists(results_file)

        # Verify content
        with open(results_file, "r") as f:
            saved_results = json.load(f)

        assert isinstance(saved_results, list)
        assert len(saved_results) == len(results)

    def test_load_evaluation_results(
        self, evaluator, temp_data_dir, sample_evaluation_data
    ):
        """Test loading evaluation results."""
        results = [sample_evaluation_data]

        # Save results first
        results_file = os.path.join(temp_data_dir, "evaluation_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f)

        # Load results
        loaded_results = evaluator.load_evaluation_results()

        assert isinstance(loaded_results, list)
        assert len(loaded_results) == len(results)

    @pytest.mark.asyncio
    async def test_batch_evaluation(self, evaluator, sample_queries, sample_contexts):
        """Test batch evaluation of multiple queries."""
        # Mock responses for all providers
        mock_responses = {
            "groq": {"response": "Test response 1", "tokens_used": 100, "latency": 1.5},
            "gemini": {
                "response": "Test response 2",
                "tokens_used": 120,
                "latency": 2.0,
            },
            "openrouter": {
                "response": "Test response 3",
                "tokens_used": 110,
                "latency": 1.8,
            },
        }

        # Mock ground truth
        with patch.object(
            evaluator.ground_truth_manager, "get_ground_truth_for_query"
        ) as mock_gt:
            mock_gt.return_value = "Expected response for testing"

            # Mock provider responses
            with patch.object(evaluator, "evaluate_multiple_responses") as mock_eval:
                mock_eval.return_value = [
                    {"provider": "groq", "metrics": {"accuracy": 0.8, "latency": 1.5}},
                    {
                        "provider": "gemini",
                        "metrics": {"accuracy": 0.75, "latency": 2.0},
                    },
                    {
                        "provider": "openrouter",
                        "metrics": {"accuracy": 0.85, "latency": 1.8},
                    },
                ]

                batch_results = await evaluator.batch_evaluate(
                    queries=sample_queries[:2],  # Test with first 2 queries
                    contexts=list(sample_contexts.values())[:2],
                )

                assert isinstance(batch_results, list)
                assert len(batch_results) > 0

    def test_evaluation_metrics_validation(self, evaluator):
        """Test validation of evaluation metrics."""
        valid_metrics = {
            "accuracy": 0.8,
            "latency": 1.5,
            "token_count": 100,
            "relevance": 0.9,
        }

        invalid_metrics = {
            "accuracy": 1.5,  # Should be <= 1.0
            "latency": -1.0,  # Should be >= 0
            "token_count": -10,  # Should be >= 0
            "relevance": 2.0,  # Should be <= 1.0
        }

        # Test valid metrics
        assert evaluator.validate_metrics(valid_metrics) is True

        # Test invalid metrics
        assert evaluator.validate_metrics(invalid_metrics) is False


class TestEvaluationIntegration:
    """Integration tests for the evaluation system."""

    @pytest.mark.asyncio
    async def test_full_evaluation_pipeline(
        self, temp_data_dir, sample_evaluation_data
    ):
        """Test the complete evaluation pipeline."""
        evaluator = Evaluator(data_dir=temp_data_dir)

        # Mock all dependencies
        with patch.object(
            evaluator.ground_truth_manager, "get_ground_truth_for_query"
        ) as mock_gt:
            mock_gt.return_value = sample_evaluation_data["ground_truth"]

            # Test evaluation
            result = await evaluator.evaluate_response(
                query=sample_evaluation_data["query"],
                context=sample_evaluation_data["context"],
                response=sample_evaluation_data["responses"][0],
            )

            # Verify result structure
            assert isinstance(result, dict)
            assert "metrics" in result
            assert "timestamp" in result

            # Save and load results
            evaluator.save_evaluation_results([result])
            loaded_results = evaluator.load_evaluation_results()

            assert len(loaded_results) == 1
            assert loaded_results[0]["query"] == sample_evaluation_data["query"]

    def test_metrics_calculation_consistency(self):
        """Test that metrics calculation is consistent across multiple calls."""
        metrics_calculator = MetricsCalculator()

        query = "What are retail KPIs?"
        response = (
            "Retail KPIs include sales per square foot and customer lifetime value."
        )
        ground_truth = "Retail KPIs include sales per square foot, customer lifetime value, and inventory turnover."

        # Calculate metrics multiple times
        metrics1 = metrics_calculator.calculate_all_metrics(
            query=query,
            response=response,
            ground_truth=ground_truth,
            start_time=1000.0,
            end_time=1002.5,
        )

        metrics2 = metrics_calculator.calculate_all_metrics(
            query=query,
            response=response,
            ground_truth=ground_truth,
            start_time=1000.0,
            end_time=1002.5,
        )

        # Metrics should be identical
        for key in metrics1:
            assert abs(metrics1[key] - metrics2[key]) < 1e-10

    def test_statistical_analysis_robustness(self):
        """Test that statistical analysis handles edge cases properly."""
        statistical_analyzer = StatisticalAnalyzer()

        # Test with empty data
        empty_data = {}
        stats = statistical_analyzer.calculate_descriptive_statistics(empty_data)
        assert stats == {}

        # Test with single value
        single_data = {"groq": [0.8]}
        stats = statistical_analyzer.calculate_descriptive_statistics(single_data)
        assert "groq" in stats
        assert stats["groq"]["mean"] == 0.8

        # Test with identical values
        identical_data = {"groq": [0.8, 0.8, 0.8], "gemini": [0.8, 0.8, 0.8]}
        anova_result = statistical_analyzer.perform_anova_test(identical_data)
        assert anova_result["f_statistic"] == 0.0  # No variance between groups


class TestEvaluationPerformance:
    """Performance tests for the evaluation system."""

    def test_metrics_calculation_performance(self):
        """Test performance of metrics calculation."""
        import time

        metrics_calculator = MetricsCalculator()

        # Test with large response
        large_response = "This is a very long response. " * 1000
        query = "What are retail KPIs?"
        ground_truth = "Retail KPIs include sales per square foot, customer lifetime value, and inventory turnover."

        start_time = time.time()
        metrics = metrics_calculator.calculate_all_metrics(
            query=query,
            response=large_response,
            ground_truth=ground_truth,
            start_time=1000.0,
            end_time=1002.5,
        )
        end_time = time.time()

        calculation_time = end_time - start_time

        # Calculation should be fast (under 1 second)
        assert calculation_time < 1.0
        assert isinstance(metrics, dict)

    def test_statistical_analysis_performance(self):
        """Test performance of statistical analysis."""
        import time

        statistical_analyzer = StatisticalAnalyzer()

        # Test with large dataset
        large_data = {
            "groq": [0.8 + 0.1 * np.random.random() for _ in range(1000)],
            "gemini": [0.75 + 0.1 * np.random.random() for _ in range(1000)],
            "openrouter": [0.7 + 0.1 * np.random.random() for _ in range(1000)],
        }

        start_time = time.time()
        stats = statistical_analyzer.calculate_descriptive_statistics(large_data)
        anova_result = statistical_analyzer.perform_anova_test(large_data)
        end_time = time.time()

        analysis_time = end_time - start_time

        # Analysis should be fast (under 5 seconds)
        assert analysis_time < 5.0
        assert isinstance(stats, dict)
        assert isinstance(anova_result, dict)


class TestEvaluationSecurity:
    """Security tests for the evaluation system."""

    def test_input_sanitization(self, malicious_inputs):
        """Test that malicious inputs are properly sanitized."""
        metrics_calculator = MetricsCalculator()

        for malicious_input in malicious_inputs:
            # Test that malicious input doesn't cause issues
            try:
                metrics = metrics_calculator.calculate_all_metrics(
                    query=malicious_input,
                    response=malicious_input,
                    ground_truth=malicious_input,
                    start_time=1000.0,
                    end_time=1002.5,
                )
                # Should not raise exceptions
                assert isinstance(metrics, dict)
            except Exception as e:
                # If exception is raised, it should be handled gracefully
                assert "security" in str(e).lower() or "invalid" in str(e).lower()

    def test_file_path_validation(self, temp_data_dir):
        """Test that file paths are properly validated."""
        evaluator = Evaluator(data_dir=temp_data_dir)

        # Test with valid path
        valid_path = os.path.join(temp_data_dir, "test.json")
        assert evaluator.validate_file_path(valid_path) is True

        # Test with malicious path
        malicious_path = "../../../etc/passwd"
        assert evaluator.validate_file_path(malicious_path) is False

    def test_data_validation(self):
        """Test that evaluation data is properly validated."""
        evaluator = Evaluator()

        # Test valid data
        valid_data = {
            "query": "What are retail KPIs?",
            "context": "retail",
            "response": {
                "response": "Test response",
                "tokens_used": 100,
                "latency": 1.5,
            },
        }
        assert evaluator.validate_evaluation_data(valid_data) is True

        # Test invalid data
        invalid_data = {
            "query": "",  # Empty query
            "context": "invalid_context",  # Invalid context
            "response": None,  # None response
        }
        assert evaluator.validate_evaluation_data(invalid_data) is False
