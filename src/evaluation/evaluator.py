#!/usr/bin/env python3
"""
Main evaluation engine for LLM performance assessment
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .ground_truth import GroundTruthManager
from .metrics import EvaluationMetrics, EvaluationMetricsCalculator
from .statistical_analysis import LLMEvaluationStats

try:
    from llm_providers.provider_manager import ProviderManager
except ImportError:
    # Fallback for when running as script
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from llm_providers.provider_manager import ProviderManager


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for individual evaluation result"""

    question_id: str
    provider_name: str
    model_name: str
    query: str
    response: str
    ground_truth: str
    metrics: EvaluationMetrics
    context_used: str = ""
    dataset_coverage: float = 0.0  # Percentage of dataset accessed
    retrieval_stats: Dict[str, Any] = None  # Retrieval quality metrics
    evaluation_timestamp: str = ""


@dataclass
class GroundTruthComparison:
    """Container for ground truth vs LLM comparison"""
    
    question_id: str
    ground_truth_response: str
    ground_truth_context: str
    ground_truth_stats: Dict[str, Any]
    llm_responses: Dict[str, EvaluationResult]  # provider_name -> result
    comparison_metrics: Dict[str, Any]
    evaluation_timestamp: str = ""


@dataclass
class BatchEvaluationResult:
    """Container for batch evaluation results"""

    evaluation_id: str
    timestamp: str
    total_questions: int
    total_providers: int
    results: List[EvaluationResult]
    summary_stats: Dict[str, Any]
    statistical_analysis: Dict[str, Any]


class LLMEvaluator:
    """Main evaluation engine for LLM performance assessment"""

    def __init__(
        self,
        ground_truth_file: str = "data/ground_truth_answers.json",
        results_dir: str = "data/evaluation_results",
    ):

        self.metrics_calculator = EvaluationMetricsCalculator()
        self.ground_truth_manager = GroundTruthManager(ground_truth_file)
        self.stats_analyzer = LLMEvaluationStats()
        self.provider_manager = ProviderManager()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info("LLM Evaluator initialized")

    def evaluate_single_response(
        self,
        question_id: str,
        provider_name: str,
        model_name: str,
        response: str,
        response_time_ms: float,
        tokens_used: int,
        context: str = "",
    ) -> EvaluationResult:
        """
        Evaluate a single LLM response

        Args:
            question_id: ID of the question being evaluated
            provider_name: Name of the LLM provider
            model_name: Name of the model used
            response: LLM response text
            response_time_ms: Response time in milliseconds
            tokens_used: Number of tokens consumed
            context: Retrieved context (optional)

        Returns:
            EvaluationResult with all metrics
        """
        try:
            # Get ground truth
            ground_truth_answer = self.ground_truth_manager.get_answer(question_id)
            if not ground_truth_answer:
                raise ValueError(f"Ground truth not found for question: {question_id}")

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(
                query=ground_truth_answer.question,
                response=response,
                ground_truth=ground_truth_answer.answer,
                response_time_ms=response_time_ms,
                tokens_used=tokens_used,
                context=context,
            )

            result = EvaluationResult(
                question_id=question_id,
                provider_name=provider_name,
                model_name=model_name,
                query=ground_truth_answer.question,
                response=response,
                ground_truth=ground_truth_answer.answer,
                metrics=metrics,
                context_used=context,
                evaluation_timestamp=datetime.now().isoformat(),
            )

            logger.info(
                f"Evaluated {provider_name}/{model_name} for question {question_id}"
            )
            return result

        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            # Return error result
            return EvaluationResult(
                question_id=question_id,
                provider_name=provider_name,
                model_name=model_name,
                query="",
                response=response,
                ground_truth="",
                metrics=EvaluationMetrics(
                    relevance_score=0.0,
                    factual_accuracy=0.0,
                    response_time_ms=response_time_ms,
                    token_efficiency=0.0,
                    coherence_score=0.0,
                    completeness_score=0.0,
                    overall_score=0.0,
                    confidence_interval=(0.0, 0.0),
                ),
                evaluation_timestamp=datetime.now().isoformat(),
            )

    def run_batch_evaluation(
        self,
        question_ids: Optional[List[str]] = None,
        provider_names: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
    ) -> BatchEvaluationResult:
        """
        Run batch evaluation across multiple questions, providers, and models

        Args:
            question_ids: Specific question IDs to evaluate (None for all)
            provider_names: Specific providers to test (None for all)
            model_names: Specific models to test (None for all)
            domains: Filter by domains (None for all)
            difficulties: Filter by difficulties (None for all)

        Returns:
            BatchEvaluationResult with comprehensive evaluation
        """
        try:
            evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Starting batch evaluation: {evaluation_id}")

            # Get questions to evaluate
            if question_ids:
                questions = [
                    self.ground_truth_manager.get_answer(qid) for qid in question_ids
                ]
                questions = [q for q in questions if q is not None]
            else:
                questions = list(self.ground_truth_manager.answers.values())

            # Apply filters
            if domains:
                questions = [q for q in questions if q.domain in domains]
            if difficulties:
                questions = [q for q in questions if q.difficulty in difficulties]

            # Get providers to test
            if provider_names:
                providers_to_test = provider_names
            else:
                providers_to_test = self.provider_manager.get_provider_names()

            # Get models to test
            if model_names:
                models_to_test = model_names
            else:
                models_to_test = []
                for provider_name in providers_to_test:
                    provider = self.provider_manager.get_provider(provider_name)
                    if provider:
                        models_to_test.extend(provider.list_models())

            logger.info(
                f"Evaluating {len(questions)} questions across {len(providers_to_test)} providers"
            )

            results = []
            total_evaluations = len(questions) * len(providers_to_test)
            completed_evaluations = 0

            for question in questions:
                for provider_name in providers_to_test:
                    provider = self.provider_manager.get_provider(provider_name)
                    if not provider:
                        logger.warning(f"Provider not found: {provider_name}")
                        continue

                    # Test health
                    if not provider.health_check():
                        logger.warning(
                            f"Provider {provider_name} is not healthy, skipping"
                        )
                        continue

                    # Get models for this provider
                    provider_models = provider.list_models()
                    if model_names:
                        provider_models = [
                            m for m in provider_models if m in model_names
                        ]

                    for model_name in provider_models:
                        try:
                            logger.info(
                                f"Evaluating {provider_name}/{model_name} for question {question.question_id}"
                            )

                            # Generate response
                            start_time = time.time()
                            llm_response = provider.generate_response(
                                query=question.question,
                                model=model_name,
                                max_tokens=500,
                                temperature=0.7,
                            )
                            response_time = (time.time() - start_time) * 1000

                            if llm_response.success:
                                # Evaluate response
                                result = self.evaluate_single_response(
                                    question_id=question.question_id,
                                    provider_name=provider_name,
                                    model_name=model_name,
                                    response=llm_response.text,
                                    response_time_ms=llm_response.latency_ms
                                    or response_time,
                                    tokens_used=llm_response.tokens_used or 0,
                                )
                                results.append(result)
                            else:
                                logger.warning(
                                    f"Failed to get response from {provider_name}/{model_name}: {llm_response.error}"
                                )

                            completed_evaluations += 1
                            logger.info(
                                f"Progress: {completed_evaluations}/{total_evaluations}"
                            )

                        except Exception as e:
                            logger.error(
                                f"Error evaluating {provider_name}/{model_name}: {e}"
                            )
                            completed_evaluations += 1

            # Generate summary statistics
            summary_stats = self._generate_summary_stats(results)

            # Perform statistical analysis
            statistical_analysis = self._perform_statistical_analysis(results)

            batch_result = BatchEvaluationResult(
                evaluation_id=evaluation_id,
                timestamp=datetime.now().isoformat(),
                total_questions=len(questions),
                total_providers=len(providers_to_test),
                results=results,
                summary_stats=summary_stats,
                statistical_analysis=statistical_analysis,
            )

            # Save results
            self._save_evaluation_results(batch_result)

            logger.info(f"Batch evaluation completed: {len(results)} results generated")
            return batch_result

        except Exception as e:
            logger.error(f"Error in batch evaluation: {e}")
            raise

    def run_ground_truth_comparison(
        self,
        question_id: str,
        provider_names: Optional[List[str]] = None,
        llm_dataset_coverage: float = 0.4,  # 40% for LLMs
        ground_truth_coverage: float = 1.0,  # 100% for ground truth
    ) -> GroundTruthComparison:
        """
        Run comparison between ground truth (100% dataset) and LLMs (40% dataset)
        
        Args:
            question_id: ID of the question to evaluate
            provider_names: Specific providers to test (None for all)
            llm_dataset_coverage: Dataset coverage percentage for LLMs (default 40%)
            ground_truth_coverage: Dataset coverage percentage for ground truth (default 100%)
            
        Returns:
            GroundTruthComparison with comprehensive comparison
        """
        try:
            # Get ground truth answer
            ground_truth_answer = self.ground_truth_manager.get_answer(question_id)
            if not ground_truth_answer:
                raise ValueError(f"Ground truth not found for question: {question_id}")
            
            # Generate ground truth context (100% dataset access)
            ground_truth_context = self._generate_full_context(
                question_id, ground_truth_answer, coverage=ground_truth_coverage
            )
            
            # Get ground truth stats
            ground_truth_stats = {
                "dataset_coverage": ground_truth_coverage,
                "context_length": len(ground_truth_context),
                "data_sources": ground_truth_answer.metadata.get("data_source", "Unknown"),
                "analysis_type": ground_truth_answer.metadata.get("analysis_type", "Unknown"),
                "key_points_count": len(ground_truth_answer.key_points),
                "factual_claims_count": len(ground_truth_answer.factual_claims),
            }
            
            # Get providers to test
            if provider_names:
                providers_to_test = provider_names
            else:
                providers_to_test = self.provider_manager.get_provider_names()
            
            # Generate LLM responses with limited dataset access
            llm_responses = {}
            for provider_name in providers_to_test:
                try:
                    provider = self.provider_manager.get_provider(provider_name)
                    if not provider:
                        continue
                    
                    # Generate limited context for LLM (40% dataset access)
                    llm_context = self._generate_limited_context(
                        question_id, ground_truth_answer, coverage=llm_dataset_coverage
                    )
                    
                    # Get LLM response
                    start_time = time.time()
                    llm_response = provider.generate_response(
                        query=ground_truth_answer.question,
                        context=llm_context,
                        max_tokens=500
                    )
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    # Calculate retrieval stats
                    retrieval_stats = {
                        "dataset_coverage": llm_dataset_coverage,
                        "context_length": len(llm_context),
                        "context_quality": self._calculate_context_quality(llm_context),
                        "retrieval_time_ms": response_time_ms,
                    }
                    
                    # Evaluate LLM response
                    evaluation_result = self.evaluate_single_response(
                        question_id=question_id,
                        provider_name=provider_name,
                        model_name=provider.model_name,
                        response=llm_response.response,
                        response_time_ms=response_time_ms,
                        tokens_used=llm_response.tokens_used,
                        context=llm_context,
                    )
                    
                    # Add dataset coverage and retrieval stats
                    evaluation_result.dataset_coverage = llm_dataset_coverage
                    evaluation_result.retrieval_stats = retrieval_stats
                    
                    llm_responses[provider_name] = evaluation_result
                    
                except Exception as e:
                    logger.error(f"Error evaluating {provider_name}: {e}")
                    continue
            
            # Calculate comparison metrics
            comparison_metrics = self._calculate_comparison_metrics(
                ground_truth_answer, llm_responses
            )
            
            result = GroundTruthComparison(
                question_id=question_id,
                ground_truth_response=ground_truth_answer.answer,
                ground_truth_context=ground_truth_context,
                ground_truth_stats=ground_truth_stats,
                llm_responses=llm_responses,
                comparison_metrics=comparison_metrics,
                evaluation_timestamp=datetime.now().isoformat(),
            )
            
            logger.info(f"Completed ground truth comparison for question {question_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in ground truth comparison: {e}")
            raise
    
    def _generate_full_context(self, question_id: str, ground_truth_answer, coverage: float = 1.0) -> str:
        """Generate full context for ground truth (100% dataset access)"""
        # For ground truth, we have access to all data
        context_parts = [
            f"Complete dataset analysis for {ground_truth_answer.domain} domain:",
            f"Data source: {ground_truth_answer.metadata.get('data_source', 'Unknown')}",
            f"Analysis type: {ground_truth_answer.metadata.get('analysis_type', 'Unknown')}",
            f"Key insights: {', '.join(ground_truth_answer.key_points)}",
            f"Factual claims: {', '.join(ground_truth_answer.factual_claims)}",
            f"Dataset coverage: {coverage * 100}% (Full access)",
        ]
        return "\n".join(context_parts)
    
    def _generate_limited_context(self, question_id: str, ground_truth_answer, coverage: float = 0.4) -> str:
        """Generate limited context for LLMs (40% dataset access)"""
        # For LLMs, we simulate limited dataset access
        # In practice, this would be implemented with actual RAG retrieval
        context_parts = [
            f"Limited dataset analysis for {ground_truth_answer.domain} domain:",
            f"Data source: {ground_truth_answer.metadata.get('data_source', 'Unknown')} (Partial access)",
            f"Analysis type: {ground_truth_answer.metadata.get('analysis_type', 'Unknown')}",
            f"Dataset coverage: {coverage * 100}% (Limited access)",
            f"Note: This analysis is based on a subset of available data.",
        ]
        
        # Add partial key points (simulating limited access)
        if ground_truth_answer.key_points:
            num_points = max(1, int(len(ground_truth_answer.key_points) * coverage))
            partial_points = ground_truth_answer.key_points[:num_points]
            context_parts.append(f"Available insights: {', '.join(partial_points)}")
        
        return "\n".join(context_parts)
    
    def _calculate_context_quality(self, context: str) -> float:
        """Calculate context quality score"""
        if not context:
            return 0.0
        
        # Simple quality metrics
        length_score = min(len(context) / 1000, 1.0)  # Normalize by expected length
        diversity_score = len(set(context.split())) / len(context.split()) if context.split() else 0
        
        return (length_score + diversity_score) / 2
    
    def _calculate_comparison_metrics(self, ground_truth_answer, llm_responses: Dict[str, EvaluationResult]) -> Dict[str, Any]:
        """Calculate comparison metrics between ground truth and LLM responses"""
        if not llm_responses:
            return {}
        
        metrics = {
            "total_llm_responses": len(llm_responses),
            "average_accuracy": 0.0,
            "best_provider": None,
            "worst_provider": None,
            "accuracy_range": (0.0, 0.0),
            "coverage_comparison": {
                "ground_truth_coverage": 1.0,
                "llm_average_coverage": 0.4,
                "coverage_gap": 0.6,
            }
        }
        
        # Calculate accuracy metrics
        accuracies = []
        for provider_name, result in llm_responses.items():
            accuracy = result.metrics.factual_accuracy
            accuracies.append(accuracy)
            
            if metrics["best_provider"] is None or accuracy > llm_responses[metrics["best_provider"]].metrics.factual_accuracy:
                metrics["best_provider"] = provider_name
            
            if metrics["worst_provider"] is None or accuracy < llm_responses[metrics["worst_provider"]].metrics.factual_accuracy:
                metrics["worst_provider"] = provider_name
        
        if accuracies:
            metrics["average_accuracy"] = sum(accuracies) / len(accuracies)
            metrics["accuracy_range"] = (min(accuracies), max(accuracies))
        
        return metrics

    def _generate_summary_stats(
        self, results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """Generate summary statistics for evaluation results"""
        try:
            if not results:
                return {}

            # Group results by provider
            provider_results = {}
            for result in results:
                provider = result.provider_name
                if provider not in provider_results:
                    provider_results[provider] = []
                provider_results[provider].append(result)

            # Calculate summary stats for each provider
            summary = {
                "total_evaluations": len(results),
                "providers_tested": len(provider_results),
                "provider_stats": {},
                "overall_stats": {},
            }

            # Provider-specific stats
            for provider, provider_results_list in provider_results.items():
                overall_scores = [
                    r.metrics.overall_score for r in provider_results_list
                ]
                relevance_scores = [
                    r.metrics.relevance_score for r in provider_results_list
                ]
                factual_scores = [
                    r.metrics.factual_accuracy for r in provider_results_list
                ]
                response_times = [
                    r.metrics.response_time_ms for r in provider_results_list
                ]

                summary["provider_stats"][provider] = {
                    "total_evaluations": len(provider_results_list),
                    "avg_overall_score": float(np.mean(overall_scores)),
                    "avg_relevance_score": float(np.mean(relevance_scores)),
                    "avg_factual_accuracy": float(np.mean(factual_scores)),
                    "avg_response_time_ms": float(np.mean(response_times)),
                    "std_overall_score": float(np.std(overall_scores)),
                    "min_overall_score": float(np.min(overall_scores)),
                    "max_overall_score": float(np.max(overall_scores)),
                }

            # Overall stats
            all_overall_scores = [r.metrics.overall_score for r in results]
            all_response_times = [r.metrics.response_time_ms for r in results]

            summary["overall_stats"] = {
                "avg_overall_score": float(np.mean(all_overall_scores)),
                "std_overall_score": float(np.std(all_overall_scores)),
                "avg_response_time_ms": float(np.mean(all_response_times)),
                "best_provider": max(
                    provider_results.keys(),
                    key=lambda p: summary["provider_stats"][p]["avg_overall_score"],
                ),
                "worst_provider": min(
                    provider_results.keys(),
                    key=lambda p: summary["provider_stats"][p]["avg_overall_score"],
                ),
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating summary stats: {e}")
            return {}

    def _perform_statistical_analysis(
        self, results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """Perform statistical analysis on evaluation results"""
        try:
            if not results:
                return {}

            # Group results by provider
            provider_results = {}
            for result in results:
                provider = result.provider_name
                if provider not in provider_results:
                    provider_results[provider] = []
                provider_results[provider].append(result.metrics.overall_score)

            # Perform statistical tests
            analysis = {
                "anova_result": self._serialize_statistical_result(
                    self.stats_analyzer.anova_test(list(provider_results.values()))
                ),
                "rankings": self.stats_analyzer.rank_llm_providers(provider_results),
                "pairwise_comparisons": self._serialize_pairwise_comparisons(
                    self.stats_analyzer.compare_providers_pairwise(provider_results)
                ),
                "effect_sizes": self.stats_analyzer.calculate_effect_sizes(
                    provider_results
                ),
            }

            return analysis

        except Exception as e:
            logger.error(f"Error performing statistical analysis: {e}")
            return {}

    def _serialize_statistical_result(self, result) -> Dict[str, Any]:
        """Convert StatisticalResult to JSON-serializable format"""
        if hasattr(result, "__dict__"):
            return {
                "test_name": result.test_name,
                "statistic": float(result.statistic),
                "p_value": float(result.p_value),
                "significant": result.significant,
                "effect_size": (
                    float(result.effect_size)
                    if result.effect_size is not None
                    else None
                ),
                "confidence_interval": result.confidence_interval,
                "description": result.description,
            }
        return {}

    def _serialize_pairwise_comparisons(
        self, comparisons: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Convert pairwise comparisons to JSON-serializable format"""
        serialized = {}
        for provider1, provider1_comparisons in comparisons.items():
            serialized[provider1] = {}
            for provider2, result in provider1_comparisons.items():
                serialized[provider1][provider2] = self._serialize_statistical_result(
                    result
                )
        return serialized

    def _save_evaluation_results(self, batch_result: BatchEvaluationResult):
        """Save evaluation results to file"""
        try:
            # Save detailed results
            results_file = (
                self.results_dir / f"{batch_result.evaluation_id}_results.json"
            )

            # Convert results to serializable format
            serializable_results = []
            for result in batch_result.results:
                result_dict = asdict(result)
                result_dict["metrics"] = asdict(result.metrics)
                serializable_results.append(result_dict)

            data = {
                "evaluation_id": batch_result.evaluation_id,
                "timestamp": batch_result.timestamp,
                "total_questions": batch_result.total_questions,
                "total_providers": batch_result.total_providers,
                "results": serializable_results,
                "summary_stats": batch_result.summary_stats,
                "statistical_analysis": batch_result.statistical_analysis,
            }

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Save summary report
            summary_file = (
                self.results_dir / f"{batch_result.evaluation_id}_summary.txt"
            )
            self._generate_text_summary(batch_result, summary_file)

            logger.info(f"Evaluation results saved to {results_file}")

        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")

    def _generate_text_summary(
        self, batch_result: BatchEvaluationResult, output_file: Path
    ):
        """Generate human-readable text summary"""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("LLM EVALUATION SUMMARY REPORT\n")
                f.write("=" * 50 + "\n\n")

                f.write(f"Evaluation ID: {batch_result.evaluation_id}\n")
                f.write(f"Timestamp: {batch_result.timestamp}\n")
                f.write(f"Total Questions: {batch_result.total_questions}\n")
                f.write(f"Total Providers: {batch_result.total_providers}\n")
                f.write(f"Total Evaluations: {len(batch_result.results)}\n\n")

                # Provider rankings
                f.write("PROVIDER RANKINGS\n")
                f.write("-" * 20 + "\n")
                rankings = batch_result.statistical_analysis.get("rankings", {})
                for provider, data in rankings.items():
                    f.write(f"{provider}:\n")
                    f.write(f"  Rank: {data.get('rank', 'N/A')}\n")
                    mean_score = data.get("mean_score", 0)
                    std_score = data.get("std_score", 0)
                    confidence_interval = data.get("confidence_interval", (0, 0))
                    f.write(f"  Mean Score: {mean_score:.4f}\n")
                    f.write(f"  Std Dev: {std_score:.4f}\n")
                    f.write(f"  Confidence Interval: {confidence_interval}\n\n")

                # Statistical significance
                f.write("STATISTICAL SIGNIFICANCE\n")
                f.write("-" * 25 + "\n")
                anova_result = batch_result.statistical_analysis.get("anova_result")
                if anova_result:
                    f.write(f"ANOVA Result: {anova_result.get('description', 'N/A')}\n")
                    effect_size = anova_result.get("effect_size", 0)
                    if effect_size is not None:
                        f.write(f"Effect Size: {effect_size:.4f}\n\n")
                    else:
                        f.write("Effect Size: N/A\n\n")

                # Best and worst providers
                overall_stats = batch_result.summary_stats.get("overall_stats", {})
                f.write("OVERALL PERFORMANCE\n")
                f.write("-" * 20 + "\n")
                f.write(f"Best Provider: {overall_stats.get('best_provider', 'N/A')}\n")
                f.write(
                    f"Worst Provider: {overall_stats.get('worst_provider', 'N/A')}\n"
                )
                avg_overall_score = overall_stats.get("avg_overall_score", 0)
                avg_response_time = overall_stats.get("avg_response_time_ms", 0)
                f.write(f"Average Overall Score: {avg_overall_score:.4f}\n")
                f.write(f"Average Response Time: {avg_response_time:.2f}ms\n\n")

        except Exception as e:
            logger.error(f"Error generating text summary: {e}")

    def load_evaluation_results(
        self, evaluation_id: str
    ) -> Optional[BatchEvaluationResult]:
        """Load evaluation results from file"""
        try:
            results_file = self.results_dir / f"{evaluation_id}_results.json"

            if not results_file.exists():
                logger.warning(f"Evaluation results not found: {evaluation_id}")
                return None

            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Reconstruct results
            results = []
            for result_data in data["results"]:
                metrics_data = result_data.pop("metrics")
                metrics = EvaluationMetrics(**metrics_data)
                result = EvaluationResult(**result_data, metrics=metrics)
                results.append(result)

            batch_result = BatchEvaluationResult(
                evaluation_id=data["evaluation_id"],
                timestamp=data["timestamp"],
                total_questions=data["total_questions"],
                total_providers=data["total_providers"],
                results=results,
                summary_stats=data["summary_stats"],
                statistical_analysis=data["statistical_analysis"],
            )

            return batch_result

        except Exception as e:
            logger.error(f"Error loading evaluation results: {e}")
            return None

    def get_evaluation_history(self) -> List[str]:
        """Get list of all evaluation IDs"""
        try:
            evaluation_files = list(self.results_dir.glob("*_results.json"))
            evaluation_ids = [f.stem.replace("_results", "") for f in evaluation_files]
            return sorted(evaluation_ids, reverse=True)
        except Exception as e:
            logger.error(f"Error getting evaluation history: {e}")
            return []
