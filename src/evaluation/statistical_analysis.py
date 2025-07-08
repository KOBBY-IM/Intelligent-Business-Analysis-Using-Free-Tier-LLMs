#!/usr/bin/env python3
"""
Statistical analysis for LLM evaluation results
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats import f_oneway, pearsonr, spearmanr, ttest_ind

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Container for statistical analysis results"""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    description: str = ""


class LLMEvaluationStats:
    """Statistical analysis for LLM evaluation results"""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def calculate_descriptive_stats(self, data: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics for a dataset"""
        try:
            data_array = np.array(data)
            return {
                "mean": float(np.mean(data_array)),
                "median": float(np.median(data_array)),
                "std": float(np.std(data_array, ddof=1)),
                "min": float(np.min(data_array)),
                "max": float(np.max(data_array)),
                "q25": float(np.percentile(data_array, 25)),
                "q75": float(np.percentile(data_array, 75)),
                "count": len(data_array),
            }
        except Exception as e:
            logger.error(f"Error calculating descriptive stats: {e}")
            return {}

    def calculate_confidence_interval(self, data: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for mean"""
        try:
            data_array = np.array(data)
            n = len(data_array)
            if n < 2:
                return (float(data_array[0]), float(data_array[0]))

            mean = np.mean(data_array)
            std_err = np.std(data_array, ddof=1) / np.sqrt(n)
            t_value = stats.t.ppf(1 - self.alpha / 2, n - 1)
            margin_of_error = t_value * std_err

            return (float(mean - margin_of_error), float(mean + margin_of_error))

        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            return (0.0, 0.0)

    def t_test_comparison(
        self, group1: List[float], group2: List[float], alternative: str = "two-sided"
    ) -> StatisticalResult:
        """
        Perform t-test comparison between two groups

        Args:
            group1: First group data
            group2: Second group data
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            StatisticalResult with test results
        """
        try:
            statistic, p_value = ttest_ind(group1, group2, alternative=alternative)

            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(
                (
                    (len(group1) - 1) * np.var(group1, ddof=1)
                    + (len(group2) - 1) * np.var(group2, ddof=1)
                )
                / (len(group1) + len(group2) - 2)
            )
            effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std

            significant = p_value < self.alpha

            description = f"T-test comparing {len(group1)} vs {len(group2)} samples"
            if significant:
                description += f" - Significant difference (p={p_value:.4f})"
            else:
                description += f" - No significant difference (p={p_value:.4f})"

            return StatisticalResult(
                test_name="Independent t-test",
                statistic=float(statistic),
                p_value=float(p_value),
                significant=significant,
                effect_size=float(effect_size),
                description=description,
            )

        except Exception as e:
            logger.error(f"Error performing t-test: {e}")
            return StatisticalResult(
                test_name="Independent t-test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                description=f"Error: {str(e)}",
            )

    def anova_test(self, groups: List[List[float]]) -> StatisticalResult:
        """
        Perform one-way ANOVA test for multiple groups

        Args:
            groups: List of groups, each containing data points

        Returns:
            StatisticalResult with test results
        """
        try:
            # Filter out empty groups
            valid_groups = [group for group in groups if len(group) > 0]

            if len(valid_groups) < 2:
                return StatisticalResult(
                    test_name="One-way ANOVA",
                    statistic=0.0,
                    p_value=1.0,
                    significant=False,
                    description="Insufficient groups for ANOVA",
                )

            statistic, p_value = f_oneway(*valid_groups)

            # Calculate effect size (eta-squared)
            total_data = np.concatenate(valid_groups)
            grand_mean = np.mean(total_data)

            ss_between = sum(
                len(group) * (np.mean(group) - grand_mean) ** 2
                for group in valid_groups
            )
            ss_total = sum((x - grand_mean) ** 2 for x in total_data)
            effect_size = ss_between / ss_total if ss_total > 0 else 0.0

            significant = p_value < self.alpha

            description = f"ANOVA comparing {len(valid_groups)} groups"
            if significant:
                description += f" - Significant differences found (p={p_value:.4f})"
            else:
                description += f" - No significant differences (p={p_value:.4f})"

            return StatisticalResult(
                test_name="One-way ANOVA",
                statistic=float(statistic),
                p_value=float(p_value),
                significant=significant,
                effect_size=float(effect_size),
                description=description,
            )

        except Exception as e:
            logger.error(f"Error performing ANOVA: {e}")
            return StatisticalResult(
                test_name="One-way ANOVA",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                description=f"Error: {str(e)}",
            )

    def correlation_analysis(
        self, x: List[float], y: List[float]
    ) -> Dict[str, StatisticalResult]:
        """
        Perform correlation analysis between two variables

        Args:
            x: First variable data
            y: Second variable data

        Returns:
            Dictionary with Pearson and Spearman correlation results
        """
        try:
            if len(x) != len(y):
                raise ValueError("Variables must have same length")

            results = {}

            # Pearson correlation
            pearson_r, pearson_p = pearsonr(x, y)
            results["pearson"] = StatisticalResult(
                test_name="Pearson Correlation",
                statistic=float(pearson_r),
                p_value=float(pearson_p),
                significant=pearson_p < self.alpha,
                description=f"Pearson r={pearson_r:.4f}, p={pearson_p:.4f}",
            )

            # Spearman correlation
            spearman_r, spearman_p = spearmanr(x, y)
            results["spearman"] = StatisticalResult(
                test_name="Spearman Correlation",
                statistic=float(spearman_r),
                p_value=float(spearman_p),
                significant=spearman_p < self.alpha,
                description=f"Spearman ρ={spearman_r:.4f}, p={spearman_p:.4f}",
            )

            return results

        except Exception as e:
            logger.error(f"Error performing correlation analysis: {e}")
            return {
                "pearson": StatisticalResult(
                    test_name="Pearson Correlation",
                    statistic=0.0,
                    p_value=1.0,
                    significant=False,
                    description=f"Error: {str(e)}",
                ),
                "spearman": StatisticalResult(
                    test_name="Spearman Correlation",
                    statistic=0.0,
                    p_value=1.0,
                    significant=False,
                    description=f"Error: {str(e)}",
                ),
            }

    def rank_llm_providers(
        self, results: Dict[str, List[float]], metric: str = "overall_score"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Rank LLM providers based on performance metrics

        Args:
            results: Dictionary with provider names as keys and metric lists as values
            metric: Metric name for ranking

        Returns:
            Dictionary with ranking information for each provider
        """
        try:
            rankings = {}

            for provider, scores in results.items():
                if not scores:
                    continue

                desc_stats = self.calculate_descriptive_stats(scores)
                ci = self.calculate_confidence_interval(scores)

                rankings[provider] = {
                    "mean_score": desc_stats.get("mean", 0.0),
                    "std_score": desc_stats.get("std", 0.0),
                    "confidence_interval": ci,
                    "sample_size": desc_stats.get("count", 0),
                    "min_score": desc_stats.get("min", 0.0),
                    "max_score": desc_stats.get("max", 0.0),
                }

            # Sort by mean score (descending)
            sorted_rankings = dict(
                sorted(rankings.items(), key=lambda x: x[1]["mean_score"], reverse=True)
            )

            # Add rank information
            for i, (provider, data) in enumerate(sorted_rankings.items(), 1):
                data["rank"] = i
                data["percentile"] = (
                    (len(sorted_rankings) - i + 1) / len(sorted_rankings) * 100
                )

            return sorted_rankings

        except Exception as e:
            logger.error(f"Error ranking providers: {e}")
            return {}

    def compare_providers_pairwise(
        self, results: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, StatisticalResult]]:
        """
        Perform pairwise comparisons between all providers

        Args:
            results: Dictionary with provider names as keys and metric lists as values

        Returns:
            Dictionary with pairwise comparison results
        """
        try:
            providers = list(results.keys())
            comparisons = {}

            for i, provider1 in enumerate(providers):
                comparisons[provider1] = {}

                for j, provider2 in enumerate(providers):
                    if i != j:
                        group1 = results[provider1]
                        group2 = results[provider2]

                        if group1 and group2:  # Only compare if both have data
                            t_test_result = self.t_test_comparison(group1, group2)
                            comparisons[provider1][provider2] = t_test_result

            return comparisons

        except Exception as e:
            logger.error(f"Error performing pairwise comparisons: {e}")
            return {}

    def calculate_effect_sizes(
        self, results: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Calculate effect sizes for provider comparisons

        Args:
            results: Dictionary with provider names as keys and metric lists as values

        Returns:
            Dictionary with effect sizes
        """
        try:
            providers = list(results.keys())
            effect_sizes = {}

            # Use the best performing provider as reference
            provider_means = {p: np.mean(results[p]) for p in providers if results[p]}
            best_provider = max(provider_means, key=provider_means.get)

            for provider in providers:
                if provider != best_provider and results[provider]:
                    # Calculate Cohen's d effect size
                    group1 = results[best_provider]
                    group2 = results[provider]

                    pooled_std = np.sqrt(
                        (
                            (len(group1) - 1) * np.var(group1, ddof=1)
                            + (len(group2) - 1) * np.var(group2, ddof=1)
                        )
                        / (len(group1) + len(group2) - 2)
                    )

                    effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std
                    effect_sizes[f"{best_provider}_vs_{provider}"] = float(effect_size)

            return effect_sizes

        except Exception as e:
            logger.error(f"Error calculating effect sizes: {e}")
            return {}

    def generate_summary_report(
        self, results: Dict[str, List[float]], metric_name: str = "overall_score"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistical summary report

        Args:
            results: Dictionary with provider names as keys and metric lists as values
            metric_name: Name of the metric being analyzed

        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            report = {
                "metric_name": metric_name,
                "total_providers": len(results),
                "total_samples": sum(len(scores) for scores in results.values()),
                "descriptive_stats": {},
                "rankings": self.rank_llm_providers(results, metric_name),
                "pairwise_comparisons": self.compare_providers_pairwise(results),
                "effect_sizes": self.calculate_effect_sizes(results),
                "anova_result": self.anova_test(list(results.values())),
                "confidence_level": self.confidence_level,
            }

            # Add descriptive stats for each provider
            for provider, scores in results.items():
                if scores:
                    report["descriptive_stats"][provider] = (
                        self.calculate_descriptive_stats(scores)
                    )

            return report

        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return {"error": str(e)}

    def interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

    def get_statistical_power(
        self, sample_size: int, effect_size: float, alpha: float = 0.05
    ) -> float:
        """
        Calculate statistical power for a given sample size and effect size

        Args:
            sample_size: Number of samples
            effect_size: Cohen's d effect size
            alpha: Significance level

        Returns:
            Statistical power (0-1)
        """
        try:
            # Simplified power calculation using normal approximation
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = effect_size * np.sqrt(sample_size / 2) - z_alpha
            power = 1 - stats.norm.cdf(z_beta)
            return float(power)

        except Exception as e:
            logger.error(f"Error calculating statistical power: {e}")
            return 0.0
