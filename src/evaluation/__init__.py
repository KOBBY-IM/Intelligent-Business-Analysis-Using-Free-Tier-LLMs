"""
Evaluation Module

This module provides comprehensive evaluation capabilities for LLM responses including:
- Metrics calculation (relevance, accuracy, coherence, etc.)
- Statistical analysis and comparison
- Ground truth management
- Automated evaluation pipelines
"""

from .evaluator import LLMEvaluator
from .ground_truth import GroundTruthManager
from .metrics import EvaluationMetricsCalculator
from .statistical_analysis import LLMEvaluationStats

__all__ = [
    "LLMEvaluator",
    "EvaluationMetricsCalculator",
    "LLMEvaluationStats",
    "GroundTruthManager",
]
