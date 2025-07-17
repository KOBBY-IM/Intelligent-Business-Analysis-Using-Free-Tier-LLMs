#!/usr/bin/env python3
"""
Configuration loader for batch evaluation system.

Provides methods to load LLM, evaluation, and other configs from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List

# Try relative import first, fallback to direct import
try:
    from ..utils.common import load_yaml_file
except ImportError:
    # Fallback for when running as script
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.common import load_yaml_file


class ConfigLoader:
    """
    Load and manage configuration files for the evaluation system.
    """

    def __init__(self, config_dir: str = "config"):
        """
        Initialize the loader with the config directory.

        Args:
            config_dir (str): Path to the config directory.
        """
        self.config_dir = Path(config_dir)

    def load_llm_config(self) -> Dict[str, Any]:
        """
        Load LLM provider configuration from llm_config.yaml.

        Returns:
            dict: LLM provider configuration.
        Raises:
            FileNotFoundError: If the config file is missing.
        """
        config_path = self.config_dir / "llm_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"LLM config not found: {config_path}")
        return load_yaml_file(config_path)

    def load_evaluation_config(self) -> Dict[str, Any]:
        """
        Load evaluation configuration from evaluation_config.yaml.

        Returns:
            dict: Evaluation configuration.
        Raises:
            FileNotFoundError: If the config file is missing.
        """
        config_path = self.config_dir / "evaluation_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Evaluation config not found: {config_path}")
        return load_yaml_file(config_path)

    def get_business_questions(self) -> List[str]:
        """
        Get standard business questions (stub for future extension).

        Returns:
            list: List of business questions (currently empty).
        """
        return []

    def get_evaluation_prompts(self) -> List[Dict[str, str]]:
        """Get structured evaluation prompts with categories"""
        prompts = [
            {
                "category": "retail",
                "prompt": "What are the key factors driving customer purchasing decisions in the e-commerce sector?",
                "context": "Focus on data-driven insights and actionable recommendations for e-commerce businesses.",
            },
            {
                "category": "finance",
                "prompt": "What are the main indicators of financial health for a small to medium-sized business?",
                "context": "Consider both quantitative metrics and qualitative factors that investors and stakeholders evaluate.",
            },
            {
                "category": "healthcare",
                "prompt": "What are the primary challenges in implementing electronic health records systems in healthcare facilities?",
                "context": "Address technical, organizational, and regulatory challenges faced by healthcare providers.",
            },
            {
                "category": "strategy",
                "prompt": "What are the critical success factors for scaling a startup from 10 to 100 employees?",
                "context": "Focus on operational, cultural, and strategic considerations during rapid growth phases.",
            },
            {
                "category": "market",
                "prompt": "What are the emerging trends in sustainable business practices and their market impact?",
                "context": "Analyze both environmental and economic implications of sustainable business models.",
            },
        ]
        return prompts
