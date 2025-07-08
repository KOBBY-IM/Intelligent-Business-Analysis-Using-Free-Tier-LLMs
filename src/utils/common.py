"""
Common utility functions for JSON/YAML loading, timing, and environment variable management.

This module provides reusable helpers for file I/O, timing, and environment handling
across the LLM evaluation codebase.
"""

import json
import os
import time
from typing import Any, Callable, Optional

import yaml


def load_json_file(path: str) -> Any:
    """
    Load a JSON file from the given path.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Any: Parsed JSON data (usually dict or list).
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml_file(path: str) -> Any:
    """
    Load a YAML file from the given path.

    Args:
        path (str): Path to the YAML file.

    Returns:
        Any: Parsed YAML data (usually dict or list).
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure execution time of a function in seconds.

    Args:
        func (Callable): Function to wrap.

    Returns:
        Callable: Wrapped function that returns (result, elapsed_time).
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Get an environment variable, optionally raising if required and missing.

    Args:
        key (str): Environment variable name.
        default (Optional[str]): Default value if not set.
        required (bool): If True, raise if variable is not set.

    Returns:
        str: The environment variable value or default.

    Raises:
        EnvironmentError: If required is True and variable is not set.
    """
    value = os.getenv(key, default)
    if required and value is None:
        raise EnvironmentError(f"Required environment variable '{key}' not set.")
    return value
