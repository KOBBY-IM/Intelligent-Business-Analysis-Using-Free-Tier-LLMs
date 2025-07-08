#!/usr/bin/env python3
"""
Abstract base class for all LLM providers with unified response format, rate limiting, and retry logic.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    success: bool
    text: str
    model: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    raw_response: Any = None


class RateLimitException(Exception):
    pass


def rate_limit_and_retry(max_retries: int = 3, min_interval: float = 1.0):
    """
    Decorator for rate limiting and retrying API calls.
    min_interval: Minimum seconds between calls.
    max_retries: Number of retries on failure.
    """

    def decorator(func: Callable):
        last_call = [0.0]

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for attempt in range(max_retries):
                now = time.time()
                elapsed = now - last_call[0]
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                try:
                    result = func(self, *args, **kwargs)
                    last_call[0] = time.time()
                    return result
                except RateLimitException as e:
                    logger.warning(
                        f"Rate limit hit: {e}. Retrying ({attempt+1}/{max_retries})..."
                    )
                    time.sleep(min_interval)
                except Exception as e:
                    logger.error(
                        f"API call failed: {e}. Retrying ({attempt+1}/{max_retries})..."
                    )
                    time.sleep(min_interval)
            return LLMResponse(
                success=False,
                text="",
                model="",
                error="Max retries exceeded",
                raw_response=None,
            )

        return wrapper

    return decorator


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.
    """

    def __init__(self, provider_name: str, models: Optional[List[str]] = None):
        self.provider_name = provider_name
        self.models = models or []
        self.last_health = None

    @abstractmethod
    @rate_limit_and_retry(max_retries=3, min_interval=1.0)
    def generate_response(
        self, query: str, context: str = "", model: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM provider.
        Returns a unified LLMResponse.
        """

    @abstractmethod
    def list_models(self) -> List[str]:
        """Return a list of available model names."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if provider is healthy and available."""

    def get_provider_name(self) -> str:
        return self.provider_name

    def get_last_health(self) -> Optional[bool]:
        return self.last_health

    def set_last_health(self, status: bool):
        self.last_health = status
