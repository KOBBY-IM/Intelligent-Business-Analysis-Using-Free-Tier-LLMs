#!/usr/bin/env python3
"""
Gemini LLM Provider Implementation with unified interface
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from .base_provider import BaseProvider, LLMResponse, RateLimitException

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Gemini LLM Provider with unified interface"""

    def __init__(self):
        super().__init__("Gemini", ["gemini-1.5-flash", "gemma-3-12b-it"])

        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not found in environment variables")

    def list_models(self) -> List[str]:
        """Return list of available Gemini models"""
        return self.models

    def health_check(self) -> bool:
        """Check if Gemini API is accessible"""
        if not self.api_key:
            self.set_last_health(False)
            return False

        try:
            # Test with a simple model list request
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.get(url, timeout=10)

            is_healthy = response.status_code == 200
            self.set_last_health(is_healthy)

            if not is_healthy:
                logger.warning(f"Gemini health check failed: {response.status_code}")

            return is_healthy

        except Exception as e:
            logger.error(f"Gemini health check error: {e}")
            self.set_last_health(False)
            return False

    def generate_response(
        self, query: str, context: str = "", model: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        """Generate response from Gemini API with unified format"""

        if not self.api_key:
            return LLMResponse(
                success=False,
                text="",
                model=model or "unknown",
                error="GOOGLE_API_KEY not found in environment variables",
            )

        # Use default model if none specified
        model_name = model or "gemini-1.5-flash"

        if model_name not in self.models:
            return LLMResponse(
                success=False,
                text="",
                model=model_name,
                error=f"Model {model_name} not available. Available models: {', '.join(self.models)}",
            )

        start_time = time.time()

        try:
            # Prepare the prompt with context if provided
            full_prompt = query
            if context:
                full_prompt = f"Context: {context}\n\nQuery: {query}"

            # Prepare URL and payload
            url = f"{self.base_url}/{model_name}:generateContent?key={self.api_key}"

            payload = {
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7),
                    "topP": kwargs.get("top_p", 0.9),
                    "topK": kwargs.get("top_k", 40),
                },
            }

            # Add safety settings if specified
            if kwargs.get("safety_settings"):
                payload["safetySettings"] = kwargs["safety_settings"]

            response = requests.post(
                url, json=payload, timeout=kwargs.get("timeout", 30)
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()

                # Extract response text
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        text = candidate["content"]["parts"][0].get("text", "")

                        # Extract token usage if available
                        tokens_used = None
                        if "usageMetadata" in result:
                            tokens_used = result["usageMetadata"].get("totalTokenCount")

                        return LLMResponse(
                            success=True,
                            text=text,
                            model=model_name,
                            tokens_used=tokens_used,
                            latency_ms=latency_ms,
                            raw_response=result,
                        )
                    else:
                        return LLMResponse(
                            success=True,
                            text="Response received but no content found",
                            model=model_name,
                            latency_ms=latency_ms,
                            raw_response=result,
                        )
                else:
                    return LLMResponse(
                        success=True,
                        text="Response received but no candidates found",
                        model=model_name,
                        latency_ms=latency_ms,
                        raw_response=result,
                    )

            elif response.status_code == 429:
                # Rate limit hit
                raise RateLimitException(f"Rate limit exceeded: {response.text}")

            else:
                return LLMResponse(
                    success=False,
                    text="",
                    model=model_name,
                    latency_ms=latency_ms,
                    error=f"API Error {response.status_code}: {response.text}",
                    raw_response=response.text,
                )

        except RateLimitException:
            raise
        except requests.exceptions.Timeout:
            return LLMResponse(
                success=False,
                text="",
                model=model_name,
                latency_ms=(time.time() - start_time) * 1000,
                error="Request timeout",
            )
        except Exception as e:
            return LLMResponse(
                success=False,
                text="",
                model=model_name,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"Request failed: {str(e)}",
            )

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if not self.api_key:
            return None

        try:
            url = f"{self.base_url}/{model_name}?key={self.api_key}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(
                    f"Failed to get model info for {model_name}: {response.status_code}"
                )
                return None

        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return None
