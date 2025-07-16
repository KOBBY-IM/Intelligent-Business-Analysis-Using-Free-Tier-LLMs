#!/usr/bin/env python3
"""
Groq LLM Provider Implementation (Unified Interface)
"""

import logging
import os
import time
from typing import List, Optional

import requests

from .base_provider import BaseProvider, LLMResponse

logger = logging.getLogger(__name__)


class GroqProvider(BaseProvider):
    def __init__(self, model_list):
        super().__init__("Groq", model_list)
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment variables")

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def list_models(self) -> List[str]:
        return self.models

    def health_check(self) -> bool:
        try:
            resp = self.generate_response("Hello", model=self.models[0], max_tokens=5)
            healthy = resp.success
            self.set_last_health(healthy)
            return healthy
        except Exception as e:
            logger.error(f"Groq health check failed: {e}")
            self.set_last_health(False)
            return False

    def generate_response(
        self,
        query: str,
        context: str = "",
        model: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                success=False,
                text="",
                model=model or "",
                error="GROQ_API_KEY not set",
                raw_response=None,
            )
        model = model or self.models[0]
        start = time.time()
        # Compose prompt
        if context:
            prompt = f"""Based on the following context, answer the question. If the context doesn't contain relevant information, say so.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
        else:
            prompt = query
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        retries = 3
        for attempt in range(retries):
            try:
                resp = requests.post(
                    self.base_url, headers=self._headers(), json=payload, timeout=30
                )
                latency = (time.time() - start) * 1000
                logger.info(f"Groq API status: {resp.status_code} {resp.reason}")
                if resp.status_code == 429:
                    logger.warning(f"Groq rate limit hit: {resp.text}")
                    if attempt < retries - 1:
                        time.sleep(2**attempt)
                        continue
                    return LLMResponse(
                        success=False,
                        text="",
                        model=model,
                        error=f"Rate limit: {resp.text}",
                        latency_ms=latency,
                        raw_response=resp.text,
                    )
                if resp.status_code >= 500:
                    logger.warning(f"Groq server error: {resp.text}")
                    if attempt < retries - 1:
                        time.sleep(2**attempt)
                        continue
                if resp.status_code != 200:
                    return LLMResponse(
                        success=False,
                        text="",
                        model=model,
                        error=f"HTTP {resp.status_code}: {resp.text}",
                        latency_ms=latency,
                        raw_response=resp.text,
                    )
                try:
                    data = resp.json()
                except Exception as e:
                    logger.error(f"Groq invalid JSON: {e}")
                    return LLMResponse(
                        success=False,
                        text="",
                        model=model,
                        error="Invalid JSON response",
                        latency_ms=latency,
                        raw_response=resp.text,
                    )
                text = ""
                if "choices" in data and data["choices"]:
                    text = data["choices"][0].get("message", {}).get("content", "")
                tokens_used = data.get("usage", {}).get("total_tokens")
                return LLMResponse(
                    success=True,
                    text=text,
                    model=model,
                    tokens_used=tokens_used,
                    latency_ms=latency,
                    error=None,
                    raw_response=data,
                )
            except requests.Timeout:
                logger.error("Groq API timeout.")
                if attempt < retries - 1:
                    time.sleep(2**attempt)
                    continue
                latency = (time.time() - start) * 1000
                return LLMResponse(
                    success=False,
                    text="",
                    model=model,
                    error="Timeout",
                    latency_ms=latency,
                    raw_response=None,
                )
            except requests.RequestException as e:
                logger.error(f"Groq API error: {e}")
                if attempt < retries - 1:
                    time.sleep(2**attempt)
                    continue
                latency = (time.time() - start) * 1000
                return LLMResponse(
                    success=False,
                    text="",
                    model=model,
                    error=str(e),
                    latency_ms=latency,
                    raw_response=None,
                )
        # If all retries fail
        latency = (time.time() - start) * 1000
        return LLMResponse(
            success=False,
            text="",
            model=model,
            error="Max retries exceeded",
            latency_ms=latency,
            raw_response=None,
        )
