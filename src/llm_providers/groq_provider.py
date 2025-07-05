#!/usr/bin/env python3
"""
Groq LLM Provider Implementation (Unified Interface)
"""

import os
import requests
import logging
import time
from typing import List, Optional
from .base_provider import BaseProvider, LLMResponse, RateLimitException

logger = logging.getLogger(__name__)

GROQ_MODELS = [
    "llama3-8b-8192",
    "llama-3.1-8b-instant",
    "qwen-qwq-32b"
]

class GroqProvider(BaseProvider):
    def __init__(self):
        super().__init__(provider_name="groq", models=GROQ_MODELS)
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def list_models(self) -> List[str]:
        return self.models

    def health_check(self) -> bool:
        # Try a minimal request to the API
        try:
            resp = self.generate_response("Hello", model=self.models[0], max_tokens=5)
            healthy = resp.success
            self.set_last_health(healthy)
            return healthy
        except Exception as e:
            logger.error(f"Groq health check failed: {e}")
            self.set_last_health(False)
            return False

    def generate_response(self, query: str, context: str = "", model: Optional[str] = None, max_tokens: int = 256, temperature: float = 0.7, **kwargs) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(success=False, text="", model=model or "", error="GROQ_API_KEY not set", raw_response=None)
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
            "temperature": temperature
        }
        try:
            resp = requests.post(self.base_url, headers=self._headers(), json=payload, timeout=30)
            latency = (time.time() - start) * 1000
            if resp.status_code == 429:
                raise RateLimitException(f"Rate limit hit: {resp.text}")
            if resp.status_code != 200:
                return LLMResponse(success=False, text="", model=model, error=f"HTTP {resp.status_code}: {resp.text}", latency_ms=latency, raw_response=resp.text)
            data = resp.json()
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
                raw_response=data
            )
        except RateLimitException as e:
            latency = (time.time() - start) * 1000
            logger.warning(f"Groq rate limit: {e}")
            raise
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"Groq API error: {e}")
            return LLMResponse(success=False, text="", model=model, error=str(e), latency_ms=latency, raw_response=None) 