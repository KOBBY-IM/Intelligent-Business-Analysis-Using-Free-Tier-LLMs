#!/usr/bin/env python3
"""
OpenRouter LLM Provider Implementation with unified interface
"""

import os
import time
import requests
import logging
from typing import List, Optional, Dict, Any
from .base_provider import BaseProvider, LLMResponse, RateLimitException

logger = logging.getLogger(__name__)

class OpenRouterProvider(BaseProvider):
    """OpenRouter LLM Provider with unified interface"""
    
    def __init__(self):
        super().__init__("OpenRouter", [
            "mistralai/mistral-7b-instruct",
            "deepseek/deepseek-r1-0528-qwen3-8b",
            "openrouter/cypher-alpha",
            "deepseek/deepseek-chat",
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwen3-14b"
        ])
        
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not found in environment variables")
    
    def list_models(self) -> List[str]:
        """Return list of available OpenRouter models"""
        return self.models
    
    def health_check(self) -> bool:
        """Check if OpenRouter API is accessible"""
        if not self.api_key:
            self.set_last_health(False)
            return False
            
        try:
            # Test with a simple model list request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=10
            )
            
            is_healthy = response.status_code == 200
            self.set_last_health(is_healthy)
            
            if not is_healthy:
                logger.warning(f"OpenRouter health check failed: {response.status_code}")
                
            return is_healthy
            
        except Exception as e:
            logger.error(f"OpenRouter health check error: {e}")
            self.set_last_health(False)
            return False
    
    def generate_response(self, query: str, context: str = "", model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate response from OpenRouter API with unified format"""
        
        if not self.api_key:
            return LLMResponse(
                success=False,
                text="",
                model=model or "unknown",
                error="OPENROUTER_API_KEY not found in environment variables"
            )
        
        # Use default model if none specified
        model_name = model or "mistralai/mistral-7b-instruct"
        
        if model_name not in self.models:
            return LLMResponse(
                success=False,
                text="",
                model=model_name,
                error=f"Model {model_name} not available. Available models: {', '.join(self.models)}"
            )
        
        start_time = time.time()
        
        try:
            # Prepare the prompt with context if provided
            full_prompt = query
            if context:
                full_prompt = f"Context: {context}\n\nQuery: {query}"
            
            # Prepare headers and payload
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/llm-business-analysis",  # Required by OpenRouter
                "X-Title": "LLM Business Analysis Project"
            }
            
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "presence_penalty": kwargs.get("presence_penalty", 0.0)
            }
            
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=payload, 
                timeout=kwargs.get("timeout", 30)
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract response text
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0].get("message", {})
                    text = message.get("content", "")
                    
                    # Extract token usage if available
                    tokens_used = None
                    if "usage" in result:
                        tokens_used = result["usage"].get("total_tokens")
                    
                    return LLMResponse(
                        success=True,
                        text=text,
                        model=model_name,
                        tokens_used=tokens_used,
                        latency_ms=latency_ms,
                        raw_response=result
                    )
                else:
                    return LLMResponse(
                        success=True,
                        text="Response received but no content found",
                        model=model_name,
                        latency_ms=latency_ms,
                        raw_response=result
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
                    raw_response=response.text
                )
                
        except RateLimitException:
            raise
        except requests.exceptions.Timeout:
            return LLMResponse(
                success=False,
                text="",
                model=model_name,
                latency_ms=(time.time() - start_time) * 1000,
                error="Request timeout"
            )
        except Exception as e:
            return LLMResponse(
                success=False,
                text="",
                model=model_name,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"Request failed: {str(e)}"
            )
    
    def get_available_models(self) -> Optional[Dict[str, Any]]:
        """Get list of all available models from OpenRouter"""
        if not self.api_key:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get available models: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return None 