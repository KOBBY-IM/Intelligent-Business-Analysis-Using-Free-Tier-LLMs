#!/usr/bin/env python3
"""
Provider Manager for handling all LLM providers uniformly (refactored for new BaseProvider)
"""

from typing import Dict, List, Optional

from .base_provider import BaseProvider, LLMResponse
from .gemini_provider import GeminiProvider
from .groq_provider import GroqProvider
from .openrouter_provider import OpenRouterProvider
from config.config_loader import ConfigLoader


class ProviderManager:
    """Manager for all LLM providers (refactored)"""

    def __init__(self, config_loader=None):
        self.providers: Dict[str, BaseProvider] = {}
        self.config_loader = config_loader if config_loader else ConfigLoader()
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all available providers using llm_config.yaml"""
        config = self.config_loader.load_llm_config()
        provider_configs = config.get("providers", {})

        self.providers = {}
        if "groq" in provider_configs and provider_configs["groq"].get("enabled", False):
            groq_models = [m["name"] for m in provider_configs["groq"].get("models", [])]
            from .groq_provider import GroqProvider
            self.providers["groq"] = GroqProvider(groq_models)
        if "gemini" in provider_configs and provider_configs["gemini"].get("enabled", False):
            gemini_models = [m["name"] for m in provider_configs["gemini"].get("models", [])]
            from .gemini_provider import GeminiProvider
            self.providers["gemini"] = GeminiProvider(gemini_models)
        if "openrouter" in provider_configs and provider_configs["openrouter"].get("enabled", False):
            openrouter_models = [m["name"] for m in provider_configs["openrouter"].get("models", [])]
            from .openrouter_provider import OpenRouterProvider
            self.providers["openrouter"] = OpenRouterProvider(openrouter_models)

    def get_provider(self, provider_name: str) -> Optional[BaseProvider]:
        """Get a specific provider by name"""
        return self.providers.get(provider_name.lower())

    def get_provider_for_model(self, model_name: str) -> Optional[BaseProvider]:
        """Get provider for a specific model"""
        # Map model names to providers
        model_to_provider = {
            # Groq models
            "llama3-8b-8192": "groq",
            # Gemini models
            "gemma-3-12b-it": "gemini",
            # OpenRouter models
            "mistralai/mistral-7b-instruct": "openrouter",
            "deepseek/deepseek-r1-0528-qwen3-8b": "openrouter",
        }
        
        provider_name = model_to_provider.get(model_name)
        if provider_name:
            return self.get_provider(provider_name)
        return None

    def get_all_providers(self) -> Dict[str, BaseProvider]:
        """Get all providers"""
        return self.providers

    def get_provider_names(self) -> List[str]:
        """Get list of all provider names"""
        return list(self.providers.keys())

    def get_all_models(self) -> Dict[str, List[str]]:
        """Get all models from all providers"""
        all_models = {}
        for provider_name, provider in self.providers.items():
            all_models[provider_name] = provider.list_models()
        return all_models

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.providers.keys())

    def generate_response(
        self, provider_name: str, query: str, model: str = None, context: str = None, **kwargs
    ) -> dict:
        """Generate response from a specific provider and model (unified interface)"""
        provider = self.get_provider(provider_name)
        if not provider:
            return {
                "success": False,
                "response": "",
                "model": model or "unknown",
                "error": f"Provider {provider_name} not found",
            }
        
        # Add context to the query if provided
        if context:
            enhanced_query = f"Context: {context}\n\nQuestion: {query}"
        else:
            enhanced_query = query
            
        llm_response = provider.generate_response(query=enhanced_query, model=model, **kwargs)
        
        # Convert LLMResponse to dict for compatibility
        return {
            "success": llm_response.success,
            "response": llm_response.text,
            "model": llm_response.model,
            "error": llm_response.error,
            "latency": getattr(llm_response, 'latency', 0),
            "token_count": len(llm_response.text.split()) if llm_response.text else 0
        }

    def health_check(self, provider_name: str) -> bool:
        provider = self.get_provider(provider_name)
        if not provider:
            return False
        return provider.health_check()

    def get_model_info(self, provider_name: str, model_name: str):
        provider = self.get_provider(provider_name)
        if not provider:
            return None
        if hasattr(provider, "get_model_info"):
            return provider.get_model_info(model_name)
        return None

    def add_custom_model(
        self,
        provider_name: str,
        model_name: str,
        description: str = "",
        max_tokens: int = 100,
        temperature: float = 0.7,
    ):
        provider = self.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name} not found")
        if hasattr(provider, "add_custom_model"):
            provider.add_custom_model(model_name, description, max_tokens, temperature)
        else:
            raise ValueError(f"Provider {provider_name} does not support custom models")
