#!/usr/bin/env python3
"""
Provider Manager for handling all LLM providers uniformly (refactored for new BaseProvider)
"""

from typing import Dict, List, Optional
from .base_provider import BaseProvider, LLMResponse
from .groq_provider import GroqProvider
from .gemini_provider import GeminiProvider
from .openrouter_provider import OpenRouterProvider

class ProviderManager:
    """Manager for all LLM providers (refactored)"""
    
    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available providers"""
        self.providers = {
            "groq": GroqProvider(),
            "gemini": GeminiProvider(),
            "openrouter": OpenRouterProvider()
        }
    
    def get_provider(self, provider_name: str) -> Optional[BaseProvider]:
        """Get a specific provider by name"""
        return self.providers.get(provider_name.lower())
    
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
    
    def generate_response(self, provider_name: str, query: str, model: str = None, **kwargs) -> LLMResponse:
        """Generate response from a specific provider and model (unified interface)"""
        provider = self.get_provider(provider_name)
        if not provider:
            return LLMResponse(success=False, text="", model=model or "unknown", error=f"Provider {provider_name} not found")
        return provider.generate_response(query=query, model=model, **kwargs)
    
    def health_check(self, provider_name: str) -> bool:
        provider = self.get_provider(provider_name)
        if not provider:
            return False
        return provider.health_check()
    
    def get_model_info(self, provider_name: str, model_name: str):
        provider = self.get_provider(provider_name)
        if not provider:
            return None
        if hasattr(provider, 'get_model_info'):
            return provider.get_model_info(model_name)
        return None
    
    def add_custom_model(self, provider_name: str, model_name: str, description: str = "", max_tokens: int = 100, temperature: float = 0.7):
        provider = self.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name} not found")
        if hasattr(provider, 'add_custom_model'):
            provider.add_custom_model(model_name, description, max_tokens, temperature)
        else:
            raise ValueError(f"Provider {provider_name} does not support custom models") 