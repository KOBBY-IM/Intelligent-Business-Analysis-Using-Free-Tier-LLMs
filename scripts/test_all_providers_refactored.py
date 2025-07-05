#!/usr/bin/env python3
"""
Test script for all refactored LLM providers using the unified interface via ProviderManager
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_providers.provider_manager import ProviderManager
from llm_providers.base_provider import RateLimitException

def test_all_providers():
    print("🧪 Testing All Refactored LLM Providers (Unified Interface)")
    print("=" * 60)
    
    manager = ProviderManager()
    provider_names = manager.get_provider_names()
    print(f"Providers found: {provider_names}\n")
    
    for provider_name in provider_names:
        print(f"=== Provider: {provider_name.upper()} ===")
        provider = manager.get_provider(provider_name)
        print(f"Available models: {provider.list_models()}")
        
        # Health check
        print("- Health check:", end=" ")
        try:
            healthy = provider.health_check()
        except Exception as e:
            print(f"❌ (Exception: {e})")
            healthy = False
        else:
            print("✅" if healthy else "❌")
        
        if not healthy:
            print(f"⚠️  {provider_name} is not healthy. Skipping response tests.\n")
            continue
        
        # Test response for each model (limit to 2 models per provider for brevity)
        test_models = provider.list_models()[:2]
        test_queries = [
            "What is the capital of France?",
            "Summarize the importance of data privacy in business.",
        ]
        for model in test_models:
            print(f"\n🤖 Model: {model}")
            for i, query in enumerate(test_queries, 1):
                print(f"  Query {i}: {query[:60]}...")
                try:
                    response = provider.generate_response(query=query, model=model, max_tokens=100)
                    if response.success:
                        print(f"    ✅ Response: {response.text[:80]}...")
                        print(f"    Tokens used: {response.tokens_used}, Latency: {response.latency_ms:.2f}ms")
                    else:
                        print(f"    ❌ Error: {response.error}")
                except RateLimitException as e:
                    print(f"    ⚠️  Rate limit: {e}")
                except Exception as e:
                    print(f"    ❌ Exception: {e}")
        print("\n" + "-" * 60 + "\n")
    print("🎉 All provider tests completed!")

if __name__ == "__main__":
    test_all_providers() 