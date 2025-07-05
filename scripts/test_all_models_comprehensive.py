#!/usr/bin/env python3
"""
Comprehensive test script for ALL models across all LLM providers using the unified interface
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_providers.provider_manager import ProviderManager
from llm_providers.base_provider import RateLimitException

def test_all_models_comprehensive():
    print("🧪 Comprehensive Testing of ALL Models Across All LLM Providers")
    print("=" * 80)
    
    manager = ProviderManager()
    provider_names = manager.get_provider_names()
    print(f"Providers found: {provider_names}\n")
    
    # Test queries for different scenarios
    test_queries = [
        "What is the capital of France?",
        "Explain the concept of machine learning in simple terms.",
        "What are the key factors for business success?",
    ]
    
    total_models = 0
    successful_tests = 0
    failed_tests = 0
    rate_limited_tests = 0
    
    for provider_name in provider_names:
        print(f"\n{'='*20} Provider: {provider_name.upper()} {'='*20}")
        provider = manager.get_provider(provider_name)
        all_models = provider.list_models()
        print(f"📋 Total models available: {len(all_models)}")
        print(f"Models: {all_models}")
        
        # Health check
        print(f"\n🏥 Health check:", end=" ")
        try:
            healthy = provider.health_check()
        except Exception as e:
            print(f"❌ (Exception: {e})")
            healthy = False
        else:
            print("✅" if healthy else "❌")
        
        if not healthy:
            print(f"⚠️  {provider_name} is not healthy. Skipping all model tests.\n")
            continue
        
        # Test ALL models for this provider
        for model_idx, model in enumerate(all_models, 1):
            print(f"\n🤖 Model {model_idx}/{len(all_models)}: {model}")
            
            for query_idx, query in enumerate(test_queries, 1):
                total_models += 1
                print(f"  📝 Query {query_idx}: {query[:50]}...")
                
                try:
                    start_time = time.time()
                    response = provider.generate_response(
                        query=query, 
                        model=model, 
                        max_tokens=150,
                        temperature=0.7
                    )
                    test_time = time.time() - start_time
                    
                    if response.success:
                        successful_tests += 1
                        print(f"    ✅ Success: {response.text[:60]}...")
                        print(f"    📊 Tokens: {response.tokens_used}, Latency: {response.latency_ms:.2f}ms")
                        print(f"    ⏱️  Test time: {test_time:.2f}s")
                    else:
                        failed_tests += 1
                        print(f"    ❌ Error: {response.error}")
                        
                except RateLimitException as e:
                    rate_limited_tests += 1
                    print(f"    ⚠️  Rate limit: {e}")
                    # Wait a bit before continuing
                    time.sleep(2)
                except Exception as e:
                    failed_tests += 1
                    print(f"    ❌ Exception: {e}")
                
                # Small delay between queries to avoid rate limits
                time.sleep(1)
        
        print(f"\n{'='*60}")
    
    # Summary
    print(f"\n📈 TEST SUMMARY")
    print(f"{'='*30}")
    print(f"Total model tests attempted: {total_models}")
    print(f"✅ Successful tests: {successful_tests}")
    print(f"❌ Failed tests: {failed_tests}")
    print(f"⚠️  Rate limited tests: {rate_limited_tests}")
    print(f"📊 Success rate: {(successful_tests/total_models*100):.1f}%" if total_models > 0 else "N/A")
    
    print(f"\n🎉 Comprehensive testing completed!")

def test_specific_provider_models(provider_name: str, models: list = None):
    """Test specific models for a specific provider"""
    print(f"🧪 Testing Specific Models for {provider_name.upper()}")
    print("=" * 60)
    
    manager = ProviderManager()
    provider = manager.get_provider(provider_name)
    
    if not provider:
        print(f"❌ Provider '{provider_name}' not found!")
        return
    
    # Use all models if none specified
    if models is None:
        models = provider.list_models()
    
    print(f"Testing models: {models}")
    
    test_query = "What is artificial intelligence and how is it used in business?"
    
    for model in models:
        print(f"\n🤖 Testing model: {model}")
        try:
            response = provider.generate_response(
                query=test_query,
                model=model,
                max_tokens=200
            )
            
            if response.success:
                print(f"✅ Success!")
                print(f"Response: {response.text}")
                print(f"Tokens: {response.tokens_used}, Latency: {response.latency_ms:.2f}ms")
            else:
                print(f"❌ Error: {response.error}")
                
        except RateLimitException as e:
            print(f"⚠️  Rate limit: {e}")
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        time.sleep(2)  # Delay between models

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM providers and models")
    parser.add_argument("--provider", help="Test specific provider only")
    parser.add_argument("--models", nargs="+", help="Test specific models only")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive test of all models")
    
    args = parser.parse_args()
    
    if args.provider:
        test_specific_provider_models(args.provider, args.models)
    elif args.comprehensive:
        test_all_models_comprehensive()
    else:
        print("Running comprehensive test by default...")
        test_all_models_comprehensive() 