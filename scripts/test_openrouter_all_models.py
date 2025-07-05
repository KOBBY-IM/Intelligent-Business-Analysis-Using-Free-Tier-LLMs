#!/usr/bin/env python3
"""
Test script specifically for OpenRouter to test ALL available models
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_providers.openrouter_provider import OpenRouterProvider

def test_openrouter_all_models():
    print("🧪 Testing ALL OpenRouter Models")
    print("=" * 50)
    
    provider = OpenRouterProvider()
    
    # List all available models
    all_models = provider.list_models()
    print(f"📋 Total OpenRouter models: {len(all_models)}")
    print(f"Models: {all_models}")
    
    # Health check
    print(f"\n🏥 Health check:", end=" ")
    try:
        healthy = provider.health_check()
        print("✅" if healthy else "❌")
    except Exception as e:
        print(f"❌ (Exception: {e})")
        healthy = False
    
    if not healthy:
        print("⚠️  OpenRouter is not healthy. Cannot proceed with tests.")
        return
    
    # Test queries
    test_queries = [
        "What is the capital of France?",
        "Explain machine learning in simple terms.",
        "What are the key factors for business success?",
    ]
    
    successful_tests = 0
    failed_tests = 0
    rate_limited_tests = 0
    
    # Test each model
    for model_idx, model in enumerate(all_models, 1):
        print(f"\n🤖 Model {model_idx}/{len(all_models)}: {model}")
        
        for query_idx, query in enumerate(test_queries, 1):
            print(f"  📝 Query {query_idx}: {query[:40]}...")
            
            try:
                response = provider.generate_response(
                    query=query,
                    model=model,
                    max_tokens=150,
                    temperature=0.7
                )
                
                if response.success:
                    successful_tests += 1
                    print(f"    ✅ Success: {response.text[:50]}...")
                    print(f"    📊 Tokens: {response.tokens_used}, Latency: {response.latency_ms:.2f}ms")
                else:
                    failed_tests += 1
                    print(f"    ❌ Error: {response.error}")
                    
            except Exception as e:
                if "rate limit" in str(e).lower():
                    rate_limited_tests += 1
                    print(f"    ⚠️  Rate limit: {e}")
                    time.sleep(5)  # Wait longer for rate limits
                else:
                    failed_tests += 1
                    print(f"    ❌ Exception: {e}")
            
            # Small delay between queries
            time.sleep(2)
    
    # Summary
    total_tests = successful_tests + failed_tests + rate_limited_tests
    print(f"\n📈 OPENROUTER TEST SUMMARY")
    print(f"{'='*30}")
    print(f"Total tests: {total_tests}")
    print(f"✅ Successful: {successful_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"⚠️  Rate limited: {rate_limited_tests}")
    if total_tests > 0:
        print(f"📊 Success rate: {(successful_tests/total_tests*100):.1f}%")
    
    print(f"\n🎉 OpenRouter testing completed!")

def test_specific_openrouter_models(models: list):
    """Test specific OpenRouter models"""
    print(f"🧪 Testing Specific OpenRouter Models: {models}")
    print("=" * 50)
    
    provider = OpenRouterProvider()
    
    # Health check
    print(f"🏥 Health check:", end=" ")
    try:
        healthy = provider.health_check()
        print("✅" if healthy else "❌")
    except Exception as e:
        print(f"❌ (Exception: {e})")
        healthy = False
    
    if not healthy:
        print("⚠️  OpenRouter is not healthy. Cannot proceed with tests.")
        return
    
    test_query = "What is artificial intelligence and how is it used in business?"
    
    for model in models:
        if model not in provider.list_models():
            print(f"❌ Model '{model}' not available in OpenRouter")
            continue
            
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
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        time.sleep(3)  # Delay between models

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OpenRouter models")
    parser.add_argument("--models", nargs="+", help="Test specific models only")
    
    args = parser.parse_args()
    
    if args.models:
        test_specific_openrouter_models(args.models)
    else:
        test_openrouter_all_models() 