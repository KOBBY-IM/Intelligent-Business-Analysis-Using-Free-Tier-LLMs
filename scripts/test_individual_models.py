#!/usr/bin/env python3
"""
Test Individual LLM Models

Tests each model within each provider to see which specific models are working.
"""

import time
import json
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from llm_providers.provider_manager import ProviderManager

def test_individual_model(provider_manager, provider_name, model_name):
    """Test a specific model within a provider"""
    try:
        print(f"🔍 Testing {provider_name}/{model_name}...")
        start_time = time.perf_counter()
        
        # Test with a simple prompt
        response = provider_manager.generate_response(
            provider_name=provider_name,
            query="Say 'Hello, I am working!' in one sentence.",
            model=model_name,
            context=""
        )
        
        latency = time.perf_counter() - start_time
        
        return {
            "provider": provider_name,
            "model": model_name,
            "status": "✅ WORKING" if response.success else "❌ FAILED",
            "latency": round(latency, 3),
            "response": response.text[:100] + "..." if len(response.text) > 100 else response.text,
            "error": response.error,
            "tokens_used": response.tokens_used,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = str(e).lower()
        status = "❌ ERROR"
        
        # Check for rate limit errors
        if any(keyword in error_msg for keyword in ["rate limit", "quota", "429", "too many requests"]):
            status = "🚨 RATE LIMITED"
        elif any(keyword in error_msg for keyword in ["overloaded", "503", "unavailable"]):
            status = "⚠️ OVERLOADED"
        
        return {
            "provider": provider_name,
            "model": model_name,
            "status": status,
            "error": str(e),
            "latency": 0,
            "tokens_used": None,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Test all models from all providers"""
    print("🤖 Individual LLM Model Testing")
    print("=" * 60)
    
    try:
        # Initialize provider manager
        provider_manager = ProviderManager()
        
        # Get all providers and their models
        all_models = provider_manager.get_all_models()
        
        if not all_models:
            print("❌ No models found!")
            return
        
        print(f"📋 Found models across {len(all_models)} providers:")
        for provider, models in all_models.items():
            print(f"   {provider}: {len(models)} models")
            for model in models:
                print(f"     - {model}")
        print()
        
        # Test each model
        results = []
        total_models = sum(len(models) for models in all_models.values())
        tested_models = 0
        
        for provider_name, models in all_models.items():
            print(f"\n🔧 Testing {provider_name.upper()} models:")
            print("-" * 40)
            
            for model_name in models:
                result = test_individual_model(provider_manager, provider_name, model_name)
                results.append(result)
                tested_models += 1
                
                # Print result immediately
                print(f"{result['model']}: {result['status']}")
                if result['status'] == "✅ WORKING":
                    print(f"   Latency: {result['latency']}s")
                    print(f"   Tokens: {result['tokens_used']}")
                    print(f"   Response: {result['response']}")
                else:
                    print(f"   Error: {result.get('error', 'Unknown error')}")
                print()
        
        # Summary
        print("\n📊 Model Testing Summary:")
        print("=" * 60)
        
        working = [r for r in results if r['status'] == "✅ WORKING"]
        failed = [r for r in results if r['status'] == "❌ FAILED"]
        errors = [r for r in results if r['status'] == "❌ ERROR"]
        rate_limited = [r for r in results if r['status'] == "🚨 RATE LIMITED"]
        overloaded = [r for r in results if r['status'] == "⚠️ OVERLOADED"]
        
        print(f"✅ Working: {len(working)}/{total_models}")
        print(f"❌ Failed: {len(failed)}/{total_models}")
        print(f"❌ Errors: {len(errors)}/{total_models}")
        print(f"🚨 Rate Limited: {len(rate_limited)}/{total_models}")
        print(f"⚠️ Overloaded: {len(overloaded)}/{total_models}")
        
        if working:
            avg_latency = sum(r['latency'] for r in working) / len(working)
            print(f"⏱️  Average Latency (working models): {avg_latency:.3f}s")
        
        # Provider breakdown
        print(f"\n📋 Provider Breakdown:")
        for provider_name in all_models.keys():
            provider_results = [r for r in results if r['provider'] == provider_name]
            working_count = len([r for r in provider_results if r['status'] == "✅ WORKING"])
            total_count = len(provider_results)
            print(f"   {provider_name}: {working_count}/{total_count} models working")
        
        # Working models list
        if working:
            print(f"\n✅ Working Models:")
            for result in working:
                print(f"   - {result['provider']}/{result['model']} ({result['latency']}s)")
        
        # Failed models list
        if failed or errors or rate_limited or overloaded:
            print(f"\n❌ Non-Working Models:")
            for result in results:
                if result['status'] != "✅ WORKING":
                    print(f"   - {result['provider']}/{result['model']}: {result['status']}")
                    if result.get('error'):
                        print(f"     Error: {result['error'][:100]}...")
        
        # Save results
        results_file = Path("data/evaluation_results/individual_model_tests.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to: {results_file}")
        
        # Recommendations
        if working:
            print(f"\n✅ Ready for testing with {len(working)} working models!")
        else:
            print(f"\n❌ No working models found. Check API keys and service status.")
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 