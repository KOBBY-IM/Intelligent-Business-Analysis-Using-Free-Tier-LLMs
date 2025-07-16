#!/usr/bin/env python3
"""
Quick Health Check for LLM Providers

Tests all available LLM providers with a simple prompt to verify they're working.
"""

import asyncio
import time
import json
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from llm_providers.provider_manager import ProviderManager

def health_check_provider(provider_manager, provider_name):
    """Test a single provider with a health check prompt"""
    try:
        print(f"🔍 Testing {provider_name}...")
        start_time = time.perf_counter()
        
        # Simple health check prompt
        response = provider_manager.generate_response(
            provider_name=provider_name,
            query="Health check: respond with 'OK' if you're working properly.",
            context=""
        )
        
        latency = time.perf_counter() - start_time
        
        return {
            "provider": provider_name,
            "status": "✅ OK" if response.success else "❌ ERROR",
            "latency": round(latency, 3),
            "response": response.text[:100] + "..." if len(response.text) > 100 else response.text,
            "error": response.error,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = str(e).lower()
        status = "❌ ERROR"
        
        # Check for rate limit errors
        if any(keyword in error_msg for keyword in ["rate limit", "quota", "429", "too many requests"]):
            status = "🚨 RATE LIMITED"
        
        return {
            "provider": provider_name,
            "status": status,
            "error": str(e),
            "latency": 0,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Run health check on all available providers"""
    print("🏥 LLM Provider Health Check")
    print("=" * 50)
    
    try:
        # Initialize provider manager
        provider_manager = ProviderManager()
        
        # Get available providers
        providers = provider_manager.get_provider_names()
        
        if not providers:
            print("❌ No LLM providers available!")
            print("Please check your API keys in .env file:")
            print("  - GROQ_API_KEY")
            print("  - GOOGLE_API_KEY") 
            print("  - HUGGINGFACE_API_KEY")
            print("  - OPENROUTER_API_KEY")
            return
        
        print(f"📋 Found {len(providers)} providers: {', '.join(providers)}")
        print("\n🔍 Running health checks...\n")
        
        # Test each provider
        results = []
        for provider in providers:
            result = health_check_provider(provider_manager, provider)
            results.append(result)
            
            # Print result immediately
            print(f"{result['provider']}: {result['status']}")
            if result['status'] == "✅ OK":
                print(f"   Latency: {result['latency']}s")
                print(f"   Response: {result['response']}")
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")
            print()
        
        # Summary
        print("📊 Health Check Summary:")
        print("-" * 30)
        
        working = [r for r in results if r['status'] == "✅ OK"]
        errors = [r for r in results if r['status'] != "✅ OK"]
        rate_limited = [r for r in results if r['status'] == "🚨 RATE LIMITED"]
        
        print(f"✅ Working: {len(working)}/{len(providers)}")
        print(f"❌ Errors: {len(errors)}/{len(providers)}")
        print(f"🚨 Rate Limited: {len(rate_limited)}/{len(providers)}")
        
        if working:
            avg_latency = sum(r['latency'] for r in working) / len(working)
            print(f"⏱️  Average Latency: {avg_latency:.3f}s")
        
        # Save results
        results_file = Path("data/evaluation_results/quick_health_check.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to: {results_file}")
        
        # Recommendations
        if rate_limited:
            print("\n⚠️  Rate Limited Providers:")
            for r in rate_limited:
                print(f"   - {r['provider']}: {r.get('error', 'Rate limit exceeded')}")
        
        if errors and not rate_limited:
            print("\n⚠️  Providers with Errors:")
            for r in errors:
                print(f"   - {r['provider']}: {r.get('error', 'Unknown error')}")
        
        if working:
            print(f"\n✅ All working providers are ready for testing!")
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 