#!/usr/bin/env python3
"""
Script to check what models are actually available from OpenRouter API
"""

import os
import requests
import json

def check_openrouter_models():
    print("🔍 Checking OpenRouter Available Models")
    print("=" * 50)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        return
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            models_data = response.json()
            
            print(f"✅ Successfully retrieved {len(models_data.get('data', []))} models")
            
            # Filter for free models
            free_models = []
            paid_models = []
            
            for model in models_data.get('data', []):
                model_id = model.get('id', '')
                pricing = model.get('pricing', {})
                
                # Check if it's a free model
                if pricing.get('prompt') == '0' and pricing.get('completion') == '0':
                    free_models.append(model_id)
                else:
                    paid_models.append(model_id)
            
            print(f"\n📋 FREE Models ({len(free_models)}):")
            for model in sorted(free_models):
                print(f"  - {model}")
            
            print(f"\n💰 PAID Models ({len(paid_models)}):")
            for model in sorted(paid_models)[:10]:  # Show first 10
                print(f"  - {model}")
            
            if len(paid_models) > 10:
                print(f"  ... and {len(paid_models) - 10} more paid models")
            
            # Check our configured models
            configured_models = [
                "mistralai/mistral-7b-instruct",
                "deepseek/deepseek-r1-0528-qwen3-8b:free",
                "openrouter/cypher-alpha:free",
                "anthropic/claude-3-haiku:free",
                "meta-llama/llama-3.1-8b-instruct:free",
                "google/gemini-flash-1.5:free",
                "openai/gpt-3.5-turbo:free"
            ]
            
            print(f"\n🔧 Our Configured Models:")
            for model in configured_models:
                if model in free_models:
                    print(f"  ✅ {model}")
                elif model in paid_models:
                    print(f"  💰 {model} (PAID)")
                else:
                    print(f"  ❌ {model} (NOT FOUND)")
            
        else:
            print(f"❌ Failed to get models: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_openrouter_models() 