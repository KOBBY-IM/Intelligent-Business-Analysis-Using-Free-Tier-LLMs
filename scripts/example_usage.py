#!/usr/bin/env python3
"""
Example usage of the unified LLM provider system
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_providers import ProviderManager

def example_basic_usage():
    """Example of basic usage"""
    print("=" * 60)
    print("BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Initialize the provider manager
    manager = ProviderManager()
    
    # Get all available models
    all_models = manager.get_all_models()
    print(f"Available models: {all_models}")
    
    # Generate a response from a specific provider and model
    provider_name = "groq"
    model_name = "llama3-8b-8192"
    prompt = "Explain quantum computing in simple terms."
    
    print(f"\nGenerating response from {provider_name}/{model_name}...")
    success, response, error = manager.generate_response(provider_name, model_name, prompt)
    
    if success:
        print(f"Response: {response}")
    else:
        print(f"Error: {error}")

def example_multiple_providers():
    """Example of using multiple providers"""
    print("\n" + "=" * 60)
    print("MULTIPLE PROVIDERS EXAMPLE")
    print("=" * 60)
    
    manager = ProviderManager()
    
    # Test the same prompt across different providers
    prompt = "What are the benefits of renewable energy?"
    
    providers_to_test = [
        ("groq", "llama3-8b-8192"),
        ("gemini", "gemini-1.5-flash"),
        ("openrouter", "mistralai/mistral-7b-instruct")
    ]
    
    for provider_name, model_name in providers_to_test:
        print(f"\n--- {provider_name.upper()} / {model_name} ---")
        success, response, error = manager.generate_response(provider_name, model_name, prompt)
        
        if success:
            print(f"Response: {response[:200]}...")
        else:
            print(f"Error: {error}")

def example_custom_model():
    """Example of adding a custom model"""
    print("\n" + "=" * 60)
    print("CUSTOM MODEL EXAMPLE")
    print("=" * 60)
    
    manager = ProviderManager()
    
    # Add a custom model to Groq (if you have access to it)
    try:
        manager.add_custom_model(
            provider_name="groq",
            model_name="custom-model",
            description="A custom model for testing",
            max_tokens=150,
            temperature=0.5
        )
        print("Custom model added successfully!")
        
        # Test the custom model
        success, response, error = manager.generate_response(
            "groq", "custom-model", "Hello, this is a test."
        )
        
        if success:
            print(f"Custom model response: {response}")
        else:
            print(f"Custom model error: {error}")
            
    except Exception as e:
        print(f"Could not add custom model: {e}")

def example_model_info():
    """Example of getting model information"""
    print("\n" + "=" * 60)
    print("MODEL INFORMATION EXAMPLE")
    print("=" * 60)
    
    manager = ProviderManager()
    
    # Get information about a specific model
    model_info = manager.get_model_info("groq", "qwen-qwq-32b")
    
    if model_info:
        print(f"Model: {model_info['name']}")
        print(f"Description: {model_info['description']}")
        print(f"Max Tokens: {model_info['max_tokens']}")
        print(f"Temperature: {model_info['temperature']}")
        print(f"Provider: {model_info['provider']}")

def main():
    """Main function to run all examples"""
    print("UNIFIED LLM PROVIDER SYSTEM - USAGE EXAMPLES")
    print("=" * 80)
    
    # Run all examples
    example_basic_usage()
    example_multiple_providers()
    example_custom_model()
    example_model_info()
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main() 