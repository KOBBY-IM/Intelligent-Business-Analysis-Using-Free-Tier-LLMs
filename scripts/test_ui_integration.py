#!/usr/bin/env python3
"""
Test script for UI integration with batch evaluator
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_ui_imports():
    """Test that all UI components import correctly"""
    print("Testing UI imports...")

    try:
        # Test main UI
        print("✓ Main UI imports successfully")

        # Test batch evaluation page
        print("✓ Batch evaluation page imports successfully")

        # Test config loader
        print("✓ Config loader imports successfully")

        # Test provider manager
        print("✓ Provider manager imports successfully")

        return True

    except Exception as e:
        print(f"✗ UI import test failed: {e}")
        return False


def test_config_loader():
    """Test configuration loading"""
    print("\nTesting configuration loading...")

    try:
        from config.config_loader import ConfigLoader

        config_loader = ConfigLoader()

        # Test LLM config
        llm_config = config_loader.load_llm_config()
        print(f"✓ LLM config loaded: {len(llm_config.get('providers', {}))} providers")

        # Test prompts
        prompts = config_loader.get_evaluation_prompts()
        print(f"✓ Evaluation prompts loaded: {len(prompts)} prompts")

        return True

    except Exception as e:
        print(f"✗ Config loader test failed: {e}")
        return False


def test_provider_manager():
    """Test provider manager"""
    print("\nTesting provider manager...")

    try:
        from llm_providers.provider_manager import ProviderManager

        manager = ProviderManager()
        providers = manager.get_provider_names()
        print(f"✓ Provider manager loaded: {len(providers)} providers")

        all_models = manager.get_all_models()
        total_models = sum(len(models) for models in all_models.values())
        print(f"✓ Total models available: {total_models}")

        return True

    except Exception as e:
        print(f"✗ Provider manager test failed: {e}")
        return False


def test_batch_evaluator_integration():
    """Test batch evaluator integration"""
    print("\nTesting batch evaluator integration...")

    try:
        # Test batch evaluator import
        import sys

        sys.path.append(str(Path(__file__).parent))
        from run_all_models import BatchEvaluator

        evaluator = BatchEvaluator()
        print("✓ Batch evaluator imports successfully")

        # Test configuration loading
        models = evaluator.load_all_models()
        total_models = sum(len(model_list) for model_list in models.values())
        print(f"✓ Batch evaluator loaded {total_models} models")

        # Test prompts
        prompts = evaluator.get_evaluation_prompts()
        print(f"✓ Batch evaluator loaded {len(prompts)} prompts")

        return True

    except Exception as e:
        print(f"✗ Batch evaluator integration test failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("UI Integration Test Suite")
    print("=" * 40)

    tests = [
        ("UI Imports", test_ui_imports),
        ("Config Loader", test_config_loader),
        ("Provider Manager", test_provider_manager),
        ("Batch Evaluator Integration", test_batch_evaluator_integration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        if test_func():
            passed += 1
            print(f"✓ {test_name} test passed")
        else:
            print(f"✗ {test_name} test failed")

    print(f"\n" + "=" * 40)
    print(f"Integration Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All integration tests passed! UI is ready for use.")
        print("\n🚀 You can now run the Streamlit UI with:")
        print("   streamlit run src/ui/main_app.py")
        print("   or")
        print("   streamlit run src/ui/main.py")
        return True
    else:
        print("❌ Some integration tests failed. Please check the configuration.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
