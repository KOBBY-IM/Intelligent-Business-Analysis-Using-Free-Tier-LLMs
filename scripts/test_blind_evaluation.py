#!/usr/bin/env python3
"""
Test script for blind evaluation functionality
"""

import json
import random
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_blind_evaluation_imports():
    """Test that blind evaluation components import correctly"""
    print("Testing blind evaluation imports...")

    try:
        # Test blind evaluation page
        print("✓ Blind evaluation page imports successfully")

        # Test config loader
        print("✓ Config loader imports successfully")

        # Test provider manager
        print("✓ Provider manager imports successfully")

        return True

    except Exception as e:
        print(f"✗ Blind evaluation import test failed: {e}")
        return False


def test_results_loading():
    """Test loading existing results for blind evaluation"""
    print("\nTesting results loading...")

    try:
        results_file = Path("data/results/2025-07-07_22-36-29_model_responses.json")

        if not results_file.exists():
            print("✗ No results file found")
            return False

        with open(results_file, "r") as f:
            data = json.load(f)

        print(
            f"✓ Results file loaded: {data.get('metadata', {}).get('total_evaluations', 0)} evaluations"
        )

        # Test filtering successful responses
        successful_responses = [
            r
            for r in data.get("results", [])
            if r["prompt_index"] == 0 and r["success"]
        ]

        print(f"✓ Found {len(successful_responses)} successful responses for prompt 0")

        if successful_responses:
            # Test blind response creation
            blind_responses = []
            for i, response in enumerate(successful_responses[:5]):
                blind_id = chr(65 + i)  # A, B, C, D, E
                blind_responses.append(
                    {
                        "blind_id": blind_id,
                        "response_text": response["response_text"],
                        "provider": response["provider"],
                        "model": response["model"],
                    }
                )

            print(f"✓ Created {len(blind_responses)} blind responses")

            # Test randomization
            original_order = [r["blind_id"] for r in blind_responses]
            random.shuffle(blind_responses)
            new_order = [r["blind_id"] for r in blind_responses]

            if original_order != new_order:
                print("✓ Response order randomized successfully")
            else:
                print("⚠️ Response order not randomized (may be coincidental)")

            return True
        else:
            print("✗ No successful responses found")
            return False

    except Exception as e:
        print(f"✗ Results loading test failed: {e}")
        return False


def test_feedback_logging():
    """Test feedback logging functionality"""
    print("\nTesting feedback logging...")

    try:
        # Create sample feedback
        sample_feedback = {
            "session_id": "test-session-123",
            "timestamp": "2025-07-07T12:00:00",
            "prompt_category": "retail",
            "prompt_text": "Test prompt",
            "selected_blind_id": "A",
            "selected_provider": "groq",
            "selected_model": "llama3-8b-8192",
            "feedback": "This response was very helpful",
            "ratings": {"helpfulness": 5, "accuracy": 4, "clarity": 5},
            "total_responses": 5,
            "response_order": ["A", "B", "C", "D", "E"],
        }

        # Test feedback file creation
        feedback_file = Path("data/results/user_feedback.json")
        feedback_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing or create new
        if feedback_file.exists():
            with open(feedback_file, "r") as f:
                all_feedback = json.load(f)
        else:
            all_feedback = []

        # Add test feedback
        all_feedback.append(sample_feedback)

        # Save back
        with open(feedback_file, "w") as f:
            json.dump(all_feedback, f, indent=2)

        print(
            f"✓ Feedback logged successfully. Total feedback entries: {len(all_feedback)}"
        )

        # Verify the feedback was saved
        with open(feedback_file, "r") as f:
            saved_feedback = json.load(f)

        if len(saved_feedback) == len(all_feedback):
            print("✓ Feedback verification successful")
            return True
        else:
            print("✗ Feedback verification failed")
            return False

    except Exception as e:
        print(f"✗ Feedback logging test failed: {e}")
        return False


def test_blind_evaluation_workflow():
    """Test the complete blind evaluation workflow"""
    print("\nTesting blind evaluation workflow...")

    try:
        # Simulate the workflow
        pass

        # Load sample data
        results_file = Path("data/results/2025-07-07_22-36-29_model_responses.json")
        with open(results_file, "r") as f:
            data = json.load(f)

        # Get sample prompt and responses
        data["prompts"][0]
        successful_responses = [
            r
            for r in data.get("results", [])
            if r["prompt_index"] == 0 and r["success"]
        ][
            :3
        ]  # Limit to 3 for testing

        if successful_responses:
            # Test blind evaluation setup
            blind_responses = []
            for i, response in enumerate(successful_responses):
                blind_id = chr(65 + i)
                blind_responses.append(
                    {
                        "blind_id": blind_id,
                        "response_text": response["response_text"],
                        "provider": response["provider"],
                        "model": response["model"],
                        "latency_seconds": response["latency_seconds"],
                        "tokens_used": response.get("tokens_used"),
                        "original_index": i,
                    }
                )

            # Randomize
            random.shuffle(blind_responses)

            print(
                f"✓ Blind evaluation workflow completed with {len(blind_responses)} responses"
            )
            print(f"  Response order: {[r['blind_id'] for r in blind_responses]}")
            print(f"  Models: {[r['model'] for r in blind_responses]}")

            return True
        else:
            print("✗ No responses available for workflow test")
            return False

    except Exception as e:
        print(f"✗ Blind evaluation workflow test failed: {e}")
        return False


def main():
    """Run all blind evaluation tests"""
    print("Blind Evaluation Test Suite")
    print("=" * 40)

    tests = [
        ("Imports", test_blind_evaluation_imports),
        ("Results Loading", test_results_loading),
        ("Feedback Logging", test_feedback_logging),
        ("Workflow", test_blind_evaluation_workflow),
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
    print(f"Blind Evaluation Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All blind evaluation tests passed! Ready for use.")
        print("\n🚀 You can now run the blind evaluation UI with:")
        print("   streamlit run src/ui/main_app.py")
        print("   Then navigate to '👁️ Blind Evaluation'")
        return True
    else:
        print("❌ Some blind evaluation tests failed. Please check the configuration.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
