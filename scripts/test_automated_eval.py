#!/usr/bin/env python3
"""
Quick Test for Automated LLM Evaluator

Runs a short test (2 minutes) to verify everything works.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from automated_llm_evaluator import AutomatedLLMEvaluator

class QuickTestEvaluator(AutomatedLLMEvaluator):
    """Quick test version with shorter duration"""
    
    def __init__(self):
        super().__init__()
        # Override for quick test
        self.test_duration_minutes = 2  # 2 minutes
        self.test_interval_seconds = 30  # 30 seconds
        self.total_tests = self.test_duration_minutes * 60 // self.test_interval_seconds  # 4 tests
        
        print(f"🔧 Quick test mode: {self.total_tests} tests over {self.test_duration_minutes} minutes")

async def main():
    """Run quick test"""
    print("🧪 Quick Test - Automated LLM Evaluator")
    print("="*50)
    print("This will run a 2-minute test to verify the system works")
    print("="*50)
    
    try:
        evaluator = QuickTestEvaluator()
        
        print(f"\n📋 Test Configuration:")
        print(f"   Duration: {evaluator.test_duration_minutes} minutes")
        print(f"   Interval: {evaluator.test_interval_seconds} seconds")
        print(f"   Total Tests: {evaluator.total_tests}")
        print(f"   Datasets: {list(evaluator.datasets.keys())}")
        print(f"   Questions: {len(evaluator.questions)}")
        
        # Check providers
        providers = evaluator.provider_manager.get_provider_names()
        print(f"   Available Providers: {providers}")
        
        if not providers:
            print("❌ No providers available - check API keys")
            return
        
        # Confirm
        response = input("\n❓ Run quick test? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Test cancelled")
            return
        
        # Run test
        await evaluator.run_automated_evaluation()
        
        print("\n✅ Quick test completed!")
        print("If successful, you can run the full evaluation with:")
        print("   python scripts/run_automated_eval.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 