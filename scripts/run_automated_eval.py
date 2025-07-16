#!/usr/bin/env python3
"""
Simple launcher for Automated LLM Evaluator

Usage: python scripts/run_automated_eval.py [options]
"""

import sys
import argparse
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from automated_llm_evaluator import AutomatedLLMEvaluator

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run automated LLM evaluation")
    
    parser.add_argument(
        "--duration", 
        type=int, 
        default=60,
        help="Test duration in minutes (default: 60)"
    )
    
    parser.add_argument(
        "--interval", 
        type=int, 
        default=120,
        help="Interval between tests in seconds (default: 120)"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/automated_eval_config.yaml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--no-confirm", 
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show configuration without running tests"
    )
    
    return parser.parse_args()

async def main():
    """Main function"""
    args = parse_arguments()
    
    print("🤖 Automated LLM Evaluator Launcher")
    print("="*50)
    
    # Validate configuration file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("Using default configuration...")
        config_path = None
    
    # Initialize evaluator
    try:
        evaluator = AutomatedLLMEvaluator(str(config_path) if config_path else None)
        
        # Override settings from command line
        if args.duration != 60:
            evaluator.test_duration_minutes = args.duration
        if args.interval != 120:
            evaluator.test_interval_seconds = args.interval
            evaluator.total_tests = evaluator.test_duration_minutes * 60 // evaluator.test_interval_seconds
        
    except Exception as e:
        print(f"❌ Failed to initialize evaluator: {e}")
        return
    
    # Show configuration
    print(f"\n📋 Configuration:")
    print(f"   Duration: {evaluator.test_duration_minutes} minutes")
    print(f"   Interval: {evaluator.test_interval_seconds} seconds")
    print(f"   Total Tests: {evaluator.total_tests}")
    print(f"   Datasets: {list(evaluator.datasets.keys())}")
    print(f"   Questions: {len(evaluator.questions)}")
    
    # Check for available providers
    providers = evaluator.provider_manager.get_available_providers()
    print(f"   Available Providers: {providers}")
    
    if not providers:
        print("❌ No LLM providers available!")
        print("Please check your API keys and configuration.")
        return
    
    if args.dry_run:
        print("\n✅ Dry run completed. Configuration looks good!")
        return
    
    # Confirm before starting
    if not args.no_confirm:
        print(f"\n⚠️  This will run {evaluator.total_tests} tests over {evaluator.test_duration_minutes} minutes")
        print("   Make sure your API keys are configured and you have sufficient quota.")
        
        response = input("\n❓ Start automated evaluation? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Evaluation cancelled")
            return
    
    # Run evaluation
    print("\n🚀 Starting automated evaluation...")
    await evaluator.run_automated_evaluation()

if __name__ == "__main__":
    asyncio.run(main()) 