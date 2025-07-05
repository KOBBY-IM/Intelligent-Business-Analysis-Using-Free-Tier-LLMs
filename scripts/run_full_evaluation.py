#!/usr/bin/env python3
"""
Comprehensive LLM evaluation script for full-scale assessment
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.evaluator import LLMEvaluator
from evaluation.ground_truth import GroundTruthManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_full_evaluation(domains=None, difficulties=None, providers=None):
    """Run full evaluation across all providers and questions"""
    
    print("🚀 COMPREHENSIVE LLM EVALUATION")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize evaluator
    evaluator = LLMEvaluator()
    
    # Get available questions and providers
    ground_truth_manager = GroundTruthManager()
    available_questions = list(ground_truth_manager.answers.keys())
    available_providers = evaluator.provider_manager.get_provider_names()
    
    print("📊 EVALUATION SCOPE")
    print("-" * 30)
    print(f"Total Questions Available: {len(available_questions)}")
    print(f"Total Providers Available: {len(available_providers)}")
    
    if domains:
        print(f"Domain Filter: {domains}")
    if difficulties:
        print(f"Difficulty Filter: {difficulties}")
    if providers:
        print(f"Provider Filter: {providers}")
    
    print()
    
    # Run evaluation
    try:
        batch_result = evaluator.run_batch_evaluation(
            domains=domains,
            difficulties=difficulties,
            provider_names=providers
        )
        
        # Print results summary
        print("📈 EVALUATION RESULTS")
        print("=" * 30)
        print(f"Evaluation ID: {batch_result.evaluation_id}")
        print(f"Total Results: {len(batch_result.results)}")
        print(f"Questions Evaluated: {batch_result.total_questions}")
        print(f"Providers Tested: {batch_result.total_providers}")
        print()
        
        # Provider performance summary
        if batch_result.summary_stats.get('provider_stats'):
            print("🏆 PROVIDER PERFORMANCE")
            print("-" * 25)
            
            # Sort providers by average score
            provider_stats = batch_result.summary_stats['provider_stats']
            sorted_providers = sorted(
                provider_stats.items(),
                key=lambda x: x[1]['avg_overall_score'],
                reverse=True
            )
            
            for i, (provider, stats) in enumerate(sorted_providers, 1):
                print(f"{i}. {provider}")
                print(f"   Average Score: {stats['avg_overall_score']:.4f}")
                print(f"   Evaluations: {stats['total_evaluations']}")
                print(f"   Response Time: {stats['avg_response_time_ms']:.2f}ms")
                print()
        
        # Statistical significance
        if batch_result.statistical_analysis.get('anova_result'):
            anova = batch_result.statistical_analysis['anova_result']
            print("📊 STATISTICAL SIGNIFICANCE")
            print("-" * 30)
            print(f"ANOVA Result: {anova.get('description', 'N/A')}")
            print(f"P-value: {anova.get('p_value', 0):.6f}")
            print(f"Effect Size: {anova.get('effect_size', 0):.4f}")
            print()
        
        # Detailed results
        print("📋 DETAILED RESULTS")
        print("-" * 20)
        for result in batch_result.results:
            print(f"Question: {result.question_id}")
            print(f"Provider: {result.provider_name}/{result.model_name}")
            print(f"Overall Score: {result.metrics.overall_score:.4f}")
            print(f"  - Relevance: {result.metrics.relevance_score:.4f}")
            print(f"  - Factual Accuracy: {result.metrics.factual_accuracy:.4f}")
            print(f"  - Coherence: {result.metrics.coherence_score:.4f}")
            print(f"  - Completeness: {result.metrics.completeness_score:.4f}")
            print(f"  - Token Efficiency: {result.metrics.token_efficiency:.4f}")
            print(f"Response Time: {result.metrics.response_time_ms:.2f}ms")
            print()
        
        print("✅ Evaluation completed successfully!")
        print(f"Results saved to: data/evaluation_results/{batch_result.evaluation_id}_results.json")
        print(f"Summary saved to: data/evaluation_results/{batch_result.evaluation_id}_summary.txt")
        
        return batch_result
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"❌ Evaluation failed: {e}")
        return None

def run_quick_evaluation():
    """Run a quick evaluation with limited scope for testing"""
    
    print("⚡ QUICK EVALUATION (Testing Mode)")
    print("=" * 50)
    print("Running evaluation with limited scope for testing...")
    print()
    
    # Test with just one question and one provider
    evaluator = LLMEvaluator()
    
    try:
        batch_result = evaluator.run_batch_evaluation(
            question_ids=["retail_001"],  # Just one question
            provider_names=["groq"],      # Just one provider
            model_names=["llama3-8b-8192"]  # Just one model
        )
        
        print("✅ Quick evaluation completed!")
        print(f"Results: {len(batch_result.results)} evaluation(s)")
        
        if batch_result.results:
            result = batch_result.results[0]
            print(f"Score: {result.metrics.overall_score:.4f}")
            print(f"Response Time: {result.metrics.response_time_ms:.2f}ms")
        
        return batch_result
        
    except Exception as e:
        logger.error(f"Quick evaluation failed: {e}")
        print(f"❌ Quick evaluation failed: {e}")
        return None

def main():
    """Main function with command line argument parsing"""
    
    parser = argparse.ArgumentParser(description="Comprehensive LLM Evaluation System")
    parser.add_argument("--mode", choices=["full", "quick"], default="quick",
                       help="Evaluation mode: full (all providers/questions) or quick (limited)")
    parser.add_argument("--domains", nargs="+", choices=["retail", "finance", "healthcare"],
                       help="Filter by domains")
    parser.add_argument("--difficulties", nargs="+", choices=["easy", "medium", "hard"],
                       help="Filter by difficulty levels")
    parser.add_argument("--providers", nargs="+", 
                       help="Filter by specific providers")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    print("🎯 LLM EVALUATION SYSTEM")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if args.mode == "quick":
        result = run_quick_evaluation()
    else:
        result = run_full_evaluation(
            domains=args.domains,
            difficulties=args.difficulties,
            providers=args.providers
        )
    
    if result:
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📁 Results saved to: data/evaluation_results/")
        print(f"📊 Evaluation ID: {result.evaluation_id}")
    else:
        print(f"\n❌ Evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 