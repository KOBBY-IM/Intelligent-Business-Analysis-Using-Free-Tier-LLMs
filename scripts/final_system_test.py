#!/usr/bin/env python3
"""
Final System-Wide Test: Process all prompts across all 9 models
- Time total execution and memory footprint
- Record final model performance stats
- Fix any inconsistencies and finalize logs
"""

import argparse
import json
import logging
import os
import psutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_providers.provider_manager import ProviderManager
from llm_providers.base_provider import RateLimitException
from utils.common import load_json_file, load_yaml_file
from utils.structured_logger import StructuredLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/final_system_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinalSystemTest:
    """Comprehensive system-wide test for all models and prompts"""
    
    def __init__(self):
        self.provider_manager = ProviderManager()
        self.results_dir = Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self.llm_config = load_yaml_file("config/llm_config.yaml")
        self.evaluation_config = load_yaml_file("config/evaluation_config.yaml")
        
        # Performance tracking
        self.start_time = None
        self.start_memory = None
        self.test_results = []
        
    def get_all_prompts(self) -> List[Dict[str, str]]:
        """Get all evaluation prompts from config"""
        prompts = [
            {
                "category": "retail",
                "prompt": "What are the top performing product categories in our retail business?",
                "context": "Analyze sales data to identify the most profitable product categories."
            },
            {
                "category": "retail", 
                "prompt": "How can we improve customer retention in our retail stores?",
                "context": "Provide actionable strategies for increasing customer loyalty and repeat purchases."
            },
            {
                "category": "finance",
                "prompt": "What are the key risk factors for our investment portfolio?",
                "context": "Analyze market conditions and portfolio composition to identify potential risks."
            },
            {
                "category": "finance",
                "prompt": "How should we optimize our cash flow management?",
                "context": "Provide recommendations for improving cash flow efficiency and working capital."
            },
            {
                "category": "healthcare",
                "prompt": "What are the most effective patient engagement strategies?",
                "context": "Recommend approaches to improve patient communication and satisfaction."
            },
            {
                "category": "healthcare",
                "prompt": "How can we reduce healthcare costs while maintaining quality?",
                "context": "Analyze cost drivers and suggest optimization strategies."
            }
        ]
        return prompts
    
    def get_all_models(self) -> Dict[str, List[str]]:
        """Get all enabled models from config"""
        all_models = {}
        for provider_name, provider_config in self.llm_config.get("providers", {}).items():
            if provider_config.get("enabled", False):
                models = [model["name"] for model in provider_config.get("models", [])]
                all_models[provider_name] = models
        return all_models
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure current memory usage"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            "percent": process.memory_percent()
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test across all models and prompts"""
        logger.info("🚀 Starting Final System-Wide Test")
        
        # Initialize performance tracking
        self.start_time = time.time()
        self.start_memory = self.measure_memory_usage()
        
        # Get test scope
        all_models = self.get_all_models()
        all_prompts = self.get_all_prompts()
        
        total_models = sum(len(models) for models in all_models.values())
        total_evaluations = total_models * len(all_prompts)
        
        logger.info(f"📊 Test Scope: {total_models} models × {len(all_prompts)} prompts = {total_evaluations} evaluations")
        
        # Initialize results
        test_summary = {
            "metadata": {
                "test_start": datetime.now().isoformat(),
                "total_models": total_models,
                "total_prompts": len(all_prompts),
                "total_evaluations": total_evaluations,
                "providers": list(all_models.keys())
            },
            "performance": {
                "start_memory_mb": self.start_memory["rss_mb"],
                "start_time": self.start_time
            },
            "results": []
        }
        
        # Track progress
        completed_evaluations = 0
        successful_evaluations = 0
        failed_evaluations = 0
        rate_limited_evaluations = 0
        
        # Test each provider
        for provider_name, models in all_models.items():
            logger.info(f"\n{'='*20} Testing Provider: {provider_name.upper()} {'='*20}")
            
            provider = self.provider_manager.get_provider(provider_name)
            if not provider:
                logger.error(f"Provider {provider_name} not found")
                continue
            
            # Health check
            try:
                healthy = provider.health_check()
                logger.info(f"Provider health: {'✅' if healthy else '❌'}")
                if not healthy:
                    logger.warning(f"Skipping unhealthy provider: {provider_name}")
                    continue
            except Exception as e:
                logger.error(f"Health check failed for {provider_name}: {e}")
                continue
            
            # Test each model
            for model_idx, model_name in enumerate(models, 1):
                logger.info(f"\n🤖 Testing Model {model_idx}/{len(models)}: {model_name}")
                
                # Test each prompt
                for prompt_idx, prompt_data in enumerate(all_prompts, 1):
                    logger.info(f"  📝 Prompt {prompt_idx}/{len(all_prompts)}: {prompt_data['category']}")
                    
                    evaluation_start = time.time()
                    
                    try:
                        # Generate response
                        response = provider.generate_response(
                            query=prompt_data["prompt"],
                            model=model_name,
                            max_tokens=300,
                            temperature=0.7
                        )
                        
                        evaluation_time = time.time() - evaluation_start
                        
                        # Record result
                        result = {
                            "provider": provider_name,
                            "model": model_name,
                            "prompt_category": prompt_data["category"],
                            "prompt_text": prompt_data["prompt"],
                            "context": prompt_data["context"],
                            "success": response.success,
                            "response_text": response.text if response.success else "",
                            "error": response.error if not response.success else None,
                            "latency_seconds": response.latency_ms / 1000 if response.success else evaluation_time,
                            "tokens_used": response.tokens_used if response.success else 0,
                            "evaluation_time": evaluation_time,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        test_summary["results"].append(result)
                        
                        if response.success:
                            successful_evaluations += 1
                            logger.info(f"    ✅ Success: {response.latency_ms:.2f}ms, {response.tokens_used} tokens")
                        else:
                            failed_evaluations += 1
                            logger.info(f"    ❌ Failed: {response.error}")
                        
                    except RateLimitException as e:
                        rate_limited_evaluations += 1
                        logger.warning(f"    ⚠️ Rate limited: {e}")
                        # Record failed result
                        result = {
                            "provider": provider_name,
                            "model": model_name,
                            "prompt_category": prompt_data["category"],
                            "prompt_text": prompt_data["prompt"],
                            "context": prompt_data["context"],
                            "success": False,
                            "response_text": "",
                            "error": f"Rate limit: {e}",
                            "latency_seconds": time.time() - evaluation_start,
                            "tokens_used": 0,
                            "evaluation_time": time.time() - evaluation_start,
                            "timestamp": datetime.now().isoformat()
                        }
                        test_summary["results"].append(result)
                        
                        # Wait before continuing
                        time.sleep(5)
                        
                    except Exception as e:
                        failed_evaluations += 1
                        logger.error(f"    ❌ Exception: {e}")
                        # Record failed result
                        result = {
                            "provider": provider_name,
                            "model": model_name,
                            "prompt_category": prompt_data["category"],
                            "prompt_text": prompt_data["prompt"],
                            "context": prompt_data["context"],
                            "success": False,
                            "response_text": "",
                            "error": f"Exception: {e}",
                            "latency_seconds": time.time() - evaluation_start,
                            "tokens_used": 0,
                            "evaluation_time": time.time() - evaluation_start,
                            "timestamp": datetime.now().isoformat()
                        }
                        test_summary["results"].append(result)
                    
                    completed_evaluations += 1
                    
                    # Progress update
                    if completed_evaluations % 10 == 0:
                        progress = (completed_evaluations / total_evaluations) * 100
                        logger.info(f"📈 Progress: {completed_evaluations}/{total_evaluations} ({progress:.1f}%)")
                    
                    # Small delay to avoid rate limits
                    time.sleep(1)
        
        # Finalize performance metrics
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        total_duration = end_time - self.start_time
        
        test_summary["performance"].update({
            "end_memory_mb": end_memory["rss_mb"],
            "end_time": end_time,
            "total_duration_seconds": total_duration,
            "memory_increase_mb": end_memory["rss_mb"] - self.start_memory["rss_mb"],
            "evaluations_per_second": completed_evaluations / total_duration if total_duration > 0 else 0
        })
        
        test_summary["summary"] = {
            "completed_evaluations": completed_evaluations,
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": failed_evaluations,
            "rate_limited_evaluations": rate_limited_evaluations,
            "success_rate": (successful_evaluations / completed_evaluations * 100) if completed_evaluations > 0 else 0
        }
        
        return test_summary
    
    def generate_performance_report(self, test_summary: Dict[str, Any]) -> str:
        """Generate a comprehensive performance report"""
        report_lines = []
        
        # Header
        report_lines.append("# Final System-Wide Test Report")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append("")
        
        metadata = test_summary["metadata"]
        summary = test_summary["summary"]
        performance = test_summary["performance"]
        
        report_lines.append(f"- **Total Models:** {metadata['total_models']}")
        report_lines.append(f"- **Total Prompts:** {metadata['total_prompts']}")
        report_lines.append(f"- **Total Evaluations:** {metadata['total_evaluations']}")
        report_lines.append(f"- **Completed Evaluations:** {summary['completed_evaluations']}")
        report_lines.append(f"- **Success Rate:** {summary['success_rate']:.1f}%")
        report_lines.append(f"- **Total Duration:** {performance['total_duration_seconds']:.2f} seconds")
        report_lines.append(f"- **Evaluations/Second:** {performance['evaluations_per_second']:.2f}")
        report_lines.append("")
        
        # Performance Metrics
        report_lines.append("## Performance Metrics")
        report_lines.append("")
        
        report_lines.append("### Memory Usage")
        report_lines.append(f"- **Start Memory:** {performance['start_memory_mb']:.2f} MB")
        report_lines.append(f"- **End Memory:** {performance['end_memory_mb']:.2f} MB")
        report_lines.append(f"- **Memory Increase:** {performance['memory_increase_mb']:.2f} MB")
        report_lines.append("")
        
        # Provider Performance
        report_lines.append("## Provider Performance")
        report_lines.append("")
        
        provider_stats = {}
        for result in test_summary["results"]:
            provider = result["provider"]
            if provider not in provider_stats:
                provider_stats[provider] = {
                    "total": 0, "successful": 0, "failed": 0, "rate_limited": 0,
                    "total_latency": 0, "total_tokens": 0
                }
            
            provider_stats[provider]["total"] += 1
            if result["success"]:
                provider_stats[provider]["successful"] += 1
                provider_stats[provider]["total_latency"] += result["latency_seconds"]
                provider_stats[provider]["total_tokens"] += result["tokens_used"]
            elif "Rate limit" in str(result.get("error", "")):
                provider_stats[provider]["rate_limited"] += 1
            else:
                provider_stats[provider]["failed"] += 1
        
        # Provider table
        report_lines.append("| Provider | Total | Successful | Failed | Rate Limited | Success Rate | Avg Latency (s) | Avg Tokens |")
        report_lines.append("|----------|-------|------------|--------|--------------|--------------|-----------------|------------|")
        
        for provider, stats in provider_stats.items():
            success_rate = (stats["successful"] / stats["total"] * 100) if stats["total"] > 0 else 0
            avg_latency = (stats["total_latency"] / stats["successful"]) if stats["successful"] > 0 else 0
            avg_tokens = (stats["total_tokens"] / stats["successful"]) if stats["successful"] > 0 else 0
            
            report_lines.append(
                f"| {provider} | {stats['total']} | {stats['successful']} | {stats['failed']} | "
                f"{stats['rate_limited']} | {success_rate:.1f}% | {avg_latency:.3f} | {avg_tokens:.1f} |"
            )
        
        report_lines.append("")
        
        # Model Performance
        report_lines.append("## Model Performance")
        report_lines.append("")
        
        model_stats = {}
        for result in test_summary["results"]:
            model_key = f"{result['provider']}/{result['model']}"
            if model_key not in model_stats:
                model_stats[model_key] = {
                    "total": 0, "successful": 0, "failed": 0,
                    "total_latency": 0, "total_tokens": 0
                }
            
            model_stats[model_key]["total"] += 1
            if result["success"]:
                model_stats[model_key]["successful"] += 1
                model_stats[model_key]["total_latency"] += result["latency_seconds"]
                model_stats[model_key]["total_tokens"] += result["tokens_used"]
            else:
                model_stats[model_key]["failed"] += 1
        
        # Model table
        report_lines.append("| Model | Total | Successful | Failed | Success Rate | Avg Latency (s) | Avg Tokens |")
        report_lines.append("|-------|-------|------------|--------|--------------|-----------------|------------|")
        
        for model, stats in model_stats.items():
            success_rate = (stats["successful"] / stats["total"] * 100) if stats["total"] > 0 else 0
            avg_latency = (stats["total_latency"] / stats["successful"]) if stats["successful"] > 0 else 0
            avg_tokens = (stats["total_tokens"] / stats["successful"]) if stats["successful"] > 0 else 0
            
            report_lines.append(
                f"| {model} | {stats['total']} | {stats['successful']} | {stats['failed']} | "
                f"{success_rate:.1f}% | {avg_latency:.3f} | {avg_tokens:.1f} |"
            )
        
        report_lines.append("")
        
        # Error Analysis
        report_lines.append("## Error Analysis")
        report_lines.append("")
        
        error_counts = {}
        for result in test_summary["results"]:
            if not result["success"]:
                error = result.get("error", "Unknown error")
                error_counts[error] = error_counts.get(error, 0) + 1
        
        if error_counts:
            report_lines.append("| Error Type | Count |")
            report_lines.append("|------------|-------|")
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"| {error} | {count} |")
        else:
            report_lines.append("No errors encountered.")
        
        report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_results(self, test_summary: Dict[str, Any], report: str):
        """Save test results and report"""
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"final_system_test_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump(test_summary, f, indent=2)
        
        # Save report
        report_file = Path("docs") / f"FinalSystemTestReport_{timestamp}.md"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, "w") as f:
            f.write(report)
        
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info(f"📄 Report saved to: {report_file}")
        
        return results_file, report_file
    
    def run(self):
        """Run the complete final system test"""
        try:
            logger.info("🚀 Starting Final System-Wide Test")
            
            # Run comprehensive test
            test_summary = self.run_comprehensive_test()
            
            # Generate report
            report = self.generate_performance_report(test_summary)
            
            # Save results
            results_file, report_file = self.save_results(test_summary, report)
            
            # Print summary
            summary = test_summary["summary"]
            performance = test_summary["performance"]
            
            print("\n" + "="*60)
            print("🎉 FINAL SYSTEM TEST COMPLETED")
            print("="*60)
            print(f"✅ Completed: {summary['completed_evaluations']} evaluations")
            print(f"✅ Success Rate: {summary['success_rate']:.1f}%")
            print(f"⏱️ Total Duration: {performance['total_duration_seconds']:.2f} seconds")
            print(f"📊 Evaluations/Second: {performance['evaluations_per_second']:.2f}")
            print(f"💾 Memory Usage: {performance['end_memory_mb']:.2f} MB")
            print(f"📁 Results: {results_file}")
            print(f"📄 Report: {report_file}")
            print("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Final system test failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Run final system-wide test")
    parser.add_argument("--quick", action="store_true", help="Run quick test with limited scope")
    
    args = parser.parse_args()
    
    tester = FinalSystemTest()
    
    if args.quick:
        logger.info("Running quick test mode...")
        # Modify prompts for quick test
        tester.get_all_prompts = lambda: [
            {
                "category": "test",
                "prompt": "What is 2+2?",
                "context": "Simple arithmetic test"
            }
        ]
    
    success = tester.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 