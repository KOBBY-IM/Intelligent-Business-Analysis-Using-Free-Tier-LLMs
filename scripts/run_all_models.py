#!/usr/bin/env python3
"""
Universal Batch Evaluator for LLM Comparison
Runs prompts through all configured models and captures structured responses with metrics.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.config_loader import ConfigLoader
from llm_providers.base_provider import LLMResponse
from llm_providers.provider_manager import ProviderManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BatchEvaluator:
    """Universal batch evaluator for all LLM models"""

    def __init__(self, config_dir: str = "config"):
        self.config_loader = ConfigLoader(config_dir)
        self.provider_manager = ProviderManager()
        self.results_dir = Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_all_models(self) -> Dict[str, List[str]]:
        """Load all configured models from config"""
        try:
            llm_config = self.config_loader.load_llm_config()
            all_models = {}

            for provider_name, provider_config in llm_config.get(
                "providers", {}
            ).items():
                if provider_config.get("enabled", False):
                    models = [
                        model["name"] for model in provider_config.get("models", [])
                    ]
                    all_models[provider_name] = models
                    logger.info(
                        f"Loaded {len(models)} models for {provider_name}: {models}"
                    )

            return all_models

        except Exception as e:
            logger.error(f"Failed to load LLM config: {e}")
            # Fallback to hardcoded models
            return {
                "groq": ["llama3-8b-8192", "llama-3.1-8b-instant", "qwen-qwq-32b"],
                "gemini": ["gemini-1.5-flash", "gemma-3-12b-it"],
                "openrouter": [
                    "mistralai/mistral-7b-instruct",
                    "deepseek/deepseek-r1-0528-qwen3-8b:free",
                    "openrouter/cypher-alpha:free",
                ],
            }

    def get_evaluation_prompts(self) -> List[Dict[str, str]]:
        """Get evaluation prompts"""
        return self.config_loader.get_evaluation_prompts()

    def evaluate_single_model(
        self, provider_name: str, model_name: str, prompt_data: Dict[str, str]
    ) -> Dict[str, Any]:
        """Evaluate a single model with a prompt"""
        start_time = time.time()

        try:
            # Get provider and generate response
            provider = self.provider_manager.get_provider(provider_name)
            if not provider:
                return {
                    "success": False,
                    "error": f"Provider {provider_name} not found",
                    "latency": 0,
                    "tokens_used": None,
                }

            # Generate response
            response: LLMResponse = provider.generate_response(
                query=prompt_data["prompt"],
                context=prompt_data.get("context", ""),
                model=model_name,
                max_tokens=1000,
                temperature=0.7,
            )

            latency = time.time() - start_time

            return {
                "success": response.success,
                "response_text": response.text if response.success else "",
                "error": response.error,
                "latency": latency,
                "tokens_used": response.tokens_used,
                "raw_response": response.raw_response,
            }

        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Error evaluating {provider_name}/{model_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "latency": latency,
                "tokens_used": None,
            }

    def run_batch_evaluation(self, output_filename: Optional[str] = None) -> str:
        """Run batch evaluation across all models and prompts"""

        # Load configuration
        all_models = self.load_all_models()
        prompts = self.get_evaluation_prompts()

        logger.info(
            f"Starting batch evaluation with {len(all_models)} providers and {len(prompts)} prompts"
        )

        # Initialize results structure
        evaluation_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_providers": len(all_models),
                "total_prompts": len(prompts),
                "total_evaluations": sum(len(models) for models in all_models.values())
                * len(prompts),
            },
            "providers": all_models,
            "prompts": prompts,
            "results": [],
        }

        # Run evaluations
        total_evaluations = 0
        successful_evaluations = 0

        for prompt_idx, prompt_data in enumerate(prompts):
            logger.info(
                f"Processing prompt {prompt_idx + 1}/{len(prompts)}: {prompt_data['category']}"
            )

            for provider_name, models in all_models.items():
                logger.info(f"  Evaluating {provider_name} provider")

                for model_name in models:
                    logger.info(f"    Testing {model_name}")

                    # Evaluate model
                    result = self.evaluate_single_model(
                        provider_name, model_name, prompt_data
                    )

                    # Record result
                    evaluation_result = {
                        "prompt_index": prompt_idx,
                        "prompt_category": prompt_data["category"],
                        "prompt_text": prompt_data["prompt"],
                        "context": prompt_data.get("context", ""),
                        "provider": provider_name,
                        "model": model_name,
                        "success": result["success"],
                        "response_text": result.get("response_text", ""),
                        "error": result.get("error"),
                        "latency_seconds": result["latency"],
                        "tokens_used": result.get("tokens_used"),
                        "timestamp": datetime.now().isoformat(),
                    }

                    evaluation_results["results"].append(evaluation_result)

                    total_evaluations += 1
                    if result["success"]:
                        successful_evaluations += 1

                    # Log progress
                    logger.info(
                        f"      {'✓' if result['success'] else '✗'} {result['latency']:.2f}s"
                    )

                    # Small delay to avoid rate limiting
                    time.sleep(0.5)

        # Add summary statistics
        evaluation_results["summary"] = {
            "total_evaluations": total_evaluations,
            "successful_evaluations": successful_evaluations,
            "success_rate": (
                successful_evaluations / total_evaluations
                if total_evaluations > 0
                else 0
            ),
            "average_latency": self._calculate_average_latency(
                evaluation_results["results"]
            ),
            "total_tokens": self._calculate_total_tokens(evaluation_results["results"]),
        }

        # Save results
        if not output_filename:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = f"{timestamp}_model_responses.json"

        output_path = self.results_dir / output_filename
        with open(output_path, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        logger.info(f"Batch evaluation completed!")
        logger.info(f"  Total evaluations: {total_evaluations}")
        logger.info(f"  Successful: {successful_evaluations}")
        logger.info(
            f"  Success rate: {evaluation_results['summary']['success_rate']:.2%}"
        )
        logger.info(f"  Results saved to: {output_path}")

        return str(output_path)

    def _calculate_average_latency(self, results: List[Dict[str, Any]]) -> float:
        """Calculate average latency from successful evaluations"""
        successful_results = [r for r in results if r["success"]]
        if not successful_results:
            return 0.0
        return sum(r["latency_seconds"] for r in successful_results) / len(
            successful_results
        )

    def _calculate_total_tokens(self, results: List[Dict[str, Any]]) -> int:
        """Calculate total tokens used"""
        total = 0
        for result in results:
            if result.get("tokens_used") is not None:
                total += result["tokens_used"]
        return total

    def print_summary_report(self, results_file: str):
        """Print a summary report of the evaluation results"""
        with open(results_file, "r") as f:
            data = json.load(f)

        print("\n" + "=" * 80)
        print("BATCH EVALUATION SUMMARY REPORT")
        print("=" * 80)

        # Metadata
        print(f"Timestamp: {data['metadata']['timestamp']}")
        print(f"Total Providers: {data['metadata']['total_providers']}")
        print(f"Total Prompts: {data['metadata']['total_prompts']}")
        print(f"Total Evaluations: {data['metadata']['total_evaluations']}")

        # Summary stats
        summary = data["summary"]
        print(f"\nOverall Statistics:")
        print(f"  Success Rate: {summary['success_rate']:.2%}")
        print(f"  Average Latency: {summary['average_latency']:.2f} seconds")
        print(f"  Total Tokens Used: {summary['total_tokens']:,}")

        # Provider breakdown
        print(f"\nProvider Breakdown:")
        provider_stats = {}
        for result in data["results"]:
            provider = result["provider"]
            if provider not in provider_stats:
                provider_stats[provider] = {
                    "total": 0,
                    "successful": 0,
                    "latencies": [],
                }

            provider_stats[provider]["total"] += 1
            if result["success"]:
                provider_stats[provider]["successful"] += 1
                provider_stats[provider]["latencies"].append(result["latency_seconds"])

        for provider, stats in provider_stats.items():
            success_rate = (
                stats["successful"] / stats["total"] if stats["total"] > 0 else 0
            )
            avg_latency = (
                sum(stats["latencies"]) / len(stats["latencies"])
                if stats["latencies"]
                else 0
            )
            print(
                f"  {provider}: {stats['successful']}/{stats['total']} ({success_rate:.2%}) - Avg: {avg_latency:.2f}s"
            )

        print("=" * 80)


def main():
    """Main entry point"""
    print("Universal Batch Evaluator for LLM Comparison")
    print("=" * 50)

    # Check environment
    required_keys = ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print(f"Warning: Missing API keys: {', '.join(missing_keys)}")
        print("Some providers may not work correctly.")
        print()

    # Initialize evaluator
    evaluator = BatchEvaluator()

    try:
        # Run batch evaluation
        results_file = evaluator.run_batch_evaluation()

        # Print summary report
        evaluator.print_summary_report(results_file)

        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {results_file}")

    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
