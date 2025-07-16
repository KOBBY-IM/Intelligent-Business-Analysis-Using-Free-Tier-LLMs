#!/usr/bin/env python3
"""
Automated LLM Evaluator

Runs comprehensive LLM performance tests every 2 minutes for 1 hour.
Uses RAG pipeline with provided datasets to test real-world performance.
"""

import asyncio
import json
import time
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
from typing import List, Dict, Any, Optional
import yaml
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from llm_providers.provider_manager import ProviderManager
from rag.pipeline import RAGPipeline
from evaluation.evaluator import LLMEvaluator
from config.config_loader import ConfigLoader
from security.secure_logger import SecureLogger
from utils.structured_logger import StructuredLogger

class AutomatedLLMEvaluator:
    """Automated system for evaluating LLM performance with RAG"""
    
    def __init__(self, config_path: str = None):
        """Initialize the automated evaluator"""
        # Initialize logger first
        self.logger = StructuredLogger("automated_evaluator")
        self.secure_logger = SecureLogger()
        
        # Load config
        try:
            if config_path:
                self.config = ConfigLoader().load_llm_config()
            else:
                self.config = {}
        except Exception as e:
            print(f"Warning: Could not load config: {e}, using defaults")
            self.config = {}
        
        # Initialize components
        self.provider_manager = ProviderManager()
        
        # Initialize RAG pipeline with default components (will be overridden per query)
        try:
            from rag.vector_store import FAISSVectorStore
            from llm_providers.groq_provider import GroqProvider
            vector_store = FAISSVectorStore()
            default_provider = GroqProvider()
            self.rag_pipeline = RAGPipeline(vector_store, default_provider)
        except Exception as e:
            print(f"Warning: Could not initialize RAG pipeline: {e}")
            self.rag_pipeline = None
        
        self.evaluator = LLMEvaluator()
        
        # Test configuration
        self.test_duration_minutes = 180  # 3 hours
        self.test_interval_seconds = 120  # 2 minutes
        self.total_tests = self.test_duration_minutes * 60 // self.test_interval_seconds  # 90 tests
        
        # Results storage
        self.results = []
        self.start_time = None
        self.end_time = None
        
        # Create results directory
        self.results_dir = Path("data/evaluation_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Health check log file
        self.health_log_file = self.results_dir / "llm_health_checks.json"
        
        # Load datasets and questions
        self.datasets = self.load_datasets()
        self.questions = self.load_questions()
        
        print(f"Automated LLM Evaluator initialized - Duration: {self.test_duration_minutes}min, Interval: {self.test_interval_seconds}s, Tests: {self.total_tests}")
    
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load available datasets for RAG testing"""
        datasets = {}
        data_dir = Path("data")
        
        # Look for CSV files
        csv_files = list(data_dir.glob("*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if len(df) > 10:  # Only use datasets with sufficient data
                    datasets[csv_file.stem] = df
                    print(f"Loaded dataset: {csv_file.stem} ({len(df)} rows)")
            except Exception as e:
                print(f"Failed to load dataset {csv_file}: {e}")
        
        # Look for JSON files with structured data
        json_files = list(data_dir.glob("*.json"))
        for json_file in json_files:
            if "feedback" not in json_file.name and "blind" not in json_file.name:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 5:
                        datasets[json_file.stem] = pd.DataFrame(data)
                        print(f"Loaded JSON dataset: {json_file.stem} ({len(data)} rows)")
                except Exception as e:
                    print(f"Failed to load JSON dataset {json_file}: {e}")
        
        if not datasets:
            print("Warning: No datasets found, will use sample data")
            # Create sample dataset
            sample_data = {
                'product_id': range(1, 101),
                'product_name': [f"Product_{i}" for i in range(1, 101)],
                'category': random.choices(['Electronics', 'Clothing', 'Home', 'Sports'], k=100),
                'price': np.random.uniform(10, 500, 100),
                'sales': np.random.randint(10, 1000, 100),
                'rating': np.random.uniform(1, 5, 100),
                'customer_satisfaction': np.random.uniform(0.6, 1.0, 100)
            }
            datasets['sample_retail'] = pd.DataFrame(sample_data)
        
        return datasets
    
    def load_questions(self) -> List[Dict[str, Any]]:
        """Load or generate test questions"""
        questions_file = Path("data/business_questions.yaml")
        
        if questions_file.exists():
            try:
                with open(questions_file, 'r') as f:
                    questions_data = yaml.safe_load(f)
                questions = questions_data.get('questions', [])
                print(f"Loaded {len(questions)} questions from {questions_file}")
            except Exception as e:
                self.logger.error(f"Failed to load questions from {questions_file}: {e}")
                questions = self.generate_default_questions()
        else:
            questions = self.generate_default_questions()
        
        return questions
    
    def generate_default_questions(self) -> List[Dict[str, Any]]:
        """Generate default business analysis questions"""
        questions = [
            {
                "question": "What are the top 5 products by sales volume?",
                "industry": "retail",
                "expected_metrics": ["sales", "product_name"],
                "difficulty": "easy"
            },
            {
                "question": "Which product category has the highest average price?",
                "industry": "retail", 
                "expected_metrics": ["category", "price"],
                "difficulty": "medium"
            },
            {
                "question": "What is the correlation between price and customer satisfaction?",
                "industry": "retail",
                "expected_metrics": ["price", "customer_satisfaction"],
                "difficulty": "hard"
            },
            {
                "question": "Which products have the highest ratings but low sales?",
                "industry": "retail",
                "expected_metrics": ["rating", "sales", "product_name"],
                "difficulty": "medium"
            },
            {
                "question": "What is the average price by product category?",
                "industry": "retail",
                "expected_metrics": ["category", "price"],
                "difficulty": "easy"
            },
            {
                "question": "Identify products with sales below average but high customer satisfaction",
                "industry": "retail",
                "expected_metrics": ["sales", "customer_satisfaction"],
                "difficulty": "hard"
            }
        ]
        
        print(f"Generated {len(questions)} default questions")
        return questions
    
    async def run_single_evaluation(self, test_number: int, total_tests: int) -> Dict[str, Any]:
        """Run a single evaluation with RAG"""
        
        # Select random dataset and question
        dataset_name = random.choice(list(self.datasets.keys()))
        dataset = self.datasets[dataset_name]
        question_data = random.choice(self.questions)
        question = question_data["question"]
        industry = question_data.get("industry", "retail")
        
        print(f"Running test {test_number}/{total_tests}", 
                        dataset=dataset_name, question=question[:50] + "...")
        
        # Prepare dataset for RAG
        dataset_path = f"data/{dataset_name}.csv"
        if not Path(dataset_path).exists():
            # Save dataset to CSV for RAG
            dataset.to_csv(dataset_path, index=False)
        
        # Get available providers
        providers = self.provider_manager.get_available_providers()
        if not providers:
            self.logger.error("No available LLM providers")
            return None
        
        evaluation_results = []
        
        for provider_name in providers:
            try:
                start_time = time.time()
                
                # Run RAG query
                rag_result = await self.rag_pipeline.query(
                    question=question,
                    dataset_path=dataset_path,
                    provider_name=provider_name,
                    max_results=5
                )
                
                end_time = time.time()
                latency = end_time - start_time
                
                # Extract response
                response = rag_result.get('answer', 'No response generated')
                context_used = rag_result.get('context', [])
                sources = rag_result.get('sources', [])
                
                # Calculate metrics
                token_count = len(response.split())  # Approximate token count
                
                # Evaluate response quality
                quality_metrics = await self.evaluator.evaluate_response(
                    question=question,
                    response=response,
                    context=context_used,
                    industry=industry
                )
                
                # Compile result
                result = {
                    'test_id': f"auto_test_{test_number:03d}",
                    'test_number': test_number,
                    'timestamp': datetime.now().isoformat(),
                    'provider': provider_name,
                    'model': self.provider_manager.get_model_name(provider_name),
                    'industry': industry,
                    'dataset': dataset_name,
                    'question': question,
                    'response': response,
                    'context_used': context_used,
                    'sources': sources,
                    'latency': round(latency, 3),
                    'token_count': token_count,
                    'quality_score': quality_metrics.get('overall_score', 0),
                    'relevance_score': quality_metrics.get('relevance', 0),
                    'coherence_score': quality_metrics.get('coherence', 0),
                    'accuracy_score': quality_metrics.get('accuracy', 0),
                    'completeness_score': quality_metrics.get('completeness', 0),
                    'error': None,
                    'test_metadata': {
                        'question_difficulty': question_data.get('difficulty', 'medium'),
                        'expected_metrics': question_data.get('expected_metrics', []),
                        'dataset_rows': len(dataset),
                        'dataset_columns': len(dataset.columns)
                    }
                }
                
                evaluation_results.append(result)
                
                print(f"Completed {provider_name} evaluation", 
                               latency=latency, quality=quality_metrics.get('overall_score', 0))
                
            except Exception as e:
                error_result = {
                    'test_id': f"auto_test_{test_number:03d}",
                    'test_number': test_number,
                    'timestamp': datetime.now().isoformat(),
                    'provider': provider_name,
                    'model': self.provider_manager.get_model_name(provider_name),
                    'industry': industry,
                    'dataset': dataset_name,
                    'question': question,
                    'response': None,
                    'context_used': [],
                    'sources': [],
                    'latency': 0,
                    'token_count': 0,
                    'quality_score': 0,
                    'relevance_score': 0,
                    'coherence_score': 0,
                    'accuracy_score': 0,
                    'completeness_score': 0,
                    'error': str(e),
                    'test_metadata': {
                        'question_difficulty': question_data.get('difficulty', 'medium'),
                        'expected_metrics': question_data.get('expected_metrics', []),
                        'dataset_rows': len(dataset),
                        'dataset_columns': len(dataset.columns)
                    }
                }
                
                evaluation_results.append(error_result)
                self.logger.error(f"Error in {provider_name} evaluation: {e}")
        
        return evaluation_results
    
    async def run_health_check(self):
        """Ping all LLM providers and individual models to check health/status."""
        health_results = []
        
        # Get all providers and their models
        all_models = self.provider_manager.get_all_models()
        
        for provider_name, models in all_models.items():
            # Test each model individually
            for model_name in models:
                try:
                    start = time.perf_counter()
                    # Use a simple, fast prompt
                    response = self.provider_manager.generate_response(
                        provider_name=provider_name,
                        query="Health check: respond with OK",
                        model=model_name,
                        context=""
                    )
                    latency = time.perf_counter() - start
                    
                    health_results.append({
                        "timestamp": datetime.now().isoformat(),
                        "provider": provider_name,
                        "model": model_name,
                        "status": "ok" if response.success else "error",
                        "latency": round(latency, 3),
                        "response": response.text[:100] + "..." if len(response.text) > 100 else response.text,
                        "tokens_used": response.tokens_used,
                        "error": response.error if not response.success else None
                    })
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    status = "error"
                    
                    # Check for specific error types
                    if any(keyword in error_msg for keyword in ["rate limit", "quota", "429", "too many requests"]):
                        status = "rate_limited"
                    elif any(keyword in error_msg for keyword in ["overloaded", "503", "unavailable"]):
                        status = "overloaded"
                    elif any(keyword in error_msg for keyword in ["decommissioned", "not found", "404"]):
                        status = "deprecated"
                    
                    health_results.append({
                        "timestamp": datetime.now().isoformat(),
                        "provider": provider_name,
                        "model": model_name,
                        "status": status,
                        "error": str(e),
                        "latency": 0,
                        "tokens_used": None,
                        "response": None
                    })
        
        # Save health check results
        if self.health_log_file.exists():
            with open(self.health_log_file, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.extend(health_results)
        with open(self.health_log_file, "w") as f:
            json.dump(data, f, indent=2)
        
        # Log summary
        working_models = [r for r in health_results if r['status'] == 'ok']
        failed_models = [r for r in health_results if r['status'] != 'ok']
        
        print(f"LLM health check completed - Total: {len(health_results)}, Working: {len(working_models)}, Failed: {len(failed_models)}")
        
        return health_results

    async def run_automated_evaluation(self) -> None:
        """Run the complete automated evaluation cycle"""
        
        self.start_time = datetime.now()
        print("Starting automated LLM evaluation", 
                        duration_minutes=self.test_duration_minutes,
                        interval_seconds=self.test_interval_seconds,
                        total_tests=self.total_tests)
        
        # Progress tracking
        completed_tests = 0
        successful_tests = 0
        failed_tests = 0
        
        try:
            for test_number in range(1, self.total_tests + 1):
                test_start = datetime.now()
                
                # Run evaluation
                try:
                    evaluation_results = await self.run_single_evaluation(test_number, self.total_tests)
                    
                    if evaluation_results:
                        self.results.extend(evaluation_results)
                        successful_tests += len([r for r in evaluation_results if r.get('error') is None])
                        failed_tests += len([r for r in evaluation_results if r.get('error') is not None])
                        completed_tests += 1
                        
                        # Save intermediate results
                        if test_number % 5 == 0:  # Save every 5 tests
                            self.save_intermediate_results(test_number)
                    
                except Exception as e:
                    self.logger.error(f"Test {test_number} failed: {e}")
                    failed_tests += 1
                
                # Health check every 10 minutes (every 5th test)
                if test_number % 5 == 0:
                    await self.run_health_check()
                
                # Calculate time until next test
                test_duration = (datetime.now() - test_start).total_seconds()
                sleep_time = max(0, self.test_interval_seconds - test_duration)
                
                # Progress update
                progress = (test_number / self.total_tests) * 100
                print(f"Test {test_number}/{self.total_tests} completed", 
                               progress=f"{progress:.1f}%",
                               successful=successful_tests,
                               failed=failed_tests,
                               sleep_time=sleep_time)
                
                # Wait for next test (unless this is the last test)
                if test_number < self.total_tests:
                    await asyncio.sleep(sleep_time)
        
        except KeyboardInterrupt:
            self.logger.warning("Automated evaluation interrupted by user")
        
        finally:
            self.end_time = datetime.now()
            self.save_final_results()
            self.generate_summary_report()
    
    def save_intermediate_results(self, test_number: int) -> None:
        """Save intermediate results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"automated_eval_intermediate_{timestamp}_test_{test_number:03d}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Saved intermediate results to {filename} ({len(self.results)} results)")
    
    def save_final_results(self) -> None:
        """Save final evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"automated_eval_final_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Also save to results directory for dashboard access
        results_file = Path("data/results") / filename
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Saved final results to {filename} ({len(self.results)} results)")
    
    def generate_summary_report(self) -> None:
        """Generate comprehensive summary report"""
        if not self.results:
            self.logger.warning("No results to generate summary")
            return
        
        # Calculate statistics
        total_evaluations = len(self.results)
        successful_evaluations = len([r for r in self.results if r.get('error') is None])
        failed_evaluations = total_evaluations - successful_evaluations
        
        # Provider statistics
        provider_stats = {}
        for result in self.results:
            provider = result.get('provider', 'Unknown')
            if provider not in provider_stats:
                provider_stats[provider] = {
                    'total': 0,
                    'successful': 0,
                    'failed': 0,
                    'latencies': [],
                    'quality_scores': [],
                    'token_counts': []
                }
            
            provider_stats[provider]['total'] += 1
            if result.get('error') is None:
                provider_stats[provider]['successful'] += 1
                provider_stats[provider]['latencies'].append(result.get('latency', 0))
                provider_stats[provider]['quality_scores'].append(result.get('quality_score', 0))
                provider_stats[provider]['token_counts'].append(result.get('token_count', 0))
            else:
                provider_stats[provider]['failed'] += 1
        
        # Calculate averages
        for provider, stats in provider_stats.items():
            if stats['latencies']:
                stats['avg_latency'] = np.mean(stats['latencies'])
                stats['avg_quality'] = np.mean(stats['quality_scores'])
                stats['avg_tokens'] = np.mean(stats['token_counts'])
                stats['success_rate'] = (stats['successful'] / stats['total']) * 100
            else:
                stats['avg_latency'] = 0
                stats['avg_quality'] = 0
                stats['avg_tokens'] = 0
                stats['success_rate'] = 0
        
        # Generate report
        report = {
            'evaluation_summary': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_minutes': self.test_duration_minutes,
                'test_interval_seconds': self.test_interval_seconds,
                'total_tests': self.total_tests,
                'total_evaluations': total_evaluations,
                'successful_evaluations': successful_evaluations,
                'failed_evaluations': failed_evaluations,
                'overall_success_rate': (successful_evaluations / total_evaluations) * 100 if total_evaluations > 0 else 0
            },
            'provider_performance': provider_stats,
            'datasets_used': list(self.datasets.keys()),
            'questions_used': len(self.questions),
            'test_metadata': {
                'rag_enabled': True,
                'evaluation_metrics': ['quality_score', 'relevance_score', 'coherence_score', 'accuracy_score'],
                'performance_metrics': ['latency', 'token_count', 'error_rate']
            }
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"automated_eval_summary_{timestamp}.json"
        report_filepath = self.results_dir / report_filename
        
        with open(report_filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("🤖 AUTOMATED LLM EVALUATION COMPLETE")
        print("="*60)
        print(f"📊 Total Evaluations: {total_evaluations}")
        print(f"✅ Successful: {successful_evaluations}")
        print(f"❌ Failed: {failed_evaluations}")
        print(f"📈 Success Rate: {report['evaluation_summary']['overall_success_rate']:.1f}%")
        print(f"⏱️  Duration: {self.test_duration_minutes} minutes")
        print(f"🔄 Test Interval: {self.test_interval_seconds} seconds")
        print(f"📁 Results saved to: {report_filepath}")
        print("\n📊 Provider Performance:")
        
        for provider, stats in provider_stats.items():
            print(f"  {provider}:")
            print(f"    Success Rate: {stats['success_rate']:.1f}%")
            print(f"    Avg Latency: {stats['avg_latency']:.3f}s")
            print(f"    Avg Quality: {stats['avg_quality']:.3f}")
            print(f"    Avg Tokens: {stats['avg_tokens']:.0f}")
        
        print("\n🎯 Next Steps:")
        print("  1. View results in Model Performance Dashboard")
        print("  2. Analyze provider comparisons")
        print("  3. Generate academic reports")
        print("="*60)
        
        print("Automated evaluation completed", 
                        total_evaluations=total_evaluations,
                        success_rate=report['evaluation_summary']['overall_success_rate'])

async def main():
    """Main function to run automated evaluation"""
    
    print("🚀 Starting Automated LLM Evaluator")
    print("="*50)
    print("This will run LLM evaluations every 2 minutes for 1 hour")
    print("Using RAG pipeline with your datasets")
    print("="*50)
    
    # Initialize evaluator
    evaluator = AutomatedLLMEvaluator()
    
    # Confirm before starting
    print(f"\n📋 Configuration:")
    print(f"   Duration: {evaluator.test_duration_minutes} minutes")
    print(f"   Interval: {evaluator.test_interval_seconds} seconds")
    print(f"   Total Tests: {evaluator.total_tests}")
    print(f"   Datasets: {list(evaluator.datasets.keys())}")
    print(f"   Questions: {len(evaluator.questions)}")
    
    response = input("\n❓ Start automated evaluation? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Evaluation cancelled")
        return
    
    # Run evaluation
    await evaluator.run_automated_evaluation()

if __name__ == "__main__":
    asyncio.run(main()) 