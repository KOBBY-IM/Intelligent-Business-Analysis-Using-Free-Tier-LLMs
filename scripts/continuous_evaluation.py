#!/usr/bin/env python3
"""
Continuous Evaluation Script
Runs performance analysis and statistical evaluation at regular intervals.
This is separate from blind evaluations to ensure consistency.
"""

import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from typing import Dict, List, Any, Optional
import sys
import os
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.structured_logger import setup_logger
from src.evaluation.statistical_analysis import StatisticalAnalyzer
from src.security.rate_limiter import APIRateLimiter, RateLimiter

class ContinuousEvaluator:
    """Runs continuous performance analysis and statistical evaluation."""
    
    def __init__(self, 
                 interval_hours: int = 2,
                 max_runs: int = None,
                 output_dir: str = "data/continuous_evaluation",
                 rate_limit_buffer: float = 0.8):
        self.interval_hours = interval_hours
        self.max_runs = max_runs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_buffer = rate_limit_buffer
        
        # Setup logging
        self.logger = setup_logger("continuous_evaluator", 
                                 log_file=self.output_dir / "continuous_eval.log")
        
        # Initialize rate limiters for each provider
        self.rate_limiters = {
            'groq': APIRateLimiter('groq'),
            'gemini': APIRateLimiter('gemini'),
            'huggingface': APIRateLimiter('huggingface')
        }
        
        # Global rate limiter for overall evaluation frequency
        self.global_limiter = RateLimiter()
        
        # Track runs
        self.run_count = 0
        self.results_summary = []
        
        # Provider-specific limits (requests per hour)
        self.provider_limits = {
            'groq': 1000,
            'gemini': 1500,
            'huggingface': 500
        }
        
        # Performance evaluation questions (different from blind evaluation)
        self.performance_questions = [
            "What is the average response time across all models?",
            "Which model has the highest accuracy based on ground truth?",
            "What is the correlation between response length and user preference?",
            "How do models perform under different load conditions?",
            "What is the statistical significance of performance differences?"
        ]
        
        self.logger.info(f"Initialized continuous evaluator with {len(self.performance_questions)} performance questions")
        
    def check_rate_limits(self) -> Dict[str, bool]:
        """Check current rate limits for all providers."""
        status = {}
        for provider, limiter in self.rate_limiters.items():
            remaining = limiter.get_remaining_api_calls('continuous_eval')
            status[provider] = {
                'per_minute': remaining['per_minute'],
                'per_hour': remaining['per_hour'],
                'can_proceed': remaining['per_hour'] > len(self.performance_questions) * 0.5
            }
        return status
    
    def calculate_safe_interval(self) -> int:
        """Calculate safe interval based on rate limits and question count."""
        # Calculate minimum time needed for all providers
        min_intervals = []
        
        for provider, limit in self.provider_limits.items():
            # Calculate how many hours needed for performance evaluation
            hours_needed = (len(self.performance_questions) * 3) / (limit * self.rate_limit_buffer)
            min_intervals.append(hours_needed)
        
        # Use the maximum (most restrictive) interval
        safe_interval = max(min_intervals)
        
        # Add some buffer and round up
        safe_interval = max(safe_interval * 1.2, self.interval_hours)
        
        self.logger.info(f"Calculated safe interval: {safe_interval:.1f} hours")
        return int(safe_interval)
    
    def wait_for_rate_limit_reset(self, provider: str) -> None:
        """Wait for rate limit reset if needed."""
        limiter = self.rate_limiters[provider]
        remaining = limiter.get_remaining_api_calls('continuous_eval')
        
        if remaining['per_hour'] < len(self.performance_questions):
            # Calculate wait time
            reset_time = limiter.rate_limiter.get_reset_time('continuous_eval', 'requests_per_hour')
            if reset_time:
                wait_seconds = (reset_time - datetime.now()).total_seconds()
                if wait_seconds > 0:
                    self.logger.info(f"Waiting {wait_seconds/3600:.1f} hours for {provider} rate limit reset")
                    time.sleep(wait_seconds + 60)  # Add 1 minute buffer
    
    def run_performance_analysis(self) -> Dict[str, Any]:
        """Run performance analysis and statistical evaluation."""
        timestamp = datetime.now().isoformat()
        self.logger.info(f"Starting performance analysis run #{self.run_count + 1} at {timestamp}")
        
        # Check rate limits before starting
        rate_status = self.check_rate_limits()
        self.logger.info(f"Rate limit status: {rate_status}")
        
        # Check if we can proceed
        can_proceed = all(status['can_proceed'] for status in rate_status.values())
        
        if not can_proceed:
            self.logger.warning("Rate limits too restrictive, waiting for reset...")
            for provider in self.rate_limiters.keys():
                self.wait_for_rate_limit_reset(provider)
        
        try:
            # Load fixed blind responses for analysis
            fixed_responses_file = Path("data/fixed_blind_responses.json")
            if not fixed_responses_file.exists():
                self.logger.warning("Fixed blind responses not found. Skipping performance analysis.")
                return {
                    'run_id': f"run_{self.run_count + 1}_{timestamp.replace(':', '-')}",
                    'timestamp': timestamp,
                    'status': 'skipped',
                    'reason': 'Fixed blind responses not found'
                }
            
            with open(fixed_responses_file, 'r') as f:
                fixed_responses = json.load(f)
            
            # Analyze performance metrics
            performance_metrics = self.analyze_performance_metrics(fixed_responses)
            
            # Generate statistical analysis
            statistical_analysis = self.generate_statistical_analysis(fixed_responses)
            
            # Save results
            run_id = f"run_{self.run_count + 1}_{timestamp.replace(':', '-')}"
            results_file = self.output_dir / f"{run_id}_performance_analysis.json"
            
            results = {
                'run_id': run_id,
                'timestamp': timestamp,
                'rate_limits': rate_status,
                'performance_metrics': performance_metrics,
                'statistical_analysis': statistical_analysis,
                'fixed_responses_analyzed': len(fixed_responses.get('responses', {}))
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Performance analysis run #{self.run_count + 1} completed successfully")
            return {
                'run_id': run_id,
                'timestamp': timestamp,
                'status': 'success',
                'rate_limits': rate_status,
                'results_file': str(results_file)
            }
            
        except Exception as e:
            self.logger.error(f"Performance analysis run #{self.run_count + 1} failed: {str(e)}")
            return {
                'run_id': f"run_{self.run_count + 1}_{timestamp.replace(':', '-')}",
                'timestamp': timestamp,
                'status': 'failed',
                'error': str(e),
                'rate_limits': rate_status
            }
    
    def analyze_performance_metrics(self, fixed_responses: Dict) -> Dict[str, Any]:
        """Analyze performance metrics from fixed responses."""
        metrics = {
            'response_times': {},
            'response_lengths': {},
            'model_coverage': {},
            'domain_performance': {}
        }
        
        responses_data = fixed_responses.get('responses', {})
        
        for question_id, question_data in responses_data.items():
            domain = question_data.get('domain', 'unknown')
            llm_responses = question_data.get('llm_responses', {})
            
            for provider, response_data in llm_responses.items():
                metadata = response_data.get('metadata', {})
                
                # Track response times
                latency = metadata.get('latency', 0)
                if provider not in metrics['response_times']:
                    metrics['response_times'][provider] = []
                metrics['response_times'][provider].append(latency)
                
                # Track response lengths
                response_text = response_data.get('response', '')
                length = len(response_text.split())
                if provider not in metrics['response_lengths']:
                    metrics['response_lengths'][provider] = []
                metrics['response_lengths'][provider].append(length)
                
                # Track model coverage
                if provider not in metrics['model_coverage']:
                    metrics['model_coverage'][provider] = 0
                metrics['model_coverage'][provider] += 1
        
        # Calculate averages
        for provider in metrics['response_times']:
            if metrics['response_times'][provider]:
                metrics['response_times'][provider] = {
                    'average': sum(metrics['response_times'][provider]) / len(metrics['response_times'][provider]),
                    'min': min(metrics['response_times'][provider]),
                    'max': max(metrics['response_times'][provider])
                }
        
        for provider in metrics['response_lengths']:
            if metrics['response_lengths'][provider]:
                metrics['response_lengths'][provider] = {
                    'average': sum(metrics['response_lengths'][provider]) / len(metrics['response_lengths'][provider]),
                    'min': min(metrics['response_lengths'][provider]),
                    'max': max(metrics['response_lengths'][provider])
                }
        
        return metrics
    
    def generate_statistical_analysis(self, fixed_responses: Dict) -> Dict[str, Any]:
        """Generate statistical analysis of the fixed responses."""
        try:
            analyzer = StatisticalAnalyzer()
            
            # This would include statistical tests, confidence intervals, etc.
            # For now, return basic statistics
            analysis = {
                'total_questions': len(fixed_responses.get('responses', {})),
                'total_responses': sum(
                    len(q_data.get('llm_responses', {})) 
                    for q_data in fixed_responses.get('responses', {}).values()
                ),
                'providers_analyzed': list(set(
                    provider
                    for q_data in fixed_responses.get('responses', {}).values()
                    for provider in q_data.get('llm_responses', {}).keys()
                )),
                'generation_timestamp': fixed_responses.get('generation_timestamp', 'unknown')
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in statistical analysis: {str(e)}")
            return {'error': str(e)}
    
    def aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results from all runs."""
        self.logger.info("Aggregating results from all runs...")
        
        # Load all result files
        result_files = list(self.output_dir.glob("*_performance_analysis.json"))
        all_results = []
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if data.get('status') == 'success':
                        all_results.append(data)
            except Exception as e:
                self.logger.warning(f"Could not load {file_path}: {e}")
        
        # Generate summary statistics
        if all_results:
            summary = {
                'total_runs': len(all_results),
                'successful_runs': len([r for r in all_results if r['status'] == 'success']),
                'failed_runs': len([r for r in all_results if r['status'] == 'failed']),
                'date_range': {
                    'start': min(r['timestamp'] for r in all_results),
                    'end': max(r['timestamp'] for r in all_results)
                },
                'rate_limit_compliance': {
                    'runs_with_rate_limits': len([r for r in all_results if 'rate_limits' in r]),
                    'average_remaining_calls': self._calculate_avg_remaining_calls(all_results)
                },
                'performance_trends': self._analyze_performance_trends(all_results),
                'run_details': all_results
            }
            
            # Save aggregated results
            summary_file = self.output_dir / "aggregated_performance_results.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Aggregated results saved to {summary_file}")
            return summary
        
        return {'error': 'No results to aggregate'}
    
    def _calculate_avg_remaining_calls(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate average remaining API calls across all runs."""
        if not results:
            return {}
        
        total_remaining = {'groq': 0, 'gemini': 0, 'huggingface': 0}
        count = 0
        
        for result in results:
            if 'rate_limits' in result:
                for provider, limits in result['rate_limits'].items():
                    if provider in total_remaining:
                        total_remaining[provider] += limits.get('per_hour', 0)
                count += 1
        
        if count == 0:
            return {}
        
        return {provider: total / count for provider, total in total_remaining.items()}
    
    def _analyze_performance_trends(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance trends across runs."""
        trends = {
            'response_time_trends': {},
            'coverage_trends': {},
            'success_rate_trends': {}
        }
        
        # Analyze trends over time
        for result in results:
            if 'performance_metrics' in result:
                metrics = result['performance_metrics']
                timestamp = result['timestamp']
                
                # Track response time trends
                for provider, times in metrics.get('response_times', {}).items():
                    if provider not in trends['response_time_trends']:
                        trends['response_time_trends'][provider] = []
                    trends['response_time_trends'][provider].append({
                        'timestamp': timestamp,
                        'average_time': times.get('average', 0)
                    })
        
        return trends
    
    def run_continuous_evaluation(self):
        """Run continuous evaluation at specified intervals with rate limiting."""
        self.logger.info(f"Starting continuous evaluation with {self.interval_hours}h intervals")
        self.logger.info(f"Max runs: {self.max_runs or 'unlimited'}")
        
        # Calculate safe interval
        safe_interval = self.calculate_safe_interval()
        if safe_interval > self.interval_hours:
            self.logger.warning(f"Rate limits require {safe_interval}h intervals (requested: {self.interval_hours}h)")
            self.interval_hours = safe_interval
        
        start_time = datetime.now()
        
        while True:
            # Check if we've reached max runs
            if self.max_runs and self.run_count >= self.max_runs:
                self.logger.info(f"Reached maximum runs ({self.max_runs}). Stopping.")
                break
            
            # Run performance analysis
            result = self.run_performance_analysis()
            self.results_summary.append(result)
            self.run_count += 1
            
            # Log progress
            elapsed = datetime.now() - start_time
            self.logger.info(f"Completed {self.run_count} runs in {elapsed}")
            
            # Wait for next interval
            if self.max_runs is None or self.run_count < self.max_runs:
                self.logger.info(f"Waiting {self.interval_hours} hours until next evaluation...")
                time.sleep(self.interval_hours * 3600)  # Convert hours to seconds
        
        # Final aggregation
        final_summary = self.aggregate_results()
        self.logger.info("Continuous evaluation completed.")
        return final_summary

def main():
    parser = argparse.ArgumentParser(description="Run continuous performance analysis with rate limiting")
    parser.add_argument("--interval", type=int, default=2, 
                       help="Interval between runs in hours (default: 2)")
    parser.add_argument("--max-runs", type=int, default=None,
                       help="Maximum number of runs (default: unlimited)")
    parser.add_argument("--output-dir", type=str, default="data/continuous_evaluation",
                       help="Output directory for results")
    parser.add_argument("--rate-limit-buffer", type=float, default=0.8,
                       help="Rate limit buffer (0.8 = use 80%% of limits, default: 0.8)")
    
    args = parser.parse_args()
    
    evaluator = ContinuousEvaluator(
        interval_hours=args.interval,
        max_runs=args.max_runs,
        output_dir=args.output_dir,
        rate_limit_buffer=args.rate_limit_buffer
    )
    
    try:
        summary = evaluator.run_continuous_evaluation()
        print(f"Continuous evaluation completed. Summary: {summary}")
    except KeyboardInterrupt:
        print("\nContinuous evaluation interrupted by user.")
        summary = evaluator.aggregate_results()
        print(f"Partial results summary: {summary}")

if __name__ == "__main__":
    main() 