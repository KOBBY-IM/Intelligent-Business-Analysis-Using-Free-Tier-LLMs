#!/usr/bin/env python3
"""
Generate Sample Performance Data

Creates sample evaluation data with performance metrics for testing
the Model Performance Dashboard.
"""

import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

def generate_sample_responses():
    """Generate sample business analysis responses"""
    responses = [
        "Based on the retail data analysis, customer preferences show seasonal patterns with 45% increase in outdoor gear during spring months.",
        "Financial analysis indicates strong correlation between market volatility and customer retention rates in Q3 2024.",
        "Healthcare sector analysis reveals significant cost savings opportunities through digital transformation initiatives.",
        "Market segmentation analysis shows premium customers contribute 60% of total revenue despite being 15% of user base.",
        "Supply chain optimization could reduce operational costs by 23% while improving delivery times.",
        "Customer sentiment analysis indicates 78% satisfaction rate with recent product launches.",
        "Revenue forecasting models predict 12% growth in next quarter based on current trends.",
        "Risk assessment shows moderate exposure to market fluctuations with diversified portfolio approach."
    ]
    return responses

def generate_sample_questions():
    """Generate sample business questions"""
    questions = [
        "What are the key trends in customer purchasing behavior?",
        "How can we optimize our pricing strategy for maximum profitability?",
        "What risks should we consider for our expansion plans?",
        "Which market segments show the highest growth potential?",
        "How effective are our current marketing campaigns?",
        "What operational improvements could reduce costs?",
        "How do seasonal patterns affect our sales performance?",
        "What competitive advantages do we have in the market?"
    ]
    return questions

def generate_performance_metrics(provider, model, base_latency=1.0):
    """Generate realistic performance metrics for a provider/model"""
    
    # Provider-specific characteristics
    provider_configs = {
        'Groq': {
            'base_latency': 0.8,
            'latency_variance': 0.3,
            'quality_mean': 0.85,
            'quality_std': 0.1,
            'token_mean': 150,
            'token_std': 40,
            'error_rate': 0.02
        },
        'Gemini': {
            'base_latency': 1.2,
            'latency_variance': 0.4,
            'quality_mean': 0.88,
            'quality_std': 0.08,
            'token_mean': 180,
            'token_std': 35,
            'error_rate': 0.01
        },
        'Hugging Face': {
            'base_latency': 2.1,
            'latency_variance': 0.8,
            'quality_mean': 0.82,
            'quality_std': 0.12,
            'token_mean': 140,
            'token_std': 50,
            'error_rate': 0.05
        },
        'OpenRouter': {
            'base_latency': 1.5,
            'latency_variance': 0.5,
            'quality_mean': 0.86,
            'quality_std': 0.09,
            'token_mean': 165,
            'token_std': 45,
            'error_rate': 0.03
        }
    }
    
    config = provider_configs.get(provider, provider_configs['Groq'])
    
    # Generate latency with realistic variance
    latency = max(0.1, np.random.normal(config['base_latency'], config['latency_variance']))
    
    # Generate quality scores (correlated with each other)
    base_quality = np.random.normal(config['quality_mean'], config['quality_std'])
    quality_score = max(0, min(1, base_quality))
    
    # Related quality metrics with some correlation
    relevance_score = max(0, min(1, quality_score + np.random.normal(0, 0.05)))
    coherence_score = max(0, min(1, quality_score + np.random.normal(0, 0.06)))
    accuracy_score = max(0, min(1, quality_score + np.random.normal(0, 0.07)))
    
    # Token count
    token_count = max(10, int(np.random.normal(config['token_mean'], config['token_std'])))
    
    # Simulate errors
    has_error = random.random() < config['error_rate']
    error = "API timeout" if has_error else None
    
    return {
        'latency': round(latency, 3),
        'token_count': token_count,
        'quality_score': round(quality_score, 3),
        'relevance_score': round(relevance_score, 3),
        'coherence_score': round(coherence_score, 3),
        'accuracy_score': round(accuracy_score, 3),
        'error': error
    }

def generate_evaluation_data(num_evaluations=200):
    """Generate comprehensive evaluation data"""
    
    providers = ['Groq', 'Gemini', 'Hugging Face', 'OpenRouter']
    models = {
        'Groq': ['mixtral-8x7b-32768', 'llama2-70b-4096'],
        'Gemini': ['gemini-pro', 'gemini-pro-vision'],
        'Hugging Face': ['microsoft/DialoGPT-large', 'facebook/blenderbot-400M'],
        'OpenRouter': ['openai/gpt-3.5-turbo', 'anthropic/claude-instant-v1']
    }
    
    industries = ['retail', 'finance', 'healthcare']
    questions = generate_sample_questions()
    responses = generate_sample_responses()
    
    evaluation_data = []
    
    # Generate data over the past 30 days
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(num_evaluations):
        provider = random.choice(providers)
        model = random.choice(models[provider])
        industry = random.choice(industries)
        question = random.choice(questions)
        response = random.choice(responses)
        
        # Generate timestamp within the last 30 days
        timestamp = start_date + timedelta(
            days=random.randint(0, 29),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        metrics = generate_performance_metrics(provider, model)
        
        evaluation_entry = {
            'id': f"eval_{i+1:04d}",
            'provider': provider,
            'model': model,
            'industry': industry,
            'question': question,
            'response': response,
            'timestamp': timestamp.isoformat(),
            'metrics': metrics,
            # Flatten metrics for easier access
            'latency': metrics['latency'],
            'token_count': metrics['token_count'],
            'quality_score': metrics['quality_score'],
            'relevance_score': metrics['relevance_score'],
            'coherence_score': metrics['coherence_score'],
            'accuracy_score': metrics['accuracy_score'],
            'error': metrics['error']
        }
        
        evaluation_data.append(evaluation_entry)
    
    return evaluation_data

def save_evaluation_data(data, filename_prefix="eval"):
    """Save evaluation data to results directory"""
    
    # Create directories
    results_dir = Path("data/results")
    eval_dir = Path("data/evaluation_results")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to results directory
    results_file = results_dir / f"{filename_prefix}_{timestamp}_results.json"
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Save to evaluation_results directory
    eval_file = eval_dir / f"{filename_prefix}_{timestamp}_results.json"
    with open(eval_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Generate summary statistics
    summary = generate_summary_stats(data)
    summary_file = eval_dir / f"{filename_prefix}_{timestamp}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    return results_file, eval_file, summary_file

def generate_summary_stats(data):
    """Generate summary statistics for the evaluation data"""
    
    total_evaluations = len(data)
    providers = list(set(item['provider'] for item in data))
    industries = list(set(item['industry'] for item in data))
    
    # Calculate averages
    avg_latency = np.mean([item['latency'] for item in data if item['error'] is None])
    avg_quality = np.mean([item['quality_score'] for item in data if item['error'] is None])
    avg_tokens = np.mean([item['token_count'] for item in data if item['error'] is None])
    
    error_rate = len([item for item in data if item['error'] is not None]) / total_evaluations * 100
    
    # Provider breakdown
    provider_stats = {}
    for provider in providers:
        provider_data = [item for item in data if item['provider'] == provider]
        provider_stats[provider] = {
            'count': len(provider_data),
            'avg_latency': np.mean([item['latency'] for item in provider_data if item['error'] is None]),
            'avg_quality': np.mean([item['quality_score'] for item in provider_data if item['error'] is None]),
            'error_rate': len([item for item in provider_data if item['error'] is not None]) / len(provider_data) * 100
        }
    
    summary = f"""
Performance Evaluation Summary
=============================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Overview:
---------
Total Evaluations: {total_evaluations}
Providers: {', '.join(providers)}
Industries: {', '.join(industries)}

Overall Performance:
-------------------
Average Latency: {avg_latency:.3f} seconds
Average Quality Score: {avg_quality:.3f}
Average Token Count: {avg_tokens:.1f}
Overall Error Rate: {error_rate:.2f}%

Provider Performance:
--------------------
"""
    
    for provider, stats in provider_stats.items():
        summary += f"""
{provider}:
  Evaluations: {stats['count']}
  Avg Latency: {stats['avg_latency']:.3f}s
  Avg Quality: {stats['avg_quality']:.3f}
  Error Rate: {stats['error_rate']:.2f}%
"""
    
    return summary

def main():
    """Main function to generate and save performance data"""
    
    print("🚀 Generating sample performance data...")
    
    # Generate evaluation data
    evaluation_data = generate_evaluation_data(num_evaluations=250)
    
    print(f"✅ Generated {len(evaluation_data)} evaluation records")
    
    # Save data
    results_file, eval_file, summary_file = save_evaluation_data(evaluation_data, "performance_test")
    
    print(f"📁 Data saved to:")
    print(f"   Results: {results_file}")
    print(f"   Evaluation: {eval_file}")
    print(f"   Summary: {summary_file}")
    
    # Print quick stats
    providers = list(set(item['provider'] for item in evaluation_data))
    industries = list(set(item['industry'] for item in evaluation_data))
    
    print(f"\n📊 Quick Stats:")
    print(f"   Providers: {', '.join(providers)}")
    print(f"   Industries: {', '.join(industries)}")
    print(f"   Date Range: {min(item['timestamp'] for item in evaluation_data)[:10]} to {max(item['timestamp'] for item in evaluation_data)[:10]}")
    
    avg_quality = np.mean([item['quality_score'] for item in evaluation_data if item['error'] is None])
    avg_latency = np.mean([item['latency'] for item in evaluation_data if item['error'] is None])
    
    print(f"   Avg Quality: {avg_quality:.3f}")
    print(f"   Avg Latency: {avg_latency:.3f}s")
    
    print("\n🎯 Performance data generated successfully!")
    print("   You can now view the Model Performance Dashboard in the Streamlit app.")

if __name__ == "__main__":
    main() 