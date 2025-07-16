#!/usr/bin/env python3
"""
Analyze LLM Health Check Logs

Summarizes rate limit errors and provider health from llm_health_checks.json.
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd

HEALTH_LOG = Path("data/evaluation_results/llm_health_checks.json")

RATE_LIMIT_KEYWORDS = ["rate limit", "quota", "429", "too many requests", "exceeded"]
OVERLOADED_KEYWORDS = ["overloaded", "503", "unavailable"]
DEPRECATED_KEYWORDS = ["decommissioned", "not found", "404"]

def load_health_checks():
    if not HEALTH_LOG.exists():
        print(f"❌ Health check log not found: {HEALTH_LOG}")
        return []
    with open(HEALTH_LOG, "r") as f:
        return json.load(f)

def is_rate_limit_error(error_msg):
    if not error_msg:
        return False
    msg = error_msg.lower()
    return any(keyword in msg for keyword in RATE_LIMIT_KEYWORDS)

def analyze_health_checks():
    data = load_health_checks()
    if not data:
        print("No health check data to analyze.")
        return
    
    df = pd.DataFrame(data)
    
    # Check if this is model-level data
    has_models = 'model' in df.columns
    
    # Basic stats
    print(f"\n📊 Total health checks: {len(df)}")
    if has_models:
        print(f"Models checked: {len(df)}")
        print(f"Providers checked: {df['provider'].unique().tolist()}")
    else:
        print(f"Providers checked: {df['provider'].unique().tolist()}")
    
    # Status counts
    if has_models:
        status_counts = df.groupby(['provider', 'status']).size().unstack(fill_value=0)
        print("\n✅ Health Check Status Counts (by Provider):")
        print(status_counts)
        
        # Model-level status
        model_status = df.groupby(['provider', 'model', 'status']).size().unstack(fill_value=0)
        print("\n🤖 Model-Level Status:")
        for provider in df['provider'].unique():
            provider_models = df[df['provider'] == provider]
            print(f"\n{provider.upper()}:")
            for _, row in provider_models.iterrows():
                status_icon = "✅" if row['status'] == 'ok' else "❌"
                print(f"  {status_icon} {row['model']}: {row['status']}")
    else:
        status_counts = df.groupby(['provider', 'status']).size().unstack(fill_value=0)
        print("\n✅ Health Check Status Counts:")
        print(status_counts)
    
    # Error analysis
    if 'error' in df.columns:
        # Rate limit errors
        df['rate_limit'] = df['error'].apply(lambda x: any(keyword in str(x).lower() for keyword in RATE_LIMIT_KEYWORDS)) if 'error' in df else False
        df['overloaded'] = df['error'].apply(lambda x: any(keyword in str(x).lower() for keyword in OVERLOADED_KEYWORDS)) if 'error' in df else False
        df['deprecated'] = df['error'].apply(lambda x: any(keyword in str(x).lower() for keyword in DEPRECATED_KEYWORDS)) if 'error' in df else False
        
        rate_limited = df[df['rate_limit']]
        overloaded = df[df['overloaded']]
        deprecated = df[df['deprecated']]
        
        if not rate_limited.empty:
            print(f"\n🚨 Rate Limited Models ({len(rate_limited)}):")
            for _, row in rate_limited.iterrows():
                model_name = row.get('model', row['provider'])
                print(f"  - {row['provider']}/{model_name}")
        
        if not overloaded.empty:
            print(f"\n⚠️  Overloaded Models ({len(overloaded)}):")
            for _, row in overloaded.iterrows():
                model_name = row.get('model', row['provider'])
                print(f"  - {row['provider']}/{model_name}")
        
        if not deprecated.empty:
            print(f"\n🗑️  Deprecated Models ({len(deprecated)}):")
            for _, row in deprecated.iterrows():
                model_name = row.get('model', row['provider'])
                print(f"  - {row['provider']}/{model_name}")
    
    # Average latency (successful checks)
    if 'latency' in df.columns:
        if has_models:
            latency_stats = df[df['status'] == 'ok'].groupby(['provider', 'model'])['latency'].agg(['mean', 'max', 'min']).round(3)
            print("\n⏱️  Latency Stats by Model (seconds):")
            print(latency_stats)
            
            # Fastest models
            working_models = df[df['status'] == 'ok']
            if not working_models.empty:
                fastest_models = working_models.nsmallest(3, 'latency')[['provider', 'model', 'latency']]
                print("\n⚡ Fastest Models:")
                for _, row in fastest_models.iterrows():
                    print(f"  - {row['provider']}/{row['model']}: {row['latency']}s")
        else:
            latency_stats = df[df['status'] == 'ok'].groupby('provider')['latency'].agg(['mean', 'max', 'min', 'count'])
            print("\n⏱️  Latency Stats (seconds):")
            print(latency_stats)
    
    # Timeline of errors
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        error_timeline = df[df['status'] != 'ok'].groupby([pd.Grouper(key='timestamp', freq='10T'), 'provider']).size().unstack(fill_value=0)
        if not error_timeline.empty:
            print("\n🕒 Error Timeline (10-min intervals):")
            print(error_timeline)
    
    # Working models summary
    if has_models:
        working_models = df[df['status'] == 'ok']
        total_models = len(df)
        working_count = len(working_models)
        
        print(f"\n📈 Working Models Summary:")
        print(f"  Total Models: {total_models}")
        print(f"  Working: {working_count}")
        print(f"  Success Rate: {(working_count/total_models)*100:.1f}%")
        
        # Provider breakdown
        provider_stats = df.groupby('provider').agg({
            'status': lambda x: (x == 'ok').sum(),
            'model': 'count'
        }).rename(columns={'status': 'working', 'model': 'total'})
        provider_stats['success_rate'] = (provider_stats['working'] / provider_stats['total'] * 100).round(1)
        
        print(f"\n📋 Provider Success Rates:")
        for provider, stats in provider_stats.iterrows():
            print(f"  {provider}: {stats['working']}/{stats['total']} ({stats['success_rate']}%)")
    
    # Save summary CSV
    summary_csv = HEALTH_LOG.parent / "llm_health_check_summary.csv"
    if has_models:
        # Save model-level summary
        model_summary = df.groupby(['provider', 'model', 'status']).size().unstack(fill_value=0)
        model_summary.to_csv(summary_csv)
    else:
        status_counts.to_csv(summary_csv)
    print(f"\n📁 Summary CSV saved to: {summary_csv}")

if __name__ == "__main__":
    analyze_health_checks() 