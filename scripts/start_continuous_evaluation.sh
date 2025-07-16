#!/bin/bash
# Start Continuous Evaluation Script
# This script runs the continuous evaluation with research-grade settings

echo "🚀 Starting Continuous Evaluation for LLM Performance Research"
echo "================================================================"

# Configuration
INTERVAL_HOURS=2           # Run every 2 hours (6 runs per day)
MAX_RUNS=42               # Run for 1 week (7 days × 6 runs/day)
RATE_LIMIT_BUFFER=0.8     # Use 80% of rate limits (conservative)
OUTPUT_DIR="data/continuous_evaluation"

echo "📊 Configuration:"
echo "   - Interval: ${INTERVAL_HOURS} hours"
echo "   - Max runs: ${MAX_RUNS} (1 week - 6 runs per day)"
echo "   - Rate limit buffer: ${RATE_LIMIT_BUFFER}"
echo "   - Output directory: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Start the continuous evaluation
echo "🔄 Starting continuous evaluation..."
python3 scripts/continuous_evaluation.py \
    --interval "${INTERVAL_HOURS}" \
    --max-runs "${MAX_RUNS}" \
    --rate-limit-buffer "${RATE_LIMIT_BUFFER}" \
    --output-dir "${OUTPUT_DIR}"

echo "✅ Continuous evaluation completed!"
echo "📁 Results saved in: ${OUTPUT_DIR}"
echo "📊 Check aggregated_results.json for summary" 