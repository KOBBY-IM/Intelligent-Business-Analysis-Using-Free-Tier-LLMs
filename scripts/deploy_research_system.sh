#!/bin/bash
# Research System Deployment Script
# Deploys the complete LLM evaluation research system with fixed blind responses

echo "🎓 LLM Evaluation Research System Deployment"
echo "============================================="

# Configuration
BLIND_EVAL_PORT=8501
CONTINUOUS_INTERVAL=2
CONTINUOUS_MAX_RUNS=42  # 1 week (6 runs per day)

echo "📊 Deployment Configuration:"
echo "   - Blind evaluation UI: Port ${BLIND_EVAL_PORT}"
echo "   - Continuous analysis: Every ${CONTINUOUS_INTERVAL}h for ${CONTINUOUS_MAX_RUNS} runs"
echo "   - Fixed responses for consistent blind evaluations"
echo ""

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "❌ Port $port is already in use"
        return 1
    else
        echo "✅ Port $port is available"
        return 0
    fi
}

# Check if ports are available
echo "🔍 Checking port availability..."
if ! check_port $BLIND_EVAL_PORT; then
    echo "💡 Try a different port: --server.port 8502"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/continuous_evaluation
mkdir -p data/results
mkdir -p logs
mkdir -p data/backups

# Phase 1: Generate Fixed Blind Responses
echo ""
echo "🎯 Phase 1: Generating Fixed Blind Responses"
echo "============================================"
echo "This ensures all users see the same responses for fair evaluation."

if [ -f "data/fixed_blind_responses.json" ]; then
    echo "✅ Fixed responses already exist. Skipping generation."
    echo "   To regenerate, delete data/fixed_blind_responses.json and run again."
else
    echo "🔄 Generating fixed blind responses..."
    python3 scripts/generate_fixed_blind_responses.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Fixed blind responses generated successfully!"
    else
        echo "❌ Failed to generate fixed responses. Please check the logs."
        exit 1
    fi
fi

# Phase 2: Start Continuous Performance Analysis
echo ""
echo "📊 Phase 2: Starting Continuous Performance Analysis"
echo "==================================================="
echo "This runs in the background and analyzes performance metrics."

# Start continuous evaluation in background
echo "🔄 Starting continuous performance analysis in background..."
python3 scripts/continuous_evaluation.py \
    --interval "${CONTINUOUS_INTERVAL}" \
    --max-runs "${CONTINUOUS_MAX_RUNS}" \
    --rate-limit-buffer 0.8 \
    --output-dir "data/continuous_evaluation" &
CONTINUOUS_PID=$!

echo "📊 Continuous analysis started (PID: $CONTINUOUS_PID)"
echo "   - Logs: data/continuous_evaluation/continuous_eval.log"
echo "   - Results: data/continuous_evaluation/"

# Wait a moment for continuous evaluation to initialize
sleep 5

# Phase 3: Start Blind Evaluation UI
echo ""
echo "🌐 Phase 3: Starting Blind Evaluation UI"
echo "========================================"
echo "Users can now participate in blind evaluations with consistent responses."

echo "🌐 Starting blind evaluation UI..."
echo "   - URL: http://localhost:${BLIND_EVAL_PORT}"
echo "   - Fixed responses ensure fair comparison"
echo "   - Press Ctrl+C to stop both processes"
echo ""

# Start Streamlit
streamlit run src/ui/main.py --server.port $BLIND_EVAL_PORT --server.headless true

# Cleanup when Streamlit is stopped
echo ""
echo "🛑 Stopping continuous analysis..."
kill $CONTINUOUS_PID 2>/dev/null

echo "✅ Research system deployment stopped"
echo "📊 Check data/continuous_evaluation/ for performance analysis results"
echo "📊 Check data/user_feedback.json for blind evaluation results" 