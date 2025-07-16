#!/bin/bash
# Research Evaluation Suite
# Runs both continuous evaluation and blind evaluation UI simultaneously

echo "🎓 Research Evaluation Suite - LLM Performance Analysis"
echo "========================================================"

# Configuration
CONTINUOUS_INTERVAL=2      # Hours between continuous evaluations (6 runs per day)
CONTINUOUS_MAX_RUNS=42     # 1 week of data collection (7 days × 6 runs/day)
BLIND_EVAL_PORT=8501       # Streamlit port for blind evaluations

echo "📊 Research Configuration:"
echo "   - Continuous evaluation: Every ${CONTINUOUS_INTERVAL}h for ${CONTINUOUS_MAX_RUNS} runs"
echo "   - Blind evaluation UI: Port ${BLIND_EVAL_PORT}"
echo "   - Both processes will run simultaneously"
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

# Start continuous evaluation in background
echo "🔄 Starting continuous evaluation in background..."
python3 scripts/continuous_evaluation.py \
    --interval "${CONTINUOUS_INTERVAL}" \
    --max-runs "${CONTINUOUS_MAX_RUNS}" \
    --rate-limit-buffer 0.8 \
    --output-dir "data/continuous_evaluation" &
CONTINUOUS_PID=$!

echo "📊 Continuous evaluation started (PID: $CONTINUOUS_PID)"
echo "   - Logs: data/continuous_evaluation/continuous_eval.log"
echo "   - Results: data/continuous_evaluation/"

# Wait a moment for continuous evaluation to initialize
sleep 5

# Start Streamlit app for blind evaluations
echo "🌐 Starting blind evaluation UI..."
echo "   - URL: http://localhost:${BLIND_EVAL_PORT}"
echo "   - Press Ctrl+C to stop both processes"
echo ""

# Start Streamlit
streamlit run src/ui/main.py --server.port $BLIND_EVAL_PORT --server.headless true

# Cleanup when Streamlit is stopped
echo ""
echo "🛑 Stopping continuous evaluation..."
kill $CONTINUOUS_PID 2>/dev/null

echo "✅ Research evaluation suite stopped"
echo "📊 Check data/continuous_evaluation/ for results" 