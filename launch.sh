#!/bin/bash

# LLM Comparison System Launcher
# Automatically finds available port and starts Streamlit

echo "🚀 LLM Comparison System Launcher"
echo "================================="

# Check if we're in the right directory
if [ ! -f "src/ui/main.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | xargs)
    echo "✅ Loaded environment variables from .env"
fi

# Function to find available port
find_port() {
    local start_port=8501
    local max_port=8510
    
    for port in $(seq $start_port $max_port); do
        if ! lsof -i :$port > /dev/null 2>&1; then
            echo $port
            return
        fi
    done
    
    echo "0"
}

# Kill existing Streamlit processes
echo "🔄 Checking for existing Streamlit processes..."
for port in 8501 8502 8503 8504 8505; do
    pid=$(lsof -t -i:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo "🔄 Killing process on port $port (PID: $pid)"
        kill $pid 2>/dev/null
        sleep 1
    fi
done

# Find available port
port=$(find_port)
if [ "$port" = "0" ]; then
    echo "❌ No available ports found in range 8501-8510"
    echo "🔧 Please manually stop any running Streamlit instances"
    exit 1
fi

echo "✅ Found available port: $port"
echo "🚀 Starting Streamlit app..."
echo "📱 Navigate to: http://localhost:$port"
echo "🔧 Use Ctrl+C to stop the server"
echo ""

# Start Streamlit
streamlit run src/ui/main.py \
    --server.port $port \
    --server.address localhost \
    --server.headless true

echo "🛑 Streamlit app stopped" 