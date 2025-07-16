# Port Management Solution

## Problem Solved

The original issue was that port 8501 was frequently occupied by existing Streamlit processes, causing startup failures. This created a poor user experience and required manual intervention.

## Solution Overview

We implemented a comprehensive port management system that:

1. **Automatically detects available ports** (8501-8510)
2. **Kills existing conflicting processes** gracefully
3. **Provides multiple launcher options** for different preferences
4. **Includes process management utilities** for debugging

## Implementation Details

### 1. Enhanced Python Launcher (`start_streamlit.py`)

**Features:**
- Scans ports 8501-8510 for availability
- Automatically kills existing Streamlit processes
- Provides clear feedback on port selection
- Graceful error handling and troubleshooting tips

**Usage:**
```bash
python start_streamlit.py
```

### 2. Shell Script Launcher (`launch.sh`)

**Features:**
- Bash-native implementation
- Same port detection and cleanup
- Works well in Unix/Linux environments
- Lightweight alternative to Python launcher

**Usage:**
```bash
./launch.sh
```

### 3. Process Manager (`scripts/manage_streamlit.py`)

**Features:**
- View status of all Streamlit processes
- Stop all Streamlit processes cleanly
- Start using the main launcher
- Detailed process information

**Usage:**
```bash
# Check what's running
python scripts/manage_streamlit.py status

# Stop all processes
python scripts/manage_streamlit.py stop

# Start fresh
python scripts/manage_streamlit.py start
```

### 4. Convenience Aliases (`streamlit_aliases.sh`)

**Features:**
- Short commands for common operations
- Easy to remember aliases
- Quick troubleshooting commands

**Usage:**
```bash
# Load aliases
source streamlit_aliases.sh

# Use convenient commands
start-llm
streamlit-status
streamlit-stop
```

## Port Detection Algorithm

The system uses the following logic:

1. **Port Scanning**: Test ports 8501-8510 using socket binding
2. **Process Identification**: Use `lsof` to find processes using ports
3. **Graceful Termination**: Send SIGTERM to conflicting processes
4. **Verification**: Confirm port availability before starting
5. **Fallback**: Provide clear error messages if no ports available

## Security Considerations

- Only kills processes on known Streamlit ports (8501-8505)
- Verifies processes are Python/Streamlit before termination
- Uses graceful SIGTERM instead of forceful SIGKILL
- Provides process information for transparency

## Troubleshooting

### Common Issues

**Port still in use after cleanup:**
```bash
# Manual port check
lsof -i :8501

# Force kill if needed
kill -9 <PID>
```

**Permission denied errors:**
```bash
# Check if process belongs to current user
ps aux | grep streamlit

# Use sudo if needed (not recommended)
sudo kill <PID>
```

**No available ports:**
```bash
# Check all common ports
python scripts/manage_streamlit.py status

# Manually stop processes
python scripts/manage_streamlit.py stop
```

## Benefits

1. **Zero-friction startup**: No manual port management needed
2. **Automatic cleanup**: Handles stale processes automatically
3. **Multiple options**: Choose the launcher that fits your workflow
4. **Debugging tools**: Easy to diagnose and fix issues
5. **Production ready**: Robust error handling and logging

## Future Enhancements

- **Docker integration**: Port mapping for containerized deployments
- **Load balancing**: Multiple instances on different ports
- **Health checks**: Automatic restart on failure
- **Monitoring**: Integration with system monitoring tools

## Configuration

The port management system can be configured via environment variables:

```bash
# .env file
STREAMLIT_PORT_START=8501
STREAMLIT_PORT_END=8510
STREAMLIT_AUTO_KILL=true
STREAMLIT_HEADLESS=true
```

This solution ensures that the LLM Evaluation System can start reliably regardless of existing processes or port conflicts. 