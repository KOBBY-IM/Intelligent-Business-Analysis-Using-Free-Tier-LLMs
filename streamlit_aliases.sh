#!/bin/bash

# Streamlit Management Aliases
# Source this file to add convenient aliases: source streamlit_aliases.sh

# Alias for starting Streamlit
alias start-llm="python start_streamlit.py"
alias start-streamlit="python start_streamlit.py"

# Alias for managing processes
alias streamlit-status="python scripts/manage_streamlit.py status"
alias streamlit-stop="python scripts/manage_streamlit.py stop"
alias streamlit-start="python scripts/manage_streamlit.py start"

# Quick launcher aliases
alias llm="python start_streamlit.py"
alias run-llm="./launch.sh"

# Port checking aliases
alias check-ports="lsof -i :8501,8502,8503,8504,8505"
alias kill-8501="lsof -t -i:8501 | xargs kill -9"

echo "✅ Streamlit aliases loaded!"
echo "Available commands:"
echo "  start-llm          - Start LLM evaluation system"
echo "  streamlit-status   - Check running processes"
echo "  streamlit-stop     - Stop all Streamlit processes"
echo "  check-ports        - Check common Streamlit ports"
echo "  llm                - Quick start (same as start-llm)" 