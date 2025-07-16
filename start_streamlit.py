#!/usr/bin/env python3
"""
Startup script for Streamlit LLM Comparison System
"""

import sys
import subprocess
import socket
import os
import signal
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def find_available_port(start_port=8501, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None


def kill_existing_streamlit_processes():
    """Kill any existing Streamlit processes on common ports"""
    common_ports = [8501, 8502, 8503, 8504, 8505]
    
    for port in common_ports:
        try:
            # Find processes using the port
            result = subprocess.run(
                ["lsof", "-t", f"-i:{port}"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        # Check if it's a streamlit process
                        proc_info = subprocess.run(
                            ["ps", "-p", pid, "-o", "comm="],
                            capture_output=True,
                            text=True
                        )
                        
                        if proc_info.returncode == 0 and 'python' in proc_info.stdout.lower():
                            print(f"🔄 Killing existing process on port {port} (PID: {pid})")
                            os.kill(int(pid), signal.SIGTERM)
                    except (ValueError, ProcessLookupError):
                        continue
        except Exception:
            continue


def main():
    """Main startup function"""
    # Set environment variables if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded environment variables from .env")

    # Kill any existing Streamlit processes
    kill_existing_streamlit_processes()

    # Find available port
    port = find_available_port()
    if port is None:
        print("❌ No available ports found in range 8501-8510")
        print("🔧 Please manually stop any running Streamlit instances")
        sys.exit(1)

    # Start Streamlit app
    try:
        print("✅ Environment setup complete")
        print(f"🚀 Starting Streamlit app on port {port}...")
        print(f"📱 Navigate to: http://localhost:{port}")
        print("🔧 Use Ctrl+C to stop the server")

        # Run Streamlit with the main app
        subprocess.run([
            "streamlit", "run", "src/ui/main.py",
            "--server.port", str(port),
            "--server.address", "localhost",
            "--server.headless", "true"
        ])

    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the project root directory")
        print("2. Check that all dependencies are installed: pip install -r requirements.txt")
        print("3. Verify your .env file has the required API keys")
        print(f"4. Try: streamlit run src/ui/main.py --server.port {port}")
        sys.exit(1)


if __name__ == "__main__":
    main()
