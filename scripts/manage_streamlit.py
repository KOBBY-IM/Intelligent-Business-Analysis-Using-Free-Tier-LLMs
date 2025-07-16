#!/usr/bin/env python3
"""
Streamlit Process Manager
Utility to start, stop, and check status of Streamlit processes
"""

import subprocess
import sys
import signal
import os
from pathlib import Path


def get_streamlit_processes():
    """Get all running Streamlit processes"""
    try:
        # Find all python processes
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True
        )
        
        processes = []
        for line in result.stdout.split('\n'):
            if 'streamlit' in line.lower() and 'python' in line.lower():
                parts = line.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    command = ' '.join(parts[10:])
                    processes.append({
                        'pid': pid,
                        'command': command
                    })
        
        return processes
    except Exception as e:
        print(f"Error getting processes: {e}")
        return []


def get_port_processes():
    """Get processes using common Streamlit ports"""
    ports = [8501, 8502, 8503, 8504, 8505]
    port_processes = {}
    
    for port in ports:
        try:
            result = subprocess.run(
                ["lsof", "-t", f"-i:{port}"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                port_processes[port] = pids
        except Exception:
            continue
    
    return port_processes


def stop_all_streamlit():
    """Stop all running Streamlit processes"""
    processes = get_streamlit_processes()
    port_processes = get_port_processes()
    
    stopped_count = 0
    
    # Stop Streamlit processes
    for proc in processes:
        try:
            pid = int(proc['pid'])
            os.kill(pid, signal.SIGTERM)
            print(f"✅ Stopped Streamlit process (PID: {pid})")
            stopped_count += 1
        except (ValueError, ProcessLookupError):
            continue
    
    # Stop processes on common ports
    for port, pids in port_processes.items():
        for pid in pids:
            try:
                pid_int = int(pid)
                os.kill(pid_int, signal.SIGTERM)
                print(f"✅ Stopped process on port {port} (PID: {pid})")
                stopped_count += 1
            except (ValueError, ProcessLookupError):
                continue
    
    if stopped_count == 0:
        print("ℹ️  No Streamlit processes found to stop")
    else:
        print(f"🛑 Stopped {stopped_count} processes")


def show_status():
    """Show status of Streamlit processes"""
    processes = get_streamlit_processes()
    port_processes = get_port_processes()
    
    print("📊 Streamlit Process Status")
    print("=" * 50)
    
    if processes:
        print("\n🔄 Running Streamlit Processes:")
        for proc in processes:
            print(f"  PID: {proc['pid']}")
            print(f"  Command: {proc['command']}")
            print()
    else:
        print("\n✅ No Streamlit processes found")
    
    if port_processes:
        print("\n🔌 Ports in use:")
        for port, pids in port_processes.items():
            print(f"  Port {port}: PIDs {', '.join(pids)}")
    else:
        print("\n✅ No processes found on common Streamlit ports")


def start_streamlit():
    """Start Streamlit using the main launcher"""
    project_root = Path(__file__).parent.parent
    launcher_script = project_root / "start_streamlit.py"
    
    if not launcher_script.exists():
        print("❌ start_streamlit.py not found")
        return
    
    print("🚀 Starting Streamlit using launcher...")
    try:
        subprocess.run([sys.executable, str(launcher_script)])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit startup cancelled")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python manage_streamlit.py [start|stop|status]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "start":
        start_streamlit()
    elif command == "stop":
        stop_all_streamlit()
    elif command == "status":
        show_status()
    else:
        print("Invalid command. Use: start, stop, or status")
        sys.exit(1)


if __name__ == "__main__":
    main() 