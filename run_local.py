#!/usr/bin/env python3
"""
Local Development Server - Simple Version
Run this to start the app locally with the same features as Streamlit Cloud
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start local development server"""
    
    print("🎯 LLM Business Intelligence - Local Development")
    print("=" * 50)
    
    # Ensure we're in the project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check if required files exist
    required_files = [
        "data/enhanced_blind_responses.json",
        "data/ground_truth_answers.json", 
        "src/ui/main.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure all required files are present.")
        return
    
    print("✅ All required files found")
    print("🚀 Starting Streamlit server...")
    print("🌐 App will be available at: http://localhost:8501")
    print("📊 Features available:")
    print("   • Enhanced RAG with 8-chunk coverage")
    print("   • 3 LLM providers (Groq, Gemini, OpenRouter)")
    print("   • Ground truth business answers")
    print("   • Side-by-side response comparison")
    print("   • User registration and feedback")
    print("\n" + "=" * 50)
    print("Press Ctrl+C to stop the server")
    print("=" * 50 + "\n")
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/ui/main.py",
            "--server.port=8501"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 