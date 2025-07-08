#!/usr/bin/env python3
"""
Startup script for Streamlit LLM Comparison System
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    # Set environment variables if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv()
        print("✅ Loaded environment variables from .env")

    # Import and run the main app
    try:
        from ui.main_app import main

        print("✅ Main app imported successfully")
        print("🚀 Starting Streamlit app...")
        print("📱 Navigate to: http://localhost:8501")
        print("🔧 Use Ctrl+C to stop the server")

        # Run the main function
        main()

    except Exception as e:
        print(f"❌ Error starting app: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the project root directory")
        print(
            "2. Check that all dependencies are installed: pip install -r requirements.txt"
        )
        print("3. Verify your .env file has the required API keys")
        print("4. Try: streamlit run src/ui/main_app.py")
        sys.exit(1)
