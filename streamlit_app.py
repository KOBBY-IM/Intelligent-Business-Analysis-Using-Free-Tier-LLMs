#!/usr/bin/env python3
"""
Streamlit Cloud Entry Point
Main entry point for deploying the LLM evaluation system on Streamlit Cloud.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Set page config for Streamlit Cloud
st.set_page_config(
    page_title="LLM Business Analysis Evaluation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main entry point for Streamlit Cloud deployment."""
    
    # Check if fixed responses exist
    fixed_responses_file = Path("data/fixed_blind_responses.json")
    if not fixed_responses_file.exists():
        st.error("❌ Fixed blind responses not found!")
        st.info("""
        **For Streamlit Cloud deployment:**
        
        Generate fixed responses locally first:
        ```bash
        python3 scripts/generate_fixed_blind_responses.py
        ```
        
        Upload the generated file to Streamlit Cloud:
        1. Upload `data/fixed_blind_responses.json` to your Streamlit Cloud app
        2. Place it in the `data/` directory
        3. Redeploy your app
        
        **Note:** Fixed responses ensure all users see the same LLM responses for fair evaluation.
        """)
        return
    
    # Import and run the main UI
    try:
        from src.ui.main import main as ui_main
        ui_main()
    except ImportError as e:
        st.error(f"❌ Import error: {str(e)}")
        st.info("Please check that all required modules are available.")
        
        # Show debug information
        st.subheader("Debug Information")
        st.write(f"Current working directory: {Path.cwd()}")
        st.write(f"Python path: {sys.path}")
        
        # Show available files
        try:
            st.write("Available files in src/:")
            src_files = list((project_root / "src").rglob("*.py"))
            for file in src_files[:20]:  # Show first 20 files
                st.write(f"- {file}")
        except Exception as ex:
            st.write(f"Could not list src files: {ex}")

if __name__ == "__main__":
    main() 