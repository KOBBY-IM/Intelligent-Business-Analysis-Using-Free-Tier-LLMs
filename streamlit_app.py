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
        
        1. **Generate fixed responses locally first:**
           ```bash
           python3 scripts/generate_fixed_blind_responses.py
           ```
        
        2. **Upload the generated file to Streamlit Cloud:**
           - Upload `data/fixed_blind_responses.json` to your Streamlit Cloud app
           - Place it in the `data/` directory
        
        3. **Redeploy your app**
        
        **Note:** Fixed responses ensure all users see the same LLM responses for fair evaluation.
        """)
        return
    
    # Import and run the main UI
    try:
        from src.ui.pages.blind_evaluation import main as blind_eval_main
        blind_eval_main()
    except ImportError:
        # Fallback to direct import
        try:
            from src.ui.main import main as ui_main
            ui_main()
        except ImportError as e:
            st.error(f"❌ Import error: {str(e)}")
            st.info("Please check that all required modules are available.")

if __name__ == "__main__":
    main() 