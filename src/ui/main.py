#!/usr/bin/env python3
"""
LLM Business Intelligence Research Platform
Simple entry point to the comparative evaluation system.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

def apply_fluent_theme():
    """Apply basic styling for the interface."""
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        border-radius: 0.5rem;
    }
    .stSuccess {
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main Streamlit interface for LLM evaluation system."""
    
    # Page configuration
    st.set_page_config(
        page_title="LLM Business Intelligence Evaluation",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom theme
    apply_fluent_theme()
    
    # Check for API keys
    api_keys_available = check_api_keys()
    
    # Main header
    st.title("🧠 Intelligent Business Analysis Using Free-Tier LLMs")
    st.markdown("### A Comparative Framework for Multi-Industry Decision Support")
    
    # Show API key status
    if not api_keys_available:
        st.error("⚠️ **API Keys Missing**")
        st.info("""
        **Environment variables not configured:**
        - `GROQ_API_KEY`
        - `GOOGLE_API_KEY` 
        - `OPENROUTER_API_KEY`
        
        **For Streamlit Cloud:**
        1. Go to your app settings
        2. Click "Secrets" tab
        3. Add your API keys in TOML format
        4. Save and redeploy
        """)
        st.markdown("---")
    
    # Introduction
    st.markdown("""
    Welcome to our research platform comparing **Groq**, **Gemini**, and **Hugging Face** LLMs 
    across retail, finance, and healthcare domains. Your participation helps evaluate these models 
    for business intelligence applications.
    
    ---
    """)
    
    # Navigation cards
    st.subheader("🚀 Choose Your Activity")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Blind Evaluation**
        - Compare LLM responses anonymously
        - Rate model performance
        - Contribute to research data
        """)
        if st.button("Start Blind Evaluation", key="blind_eval", disabled=not api_keys_available):
            if api_keys_available:
                st.switch_page("pages/1_Blind_Evaluation.py")
    
    with col2:
        st.markdown("""
        **📈 Metrics Dashboard**
        - View performance statistics
        - Compare model accuracy
        - Analyze response quality
        """)
        if st.button("View Metrics", key="metrics", disabled=not api_keys_available):
            if api_keys_available:
                st.switch_page("pages/2_Metrics_Dashboard.py")
    
    with col3:
        st.markdown("""
        **📤 Export Results**
        - Download evaluation data
        - Access research findings
        - Generate reports
        """)
        if st.button("Export Data", key="export", disabled=not api_keys_available):
            if api_keys_available:
                st.switch_page("pages/3_Export_Results.py")
    
    # Quick system status
    st.markdown("---")
    st.markdown("### System Status")
    
    try:
        # Check fixed responses
        from pathlib import Path
        fixed_responses_file = Path("data/fixed_blind_responses.json")
        if fixed_responses_file.exists():
            st.success("✅ Fixed blind responses loaded")
        else:
            st.error("❌ Fixed blind responses not found")
            
        # Check core modules
        from src.config.config_loader import ConfigManager
        config = ConfigManager()
        st.success("✅ Configuration system loaded")
        
        # Check data files
        data_files = ["shopping_trends.csv", "Tesla_stock_data.csv"]
        for data_file in data_files:
            if Path(f"data/{data_file}").exists():
                st.success(f"✅ {data_file} available")
            else:
                st.warning(f"⚠️ {data_file} missing")
                
    except Exception as e:
        st.error(f"❌ System check failed: {str(e)}")

def check_api_keys():
    """Check if required API keys are available."""
    import os
    required_keys = ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"]
    return all(os.getenv(key) for key in required_keys)


if __name__ == "__main__":
    main() 