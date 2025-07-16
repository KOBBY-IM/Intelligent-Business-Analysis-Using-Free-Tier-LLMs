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

st.set_page_config(
    page_title="LLM Business Intelligence Research",
    page_icon="🧠",
    layout="wide"
)

def main():
    """Main application entry point."""
    
    st.title("🧠 LLM Business Intelligence Research")
    st.subheader("Comparative Framework for Multi-Industry Decision Support")
    
    st.markdown("""
    ## Welcome to the Research Platform
    
    This system evaluates **free-tier Large Language Models** across business domains:
    - **Retail**: Customer behavior and sales analysis
    - **Finance**: Market trends and investment insights
    
    ### Available Tools:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Blind Evaluation**
        - Compare LLM responses anonymously
        - Rate model performance
        - Contribute to research data
        """)
        if st.button("Start Blind Evaluation", key="blind_eval"):
            st.switch_page("pages/1_Blind_Evaluation.py")
    
    with col2:
        st.markdown("""
        **📈 Metrics Dashboard**
        - View performance statistics
        - Compare model accuracy
        - Analyze response quality
        """)
        if st.button("View Metrics", key="metrics"):
            st.switch_page("pages/2_Metrics_Dashboard.py")
    
    with col3:
        st.markdown("""
        **📤 Export Results**
        - Download evaluation data
        - Access research findings
        - Generate reports
        """)
        if st.button("Export Data", key="export"):
            st.switch_page("pages/3_Export_Results.py")
    
    # Quick system status
    st.markdown("---")
    st.markdown("### System Status")
    
    try:
        from llm_providers.provider_manager import ProviderManager
        pm = ProviderManager()
        all_providers = pm.get_all_providers()
        total_models = sum(len(provider.models) for provider in all_providers.values())
        st.success(f"✅ {len(all_providers)} providers, {total_models} models available")
    except Exception as e:
        st.error(f"⚠️ System check failed: {str(e)}")
        
    # Research info
    with st.expander("ℹ️ About This Research"):
        st.markdown("""
        **Objective**: Compare the effectiveness of free-tier LLMs for business intelligence tasks.
        
        **Models Evaluated**:
        - Groq (Mixtral, Llama)
        - Google Gemini 
        - OpenRouter models
        
        **Evaluation Criteria**:
        - Accuracy and relevance
        - Response coherence
        - Processing speed
        - Domain-specific insights
        
        **Ethics**: All data is anonymized. Participation is voluntary.
        """)


if __name__ == "__main__":
    main() 