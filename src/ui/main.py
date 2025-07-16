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
    
    # Initialize session state for navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🧠 LLM Evaluation")
        
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_page = "home"
            
        if st.button("📊 Blind Evaluation", use_container_width=True, disabled=not api_keys_available):
            st.session_state.current_page = "blind_eval"
            
        if st.button("📈 Metrics Dashboard", use_container_width=True, disabled=not api_keys_available):
            st.session_state.current_page = "metrics"
            
        if st.button("📤 Export Results", use_container_width=True, disabled=not api_keys_available):
            st.session_state.current_page = "export"
    
    # Main content based on current page
    if st.session_state.current_page == "home":
        show_home_page(api_keys_available)
    elif st.session_state.current_page == "blind_eval":
        show_blind_evaluation()
    elif st.session_state.current_page == "metrics":
        show_metrics_dashboard()
    elif st.session_state.current_page == "export":
        show_export_page()

def show_home_page(api_keys_available):
    """Display the home page content."""
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
        if st.button("Start Blind Evaluation", key="blind_eval_home", disabled=not api_keys_available):
            if api_keys_available:
                st.session_state.current_page = "blind_eval"
                st.rerun()
    
    with col2:
        st.markdown("""
        **📈 Metrics Dashboard**
        - View performance statistics
        - Compare model accuracy
        - Analyze response quality
        """)
        if st.button("View Metrics", key="metrics_home", disabled=not api_keys_available):
            if api_keys_available:
                st.session_state.current_page = "metrics"
                st.rerun()
    
    with col3:
        st.markdown("""
        **📤 Export Results**
        - Download evaluation data
        - Access research findings
        - Generate reports
        """)
        if st.button("Export Data", key="export_home", disabled=not api_keys_available):
            if api_keys_available:
                st.session_state.current_page = "export"
                st.rerun()
    
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

def show_blind_evaluation():
    """Display the blind evaluation page."""
    st.title("📊 Blind Evaluation")
    
    try:
        # Import and run the blind evaluation
        import sys
        from pathlib import Path
        
        # Add src to path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root / "src"))
        
        # Import the blind evaluation components
        from src.rag.blind_test_generator import BlindTestGenerator
        from src.utils.question_sampler import QuestionSampler
        from src.llm_providers.provider_manager import ProviderManager
        
        st.info("Loading blind evaluation interface...")
        
        # Load fixed responses
        import json
        fixed_responses_file = Path("data/fixed_blind_responses.json")
        
        if not fixed_responses_file.exists():
            st.error("❌ Fixed blind responses not found!")
            st.info("Please generate fixed responses first using the setup script.")
            return
            
        with open(fixed_responses_file, 'r') as f:
            fixed_data = json.load(f)
            
        st.success(f"✅ Loaded {len(fixed_data['responses'])} fixed responses")
        
        # Simple evaluation interface
        st.markdown("### Compare LLM Responses")
        
        # Select a random question
        questions = list(fixed_data['responses'].keys())
        if "selected_question" not in st.session_state:
            import random
            st.session_state.selected_question = random.choice(questions)
            
        question_data = fixed_data['responses'][st.session_state.selected_question]
        
        st.markdown(f"**Question:** {question_data['question']}")
        st.markdown(f"**Domain:** {question_data['domain'].title()}")
        
        # Show responses
        responses = question_data['llm_responses']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Response A")
            st.write(responses['groq']['response'])
            
        with col2:
            st.markdown("#### Response B") 
            st.write(responses['gemini']['response'])
            
        with col3:
            st.markdown("#### Response C")
            st.write(responses['openrouter']['response'])
            
        # Rating interface
        st.markdown("---")
        st.markdown("### Rate the Responses")
        
        rating = st.radio(
            "Which response is most helpful for business decision-making?",
            ["Response A", "Response B", "Response C"],
            key="response_rating"
        )
        
        if st.button("Submit Rating", key="submit_rating"):
            st.success(f"✅ Thank you! You selected {rating}")
            # Here you could save the rating to a file
            
            # Load next question
            st.session_state.selected_question = random.choice(questions)
            st.rerun()
            
        if st.button("Next Question", key="next_question"):
            import random
            st.session_state.selected_question = random.choice(questions)
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error loading blind evaluation: {str(e)}")
        st.info("Please check that all required files and modules are available.")

def show_metrics_dashboard():
    """Display the metrics dashboard."""
    st.title("📈 Metrics Dashboard")
    st.info("Metrics dashboard functionality will be implemented here.")
    
    # Basic placeholder content
    st.markdown("### Performance Overview")
    st.markdown("This section will show:")
    st.markdown("- Response quality metrics")
    st.markdown("- User preference statistics") 
    st.markdown("- Model comparison charts")

def show_export_page():
    """Display the export page."""
    st.title("📤 Export Results")
    st.info("Export functionality will be implemented here.")
    
    # Basic placeholder content
    st.markdown("### Available Exports")
    st.markdown("This section will provide:")
    st.markdown("- CSV download of evaluation results")
    st.markdown("- Summary reports")
    st.markdown("- Statistical analysis files")

def check_api_keys():
    """Check if required API keys are available."""
    import os
    required_keys = ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"]
    return all(os.getenv(key) for key in required_keys)


if __name__ == "__main__":
    main() 