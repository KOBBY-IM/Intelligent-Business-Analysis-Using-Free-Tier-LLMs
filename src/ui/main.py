#!/usr/bin/env python3
"""
LLM Business Intelligence Research Platform
Simple entry point to the comparative evaluation system.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

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
    
    status_checks = []
    
    # Check fixed responses
    try:
        from pathlib import Path
        fixed_responses_file = Path("data/fixed_blind_responses.json")
        if fixed_responses_file.exists():
            status_checks.append(("✅", "Fixed blind responses loaded"))
        else:
            status_checks.append(("❌", "Fixed blind responses not found"))
    except Exception as e:
        status_checks.append(("⚠️", f"Could not check fixed responses: {str(e)}"))
            
    # Check core modules
    try:
        from src.config.config_loader import ConfigLoader
        config = ConfigLoader()
        status_checks.append(("✅", "Configuration system loaded"))
    except Exception as e:
        status_checks.append(("⚠️", f"Configuration system issue: {str(e)}"))
        
    # Check data files
    try:
        data_files = ["shopping_trends.csv", "Tesla_stock_data.csv"]
        for data_file in data_files:
            if Path(f"data/{data_file}").exists():
                status_checks.append(("✅", f"{data_file} available"))
            else:
                status_checks.append(("⚠️", f"{data_file} missing"))
    except Exception as e:
        status_checks.append(("⚠️", f"Could not check data files: {str(e)}"))
        
    # Display all status checks
    for icon, message in status_checks:
        if icon == "✅":
            st.success(f"{icon} {message}")
        elif icon == "❌":
            st.error(f"{icon} {message}")
        else:
            st.warning(f"{icon} {message}")

def show_blind_evaluation():
    """Display the blind evaluation page."""
    st.title("📊 Blind Evaluation")
    
    try:
        # Load fixed responses directly without complex imports
        import json
        import random
        from pathlib import Path
        
        st.info("Loading blind evaluation interface...")
        
        # Load fixed responses
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
            st.session_state.selected_question = random.choice(questions)
            
        question_data = fixed_data['responses'][st.session_state.selected_question]
        
        # Display question info
        st.markdown(f"**Question:** {question_data['question']}")
        st.markdown(f"**Domain:** {question_data['domain'].title()}")
        
        # Show RAG context in expander
        with st.expander("📋 Business Context (Click to view)"):
            st.text(question_data['rag_context'])
        
        # Show responses in randomized order
        responses = question_data['llm_responses']
        provider_names = list(responses.keys())
        
        # Randomize provider order for blind evaluation
        if "provider_order" not in st.session_state:
            st.session_state.provider_order = provider_names.copy()
            random.shuffle(st.session_state.provider_order)
        
        # Display responses
        col1, col2, col3 = st.columns(3)
        
        response_labels = ["Response A", "Response B", "Response C"]
        
        for i, (col, label) in enumerate(zip([col1, col2, col3], response_labels)):
            provider = st.session_state.provider_order[i]
            response_data = responses[provider]
            
            with col:
                st.markdown(f"#### {label}")
                st.write(response_data['response'])
                
                # Show metadata in small text
                st.caption(f"Length: {len(response_data['response'])} chars | "
                          f"Tokens: {response_data['metadata'].get('token_count', 'N/A')}")
        
        # Rating interface
        st.markdown("---")
        st.markdown("### Rate the Responses")
        
        rating = st.radio(
            "Which response is most helpful for business decision-making?",
            response_labels,
            key="response_rating"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Submit Rating", key="submit_rating", type="primary"):
                # Map response to actual provider
                selected_index = response_labels.index(rating)
                actual_provider = st.session_state.provider_order[selected_index]
                
                st.success(f"✅ Thank you! You selected {rating}")
                st.info(f"You chose the response from: **{actual_provider.title()}**")
                
                # Save rating (could be expanded to save to file)
                if "ratings" not in st.session_state:
                    st.session_state.ratings = []
                
                st.session_state.ratings.append({
                    "question_id": st.session_state.selected_question,
                    "selected_response": rating,
                    "actual_provider": actual_provider,
                    "timestamp": str(datetime.now())
                })
                
                st.write(f"Total evaluations completed: {len(st.session_state.ratings)}")
                
        with col2:
            if st.button("Next Question", key="next_question"):
                # Load next question and randomize provider order
                st.session_state.selected_question = random.choice(questions)
                st.session_state.provider_order = provider_names.copy()
                random.shuffle(st.session_state.provider_order)
                st.rerun()
                
        with col3:
            if st.button("Show Statistics", key="show_stats"):
                if "ratings" in st.session_state and st.session_state.ratings:
                    st.subheader("Your Evaluation Statistics")
                    
                    # Count provider preferences
                    provider_counts = {}
                    for rating in st.session_state.ratings:
                        provider = rating["actual_provider"]
                        provider_counts[provider] = provider_counts.get(provider, 0) + 1
                    
                    for provider, count in provider_counts.items():
                        percentage = (count / len(st.session_state.ratings)) * 100
                        st.write(f"**{provider.title()}**: {count} selections ({percentage:.1f}%)")
                else:
                    st.info("Complete some evaluations first to see statistics!")
            
    except Exception as e:
        st.error(f"❌ Error loading blind evaluation: {str(e)}")
        st.info("The evaluation system is using a simplified approach to avoid import issues.")
        
        # Show debug information
        with st.expander("Debug Information"):
            st.write(f"Error details: {str(e)}")
            st.write(f"Error type: {type(e).__name__}")
            
            # Check file existence
            import os
            st.write(f"Current directory: {os.getcwd()}")
            st.write(f"Data directory exists: {os.path.exists('data')}")
            if os.path.exists('data'):
                st.write(f"Data files: {os.listdir('data')}")

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