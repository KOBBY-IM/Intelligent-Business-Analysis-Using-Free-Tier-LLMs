"""
Blind Response Cards Component

This module provides functionality to display shuffled, anonymized response cards
for blind evaluation of LLM responses across different industries.
"""

import streamlit as st
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from .styles import UIStyles, apply_base_styles


def load_blind_responses(industry: str) -> Optional[Dict[str, Any]]:
    """
    Load blind responses for a specific industry from JSON file.
    
    Args:
        industry: The industry to load responses for ('retail', 'finance', 'healthcare')
        
    Returns:
        Dictionary containing prompt, context, and responses, or None if not found
    """
    try:
        responses_file = Path("data/blind_responses.json")
        if not responses_file.exists():
            st.error("Blind responses file not found. Please ensure data/blind_responses.json exists.")
            return None
            
        with open(responses_file, 'r') as f:
            data = json.load(f)
            
        if industry not in data:
            st.error(f"No responses found for industry: {industry}")
            return None
            
        return data[industry]
        
    except Exception as e:
        st.error(f"Error loading blind responses: {e}")
        return None


def create_blind_map(responses: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Create a mapping between blind labels (A-F) and response IDs.
    
    Args:
        responses: List of response dictionaries
        
    Returns:
        Dictionary mapping blind labels to response IDs
    """
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    response_ids = [resp['id'] for resp in responses]
    
    # Create shuffled mapping
    shuffled_ids = response_ids.copy()
    random.shuffle(shuffled_ids)
    
    return dict(zip(labels, shuffled_ids))


def render_response_card(response_id: str, response_text: str, selected: bool = False) -> None:
    """
    Render a single response card with consistent styling.
    
    Args:
        response_id: Unique identifier for the response (e.g., "A", "B", "C")
        response_text: The response text to display
        selected: Whether this card is currently selected
    """
    # Apply base styles
    apply_base_styles()
    
    # Card styling based on selection state
    card_style = "border: 2px solid #1f77b4; background: #f0f8ff;" if selected else "border: 1px solid #ddd;"
    
    # Use Streamlit's built-in components instead of HTML
    with st.container():
        st.markdown(f"#### Response {response_id}")
        st.markdown(response_text)
        st.markdown("")  # Add spacing


def render_blind_response_cards(industry: str, metrics: bool = False) -> Optional[Dict[str, str]]:
    """
    Render shuffled, anonymized response cards for blind evaluation.
    Args:
        industry: Industry to evaluate ('retail', 'finance', 'healthcare', ...)
        metrics: Whether to show metrics (for debugging)
    Returns:
        Blind mapping dictionary (label -> response_id) or None if error
    """
    # Load responses
    industry_data = load_blind_responses(industry)
    if not industry_data:
        return None
    responses = industry_data['responses']
    # Create blind mapping
    blind_map = create_blind_map(responses)
    # Store mapping in session state
    st.session_state['blind_map'] = blind_map
    # Display industry context using pure Streamlit
    st.markdown(f"## 📊 {industry.title()} Industry Evaluation")
    st.markdown(f"**Query:** {industry_data['prompt']}")
    st.markdown(f"**Context:** {industry_data['context']}")
    st.markdown("")
    
    # Instructions using pure Streamlit
    st.markdown("### 🤖 AI Response Comparison")
    st.markdown("""
    Below are 6 AI-generated responses to the business scenario above.
    Each response is labeled A-F and presented in random order to ensure unbiased evaluation.
    
    **Please review each response carefully and consider:**
    - Relevance to the business problem
    - Practicality of the recommendations
    - Clarity and completeness of the analysis
    - Overall helpfulness for decision-making
    """)
    # Display response cards as tiles (2 per row)
    st.divider()
    response_lookup = {resp['id']: resp for resp in responses}
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for i in range(0, len(labels), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(labels):
                label = labels[i + j]
                response_id = blind_map[label]
                with cols[j]:
                    render_response_card(label, response_lookup[response_id]['content'], selected=False)
    return blind_map


def get_selected_response_id(selected_label: str) -> Optional[str]:
    """
    Get the actual response ID for a selected blind label.
    
    Args:
        selected_label: The blind label selected by user (A-F)
        
    Returns:
        The actual response ID, or None if not found
    """
    blind_map = st.session_state.get('blind_map', {})
    return blind_map.get(selected_label)


def get_response_by_id(response_id: str, industry: str) -> Optional[Dict[str, Any]]:
    """
    Get response data by ID for a specific industry.
    
    Args:
        response_id: The response ID to look up
        industry: The industry context
        
    Returns:
        Response dictionary or None if not found
    """
    industry_data = load_blind_responses(industry)
    if not industry_data:
        return None
    
    for response in industry_data['responses']:
        if response['id'] == response_id:
            return response
    
    return None


def display_evaluation_summary(selected_label: str, industry: str) -> None:
    """
    Display a summary of the user's selection with response details.
    
    Args:
        selected_label: The blind label selected by user
        industry: The industry being evaluated
    """
    response_id = get_selected_response_id(selected_label)
    if not response_id:
        st.error("Could not retrieve selected response details.")
        return
    
    response = get_response_by_id(response_id, industry)
    if not response:
        st.error("Could not load response details.")
        return
    
    st.markdown("---")
    st.markdown("### 📋 Your Selection Summary")
    
    # Use Streamlit components instead of HTML
    st.success("Selected Response")
    st.markdown(f"## {selected_label}")
    st.markdown("")


# Example usage function for testing
def test_blind_cards():
    """Test function to demonstrate blind response cards."""
    st.title("🧪 Blind Response Cards Test")
    
    industry = st.selectbox(
        "Select Industry",
        ["retail", "finance", "healthcare"],
        key="test_industry"
    )
    
    show_metrics = st.checkbox("Show Debug Metrics", key="test_metrics")
    
    if st.button("Load Blind Responses"):
        # Assuming load_blind_responses now returns a list of questions
        questions = load_blind_responses(industry)
        if questions:
            # Select a random question from the list
            selected_question = random.choice(questions)
            blind_map = render_blind_response_cards(selected_question, show_metrics)
            
            if blind_map:
                st.success("Blind responses loaded successfully!")
                st.json(blind_map)
                # Example of how to get the selected response ID
                selected_label = st.selectbox(
                    "Select a Response Label (A-F)",
                    ['A', 'B', 'C', 'D', 'E', 'F'],
                    key="test_selected_label"
                )
                response_id = get_selected_response_id(selected_label)
                if response_id:
                    st.info(f"Selected Response ID: {response_id}")
                    display_evaluation_summary(selected_label, industry)
        else:
            st.error("Could not load blind responses.")


if __name__ == "__main__":
    test_blind_cards() 