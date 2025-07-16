"""
User Feedback Form Component

This module provides functionality to collect user feedback on blind evaluation
responses, including response selection and comments.
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from .styles import UIStyles, apply_base_styles


def save_user_feedback(
    industry: str,
    selected_label: str,
    comment: str,
    blind_map: Dict[str, str],
    prompt: str
) -> bool:
    """
    Save user feedback to JSON file with all required metadata.
    
    Args:
        industry: The industry being evaluated
        selected_label: The blind label selected by user (A-F)
        comment: User's comment/feedback
        blind_map: Mapping of blind labels to response IDs
        prompt: The original prompt/question
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Create feedback data structure
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": st.session_state.get("session_id", "unknown"),
            "industry": industry,
            "prompt": prompt,
            "selected_label": selected_label,
            "selected_response_id": blind_map.get(selected_label, "unknown"),
            "comment": comment,
            "blind_map": blind_map,
            "step": st.session_state.get("step", 0)
        }
        
        # Create results directory if it doesn't exist
        results_dir = Path("data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        feedback_file = results_dir / "user_feedback.json"
        
        # Load existing feedback or create new list
        if feedback_file.exists():
            with open(feedback_file, 'r') as f:
                existing_feedback = json.load(f)
        else:
            existing_feedback = []
        
        existing_feedback.append(feedback_data)
        
        # Save updated feedback
        with open(feedback_file, 'w') as f:
            json.dump(existing_feedback, f, indent=2)
        
        return True
        
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return False


def render_feedback_form(
    industry: str,
    blind_map: Dict[str, str],
    prompt: str,
    on_submit_callback: Optional[callable] = None
) -> bool:
    """
    Render the user feedback form with consistent styling.
    
    Args:
        industry: The industry context
        blind_map: Mapping of labels to response IDs
        prompt: The question prompt
        on_submit_callback: Optional callback function to execute on successful submission
        
    Returns:
        bool: True if form was successfully submitted, False otherwise
    """
    # Apply base styles
    apply_base_styles()
    
    # Form header
    UIStyles.render_header(
        title="Select Your Preferred Response",
        subtitle=f"Industry: {industry.title()}",
        icon="📝",
        color="primary"
    )
    
    # Response selection
    st.markdown(
        """
        **Which response do you think is most helpful for this business scenario?**
        
        Please consider:
        - How well the response addresses the business problem
        - The practicality and feasibility of the recommendations
        - The clarity and completeness of the analysis
        - Overall usefulness for decision-making
        """
    )
    
    # Radio buttons for response selection
    selected_label = st.radio(
        "Select the best response:",
        options=['A', 'B', 'C', 'D', 'E', 'F'],
        key=f"response_selection_{industry}",
        horizontal=True,
        format_func=lambda x: f"Response {x}"
    )
    
    st.markdown("---")
    
    # Comment section
    st.markdown(
        """
        **Please provide your reasoning (optional but helpful):**
        
        Tell us why you selected this response. What made it stand out?
        """
    )
    
    comment = st.text_area(
        "Your feedback:",
        placeholder="I selected this response because...",
        key=f"user_comment_{industry}",
        height=120,
        help="Share your thoughts on why this response is most helpful"
    )
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button(
            "✅ Submit Evaluation",
            type="primary",
            use_container_width=True,
            key=f"submit_{industry}"
        ):
            if selected_label:
                # Save feedback
                success = save_user_feedback(
                    industry=industry,
                    selected_label=selected_label,
                    comment=comment,
                    blind_map=blind_map,
                    prompt=prompt
                )
                
                if success:
                    st.success("✅ Evaluation submitted successfully!")
                    
                    # Store selection in session state
                    st.session_state[f"selected_{industry}"] = selected_label
                    st.session_state[f"comment_{industry}"] = comment
                    
                    # Execute callback if provided
                    if on_submit_callback:
                        on_submit_callback()
                    
                    return True
                else:
                    st.error("❌ Failed to save evaluation. Please try again.")
                    return False
            else:
                st.error("Please select a response before submitting.")
                return False
    
    return False


def display_feedback_summary(industry: str) -> None:
    """
    Display a summary of the user's feedback for a specific industry.
    
    Args:
        industry: The industry to display summary for
    """
    selected_label = st.session_state.get(f"selected_{industry}")
    comment = st.session_state.get(f"comment_{industry}")
    
    if selected_label:
        st.markdown("---")
        st.markdown(f"### 📋 {industry.title()} Evaluation Summary")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(
                f"""
                <div style="
                    background: #e3f2fd;
                    border: 2px solid #2196f3;
                    border-radius: 8px;
                    padding: 1rem;
                    text-align: center;
                ">
                    <h3 style="margin: 0; color: #1976d2;">Selected</h3>
                    <h2 style="margin: 0.5rem 0; color: #1976d2; font-size: 2.5rem;">{selected_label}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            if comment:
                st.markdown("**Your Feedback:**")
                st.markdown(f"> {comment}")
            else:
                st.info("No comment provided")


def get_industry_progress() -> Dict[str, bool]:
    """
    Get the progress status for each industry evaluation.
    
    Returns:
        Dictionary mapping industry names to completion status
    """
    industries = ["retail", "finance", "healthcare"]
    progress = {}
    
    for industry in industries:
        progress[industry] = f"selected_{industry}" in st.session_state
    
    return progress


def render_progress_indicator() -> None:
    """
    Render a visual progress indicator for industry evaluations.
    """
    progress = get_industry_progress()
    total_industries = len(progress)
    completed = sum(progress.values())
    
    st.markdown("---")
    st.markdown("### 📊 Evaluation Progress")
    
    # Progress bar
    progress_percentage = completed / total_industries
    st.progress(progress_percentage, text=f"{completed}/{total_industries} industries completed")
    
    # Industry status
    col1, col2, col3 = st.columns(3)
    
    industries = list(progress.keys())
    for i, industry in enumerate(industries):
        with [col1, col2, col3][i]:
            if progress[industry]:
                st.markdown(
                    f"""
                    <div style="
                        background: #e8f5e8;
                        border: 2px solid #4caf50;
                        border-radius: 8px;
                        padding: 0.5rem;
                        text-align: center;
                    ">
                        <strong>✅ {industry.title()}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        background: #fff3e0;
                        border: 2px solid #ff9800;
                        border-radius: 8px;
                        padding: 0.5rem;
                        text-align: center;
                    ">
                        <strong>⏳ {industry.title()}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def validate_feedback_form(selected_label: str, comment: str) -> List[str]:
    """
    Validate user feedback form inputs.
    
    Args:
        selected_label: The selected response label
        comment: User's comment
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if not selected_label:
        errors.append("Please select a response")
    
    if comment and len(comment.strip()) < 10:
        errors.append("Comment should be at least 10 characters long")
    
    if comment and len(comment) > 1000:
        errors.append("Comment should be less than 1000 characters")
    
    return errors


def render_industry_evaluation_page(industry: str) -> bool:
    """
    Render a complete industry evaluation page with feedback form.
    
    Args:
        industry: The industry to evaluate
        
    Returns:
        True if evaluation was completed, False otherwise
    """
    from .blind_response_cards import render_blind_response_cards, load_blind_responses
    
    # Load industry data
    industry_data = load_blind_responses(industry)
    if not industry_data:
        return False
    
    # Render blind response cards
    blind_map = render_blind_response_cards(industry)
    if not blind_map:
        return False
    
    # Render feedback form
    def on_submit():
        # Don't change step here - let the main evaluation function handle progression
        st.rerun()
    
    submitted = render_feedback_form(
        industry=industry,
        blind_map=blind_map,
        prompt=industry_data['prompt'],
        on_submit_callback=on_submit
    )
    
    if submitted:
        # Display summary
        display_feedback_summary(industry)
        
        # Show progress
        render_progress_indicator()
        
        return True
    
    return False


# Test function for development
def test_feedback_form():
    """Test function to demonstrate feedback form functionality."""
    st.title("🧪 User Feedback Form Test")
    
    # Mock data
    industry = "retail"
    blind_map = {
        'A': 'retail_response_1',
        'B': 'retail_response_2', 
        'C': 'retail_response_3',
        'D': 'retail_response_4',
        'E': 'retail_response_5',
        'F': 'retail_response_6'
    }
    prompt = "Analyze customer purchase patterns and recommend inventory optimization strategies."
    
    st.markdown("### Test Feedback Form")
    
    submitted = render_feedback_form(
        industry=industry,
        blind_map=blind_map,
        prompt=prompt
    )
    
    if submitted:
        st.success("Form submitted successfully!")
        st.json(st.session_state)


if __name__ == "__main__":
    test_feedback_form() 