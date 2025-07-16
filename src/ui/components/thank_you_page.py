#!/usr/bin/env python3
"""
Thank You Page Component

Displays a clean post-submission screen with friendly success message
and optional external link for further feedback.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from utils.feedback_logger import FeedbackLogger, create_session_id


def render_thank_you_page(
    session_id: str,
    completed_industries: List[str],
    user_selections: Optional[Dict[str, str]] = None,
    total_time_seconds: Optional[float] = None,
    show_further_feedback_link: bool = True
) -> None:
    """
    Render the thank you page with completion summary.
    
    Args:
        session_id: Unique session identifier
        completed_industries: List of industries that were evaluated
        user_selections: Dictionary of user selections by industry
        total_time_seconds: Total time spent on evaluation
        show_further_feedback_link: Whether to show external feedback link
    """
    
    # Log session completion
    logger = FeedbackLogger()
    logger.log_session_completion(
        session_id=session_id,
        completed_industries=completed_industries,
        total_time_seconds=total_time_seconds
    )
    
    # Main thank you message
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 0;">
            <h1 style="color: #0078d4; margin-bottom: 1rem;">🎉 Thank You!</h1>
            <p style="font-size: 1.2rem; color: #201f1e; margin-bottom: 2rem;">
                Your participation in our research study is greatly appreciated.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Completion summary
    st.markdown("### 📊 Evaluation Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Industries Evaluated",
            value=len(completed_industries),
            delta=f"{len(completed_industries)}/3 completed"
        )
        
        if total_time_seconds:
            minutes = int(total_time_seconds // 60)
            seconds = int(total_time_seconds % 60)
            st.metric(
                label="Total Time",
                value=f"{minutes}m {seconds}s",
                delta="Time spent on evaluation"
            )
    
    with col2:
        st.metric(
            label="Session ID",
            value=session_id[:12] + "...",
            delta="Your unique session identifier"
        )
        
        st.metric(
            label="Data Saved",
            value="✅ Secure",
            delta="All responses logged successfully"
        )
    
    # User selections summary
    if user_selections:
        st.markdown("### 📋 Your Selections")
        
        for industry in completed_industries:
            selection = user_selections.get(industry, "Not selected")
            st.markdown(
                f"""
                <div style="
                    background: #f8f9fa;
                    padding: 1rem;
                    border-radius: 8px;
                    margin: 0.5rem 0;
                    border-left: 4px solid #0078d4;
                ">
                    <strong>{industry.title()}:</strong> Response {selection}
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Research impact section
    st.markdown("### 🔬 Research Impact")
    
    st.markdown(
        """
        Your feedback contributes to:
        
        - **Understanding LLM Performance**: How different AI models perform in real business scenarios
        - **Improving Decision Support**: Better AI tools for business intelligence
        - **Cross-Industry Analysis**: Performance comparison across retail, finance, and healthcare
        - **Academic Research**: Advancing the field of AI evaluation methodologies
        
        Your responses will be analyzed alongside other participants to identify patterns
        and insights that can improve AI systems for business applications.
        """
    )
    
    # Data privacy and next steps
    st.markdown("### 🔒 Data Privacy & Next Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            **Data Protection:**
            - All responses are anonymized
            - No personal information collected
            - Data stored securely on local servers
            - Used only for academic research
            """
        )
    
    with col2:
        st.markdown(
            """
            **What Happens Next:**
            - Research team analyzes the data
            - Results published in academic format
            - No personal information shared
            - Study contributes to AI research
            """
        )
    
    # External feedback link (optional)
    if show_further_feedback_link:
        st.markdown("---")
        st.markdown("### 💬 Additional Feedback")
        
        st.markdown(
            """
            We value your input! If you'd like to provide additional feedback
            or have suggestions for improving this research, please consider:
            """
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📧 Email Feedback", key="email_feedback"):
                st.info(
                    "Please email your feedback to: research@university.edu\n\n"
                    "Include your session ID for reference: " + session_id
                )
        
        with col2:
            if st.button("📝 Survey Link", key="survey_link"):
                st.info(
                    "Complete our optional feedback survey:\n\n"
                    "https://forms.example.com/llm-feedback-survey\n\n"
                    "This helps us improve future research studies."
                )
    
    # Return to main app
    st.markdown("---")
    st.markdown("### 🏠 Return to Dashboard")
    
    if st.button("🏠 Return to Main Dashboard", type="primary", use_container_width=True):
        # Reset session state
        for key in list(st.session_state.keys()):
            if key.startswith(('step', 'consent', 'evaluation', 'current', 'responses', 'user_selection', 'industry_order', 'current_industry_index')):
                del st.session_state[key]
        
        # Redirect to main page
        st.switch_page("streamlit_app.py")


def render_simple_thank_you(
    session_id: str,
    message: str = "Thank you for participating in our research study!"
) -> None:
    """
    Render a simple thank you message.
    
    Args:
        session_id: Session identifier
        message: Custom thank you message
    """
    
    st.markdown(
        f"""
        <div style="text-align: center; padding: 4rem 0;">
            <h1 style="color: #0078d4; margin-bottom: 1rem;">🎉 Thank You!</h1>
            <p style="font-size: 1.2rem; color: #201f1e; margin-bottom: 2rem;">
                {message}
            </p>
            <p style="color: #666; font-size: 0.9rem;">
                Session ID: {session_id}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("🏠 Return to Dashboard", type="primary", use_container_width=True):
        st.switch_page("streamlit_app.py")


def render_completion_stats(session_id: str) -> Dict[str, any]:
    """
    Render completion statistics for the session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Dictionary with completion statistics
    """
    
    logger = FeedbackLogger()
    session_feedback = logger.get_session_feedback(session_id)
    
    # Calculate statistics
    industry_feedback = [f for f in session_feedback if f.get("record_type") == "industry_feedback"]
    session_events = [f for f in session_feedback if f.get("record_type") == "session_event"]
    
    stats = {
        "total_responses": len(industry_feedback),
        "industries_evaluated": list(set(f.get("industry") for f in industry_feedback)),
        "session_started": any(e.get("event_type") == "session_start" for e in session_events),
        "session_completed": any(e.get("event_type") == "session_completion" for e in session_events),
        "average_rating": 0
    }
    
    # Calculate average rating if ratings exist
    all_ratings = []
    for feedback in industry_feedback:
        ratings = feedback.get("ratings", {})
        if ratings:
            all_ratings.extend(ratings.values())
    
    if all_ratings:
        stats["average_rating"] = sum(all_ratings) / len(all_ratings)
    
    return stats


if __name__ == "__main__":
    # Test the thank you page component
    st.set_page_config(page_title="Thank You", page_icon="🎉")
    
    # Simulate a completed session
    test_session_id = create_session_id()
    test_industries = ["retail", "finance", "healthcare"]
    test_selections = {
        "retail": "A",
        "finance": "C", 
        "healthcare": "B"
    }
    
    render_thank_you_page(
        session_id=test_session_id,
        completed_industries=test_industries,
        user_selections=test_selections,
        total_time_seconds=450.5  # 7 minutes 30 seconds
    ) 