#!/usr/bin/env python3
"""
Enhanced Blind Evaluation Interface

Combines the essential blind testing structure from main.py including custom UI styling,
email notifications, and complex session state management with our enhanced 
dataset context functionality.
"""

import streamlit as st
import random
import json
import os
import re
import sys
import time
import smtplib
from datetime import datetime
from pathlib import Path
from typing import Dict
from email.message import EmailMessage
import streamlit_sortables as sortables

# Add src to path for imports
src_path = str(Path(__file__).parent.parent.parent / "src")
sys.path.insert(0, src_path)

# Import with absolute path resolution
try:
    from utils.question_sampler import QuestionSampler
    from llm_providers.provider_manager import ProviderManager
    from security.input_validator import InputValidator
    from ui.components.styles import UIStyles, PageTemplates, apply_base_styles
    from evaluation.ground_truth_generator import GroundTruthGenerator
except ImportError as e:
    # Fallback: direct file imports
    print(f"Import error: {e}")
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.question_sampler import QuestionSampler
    from src.llm_providers.provider_manager import ProviderManager
    from src.security.input_validator import InputValidator
    from src.ui.components.styles import UIStyles, PageTemplates, apply_base_styles
    from src.evaluation.ground_truth_generator import GroundTruthGenerator


def can_user_continue_session(email: str) -> bool:
    """Check if user has registered but not completed the evaluation (can continue)."""
    # Check if user is registered
    reg_file = Path("data/tester_registrations.json")
    is_registered = False
    
    if reg_file.exists():
        try:
            with open(reg_file, "r") as f:
                reg_data = json.load(f)
                for entry in reg_data:
                    if entry.get('email', '').lower() == email.lower():
                        is_registered = True
                        break
        except Exception:
            pass
    
    # If registered but hasn't completed evaluation, they can continue
    return is_registered and not check_email_completed_evaluation(email)


def check_email_completed_evaluation(email: str) -> bool:
    """Check if email has already COMPLETED the full evaluation (both retail and finance)."""
    # Check user feedback data for completed evaluations only
    feedback_files = [
        Path("data/enhanced_user_feedback.json"),
        Path("data/results/user_feedback.json"),
        Path("data/user_feedback.json")
    ]
    
    for feedback_file in feedback_files:
        if feedback_file.exists():
            try:
                with open(feedback_file, "r") as f:
                    feedback_data = json.load(f)
                    
                    # Count submissions by domain for this email
                    email_lower = email.lower()
                    retail_count = 0
                    finance_count = 0
                    
                    for entry in feedback_data:
                        if entry.get('tester_email', '').lower() == email_lower:
                            domain = entry.get('domain', '') or entry.get('industry', '')
                            if domain == 'retail':
                                retail_count += 1
                            elif domain == 'finance':
                                finance_count += 1
                    
                    # Check if user completed both domains (assuming 5 questions each)
                    # We'll check if they have at least 4 submissions in each domain to account for potential skips
                    if retail_count >= 4 and finance_count >= 4:
                        return True
                        
            except Exception:
                continue
    
    return False


def save_registration(email: str, signature: str):
    """Save user registration data."""
    reg_file = Path("data/tester_registrations.json")
    reg_file.parent.mkdir(parents=True, exist_ok=True)
    
    if reg_file.exists():
        with open(reg_file, "r") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
    else:
        data = []
    
    data.append({
        "email": email,
        "signature": signature,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    with open(reg_file, "w") as f:
        json.dump(data, f, indent=2)


def save_feedback(domain: str, question_idx: int, selected: str, comment: str, blind_map: Dict):
    """Save user feedback with enhanced data structure."""
    feedback_file = Path("data/user_feedback.json")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    
    if feedback_file.exists():
        with open(feedback_file, "r") as f:
            try:
                feedback_data = json.load(f)
            except Exception:
                feedback_data = []
    else:
        feedback_data = []
    
    feedback_data.append({
        "tester_email": st.session_state.get('tester_email'),
        "session_id": st.session_state.get('session_id'),
        "domain": domain,
        "question_idx": question_idx,
        "selected_response": selected,
        "comment": comment,
        "blind_map": blind_map,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=2)


def save_enhanced_feedback(domain: str, question_idx: int, selected: list, comment: str, 
                          blind_map: Dict, question: Dict, all_responses: list, 
                          selected_response_metadata: Dict):
    """Save enhanced user feedback with response metadata and quality metrics, supporting ranking."""
    feedback_file = Path("data/enhanced_user_feedback.json")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    
    if feedback_file.exists():
        with open(feedback_file, "r") as f:
            try:
                feedback_data = json.load(f)
            except Exception:
                feedback_data = []
    else:
        feedback_data = []

    # Map response_id to model for this question
    response_lookup = {f"response_{i}": resp for i, resp in enumerate(all_responses)}
    # Convert ranking (list of labels) to model order
    ranking_model_order = [
        response_lookup[blind_map[label]].get('model', 'unknown')
        for label in selected
    ]

    # Calculate response comparison metrics (unchanged)
    response_metrics = []
    for i, resp in enumerate(all_responses):
        resp_metadata = resp.get('metadata', {})
        response_metrics.append({
            'response_id': f"response_{i}",
            'word_count': resp_metadata.get('word_count', 0),
            'latency': resp_metadata.get('latency', 0),
            'structure_score': resp_metadata.get('response_structure_score', 0),
            'has_error': resp_metadata.get('has_error', False),
            'has_formatting': resp_metadata.get('has_bullet_points', False) or resp_metadata.get('has_code_blocks', False)
        })

    enhanced_feedback = {
        "tester_email": st.session_state.get('tester_email'),
        "session_id": st.session_state.get('session_id'),
        "domain": domain,
        "question_idx": question_idx,
        "question_text": question.get('question', ''),
        "question_context": question.get('context', ''),
        "ranking_label_order": selected,  # e.g., ['B', 'A', 'D', 'C']
        "ranking_model_order": ranking_model_order,  # e.g., ['llama3-8b-8192', ...]
        "comment": comment,
        "blind_map": blind_map,
        "timestamp": datetime.utcnow().isoformat(),
        # Deprecated fields for compatibility
        "selected_response": None,
        "selected_response_id": None,
        # Enhanced metrics
        "selected_model": None,
        "selected_provider": None,
        "selected_response_metadata": {},
        # Comparison context
        "total_responses_shown": len(all_responses),
        "successful_responses": len([r for r in all_responses if not r.get('metadata', {}).get('has_error', False)]),
        "all_response_metrics": response_metrics,
        # Quality analysis (not used for ranking)
        "selection_quality_analysis": {}
    }

    feedback_data.append(enhanced_feedback)
    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=2)
    
    # Also save to the original feedback file for backward compatibility
    save_feedback(domain, question_idx, selected, comment, blind_map)


def send_admin_email(subject: str, body: str) -> bool:
    """Send email notification to admin."""
    admin_email = os.getenv("ADMIN_EMAIL")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    
    if not (admin_email and smtp_server and smtp_user and smtp_pass):
        return False
    
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = admin_email
    msg.set_content(body)
    
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Email notification failed: {e}")
        return False


def initialize_complex_session_state():
    """Initialize complex session state management from main.py."""
    if "step" not in st.session_state:
        st.session_state["step"] = 0  # 0: Introduction, 1: Evaluation, 2: Complete
    if "registered" not in st.session_state:
        st.session_state["registered"] = False
    if "current_domain" not in st.session_state:
        st.session_state["current_domain"] = "retail"
    if "retail_question_idx" not in st.session_state:
        st.session_state["retail_question_idx"] = 0
    if "finance_question_idx" not in st.session_state:
        st.session_state["finance_question_idx"] = 0
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = f"session_{int(time.time())}"
    if "admin_email_sent" not in st.session_state:
        st.session_state["admin_email_sent"] = False
    if "session_data" not in st.session_state:
        st.session_state["session_data"] = None


def display_ground_truth_context(question_id: str, project_root: Path):
    """Display ground truth answer context to help evaluators."""
    import plotly.graph_objects as go
    import pandas as pd
    try:
        generator = GroundTruthGenerator(str(project_root))
        ground_truth = generator.get_ground_truth_for_question(question_id)
        if ground_truth:
            # --- Summary Card ---
            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                color: white;
                padding: 20px 25px;
                border-radius: 12px;
                margin: 15px 0;
                box-shadow: 0 6px 20px rgba(0,0,0,0.1);
                border-left: 6px solid #2e7d32;
            '>
                <h4 style='margin: 0 0 10px 0; font-weight: bold; font-size: 1.3rem;'>
                    💡 GROUND TRUTH CONTEXT
                </h4>
                <p style='margin: 10px 0; font-size: 1.1rem; line-height: 1.6;'>
                    <strong>Expected Answer:</strong> {ground_truth.get('answer', 'No ground truth available')}<br/>
                    <strong>Business Insight:</strong> {ground_truth.get('business_insight', '')}
                </p>
            </div>
            """, unsafe_allow_html=True)
            # --- Details Section ---
            if 'details' in ground_truth:
                with st.expander("📊 See Detailed Ground Truth Data", expanded=False):
                    details = ground_truth['details']
                    # --- Generalized Table/Chart Display ---
                    for key, value in details.items():
                        # Skip if already handled below
                        if not isinstance(value, dict):
                            st.write(f"**{key.replace('_', ' ').title()}**: {value}")
                            continue
                        # Try to display as table if all values are dicts with same keys
                        if all(isinstance(v, dict) for v in value.values()):
                            df = pd.DataFrame(value).T
                            st.markdown(f"### 📋 {key.replace('_', ' ').title()}")
                            st.dataframe(df, use_container_width=True)
                            # If numeric, plot
                            if df.select_dtypes(include='number').shape[1] > 0 and df.shape[0] > 1:
                                fig = go.Figure()
                                for col in df.select_dtypes(include='number').columns:
                                    fig.add_trace(go.Bar(x=df.index, y=df[col], name=col))
                                fig.update_layout(title=f"{key.replace('_', ' ').title()} (Bar Chart)", barmode='group', height=350)
                                st.plotly_chart(fig, use_container_width=True)
                            continue
                        # If value is a dict of numbers (e.g., category: value)
                        if all(isinstance(v, (int, float)) for v in value.values()):
                            st.markdown(f"### 📋 {key.replace('_', ' ').title()}")
                            df = pd.DataFrame(list(value.items()), columns=[key.split('_by_')[-1].title(), 'Value'])
                            st.dataframe(df, use_container_width=True)
                            fig = go.Figure(go.Bar(x=df.iloc[:,0], y=df['Value'], marker_color='#4CAF50'))
                            fig.update_layout(title=f"{key.replace('_', ' ').title()} (Bar Chart)", height=350)
                            st.plotly_chart(fig, use_container_width=True)
                            continue
                        # Otherwise, show as JSON
                        st.markdown(f"### 📋 {key.replace('_', ' ').title()}")
                        st.json(value)
                    # --- Raw JSON for researchers ---
                    with st.expander("🔍 View Raw JSON Data", expanded=False):
                        st.json(details)
    except Exception as e:
        st.warning(f"⚠️ Could not load ground truth context: {str(e)}")


def display_dataset_context_styled(sampler: QuestionSampler):
    """Display dataset context information with custom styling."""
    # Header with consistent styling
    UIStyles.render_header(
        title="Dataset Context for Evaluation",
        subtitle="Understanding the data you'll be evaluating",
        icon="📊",
        color="secondary"
    )
    
    # Show evaluation guidelines
    guidelines = sampler.get_evaluation_guidelines()
    if guidelines:
        st.markdown("### 📋 Evaluation Guidelines")
        
        if 'rating_scale' in guidelines:
            st.markdown("**Rating Scale:**")
            for rating, description in guidelines['rating_scale'].items():
                rating_num = rating.split('_')[0]
                st.markdown(f"- **{rating_num}**: {description}")
        
        if 'what_to_look_for' in guidelines:
            st.markdown("**What to Look For in Responses:**")
            for criterion in guidelines['what_to_look_for']:
                st.markdown(f"- {criterion}")
    
    # Display dataset information with styling
    st.markdown("### 📈 Datasets You'll Be Evaluating")
    
    context = sampler.get_dataset_context()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'retail' in context:
            retail_ctx = context['retail']
            st.markdown("#### 🛍️ Retail Dataset")
            st.markdown(f"**{retail_ctx.get('name', 'Retail Data')}**")
            st.markdown(f"*{retail_ctx.get('description', 'Customer shopping data')}*")
            st.markdown(f"📊 **Size:** {retail_ctx.get('size', 'N/A')}")
            
            if 'key_fields' in retail_ctx:
                with st.expander("📋 Key Data Fields"):
                    fields = retail_ctx['key_fields']
                    if 'demographic' in fields:
                        st.markdown("**Demographics:**")
                        for field in fields['demographic']:
                            st.markdown(f"- {field}")
                    if 'transaction' in fields:
                        st.markdown("**Transaction Data:**")
                        for field in fields['transaction']:
                            st.markdown(f"- {field}")
                    if 'behavior' in fields:
                        st.markdown("**Customer Behavior:**")
                        for field in fields['behavior']:
                            st.markdown(f"- {field}")
    
    with col2:
        if 'finance' in context:
            finance_ctx = context['finance']
            st.markdown("#### 📈 Finance Dataset")
            st.markdown(f"**{finance_ctx.get('name', 'Finance Data')}**")
            st.markdown(f"*{finance_ctx.get('description', 'Stock market data')}*")
            st.markdown(f"📊 **Size:** {finance_ctx.get('size', 'N/A')}")
            
            if 'key_fields' in finance_ctx:
                with st.expander("📋 Key Data Fields"):
                    fields = finance_ctx['key_fields']
                    if 'pricing' in fields:
                        st.markdown("**Price Data:**")
                        for field in fields['pricing']:
                            st.markdown(f"- {field}")
                    if 'volume' in fields:
                        st.markdown("**Volume Data:**")
                        for field in fields['volume']:
                            st.markdown(f"- {field}")
                    if 'calculated_metrics' in fields:
                        st.markdown("**Calculated Metrics:**")
                        for field in fields['calculated_metrics']:
                            st.markdown(f"- {field}")
    
    UIStyles.render_section_divider()


def show_introduction():
    """Display introduction and registration form with improved two-column layout and spacing."""
    # Apply base styles
    apply_base_styles()
    
    # Enhanced CSS for the registration page with darker right column
    st.markdown("""
    <style>
    /* Enhanced styling for the registration page */
    .registration-dark-section {
        background-color: #f8f9fa;
        padding: 25px 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #dee2e6;
    }
    
    /* Better form styling in dark section */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: white;
        border: 2px solid #dee2e6;
        border-radius: 8px;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Remove any extra blank lines before the header
    UIStyles.render_header(
        title="Blind LLM Evaluation",
        subtitle="Research on Free-Tier LLM Performance in Business Intelligence",
        icon="🔬",
        color="primary"
    )

    # Main two-column layout
    left, right = st.columns([2, 1], gap="large")

    with left:
        # Start darker background container for study purpose section
        st.markdown('<div class="registration-dark-section">', unsafe_allow_html=True)
        
        st.markdown("## 🎯 Study Purpose")
        st.markdown("""
        - Blind evaluation of LLMs for business intelligence
        - Real business scenarios (retail, finance)
        - Your feedback advances AI research
        """)
        UIStyles.render_section_divider()
        try:
            # Use absolute path from project root 
            project_root = Path(__file__).parent.parent.parent.parent
            questions_file = project_root / "data" / "evaluation_questions.yaml"
            sampler = QuestionSampler(str(questions_file))
            display_dataset_context_styled(sampler)
        except Exception as e:
            st.warning(f"Could not load dataset context: {str(e)}")
        
        # End darker background container for study purpose section
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        # Remove any blank space before the header in the right column
        st.markdown("## 📝 Register & Consent", unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 16px; border-radius: 10px; border-left: 4px solid #1f77b4; margin-bottom: 16px;">
        <b>Required Registration</b><br>
        Please provide your email and agree to the consent form below. Your responses will be anonymized and used for research purposes only.
        </div>
        """, unsafe_allow_html=True)
        with st.expander("📋 View Consent Form", expanded=False):
            st.markdown("""
            <div style="background-color: #fff3cd; padding: 12px; border-radius: 8px; border: 1px solid #ffeaa7; margin: 10px 0;">
            <b>Consent to Participate in Research</b><br>
            <ul>
            <li><b>Study Purpose:</b> You are invited to participate in a research study evaluating AI-generated business analysis responses from different LLM providers.</li>
            <li><b>Voluntary Participation:</b> Your participation is completely voluntary and you may withdraw at any time without penalty.</li>
            <li><b>Data Privacy:</b> All data will be anonymized and used solely for academic research purposes. No personal information will be shared.</li>
            <li><b>Consent:</b> By checking the box below and providing your email, you consent to participate in this research study.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        # Error state variables
        if 'email_error' not in st.session_state:
            st.session_state['email_error'] = ""
        if 'consent_warning' not in st.session_state:
            st.session_state['consent_warning'] = False
        if 'signature_error' not in st.session_state:
            st.session_state['signature_error'] = ""
        if 'name_length_error' not in st.session_state:
            st.session_state['name_length_error'] = ""
        if 'duplicate_email_error' not in st.session_state:
            st.session_state['duplicate_email_error'] = ""
        # Add helpful info for returning users
        st.markdown("""
        <div style='background-color:#e0f2fe; color:#01579b; padding:15px; border-radius:8px; margin-bottom:15px; border-left:4px solid #0288d1;'>
            <strong>💡 Returning User?</strong><br/>
            Simply enter your email address to continue your evaluation where you left off.<br/>
            <strong>✨ New User?</strong> Fill out all fields below to start your first evaluation.
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("registration_form"):
            st.markdown("### Registration Details")
            # Custom error/warning display for visibility
            if st.session_state['email_error']:
                st.markdown(f"""
                <div style='background-color:#ffeaea; color:#b30000; font-size:1.1rem; padding:10px 16px; border-radius:8px; border-left:5px solid #b30000; margin-bottom:8px; font-weight:bold;'>
                {st.session_state['email_error']}
                </div>
                """, unsafe_allow_html=True)
            if st.session_state['signature_error']:
                st.markdown(f"""
                <div style='background-color:#ffeaea; color:#b30000; font-size:1.1rem; padding:10px 16px; border-radius:8px; border-left:5px solid #b30000; margin-bottom:8px; font-weight:bold;'>
                {st.session_state['signature_error']}
                </div>
                """, unsafe_allow_html=True)
            if st.session_state['name_length_error']:
                st.markdown(f"""
                <div style='background-color:#ffeaea; color:#b30000; font-size:1.1rem; padding:10px 16px; border-radius:8px; border-left:5px solid #b30000; margin-bottom:8px; font-weight:bold;'>
                {st.session_state['name_length_error']}
                </div>
                """, unsafe_allow_html=True)
            if st.session_state['duplicate_email_error']:
                st.markdown(f"""
                <div style='background-color:#ffeaea; color:#b30000; font-size:1.1rem; padding:10px 16px; border-radius:8px; border-left:5px solid #b30000; margin-bottom:8px; font-weight:bold;'>
                {st.session_state['duplicate_email_error']}
                </div>
                """, unsafe_allow_html=True)
            email = st.text_input("Email", placeholder="your@email.com")
            signature = st.text_input("Full Name", placeholder="John Smith")
            # Visible caution above consent checkbox
            st.markdown("""
            <div style='background-color:#fffbe6; color:#b38f00; font-size:1.05rem; padding:8px 14px; border-radius:8px; border-left:5px solid #b38f00; margin-bottom:8px; font-weight:bold;'>
            You must tick the box below to agree to the consent form and participate in this research study.
            </div>
            """, unsafe_allow_html=True)
            consent = st.checkbox("I have read and agree to the consent form above.", value=False)
            if st.session_state['consent_warning']:
                st.markdown("""
                <div style='background-color:#ffeaea; color:#b30000; font-size:1.1rem; padding:10px 16px; border-radius:8px; border-left:5px solid #b30000; margin-bottom:8px; font-weight:bold;'>
                You must agree to the consent form to participate in this research study.
                </div>
                """, unsafe_allow_html=True)
            submit = st.form_submit_button("🚀 Register and Start Test", type="primary", use_container_width=True)
            if submit:
                validator = InputValidator()
                email = validator.sanitize_text(email)
                signature = validator.sanitize_text(signature)
                email_valid = re.match(r"[^@]+@[^@]+\.[^@]+", email)
                # Reset error states
                st.session_state['email_error'] = ""
                st.session_state['consent_warning'] = False
                st.session_state['signature_error'] = ""
                st.session_state['name_length_error'] = ""
                st.session_state['duplicate_email_error'] = ""
                
                # Check if email has completed the full evaluation
                evaluation_completed = check_email_completed_evaluation(email) if email_valid else False
                
                if not email_valid:
                    st.session_state['email_error'] = "Please enter a valid email address in the format: user@domain.com"
                elif evaluation_completed:
                    st.session_state['duplicate_email_error'] = "This email address has already completed the full evaluation. Each person can only participate once. Thank you for your previous participation!"
                elif can_user_continue_session(email):
                    # User can continue their existing session (no need for signature/consent validation)
                    st.session_state['registered'] = True
                    st.session_state['tester_email'] = email
                    st.session_state['step'] = 1
                    start_evaluation_session()
                    st.success("✅ Welcome back! Continuing your evaluation session...")
                    st.info("📍 You'll resume from where you left off in your evaluation.")
                    st.balloons()
                    st.rerun()
                elif not signature.strip():
                    st.session_state['signature_error'] = "Please type your full name as a digital signature to proceed."
                elif len(signature) > 100:
                    st.session_state['name_length_error'] = "Name too long. Please use a shorter version (maximum 100 characters)."
                elif not consent:
                    st.session_state['consent_warning'] = True
                else:
                    save_registration(email, signature)
                    st.session_state['registered'] = True
                    st.session_state['tester_email'] = email
                    st.session_state['step'] = 1
                    start_evaluation_session()
                    st.success("Registration complete! Starting the blind test...")
                    st.balloons()
                    st.rerun()
        # Highly visible info message at the bottom of the right column
        st.markdown("""
        <div style='background-color:#e3f0ff; color:#174ea6; font-size:1.15rem; padding:16px 18px; border-radius:10px; border-left:6px solid #174ea6; margin-top:18px; font-weight:bold; text-align:center;'>
        Your responses are anonymized and used for research only. Thank you for participating!
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")


def show_evaluation():
    """Display the main evaluation interface with complex session state management."""
    # Header section using styled components
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🔬 Blind LLM Evaluation")
        st.markdown("### Compare AI responses and select the best one for each scenario")
        st.markdown("")

    # Show registered email
    if st.session_state.get('tester_email'):
        st.info(f"You are participating as: {st.session_state['tester_email']}")

    # Complex session state management for industry switching
    current_domain = st.session_state["current_domain"]
    domains = ["retail", "finance"]

    # Progress indicator with complex state tracking
    retail_progress = st.session_state["retail_question_idx"]
    finance_progress = st.session_state["finance_question_idx"]
    
    # Get questions from session data
    if not st.session_state.get('session_data'):
        st.error("❌ Session data not found. Please restart the evaluation.")
        if st.button("🔄 Restart Evaluation"):
            for key in list(st.session_state.keys()):
                if key.startswith(('retail_', 'finance_', 'current_', 'session_')):
                    del st.session_state[key]
            st.session_state['step'] = 0
            st.rerun()
        return
    
    # Calculate total questions from session data
    total_retail = len(st.session_state.session_data['domains']['retail']['questions'])
    total_finance = len(st.session_state.session_data['domains']['finance']['questions'])
    total_questions = total_retail + total_finance
    completed_questions = retail_progress + finance_progress

    # Calculate overall progress
    progress_percentage = completed_questions / total_questions if total_questions > 0 else 0
    
    # Progress indicator with styling
    progress_col1, progress_col2, progress_col3 = st.columns([1, 2, 1])
    with progress_col2:
        st.progress(progress_percentage, text=f"Overall Progress: {completed_questions}/{total_questions} questions")

    st.markdown(f"### 📊 Progress: {completed_questions}/{total_questions} questions completed")

    # Industry progress bars
    col1, col2 = st.columns(2)
    with col1:
        retail_percent = (retail_progress / total_retail) * 100 if total_retail > 0 else 0
        # Show current question number (1-based) for better user understanding
        if retail_progress >= total_retail:
            retail_display = f"🛍️ Retail: {total_retail}/{total_retail} Complete ✅"
        else:
            current_question = min(retail_progress + 1, total_retail)
            retail_display = f"🛍️ Retail: Question {current_question} of {total_retail}"
        st.progress(retail_percent / 100, text=retail_display)

    with col2:
        finance_percent = (finance_progress / total_finance) * 100 if total_finance > 0 else 0
        # Show current question number (1-based) for better user understanding  
        if finance_progress >= total_finance:
            finance_display = f"💰 Finance: {total_finance}/{total_finance} Complete ✅"
        else:
            current_question = min(finance_progress + 1, total_finance)
            finance_display = f"💰 Finance: Question {current_question} of {total_finance}"
        st.progress(finance_percent / 100, text=finance_display)
    
    # Debug information (only in development) 
    debug_mode = st.secrets.get("DEBUG_MODE", False) or st.sidebar.checkbox("🐛 Show Debug Info", key="debug_mode_toggle")
    if debug_mode:
        st.sidebar.write("🐛 Debug Info:")
        st.sidebar.write(f"Retail progress: {retail_progress}")
        st.sidebar.write(f"Total retail: {total_retail}")
        st.sidebar.write(f"Finance progress: {finance_progress}")
        st.sidebar.write(f"Total finance: {total_finance}")
        st.sidebar.write(f"Current domain: {current_domain}")
        if st.session_state.get('session_data'):
            retail_actual = len(st.session_state.session_data['domains']['retail']['questions'])
            finance_actual = len(st.session_state.session_data['domains']['finance']['questions'])
            st.sidebar.write(f"Actual retail count: {retail_actual}")
            st.sidebar.write(f"Actual finance count: {finance_actual}")
            if retail_actual != total_retail:
                st.sidebar.error(f"❌ Mismatch: total_retail={total_retail} vs actual={retail_actual}")
            if finance_actual != total_finance:
                st.sidebar.error(f"❌ Mismatch: total_finance={total_finance} vs actual={finance_actual}")
            # Show the actual questions for debugging
            st.sidebar.write("📝 Questions in session:")
            for i, q in enumerate(st.session_state.session_data['domains'][current_domain]['questions']):
                marker = "👉" if i == st.session_state.get(f"{current_domain}_question_idx", 0) else "  "
                st.sidebar.write(f"{marker} {i+1}. {q.get('id', 'no_id')}")
        
        # Also show at the top for easier visibility
        st.info(f"🐛 Debug: {current_domain.title()} Progress = {st.session_state.get(f'{current_domain}_question_idx', 0)} / {len(st.session_state.session_data['domains'][current_domain]['questions']) if st.session_state.get('session_data') else 'No data'}")

    # Current domain and question with complex state management
    if current_domain == "retail":
        question_idx = st.session_state["retail_question_idx"]
        questions = st.session_state.session_data['domains']['retail']['questions']
        domain_name = "🛍️ Retail"
    elif current_domain == "finance":
        question_idx = st.session_state["finance_question_idx"]
        questions = st.session_state.session_data['domains']['finance']['questions']
        domain_name = "💰 Finance"

    if question_idx < len(questions):
        st.markdown(f"## {domain_name}: Question {question_idx + 1} of {len(questions)}")
        
        question = questions[question_idx]
        
        # Display question with enhanced styling for maximum visibility
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px 30px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border-left: 8px solid #4f46e5;
        '>
            <h2 style='
                margin: 0;
                font-size: 1.8rem;
                font-weight: bold;
                color: #ffffff;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                line-height: 1.4;
            '>
                📋 BUSINESS SCENARIO
            </h2>
            <p style='
                margin: 15px 0 0 0;
                font-size: 1.3rem;
                font-weight: 600;
                color: #f8fafc;
                line-height: 1.6;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
            '>
                {question['question']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show domain context with enhanced styling
        try:
            # Use absolute path from project root
            project_root = Path(__file__).parent.parent.parent.parent
            questions_file = project_root / "data" / "evaluation_questions.yaml"
            sampler = QuestionSampler(str(questions_file))
            context = sampler.get_dataset_context(current_domain)
            if context:
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    padding: 20px 25px;
                    border-radius: 12px;
                    margin: 15px 0;
                    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
                    border-left: 6px solid #e91e63;
                '>
                    <h3 style='
                        margin: 0 0 10px 0;
                        font-size: 1.2rem;
                        font-weight: bold;
                        color: #ffffff;
                        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                    '>
                        💡 DATASET CONTEXT
                    </h3>
                    <p style='
                        margin: 0;
                        font-size: 1.1rem;
                        font-weight: 500;
                        color: #fef7f7;
                        line-height: 1.5;
                        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
                    '>
                        {context.get('description', 'Business data analysis')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            pass
        
        # Display ground truth context to help evaluators
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            display_ground_truth_context(question.get('id', ''), project_root)
        except Exception as e:
            st.warning(f"⚠️ Could not load ground truth: {str(e)}")
        
        # Load pre-generated responses
        responses = load_pre_generated_responses(question, current_domain)
        
        if responses:
            # Create blind mapping
            response_ids = [f"response_{i}" for i in range(len(responses))]
            labels = ['A', 'B', 'C', 'D', 'E', 'F'][:len(responses)]
            
            # Create shuffled mapping
            shuffled_ids = response_ids.copy()
            random.shuffle(shuffled_ids)
            blind_map = dict(zip(labels, shuffled_ids))
            
            # Store mapping in session state
            st.session_state['blind_map'] = blind_map
            
            # Display responses as cards with enhanced styling
            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                color: white;
                padding: 20px 25px;
                border-radius: 12px;
                margin: 25px 0 20px 0;
                box-shadow: 0 6px 20px rgba(0,0,0,0.1);
                border-left: 6px solid #059669;
            '>
                <h3 style='
                    margin: 0 0 8px 0;
                    font-size: 1.4rem;
                    font-weight: bold;
                    color: #ffffff;
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                '>
                    🤖 AI RESPONSE COMPARISON
                </h3>
                <p style='
                    margin: 0;
                    font-size: 1.05rem;
                    font-weight: 500;
                    color: #f0fdf4;
                    line-height: 1.5;
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
                '>
                    Below are AI-generated responses to the business scenario above. Each response is labeled A-F and presented in random order to ensure unbiased evaluation.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display response cards as tiles (2 per row) with enhanced styling
            response_lookup = {f"response_{i}": resp for i, resp in enumerate(responses)}
            
            for i in range(0, len(labels), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(labels):
                        label = labels[i + j]
                        response_id = blind_map[label]
                        response = response_lookup[response_id]
                        
                        with cols[j]:
                            # Get response metadata
                            metadata = response.get('metadata', {})
                            has_error = metadata.get('has_error', False)
                            
                            # Calculate quality indicators for display
                            word_count = metadata.get('word_count', 0)
                            latency = metadata.get('latency', 0)
                            structure_score = metadata.get('response_structure_score', 0)
                            has_formatting = metadata.get('has_bullet_points', False) or metadata.get('has_code_blocks', False)
                            
                            # Color coding based on quality without revealing model
                            if has_error:
                                border_color = "#dc3545"
                                badge_color = "#dc3545"
                                quality_icon = "❌"
                            elif structure_score > 0.7:
                                border_color = "#28a745"
                                badge_color = "#28a745"
                                quality_icon = "⭐"
                            elif structure_score > 0.4:
                                border_color = "#ffc107"
                                badge_color = "#ffc107"
                                quality_icon = "📝"
                            else:
                                border_color = "#1f77b4"
                                badge_color = "#1f77b4"
                                quality_icon = "🤖"
                            
                            # Enhanced card with quality indicators
                            st.markdown(f"""
                            <div style="background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 10px 0; border: 2px solid {border_color};">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #f0f0f0;">
                                    <h3 style="color: {badge_color}; margin: 0; font-size: 1.4rem; font-weight: 600;">{quality_icon} Response {label}</h3>
                                    <span style="background-color: {badge_color}; color: white; padding: 5px 12px; border-radius: 20px; font-weight: 600; font-size: 1rem;">{label}</span>
                                </div>
                                <div style="line-height: 1.7; font-size: 1.1rem; color: #333; text-align: justify; margin-bottom: 15px;">
                                    {response.get('response', 'No response available')}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Add quality metrics footer (without revealing model identity)
                            if not has_error:
                                quality_bars = []
                                
                                # Length indicator
                                length_indicator = "📏 Comprehensive" if word_count > 150 else "📏 Concise" if word_count > 50 else "📏 Brief"
                                quality_bars.append(length_indicator)
                                
                                # Speed indicator  
                                speed_indicator = "⚡ Fast" if latency < 2 else "⚡ Medium" if latency < 5 else "⚡ Slow"
                                quality_bars.append(speed_indicator)
                                
                                # Structure indicator
                                if has_formatting:
                                    quality_bars.append("📋 Well-Structured")
                                
                                st.markdown(f"""
                                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 8px; margin-top: 10px; border-left: 4px solid {border_color};">
                                    <div style="font-size: 0.9rem; color: #495057; display: flex; gap: 15px; flex-wrap: wrap;">
                                        {' • '.join(quality_bars)}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("</div>", unsafe_allow_html=True)
            
            # Clear visual separation between responses and ranking
            st.markdown("---")
            st.markdown("""
            <div style='
                text-align: center;
                padding: 15px 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin: 30px 0 20px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            '>
                <h3 style='
                    margin: 0;
                    font-size: 1.4rem;
                    font-weight: bold;
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                '>
                    ⬇️ NOW RANK THE RESPONSES BELOW ⬇️
                </h3>
                <p style='
                    margin: 5px 0 0 0;
                    font-size: 1rem;
                    opacity: 0.9;
                '>
                    The responses remain in their original positions above for reference
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Selection interface
            st.markdown("---")
            st.markdown("###  Your Evaluation")
            # Enhanced ranking instructions
            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
                color: #831843;
                padding: 20px 25px;
                border-radius: 12px;
                margin: 25px 0 20px 0;
                box-shadow: 0 6px 20px rgba(0,0,0,0.1);
                border-left: 6px solid #be185d;
                border: 2px solid #ec4899;
            '>
                <h3 style='
                    margin: 0 0 8px 0;
                    font-size: 1.3rem;
                    font-weight: bold;
                    color: #831843;
                    text-shadow: 1px 1px 2px rgba(255,255,255,0.3);
                '>
                    📊 RANK THE RESPONSES
                </h3>
                <p style='
                    margin: 0;
                    font-size: 1.1rem;
                    font-weight: 600;
                    color: #9f1239;
                    line-height: 1.5;
                    text-shadow: 1px 1px 2px rgba(255,255,255,0.2);
                '>
                    Drag and drop the responses below to rank them from <strong>BEST</strong> (top) to <strong>WORST</strong> (bottom). You must rank all responses before submitting.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Compact, centered ranking UI
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                ranking_labels = labels[:len(responses)]
                default_order = [f"Response {l}" for l in ranking_labels]
                # Map label to blind_map for lookup
                label_to_id = {f"Response {l}": l for l in ranking_labels}
                
                # Use session state to persist ranking
                if f"ranking_{current_domain}_{question_idx}" not in st.session_state:
                    st.session_state[f"ranking_{current_domain}_{question_idx}"] = default_order
                
                # Compact ranking display with better styling
                st.markdown("""
                <div style='
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    padding: 25px;
                    border-radius: 15px;
                    border: 3px solid #dee2e6;
                    margin: 20px 0;
                    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
                '>
                    <h4 style='
                        text-align: center;
                        color: #495057;
                        margin-bottom: 20px;
                        font-size: 1.2rem;
                        font-weight: bold;
                    '>
                        🏆 Drag to Rank: Best (Top) → Worst (Bottom)
                    </h4>
                    <div style='
                        text-align: center;
                        color: #6c757d;
                        font-size: 0.9rem;
                        margin-bottom: 15px;
                        font-style: italic;
                    '>
                        Drag the items below to reorder them
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                ranking = sortables.sort_items(
                    items=st.session_state[f"ranking_{current_domain}_{question_idx}"],
                    direction="vertical",
                    key=f"sortable_{current_domain}_{question_idx}"
                )
                st.session_state[f"ranking_{current_domain}_{question_idx}"] = ranking

            # Keep comment and submit button in the same centered column
            with col2:
                comment = st.text_area(
                    "**Please provide your reasoning (optional but helpful):**",
                    placeholder="I ranked these responses this way because...",
                    height=100
                )

                if st.button("Submit and Continue", type="primary", use_container_width=True):
                    # Validate ranking
                    if len(ranking) != len(ranking_labels) or set(ranking) != set(default_order):
                        st.error("Please rank all responses before submitting.")
                    else:
                        # Convert ranking to label order (e.g., ['B', 'A', 'D', 'C'])
                        ranking_labels_order = [label_to_id[item] for item in ranking]
                        # Save enhanced feedback with ranking
                        save_enhanced_feedback(
                            domain=current_domain,
                            question_idx=question_idx,
                            selected=ranking_labels_order,  # Now a list
                            comment=comment,
                            blind_map=blind_map,
                            question=question,
                            all_responses=responses,
                            selected_response_metadata={}
                        )
                        # Progression logic (unchanged)
                        if current_domain == "retail":
                            st.session_state["retail_question_idx"] += 1
                        elif current_domain == "finance":
                            st.session_state["finance_question_idx"] += 1
                        st.rerun()
        
        else:
            st.error("❌ Could not generate responses for this question.")
            if st.button("Skip Question"):
                # Complex state management for skipping
                if current_domain == "retail":
                    st.session_state["retail_question_idx"] += 1
                elif current_domain == "finance":
                    st.session_state["finance_question_idx"] += 1
                st.rerun()

    else:
        # All questions completed for current domain
        st.success(f"✅ All {domain_name} questions completed!")
        
        # Check if we need to move to next domain or complete the evaluation
        if current_domain == "retail":
            # Move to finance domain
            st.session_state["current_domain"] = "finance"
            st.info("🔄 Moving to Finance domain...")
            time.sleep(1)  # Brief pause to show the transition message
            st.rerun()
        elif current_domain == "finance":
            # All domains completed - move to completion step
            st.session_state["step"] = 2
            st.info("🎉 All evaluations completed! Redirecting to completion page...")
            time.sleep(1)  # Brief pause to show the completion message
            st.rerun()
        else:
            # Fallback for any unexpected domain
            st.error("❌ Unknown domain. Please restart the evaluation.")
            if st.button("🔄 Restart Evaluation"):
                for key in list(st.session_state.keys()):
                    if key.startswith(('retail_', 'finance_', 'current_', 'session_')):
                        del st.session_state[key]
                st.session_state['step'] = 0
                st.rerun()


def show_completion():
    """Display completion screen with enhanced styling."""
    # Create user summary with complex state
    user_summary = {
        'retail_completed': st.session_state.get("retail_question_idx", 0) >= len(st.session_state.session_data['domains']['retail']['questions']),
        'finance_completed': st.session_state.get("finance_question_idx", 0) >= len(st.session_state.session_data['domains']['finance']['questions'])
    }
    
    # Use the centralized completion page template
    PageTemplates.render_completion_page(user_summary=user_summary, show_restart_button=True)


def load_pre_generated_responses(question, domain):
    """Load fixed pre-generated responses for consistent blind evaluations."""
    try:
        # Load from the fixed responses file
        responses_file = Path("data/fixed_blind_responses.json")
        
        if not responses_file.exists():
            st.error("❌ Fixed blind responses not found. Please run the fixed response generation script first.")
            st.code("python3 scripts/generate_fixed_blind_responses.py", language="bash")
            return []
        
        with open(responses_file, 'r') as f:
            data = json.load(f)
        
        # Find matching question in fixed responses
        question_id = f"{domain}_{question['question_idx']}"
        if question_id in data.get("responses", {}):
            responses = data["responses"][question_id].get("llm_responses", {})
            st.info(f"📋 Using fixed responses ({len(responses)} models) for consistent evaluation")
            return responses
        
        # Fallback: try to match by question text
        question_text = question['question']
        for q_id, q_data in data.get("responses", {}).items():
            if q_data.get('question') == question_text:
                responses = q_data.get("llm_responses", {})
                st.info(f"📋 Using fixed responses ({len(responses)} models) for consistent evaluation")
                return responses
        
        st.error(f"❌ No fixed responses found for {domain} question {question['question_idx']}")
        return []
        
    except Exception as e:
        st.error(f"❌ Error loading fixed responses: {str(e)}")
        return []


def generate_llm_responses(question):
    """Generate enhanced responses from available LLMs with RAG and quality metrics."""
    import hashlib
    import time
    from pathlib import Path
    from rag.csv_rag_pipeline import CSVRAGPipeline
    
    # Create a cache key based on question content
    question_text = f"{question['question']}_{question.get('context', '')}"
    cache_key = hashlib.md5(question_text.encode()).hexdigest()
    cache_file = Path(f"data/response_cache/{cache_key}.json")
    
    # Try to load from cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_responses = json.load(f)
                st.info("💡 Using cached responses for faster loading")
                return cached_responses
        except Exception:
            pass  # If cache is corrupted, regenerate
    
    # Set up RAG pipeline for dataset access
    current_domain = st.session_state.get("current_domain", "retail")
    
    # Map domain to appropriate dataset
    dataset_files = {
        "retail": ["data/shopping_trends.csv"],
        "finance": ["data/Tesla_stock_data.csv"]
    }
    
    csv_files = dataset_files.get(current_domain, ["data/shopping_trends.csv"])
    
    # Initialize RAG pipeline with optimized parameters
    try:
        rag_pipeline = CSVRAGPipeline()
        with st.spinner("🔍 Indexing dataset for context retrieval..."):
            # OPTIMIZED: Larger chunks for better coverage, more efficient retrieval
            rag_pipeline.build_index(csv_files, chunk_size=200)
        
        # Generate RAG context for the question with optimized retrieval
        # OPTIMIZED: More chunks for 25.6% coverage vs 3.8% 
        rag_context = rag_pipeline.generate_context(question['question'], top_k=5)
        
        # Calculate and display coverage stats
        total_rows = 3900 if current_domain == "retail" else 3782
        retrieved_rows = min(5 * 200, total_rows)
        coverage_percent = (retrieved_rows / total_rows) * 100
        
        st.success(f"📊 Retrieved {retrieved_rows:,} rows ({coverage_percent:.1f}% coverage) from {current_domain} dataset")
        
    except Exception as e:
        st.warning(f"⚠️ Could not initialize RAG pipeline: {str(e)}. Proceeding without dataset context.")
        rag_context = "No specific dataset context available."
    
    try:
        with st.spinner("🤖 Generating AI responses..."):
            # Initialize ProviderManager with absolute config path
            project_root = Path(__file__).parent.parent.parent.parent
            config_dir = project_root / "config"
            
            # Import ConfigLoader and initialize with absolute path
            from config.config_loader import ConfigLoader
            config_loader = ConfigLoader(str(config_dir))
            
            manager = ProviderManager(config_loader)
            responses = []
            errors = []
            
            # Get available models and their providers
            all_models = manager.get_all_models()
            model_provider_pairs = []
            for provider_name, provider_models in all_models.items():
                for model_name in provider_models:
                    model_provider_pairs.append((provider_name, model_name))
            
            # Progress tracking
            progress_bar = st.progress(0)
            total_models = len(model_provider_pairs)
            
            for idx, (provider_name, model_name) in enumerate(model_provider_pairs):
                progress_bar.progress((idx + 1) / total_models, 
                                    text=f"Generating response from {model_name}...")
                
                try:
                    start_time = time.time()
                    
                    # Create enhanced prompt with RAG context
                    enhanced_prompt = f"""
Business Scenario: {question['question']}

Dataset Context: {rag_context}

Additional Context: {question.get('context', '')}

Based on the business scenario above and the relevant data insights from the {current_domain} dataset, please provide a comprehensive analysis and actionable recommendations. Focus on:
1. Data-driven insights from the provided context
2. Practical business recommendations
3. Specific actions that can be taken based on the data patterns
4. Key performance indicators to monitor

Please ensure your response is grounded in the actual data provided and offers specific, actionable business intelligence.
"""
                    
                    response = manager.generate_response(
                        provider_name=provider_name,
                        query=enhanced_prompt,
                        model=model_name
                    )
                    end_time = time.time()
                    
                    if response and response.success and response.text:
                        # Calculate basic quality metrics
                        response_text = response.text
                        word_count = len(response_text.split())
                        sentence_count = len([s for s in response_text.split('.') if s.strip()])
                        
                        # Provider name is already available from the loop
                        provider_display_name = provider_name.lower()
                        
                        enhanced_response = {
                            'model': model_name,
                            'provider': provider_display_name,
                            'response': response_text,
                            'metadata': {
                                'latency': round(end_time - start_time, 2),
                                'word_count': word_count,
                                'sentence_count': sentence_count,
                                'character_count': len(response_text),
                                'timestamp': datetime.utcnow().isoformat(),
                                'has_code_blocks': '```' in response_text,
                                'has_bullet_points': any(line.strip().startswith(('•', '-', '*', '1.', '2.')) 
                                                       for line in response_text.split('\n')),
                                'response_structure_score': min(1.0, (word_count / 100) * 0.5 + 
                                                              (sentence_count / 10) * 0.3 + 
                                                              (0.2 if '```' in response_text else 0)),
                                'uses_rag': True,
                                'domain': current_domain,
                                'dataset_files': csv_files,
                                'rag_context_length': len(rag_context),
                                'rag_context_preview': rag_context[:200] + "..." if len(rag_context) > 200 else rag_context
                            }
                        }
                        
                        responses.append(enhanced_response)
                    else:
                        # Handle failed response (not successful or no text)
                        error_msg = response.error if response and hasattr(response, 'error') else "No response generated"
                        errors.append({'model': model_name, 'error': error_msg})
                        responses.append({
                            'model': model_name,
                            'provider': provider_name.lower(),
                            'response': f"❌ Error: {error_msg}",
                            'metadata': {
                                'latency': 0,
                                'word_count': 0,
                                'sentence_count': 0,
                                'character_count': 0,
                                'timestamp': datetime.utcnow().isoformat(),
                                'has_error': True,
                                'error_message': error_msg,
                                'uses_rag': True,
                                'domain': current_domain,
                                'dataset_files': csv_files,
                                'rag_context_length': len(rag_context),
                                'rag_context_preview': rag_context[:200] + "..." if len(rag_context) > 200 else rag_context
                            }
                        })
                        
                except Exception as e:
                    error_msg = f"Failed to get response from {model_name}: {str(e)}"
                    errors.append({'model': model_name, 'error': error_msg})
                    # Add a placeholder response for failed models
                    responses.append({
                        'model': model_name,
                        'provider': provider_name.lower(),
                        'response': f"❌ Error: Unable to generate response from {model_name}",
                        'metadata': {
                            'latency': 0,
                            'word_count': 0,
                            'sentence_count': 0,
                            'character_count': 0,
                            'timestamp': datetime.utcnow().isoformat(),
                            'has_error': True,
                            'error_message': str(e),
                            'uses_rag': True,
                            'domain': current_domain,
                            'dataset_files': csv_files,
                            'rag_context_length': len(rag_context) if 'rag_context' in locals() else 0,
                            'rag_context_preview': rag_context[:200] + "..." if 'rag_context' in locals() and len(rag_context) > 200 else ""
                        }
                    })
                    continue
            
            progress_bar.empty()
            
            # Show summary
            successful_responses = len([r for r in responses if not r.get('metadata', {}).get('has_error')])
            if errors:
                with st.expander(f"⚠️ {len(errors)} models failed to respond", expanded=False):
                    for error in errors:
                        st.error(f"**{error['model']}**: {error['error']}")
            
            if successful_responses > 0:
                st.success(f"✅ Generated {successful_responses} responses successfully")
                
                # Cache the responses for future use
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(responses, f, indent=2)
                except Exception:
                    pass  # Cache failure shouldn't stop the process
                
                return responses
            else:
                st.error("❌ No models could generate responses. Please check your API configuration.")
                return []
            
    except Exception as e:
        st.error(f"Error generating responses: {str(e)}")
        return []


def start_evaluation_session():
    """Start a new evaluation session."""
    try:
        # Use absolute path from project root
        project_root = Path(__file__).parent.parent.parent.parent
        questions_file = project_root / "data" / "evaluation_questions.yaml"
        sampler = QuestionSampler(str(questions_file))
        session_data = sampler.sample_multi_domain(['retail', 'finance'], sample_size=5)
        
        st.session_state.session_data = session_data
        return True
    except Exception as e:
        st.error(f"Failed to start evaluation: {str(e)}")
        return False


def main():
    """Main application function with complex session state management."""
    st.set_page_config(
        page_title="LLM Blind Evaluation",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_complex_session_state()
    
    # Sidebar navigation using centralized template
    current_step = st.session_state.get("step", 0)
    session_id = st.session_state.get("session_id")
    PageTemplates.render_sidebar_progress(current_step, session_id)
    
    # Main content based on current step (complex state management)
    if st.session_state["step"] == 0:
        show_introduction()
    elif st.session_state["step"] == 1:
        show_evaluation()
    elif st.session_state["step"] == 2:
        show_completion()
    else:
        st.error("Invalid step. Please refresh the page.")


if __name__ == "__main__":
    main() 