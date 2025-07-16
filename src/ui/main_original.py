import streamlit as st
import json
import re
import random
import time
from datetime import datetime
from pathlib import Path
import os
import sys
import smtplib
from email.message import EmailMessage

# Add src to path if not already there
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from security.input_validator import InputValidator
from ui.components.styles import UIStyles, PageTemplates, apply_base_styles

st.set_page_config(
    page_title="LLM Blind Evaluation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def save_registration(email, signature):
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

def save_feedback(industry, question_idx, selected, comment, blind_map):
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
        "industry": industry,
        "question_idx": question_idx,
        "selected_response": selected,
        "comment": comment,
        "blind_map": blind_map,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=2)

def send_admin_email(subject, body):
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

# Initialize session state
if "step" not in st.session_state:
    st.session_state["step"] = 0  # 0: Introduction, 1: Evaluation, 2: Complete
if "registered" not in st.session_state:
    st.session_state["registered"] = False
if "current_industry" not in st.session_state:
    st.session_state["current_industry"] = "retail"
if "retail_question_idx" not in st.session_state:
    st.session_state["retail_question_idx"] = 0
if "finance_question_idx" not in st.session_state:
    st.session_state["finance_question_idx"] = 0
if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"session_{int(time.time())}"

# Load blind test data
try:
    with open("data/blind_responses.json") as f:
        blind_data = json.load(f)
except FileNotFoundError:
    st.error("❌ Blind test data not found. Please run the data generator first.")
    st.info("Run: python scripts/generate_6_questions_data.py")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading blind test data: {e}")
    st.stop()

def show_introduction():
    """Display introduction and registration form"""
    
    # Apply base styles
    apply_base_styles()
    
    # Header with consistent styling
    UIStyles.render_header(
        title="Blind LLM Evaluation",
        subtitle="Research on Free-Tier LLM Performance in Business Intelligence",
        icon="🔬",
        color="primary"
    )
    
    # Study purpose section
    st.markdown("""
    ## 🎯 Study Purpose
    
    This research evaluates the performance of free-tier Large Language Models (LLMs) in business intelligence contexts. 
    You will evaluate AI-generated responses to business questions across **retail** and **finance** industries.
    
    **Key Features:**
    - **Blind evaluation**: Response sources are hidden to ensure unbiased assessment
    - **Real business scenarios**: Questions reflect actual industry challenges
    - **Multiple AI models**: Responses from leading free-tier LLMs
    - **Research contribution**: Your feedback advances AI evaluation methodologies
    """)
    
    UIStyles.render_section_divider()
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(
            """
            ## Study Purpose
            
            This research evaluates the performance of free-tier Large Language Models (LLMs) 
            in business intelligence tasks across two key industries:
            
            - **🛍️ Retail**: Customer behavior analysis, inventory optimization, and market insights
            - **💰 Finance**: Risk assessment, fraud detection, and investment recommendations
            
            ### What You'll Do
            
            1. **Review business scenarios** from different industries
            2. **Compare AI responses** from multiple LLM providers
            3. **Select the best response** based on relevance and quality
            4. **Provide feedback** on response effectiveness
            
            ### Expected Duration
            - **10-15 minutes** for the complete evaluation (12 questions total)
            - You can pause and resume at any time
            
            ### Data Privacy
            - All responses are anonymized
            - No personal information is collected
            - Results are used for academic research only
            """
        )
        
        st.markdown("---")
        
        st.markdown("# 📝 Registration & Consent")
        
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #1f77b4; margin: 20px 0;">
        <h3 style="color: #1f77b4; margin-bottom: 10px;">📋 Important: Required Registration</h3>
        <p style="font-size: 1.2rem; line-height: 1.6; margin-bottom: 15px;">
        Before participating in this blind evaluation, you must provide your email and agree to the consent form below. 
        Your responses will be anonymized and used for research purposes only.
        </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 View Consent Form", expanded=False):
            st.markdown("""
            <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; border: 1px solid #ffeaa7; margin: 10px 0;">
            <h4 style="color: #856404; margin-bottom: 15px;">📜 Consent to Participate in Research</h4>
            <div style="font-size: 1.1rem; line-height: 1.7; color: #333;">
            <p><strong>Study Purpose:</strong> You are invited to participate in a research study evaluating AI-generated business analysis responses from different LLM providers.</p>
            <p><strong>Voluntary Participation:</strong> Your participation is completely voluntary and you may withdraw at any time without penalty.</p>
            <p><strong>Data Privacy:</strong> All data will be anonymized and used solely for academic research purposes. No personal information will be shared.</p>
            <p><strong>Consent:</strong> By checking the box below and providing your email, you consent to participate in this research study.</p>
            </div>
            </div>
            """, unsafe_allow_html=True)

        with st.form("registration_form"):
            st.markdown("### 📝 Registration Details")
            
            # Email field with prominent label
            st.markdown("""
            <div style="background-color: #e3f2fd; padding: 10px; border-radius: 8px; margin: 10px 0;">
            <h4 style="color: #1565c0; margin-bottom: 5px; font-size: 1.3rem;">📧 Email Address (Required)</h4>
            <p style="color: #1565c0; font-size: 1rem; margin-bottom: 0;">Please enter your email address to participate in this study</p>
            </div>
            """, unsafe_allow_html=True)
            email = st.text_input("Email", placeholder="your@email.com", label_visibility="collapsed")
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            
            # Signature field with prominent label
            st.markdown("""
            <div style="background-color: #e8f5e8; padding: 10px; border-radius: 8px; margin: 10px 0;">
            <h4 style="color: #2e7d32; margin-bottom: 5px; font-size: 1.3rem;">✍️ Digital Signature (Required)</h4>
            <p style="color: #2e7d32; font-size: 1rem; margin-bottom: 0;">Type your full name as a digital signature to consent to participate</p>
            </div>
            """, unsafe_allow_html=True)
            signature = st.text_input("Full Name", placeholder="John Smith", label_visibility="collapsed")
            
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            
            st.markdown("### ✅ Consent Agreement")
            st.markdown("""
            <div style="background-color: #fff3e0; padding: 10px; border-radius: 8px; margin: 10px 0;">
            <h4 style="color: #e65100; margin-bottom: 5px; font-size: 1.3rem;">✅ Required Consent</h4>
            <p style="color: #e65100; font-size: 1rem; margin-bottom: 0;">You must agree to the consent form to participate in this research study</p>
            </div>
            """, unsafe_allow_html=True)
            consent = st.checkbox("✅ I have read and agree to the consent form above.", value=False)
            st.markdown("---")
            submit = st.form_submit_button("🚀 Register and Start Test", type="primary", use_container_width=True)

            if submit:
                validator = InputValidator()
                
                # Validate and sanitize inputs
                email = validator.sanitize_text(email)
                signature = validator.sanitize_text(signature)
                
                email_valid = re.match(r"[^@]+@[^@]+\.[^@]+", email)
                if not email_valid:
                    st.markdown("""
                    <div style="background-color: #f8d7da; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545; margin: 20px 0;">
                    <h4 style="color: #721c24; margin-bottom: 10px;">❌ Invalid Email Address</h4>
                    <p style="font-size: 1.1rem; line-height: 1.6; color: #721c24; margin-bottom: 0;">
                    Please enter a valid email address in the format: user@domain.com
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif not signature.strip():
                    st.markdown("""
                    <div style="background-color: #f8d7da; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545; margin: 20px 0;">
                    <h4 style="color: #721c24; margin-bottom: 10px;">❌ Digital Signature Required</h4>
                    <p style="font-size: 1.1rem; line-height: 1.6; color: #721c24; margin-bottom: 0;">
                    Please type your full name as a digital signature to proceed.
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif len(signature) > 100:  # Reasonable name length limit
                    st.markdown("""
                    <div style="background-color: #f8d7da; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545; margin: 20px 0;">
                    <h4 style="color: #721c24; margin-bottom: 10px;">❌ Name Too Long</h4>
                    <p style="font-size: 1.1rem; line-height: 1.6; color: #721c24; margin-bottom: 0;">
                    Name too long. Please use a shorter version (maximum 100 characters).
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif not consent:
                    st.markdown("""
                    <div style="background-color: #f8d7da; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545; margin: 20px 0;">
                    <h4 style="color: #721c24; margin-bottom: 10px;">❌ Consent Required</h4>
                    <p style="font-size: 1.1rem; line-height: 1.6; color: #721c24; margin-bottom: 0;">
                    You must agree to the consent form to participate in this research study.
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    save_registration(email, signature)
                    st.session_state['registered'] = True
                    st.session_state['tester_email'] = email
                    st.session_state['step'] = 1
                    st.success("Registration complete! Starting the blind test...")
                    st.balloons()
                    st.rerun()

        if not st.session_state.get('registered'):
            st.markdown("""
            <div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 20px 0;">
            <h4 style="color: #0c5460; margin-bottom: 10px;">⚠️ Registration Required</h4>
            <p style="font-size: 1.1rem; line-height: 1.6; color: #0c5460; margin-bottom: 0;">
            Please complete the registration and consent form above to continue with the blind evaluation.
            </p>
            </div>
            """, unsafe_allow_html=True)

def show_evaluation():
    """Display the main evaluation interface"""
    # Header section using pure Streamlit
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🔬 Blind LLM Evaluation")
        st.markdown("### Compare AI responses and select the best one for each scenario")
        st.markdown("")

    # Show registered email
    if st.session_state.get('tester_email'):
        st.info(f"You are participating as: {st.session_state['tester_email']}")

    # Industry selection
    current_industry = st.session_state["current_industry"]
    industries = ["retail", "finance"]

    # Progress indicator
    retail_progress = st.session_state["retail_question_idx"]
    finance_progress = st.session_state["finance_question_idx"]
    total_questions = len(blind_data["retail"]) + len(blind_data["finance"])
    completed_questions = retail_progress + finance_progress

    # Calculate overall progress
    progress_percentage = completed_questions / total_questions if total_questions > 0 else 0
    
    # Progress indicator
    progress_col1, progress_col2, progress_col3 = st.columns([1, 2, 1])
    with progress_col2:
        st.progress(progress_percentage, text=f"Overall Progress: {completed_questions}/{total_questions} questions")

    st.markdown(f"### 📊 Progress: {completed_questions}/{total_questions} questions completed")

    # Industry progress bars
    col1, col2 = st.columns(2)
    with col1:
        retail_percent = (retail_progress / len(blind_data["retail"])) * 100
        st.progress(retail_percent / 100, text=f"🛍️ Retail: {retail_progress}/{len(blind_data['retail'])}")

    with col2:
        finance_percent = (finance_progress / len(blind_data["finance"])) * 100
        st.progress(finance_percent / 100, text=f"💰 Finance: {finance_progress}/{len(blind_data['finance'])}")

    # Current industry and question
    if current_industry == "retail":
        question_idx = st.session_state["retail_question_idx"]
        questions = blind_data["retail"]
        industry_name = "🛍️ Retail"
    elif current_industry == "finance":
        question_idx = st.session_state["finance_question_idx"]
        questions = blind_data["finance"]
        industry_name = "💰 Finance"

    if question_idx < len(questions):
        st.markdown(f"## {industry_name}: Question {question_idx + 1} of {len(questions)}")
        
        question = questions[question_idx]
        
        st.markdown(f"**Scenario:** {question['prompt']}")
        st.markdown(f"**Context:** {question['context']}")
        
        # Get responses and create blind mapping
        responses = question["responses"]
        response_ids = [resp['id'] for resp in responses]
        labels = ['A', 'B', 'C', 'D', 'E', 'F'][:len(responses)]
        
        # Create shuffled mapping
        shuffled_ids = response_ids.copy()
        random.shuffle(shuffled_ids)
        blind_map = dict(zip(labels, shuffled_ids))
        
        # Store mapping in session state
        st.session_state['blind_map'] = blind_map
        
        # Display responses as cards
        st.markdown("### 🤖 AI Response Comparison")
        st.markdown("Below are AI-generated responses to the business scenario above. Each response is labeled A-F and presented in random order to ensure unbiased evaluation.")
        
        # Display response cards as tiles (2 per row)
        response_lookup = {resp['id']: resp for resp in responses}
        
        for i in range(0, len(labels), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(labels):
                    label = labels[i + j]
                    response_id = blind_map[label]
                    response = response_lookup[response_id]
                    
                    with cols[j]:
                        # Use enhanced card styling
                        st.markdown(f"""
                        <div style="background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 10px 0; border: 1px solid #e0e0e0;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #f0f0f0;">
                                <h3 style="color: #1f77b4; margin: 0; font-size: 1.4rem; font-weight: 600;">🤖 Response {label}</h3>
                                <span style="background-color: #1f77b4; color: white; padding: 5px 12px; border-radius: 20px; font-weight: 600; font-size: 1rem;">{label}</span>
                            </div>
                            <div style="line-height: 1.7; font-size: 1.1rem; color: #333; text-align: justify;">
                                {response['content']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Selection interface
        st.markdown("---")
        st.markdown("### 📝 Your Evaluation")
        
        selected = st.radio(
            "**Which response do you think is most helpful for this business scenario?**",
            labels,
            format_func=lambda x: f"Response {x}"
        )
        
        comment = st.text_area(
            "**Please provide your reasoning (optional but helpful):**",
            placeholder="I selected this response because...",
            height=100
        )
        
        if st.button("Submit and Continue", type="primary", use_container_width=True):
            # Save selection
            save_feedback(current_industry, question_idx, selected, comment, blind_map)
            
            # Move to next question or industry
            if current_industry == "retail":
                if question_idx + 1 < len(questions):
                    st.session_state["retail_question_idx"] += 1
                else:
                    # Move to finance
                    st.session_state["current_industry"] = "finance"
                    st.session_state["retail_question_idx"] = 0
            elif current_industry == "finance":
                if question_idx + 1 < len(questions):
                    st.session_state["finance_question_idx"] += 1
                else:
                    # Test complete
                    st.session_state["finance_question_idx"] = 0
                    st.session_state["current_industry"] = "retail"
                    st.session_state["step"] = 2
                    
                    # Send admin email notification
                    if not st.session_state.get("admin_email_sent"):
                        sent = send_admin_email(
                            subject="Blind Test Evaluation Complete",
                            body=f"A tester ({st.session_state.get('tester_email','unknown')}) has completed the full blind evaluation. Please log in to the admin dashboard to review."
                        )
                        st.session_state["admin_email_sent"] = True
                        if sent:
                            st.toast("Admin notified of completion.")
                        else:
                            st.toast("Admin notification failed. Check SMTP settings.")
                    
                    st.rerun()
            
            st.rerun()

    else:
        # All questions completed for current industry
        st.success(f"✅ All {industry_name} questions completed!")
        st.info("Moving to next industry...")
        st.rerun()

def show_completion():
    """Display completion screen with evaluation summary"""
    # Create user summary
    user_summary = {
        'retail_completed': st.session_state.get("retail_question_idx", 0) >= 6,
        'finance_completed': st.session_state.get("finance_question_idx", 0) >= 6
    }
    
    # Use the centralized completion page template
    PageTemplates.render_completion_page(user_summary=user_summary, show_restart_button=True)

# Sidebar navigation using centralized template
current_step = st.session_state.get("step", 0)
session_id = st.session_state.get("session_id")
PageTemplates.render_sidebar_progress(current_step, session_id)

# Main content based on current step
if st.session_state["step"] == 0:
    show_introduction()
elif st.session_state["step"] == 1:
    show_evaluation()
elif st.session_state["step"] == 2:
    show_completion()
else:
    st.error("Invalid step. Please refresh the page.")
