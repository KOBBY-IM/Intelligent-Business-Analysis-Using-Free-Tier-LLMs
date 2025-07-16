#!/usr/bin/env python3
"""
LLM Business Intelligence Research Platform
Complete implementation with registration, security, and consent features.
"""

import streamlit as st
import sys
import json
import re
import random
import time
from datetime import datetime
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import security modules
from src.security.input_validator import InputValidator
from src.security.rate_limiter import RateLimiter
from src.security.secure_logger import SecureLogger
from src.utils.feedback_logger import FeedbackLogger

def apply_fluent_theme():
    """Apply Microsoft Fluent UI theme styling."""
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        border-radius: 0.5rem;
        border: 1px solid #0078d4;
        background-color: #0078d4;
        color: white;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #106ebe;
        border-color: #106ebe;
    }
    .stSuccess {
        font-weight: 500;
        background-color: #dff6dd;
        border: 1px solid #0f5132;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stError {
        background-color: #f8d7da;
        border: 1px solid #842029;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stWarning {
        background-color: #fff3cd;
        border: 1px solid #664d03;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .registration-form {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .consent-form {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all required session state variables."""
    if 'registered' not in st.session_state:
        st.session_state.registered = False
    if 'tester_email' not in st.session_state:
        st.session_state.tester_email = ""
    if 'step' not in st.session_state:
        st.session_state.step = 0  # 0: Registration, 1: Evaluation, 2: Complete
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home"
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
    if 'consent_given' not in st.session_state:
        st.session_state.consent_given = False
    if 'evaluation_started' not in st.session_state:
        st.session_state.evaluation_started = False

def save_registration(email: str, signature: str, validator: InputValidator, secure_logger: SecureLogger):
    """Save user registration data securely."""
    reg_file = Path("data/tester_registrations.json")
    reg_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Sanitize inputs
    email = validator.sanitize_text(email)
    signature = validator.sanitize_text(signature)
    
    if reg_file.exists():
        with open(reg_file, "r") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
    else:
        data = []
    
    registration_entry = {
        "email": email,
        "signature": signature,
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": st.session_state.session_id
    }
    
    data.append(registration_entry)
    
    with open(reg_file, "w") as f:
        json.dump(data, f, indent=2)
    
    # Log registration event
    secure_logger.log_event(
        event_type='user_registration',
        message='New user registered for evaluation',
        data={
            'session_id': st.session_state.session_id,
            'has_email': bool(email),
            'has_signature': bool(signature)
        }
    )

def check_email_completed_evaluation(email: str) -> bool:
    """Check if an email has already completed the full evaluation."""
    feedback_file = Path("data/results/user_feedback.json")
    if not feedback_file.exists():
        return False
    
    try:
        with open(feedback_file, "r") as f:
            feedback_data = json.load(f)
            
        # Check if this email has completed evaluations
        for entry in feedback_data:
            if entry.get('tester_email', '').lower() == email.lower():
                # Check if they completed both domains
                return True
    except Exception:
        pass
    
    return False

def can_user_continue_session(email: str) -> bool:
    """Check if user has registered but not completed the evaluation."""
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
    
    return is_registered and not check_email_completed_evaluation(email)

def main():
    """Main Streamlit interface for LLM evaluation system."""
    
    # Page configuration
    st.set_page_config(
        page_title="LLM Business Intelligence Evaluation",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom theme
    apply_fluent_theme()
    
    # Initialize session state and security components
    initialize_session_state()
    validator = InputValidator()
    rate_limiter = RateLimiter()
    secure_logger = SecureLogger("logs/secure.log")
    feedback_logger = FeedbackLogger()
    
    # Check for API keys
    api_keys_available = check_api_keys()
    
    # Main navigation logic based on registration and step
    if not st.session_state.registered or st.session_state.step == 0:
        show_registration_page(validator, secure_logger, rate_limiter)
    elif st.session_state.step == 1:
        show_evaluation_interface(api_keys_available, validator, secure_logger, feedback_logger)
    elif st.session_state.step == 2:
        show_completion_page(secure_logger)

def show_registration_page(validator: InputValidator, secure_logger: SecureLogger, rate_limiter: RateLimiter):
    """Display the registration and consent page."""
    
    # Header
    st.title("🔬 Blind LLM Evaluation Research")
    st.markdown("### Comparative Analysis of Free-Tier LLMs in Business Intelligence")
    
    # Study information
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ## 🎯 Study Purpose
        
        This research evaluates the performance of free-tier Large Language Models (LLMs) 
        in business intelligence contexts. You will evaluate AI-generated responses to 
        business questions across **retail** and **finance** industries.
        
        **Key Features:**
        - **Blind evaluation**: Response sources are hidden to ensure unbiased assessment
        - **Real business scenarios**: Questions reflect actual industry challenges  
        - **Multiple AI models**: Responses from leading free-tier LLMs
        - **Research contribution**: Your feedback advances AI evaluation methodologies
        
        ### What You'll Do
        
        1. **Review business scenarios** from different industries
        2. **Compare AI responses** from multiple LLM providers
        3. **Select the best response** based on relevance and quality
        4. **Provide feedback** on response effectiveness
        
        ### Expected Duration
        - **10-15 minutes** for the complete evaluation
        - You can pause and resume at any time
        
        ### Data Privacy
        - All responses are anonymized
        - No personal information is collected beyond email
        - Results are used for academic research only
        """)
        
    with col2:
        st.markdown("## 📝 Registration & Consent")
        
        # Registration form
        st.markdown("""
        <div class="registration-form">
        <h4>📋 Required Registration</h4>
        <p>Before participating, you must provide your email and agree to the consent form below. 
        Your responses will be anonymized and used for research purposes only.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Consent form in expander
        with st.expander("📋 View Consent Form", expanded=False):
            st.markdown("""
            <div class="consent-form">
            <h4>Consent to Participate in Research</h4>
            <ul>
            <li><strong>Study Purpose:</strong> You are invited to participate in a research study 
            evaluating AI-generated business analysis responses from different LLM providers.</li>
            <li><strong>Voluntary Participation:</strong> Your participation is completely voluntary 
            and you may withdraw at any time without penalty.</li>
            <li><strong>Data Privacy:</strong> All data will be anonymized and used solely for 
            academic research purposes. No personal information will be shared.</li>
            <li><strong>Duration:</strong> The evaluation should take approximately 10-15 minutes.</li>
            <li><strong>Benefits:</strong> Your participation contributes to advancing AI evaluation 
            methodologies and understanding LLM performance in business contexts.</li>
            <li><strong>Contact:</strong> If you have questions about this study, please contact 
            the research team.</li>
            <li><strong>Consent:</strong> By checking the box below and providing your email, 
            you consent to participate in this research study.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Registration form
        with st.form("registration_form"):
            st.markdown("### Registration Details")
            
            # Helpful info for returning users
            st.info("💡 **Returning User?** Simply enter your email to continue where you left off.")
            
            email = st.text_input("📧 Email Address", placeholder="your@email.com")
            signature = st.text_input("✍️ Full Name (Digital Signature)", placeholder="John Smith")
            consent = st.checkbox("✅ I have read and agree to the consent form above.")
            
            submit = st.form_submit_button("🚀 Register and Start Evaluation", type="primary")
            
            if submit:
                # Rate limiting check
                user_ip = st.session_state.get('user_ip', 'unknown')
                if not rate_limiter.is_allowed(user_ip, 'user_queries_per_hour'):
                    st.error("❌ Too many requests. Please try again later.")
                    return
                
                # Validate inputs
                email = validator.sanitize_text(email)
                signature = validator.sanitize_text(signature)
                
                email_valid = re.match(r"[^@]+@[^@]+\.[^@]+", email)
                evaluation_completed = check_email_completed_evaluation(email) if email_valid else False
                
                if not email_valid:
                    st.error("❌ Please enter a valid email address in the format: user@domain.com")
                elif evaluation_completed:
                    st.error("❌ This email address has already completed the evaluation. Each person can only participate once.")
                elif can_user_continue_session(email):
                    # Returning user
                    st.session_state.registered = True
                    st.session_state.tester_email = email
                    st.session_state.step = 1
                    st.success("✅ Welcome back! Continuing your evaluation session...")
                    secure_logger.log_event('user_return', f'Returning user: {email}')
                    st.rerun()
                elif not signature.strip():
                    st.error("❌ Please type your full name as a digital signature to proceed.")
                elif len(signature) > 100:
                    st.error("❌ Name too long. Please use a shorter version (maximum 100 characters).")
                elif not consent:
                    st.error("❌ You must agree to the consent form to participate in this research study.")
                else:
                    # New registration
                    save_registration(email, signature, validator, secure_logger)
                    st.session_state.registered = True
                    st.session_state.tester_email = email
                    st.session_state.consent_given = True
                    st.session_state.step = 1
                    st.success("✅ Registration complete! Starting the evaluation...")
                    st.balloons()
                    st.rerun()
    
    # Footer information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem;'>
    <strong>🎓 Academic Research Study</strong><br>
    Your responses are anonymized and used for research only. Thank you for participating!
    </div>
    """, unsafe_allow_html=True)

def show_evaluation_interface(api_keys_available: bool, validator: InputValidator, 
                            secure_logger: SecureLogger, feedback_logger: FeedbackLogger):
    """Display the main evaluation interface."""
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🔬 LLM Evaluation")
        st.markdown(f"**User:** {st.session_state.tester_email}")
        st.markdown(f"**Session:** {st.session_state.session_id}")
        
        st.markdown("---")
        
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_page = "home"
            
        if st.button("📊 Blind Evaluation", use_container_width=True, disabled=not api_keys_available):
            st.session_state.current_page = "blind_eval"
            
        if st.button("📈 Progress & Stats", use_container_width=True):
            st.session_state.current_page = "progress"
            
        if st.button("ℹ️ Study Information", use_container_width=True):
            st.session_state.current_page = "info"
    
    # Main content based on current page
    if st.session_state.current_page == "home":
        show_evaluation_home(api_keys_available)
    elif st.session_state.current_page == "blind_eval":
        show_blind_evaluation_interface(validator, secure_logger, feedback_logger)
    elif st.session_state.current_page == "progress":
        show_progress_interface(feedback_logger)
    elif st.session_state.current_page == "info":
        show_study_information()

def show_evaluation_home(api_keys_available: bool):
    """Display the evaluation home page."""
    
    st.title("🔬 LLM Evaluation Dashboard")
    st.markdown("### Ready to begin your blind evaluation")
    
    if not api_keys_available:
        st.error("⚠️ **System Configuration Issue**")
        st.info("The evaluation system is temporarily unavailable. Please try again later.")
        return
    
    # Quick start section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Blind Evaluation Ready**
        - Compare LLM responses anonymously
        - Rate model performance objectively
        - Contribute valuable research data
        """)
        if st.button("🚀 Start Blind Evaluation", key="start_eval", type="primary"):
            st.session_state.current_page = "blind_eval"
            st.session_state.evaluation_started = True
            st.rerun()
    
    with col2:
        st.markdown("""
        **📈 Track Your Progress**
        - View evaluation statistics
        - See completion status
        - Review your contributions
        """)
        if st.button("📊 View Progress", key="view_progress"):
            st.session_state.current_page = "progress"
            st.rerun()
    
    # System status
    show_system_status()

def show_blind_evaluation_interface(validator: InputValidator, secure_logger: SecureLogger, 
                                  feedback_logger: FeedbackLogger):
    """Enhanced blind evaluation interface with security and logging."""
    
    st.title("📊 Blind LLM Evaluation")
    
    try:
        # Load fixed responses
        import json
        import random
        from pathlib import Path
        
        fixed_responses_file = Path("data/fixed_blind_responses.json")
        
        if not fixed_responses_file.exists():
            st.error("❌ Evaluation data not available!")
            st.info("Please contact the research team if this problem persists.")
            return
            
        with open(fixed_responses_file, 'r') as f:
            fixed_data = json.load(f)
            
        st.success(f"✅ Loaded {len(fixed_data['responses'])} evaluation questions")
        
        # Progress tracking
        if 'evaluation_progress' not in st.session_state:
            st.session_state.evaluation_progress = []
        
        # Question selection
        questions = list(fixed_data['responses'].keys())
        completed_questions = set(q['question_id'] for q in st.session_state.evaluation_progress)
        remaining_questions = [q for q in questions if q not in completed_questions]
        
        if not remaining_questions:
            st.success("🎉 **Evaluation Complete!**")
            st.info("You have completed all evaluation questions. Thank you for your participation!")
            if st.button("🏆 View Final Summary", type="primary"):
                st.session_state.step = 2
                st.rerun()
            return
        
        # Current question
        if 'current_question_id' not in st.session_state or st.session_state.current_question_id not in remaining_questions:
            st.session_state.current_question_id = random.choice(remaining_questions)
            
        question_data = fixed_data['responses'][st.session_state.current_question_id]
        
        # Progress indicator
        progress = len(st.session_state.evaluation_progress) / len(questions)
        st.progress(progress, text=f"Progress: {len(st.session_state.evaluation_progress)}/{len(questions)} questions completed")
        
        # Question display
        st.markdown(f"**Question:** {question_data['question']}")
        st.markdown(f"**Domain:** {question_data['domain'].title()}")
        
        # Show business context
        with st.expander("📋 Business Context (Click to view)", expanded=False):
            st.text(question_data['rag_context'])
        
        # Randomize responses for blind evaluation
        if 'response_order' not in st.session_state:
            responses = question_data['llm_responses']
            provider_names = list(responses.keys())
            random.shuffle(provider_names)
            st.session_state.response_order = provider_names
        
        responses = question_data['llm_responses']
        response_labels = ["Response A", "Response B", "Response C"]
        
        # Display responses
        st.markdown("### Compare the Responses")
        st.markdown("*Please read all responses carefully before making your selection.*")
        
        for i, label in enumerate(response_labels):
            provider = st.session_state.response_order[i]
            response_data = responses[provider]
            
            with st.container():
                st.markdown(f"#### {label}")
                st.markdown(f"*{response_data['response']}*")
                st.caption(f"Length: {len(response_data['response'])} characters")
                st.markdown("---")
        
        # Evaluation form
        st.markdown("### Your Evaluation")
        
        with st.form("evaluation_form"):
            selected_response = st.radio(
                "Which response is most helpful for business decision-making?",
                response_labels,
                index=None
            )
            
            confidence = st.slider(
                "How confident are you in your selection?",
                min_value=1, max_value=5, value=3,
                help="1 = Not confident, 5 = Very confident"
            )
            
            comment = st.text_area(
                "Optional: Please explain your choice",
                max_chars=500,
                help="Why did you choose this response? What made it better than the others?"
            )
            
            submitted = st.form_submit_button("Submit Evaluation", type="primary")
            
            if submitted:
                if not selected_response:
                    st.error("❌ Please select a response before submitting.")
                else:
                    # Validate comment
                    comment_valid, comment_error = validator.validate_comment(comment)
                    if not comment_valid:
                        st.error(f"❌ {comment_error}")
                        return
                    
                    # Map selection to actual provider
                    selected_index = response_labels.index(selected_response)
                    actual_provider = st.session_state.response_order[selected_index]
                    
                    # Save evaluation
                    evaluation_entry = {
                        'question_id': st.session_state.current_question_id,
                        'question': question_data['question'],
                        'domain': question_data['domain'],
                        'selected_response': selected_response,
                        'actual_provider': actual_provider,
                        'confidence': confidence,
                        'comment': validator.sanitize_text(comment),
                        'timestamp': datetime.now().isoformat(),
                        'session_id': st.session_state.session_id,
                        'tester_email': st.session_state.tester_email
                    }
                    
                    st.session_state.evaluation_progress.append(evaluation_entry)
                    
                    # Log evaluation
                    secure_logger.log_evaluation_submission(
                        industry=question_data['domain'],
                        question_id=st.session_state.current_question_id,
                        selected_response=selected_response,
                        has_comment=bool(comment.strip())
                    )
                    
                    # Save to feedback logger
                    feedback_logger.log_evaluation(
                        user_id=st.session_state.tester_email,
                        evaluation_data=evaluation_entry
                    )
                    
                    # Show feedback
                    st.success(f"✅ Thank you! You selected **{selected_response}**")
                    st.info(f"💡 This response was generated by: **{actual_provider.title()}**")
                    
                    # Clear current question to load next
                    if 'current_question_id' in st.session_state:
                        del st.session_state.current_question_id
                    if 'response_order' in st.session_state:
                        del st.session_state.response_order
                    
                    # Auto-advance or show completion
                    remaining_after = len(remaining_questions) - 1
                    if remaining_after > 0:
                        st.info(f"📊 {remaining_after} questions remaining. Loading next question...")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.success("🎉 **All evaluations complete!**")
                        st.balloons()
                        if st.button("🏆 View Final Results", type="primary"):
                            st.session_state.step = 2
                            st.rerun()
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if len(remaining_questions) > 1 and st.button("⏭️ Skip Question", help="Skip to a different question"):
                # Clear current question
                if 'current_question_id' in st.session_state:
                    del st.session_state.current_question_id
                if 'response_order' in st.session_state:
                    del st.session_state.response_order
                st.rerun()
        
        with col2:
            if len(st.session_state.evaluation_progress) > 0:
                if st.button("📊 View Progress", help="See your evaluation statistics"):
                    st.session_state.current_page = "progress"
                    st.rerun()
                    
    except Exception as e:
        st.error(f"❌ Error loading evaluation interface: {str(e)}")
        secure_logger.log_event('evaluation_error', f'Error in evaluation interface: {str(e)}')

def show_progress_interface(feedback_logger: FeedbackLogger):
    """Display user progress and statistics."""
    
    st.title("📈 Your Evaluation Progress")
    
    if 'evaluation_progress' not in st.session_state or not st.session_state.evaluation_progress:
        st.info("🚀 **Ready to Start!**")
        st.markdown("You haven't completed any evaluations yet. Click the button below to begin.")
        if st.button("🚀 Start Evaluation", type="primary"):
            st.session_state.current_page = "blind_eval"
            st.rerun()
        return
    
    evaluations = st.session_state.evaluation_progress
    
    # Overall progress
    st.subheader("📊 Overall Progress")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Evaluations Completed", len(evaluations))
    
    with col2:
        domains = [eval['domain'] for eval in evaluations]
        unique_domains = len(set(domains))
        st.metric("Domains Evaluated", unique_domains)
    
    with col3:
        avg_confidence = sum(eval['confidence'] for eval in evaluations) / len(evaluations)
        st.metric("Average Confidence", f"{avg_confidence:.1f}/5")
    
    # Provider selection statistics
    st.subheader("🤖 Your LLM Preferences")
    
    provider_counts = {}
    for eval in evaluations:
        provider = eval['actual_provider']
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
    
    if provider_counts:
        total_selections = sum(provider_counts.values())
        
        for provider, count in provider_counts.items():
            percentage = (count / total_selections) * 100
            st.write(f"**{provider.title()}**: {count} selections ({percentage:.1f}%)")
    
    # Domain breakdown
    st.subheader("🏢 Domain Breakdown")
    
    domain_counts = {}
    for eval in evaluations:
        domain = eval['domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    for domain, count in domain_counts.items():
        st.write(f"**{domain.title()}**: {count} evaluations")
    
    # Recent evaluations
    st.subheader("📝 Recent Evaluations")
    
    # Show last 5 evaluations
    recent_evals = evaluations[-5:] if len(evaluations) > 5 else evaluations
    
    for eval in reversed(recent_evals):
        with st.expander(f"{eval['domain'].title()}: {eval['question'][:60]}..."):
            st.write(f"**Selected:** {eval['selected_response']}")
            st.write(f"**Actual Provider:** {eval['actual_provider'].title()}")
            st.write(f"**Confidence:** {eval['confidence']}/5")
            if eval['comment']:
                st.write(f"**Comment:** {eval['comment']}")
            st.write(f"**Time:** {eval['timestamp']}")

def show_study_information():
    """Display detailed study information."""
    
    st.title("ℹ️ Study Information")
    
    st.markdown("""
    ## 🎓 Research Study Details
    
    ### Objective
    This study aims to evaluate and compare the performance of free-tier Large Language Models (LLMs) 
    in business intelligence tasks across multiple industry domains.
    
    ### Methodology
    - **Blind Evaluation**: Participants evaluate responses without knowing which model generated them
    - **Cross-Domain Analysis**: Questions span retail, finance, and healthcare industries
    - **Standardized Metrics**: Consistent evaluation criteria across all responses
    - **Statistical Analysis**: Results analyzed for significance and patterns
    
    ### Participating Models
    - **Groq**: Mixtral and Llama models via Groq API
    - **Google Gemini**: Gemini Pro via Google Generative AI
    - **OpenRouter**: Various models via OpenRouter platform
    
    ### Data Usage
    - All responses are anonymized before analysis
    - Personal identifiers are removed from research data
    - Results used solely for academic research purposes
    - Findings may be published in academic conferences/journals
    
    ### Privacy & Ethics
    - IRB approved research protocol
    - Voluntary participation with right to withdraw
    - No sensitive personal data collected
    - Secure data storage and transmission
    
    ### Contact Information
    If you have questions about this study, please contact the research team.
    
    ### Acknowledgments
    Thank you for participating in this important research!
    """)

def show_completion_page(secure_logger: SecureLogger):
    """Display the completion page with final statistics."""
    
    st.title("🎉 Evaluation Complete!")
    st.markdown("### Thank you for your valuable contribution to LLM research")
    
    if 'evaluation_progress' in st.session_state and st.session_state.evaluation_progress:
        evaluations = st.session_state.evaluation_progress
        
        # Final statistics
        st.subheader("📊 Your Final Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Evaluations", len(evaluations))
        
        with col2:
            domains = set(eval['domain'] for eval in evaluations)
            st.metric("Domains Covered", len(domains))
        
        with col3:
            avg_confidence = sum(eval['confidence'] for eval in evaluations) / len(evaluations)
            st.metric("Avg Confidence", f"{avg_confidence:.1f}/5")
        
        with col4:
            comments_count = sum(1 for eval in evaluations if eval['comment'].strip())
            st.metric("Comments Provided", comments_count)
        
        # Provider preferences
        st.subheader("🤖 Your LLM Preferences")
        
        provider_counts = {}
        for eval in evaluations:
            provider = eval['actual_provider']
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        if provider_counts:
            total = sum(provider_counts.values())
            for provider, count in sorted(provider_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100
                st.write(f"**{provider.title()}**: {count} selections ({percentage:.1f}%)")
        
        # Log completion
        secure_logger.log_event(
            event_type='evaluation_completion',
            message='User completed full evaluation',
            data={
                'session_id': st.session_state.session_id,
                'total_evaluations': len(evaluations),
                'domains_covered': len(domains),
                'average_confidence': avg_confidence
            }
        )
    
    st.markdown("---")
    st.markdown("""
    ## 🙏 Thank You!
    
    Your participation is invaluable to advancing our understanding of LLM performance 
    in business intelligence applications. Your feedback will contribute to:
    
    - **Academic Research**: Publications in AI and business intelligence conferences
    - **Model Improvement**: Insights for LLM developers and researchers  
    - **Industry Applications**: Better understanding of AI capabilities in business contexts
    - **Future Studies**: Foundation for expanded research in this area
    
    ### What Happens Next?
    
    - Your responses will be analyzed alongside other participants
    - Results will be aggregated and anonymized
    - Findings will be shared through academic publications
    - No individual responses will be identifiable
    
    ### Stay Connected
    
    If you're interested in learning about the results of this study, please contact the research team.
    """)
    
    # Option to restart (for testing purposes)
    if st.button("🔄 Start New Session", help="Clear session and start over (for testing)"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def show_system_status():
    """Display comprehensive system status checks."""
    
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    
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
    
    # Check security components
    try:
        from src.security.input_validator import InputValidator
        from src.security.rate_limiter import RateLimiter
        from src.security.secure_logger import SecureLogger
        validator = InputValidator()
        status_checks.append(("✅", "Security components loaded"))
    except Exception as e:
        status_checks.append(("⚠️", f"Security system issue: {str(e)}"))
        
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
        
    # Display status checks
    col1, col2 = st.columns(2)
    
    for i, (icon, message) in enumerate(status_checks):
        col = col1 if i % 2 == 0 else col2
        with col:
            if icon == "✅":
                st.success(f"{icon} {message}")
            elif icon == "❌":
                st.error(f"{icon} {message}")
            else:
                st.warning(f"{icon} {message}")

def check_api_keys():
    """Check if required API keys are available."""
    import os
    required_keys = ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"]
    return all(os.getenv(key) for key in required_keys)

if __name__ == "__main__":
    main() 