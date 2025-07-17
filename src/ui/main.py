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
    
    # Initialize secure logger with fallback for Streamlit Cloud
    try:
        secure_logger = SecureLogger("logs/secure.log")
    except Exception as e:
        # Fallback to console-only logging on Streamlit Cloud
        secure_logger = SecureLogger(None)
        st.warning(f"Using console logging due to file system restrictions: {e}")
    
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
    elif st.session_state.step == 3:
        show_study_information()

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
            st.session_state.step = 0
            st.rerun()
            
        if st.button("📊 Blind Evaluation", use_container_width=True, disabled=not api_keys_available):
            st.session_state.step = 1
            st.rerun()
            
        if st.button("📈 Progress & Stats", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
            
        if st.button("ℹ️ Study Information", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    
    # Main content based on step
    if st.session_state.step == 0:
        show_evaluation_home(api_keys_available)
    elif st.session_state.step == 1:
        show_blind_evaluation_interface(validator, secure_logger, feedback_logger)
    elif st.session_state.step == 2:
        show_progress_interface(feedback_logger)
    elif st.session_state.step == 3:
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
            st.session_state.step = 1
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
            st.session_state.step = 2
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
        
        # Try to load enhanced responses first, then fallback to others
        fixed_responses_file = Path("data/enhanced_blind_responses.json")
        if not fixed_responses_file.exists():
            fixed_responses_file = Path("data/fixed_blind_responses_with_guidance.json")
        if not fixed_responses_file.exists():
            fixed_responses_file = Path("data/fixed_blind_responses.json")
        
        if not fixed_responses_file.exists():
            st.error("❌ Evaluation data not available!")
            st.info("Please contact the research team if this problem persists.")
            return
            
        with open(fixed_responses_file, 'r') as f:
            fixed_data = json.load(f)
            
        # Check if this is enhanced data
        is_enhanced = "enhanced_blind_responses.json" in str(fixed_responses_file)
        if is_enhanced and "improvements" in fixed_data:
            st.success("🚀 **Enhanced RAG System Active!**")
            with st.expander("📊 Enhanced Features", expanded=False):
                st.markdown("### RAG Improvements")
                for feature, description in fixed_data["improvements"].items():
                    st.markdown(f"• **{feature.replace('_', ' ').title()}**: {description}")
                st.markdown("### Benefits")
                st.markdown("• **Better Coverage**: 8 chunks instead of 5")
                st.markdown("• **Higher Quality**: 150 rows per chunk instead of 200")
                st.markdown("• **Dynamic Guidance**: Query-based analysis focus")
                st.markdown("• **Enhanced Analysis**: Outlier detection and trend analysis")
                st.markdown("• **Cross-Chunk Patterns**: Pattern detection across chunks")
        
        # Initialize user's question set if not already done
        if 'user_question_set' not in st.session_state:
            # Separate questions by domain
            retail_questions = [q for q in fixed_data['responses'].keys() if q.startswith('retail_')]
            finance_questions = [q for q in fixed_data['responses'].keys() if q.startswith('finance_')]
            
            # Randomly select 5 from each domain
            selected_retail = random.sample(retail_questions, min(5, len(retail_questions)))
            selected_finance = random.sample(finance_questions, min(5, len(finance_questions)))
            
            st.session_state.user_question_set = selected_retail + selected_finance
            st.session_state.total_questions = len(st.session_state.user_question_set)
            
            # Log the question selection
            secure_logger.log_event(
                event_type='question_set_assigned',
                message=f"Assigned {len(selected_retail)} retail + {len(selected_finance)} finance questions",
                data={
                    'retail_questions': selected_retail,
                    'finance_questions': selected_finance,
                    'total_questions': st.session_state.total_questions
                }
            )
        
        # Progress tracking
        if 'evaluation_progress' not in st.session_state:
            st.session_state.evaluation_progress = []
        
        # Question selection from user's assigned set
        completed_questions = set(q['question_id'] for q in st.session_state.evaluation_progress)
        remaining_questions = [q for q in st.session_state.user_question_set if q not in completed_questions]
        
        if not remaining_questions:
            st.success("🎉 **Evaluation Complete!**")
            st.info("You have completed all 10 evaluation questions. Thank you for your participation!")
            if st.button("🏆 View Final Summary", type="primary"):
                st.session_state.step = 2
                st.rerun()
            return
        
        # Current question
        if 'current_question_id' not in st.session_state or st.session_state.current_question_id not in remaining_questions:
            st.session_state.current_question_id = random.choice(remaining_questions)
            
        question_data = fixed_data['responses'][st.session_state.current_question_id]
        
        # Progress indicator
        progress = len(st.session_state.evaluation_progress) / st.session_state.total_questions
        st.progress(progress, text=f"Progress: {len(st.session_state.evaluation_progress)}/{st.session_state.total_questions} questions completed")
        
        # Question display
        st.markdown(f"**Question:** {question_data['question']}")
        st.markdown(f"**Domain:** {question_data['domain'].title()}")
        
        # Show ground truth comparison
        with st.expander("🎯 Ground Truth & Business Context", expanded=False):
            # Load actual ground truth from ground_truth_answers.json
            try:
                ground_truth_file = Path("data/ground_truth_answers.json")
                if ground_truth_file.exists():
                    with open(ground_truth_file, 'r') as f:
                        ground_truth_data = json.load(f)
                    
                    domain = question_data['domain']
                    question_id = st.session_state.current_question_id
                    
                    if domain in ground_truth_data and question_id in ground_truth_data[domain]:
                        gt_answer = ground_truth_data[domain][question_id]
                        
                        st.markdown("### 🎯 Expert Ground Truth Answer")
                        st.success(f"**Answer:** {gt_answer['answer']}")
                        
                        # Show key points if available
                        if 'key_points' in gt_answer and gt_answer['key_points']:
                            st.markdown("**🔑 Key Data Points:**")
                            for point in gt_answer['key_points']:
                                st.markdown(f"• {point}")
                        
                        # Show factual claims if available  
                        if 'factual_claims' in gt_answer and gt_answer['factual_claims']:
                            st.markdown("**📊 Factual Claims to Verify:**")
                            for claim in gt_answer['factual_claims']:
                                st.markdown(f"• {claim}")
                        
                        st.info("💡 **Note:** This expert analysis uses 100% of the dataset and serves as the gold standard for comparison.")
                        
                    else:
                        st.warning("No ground truth available for this question.")
                else:
                    st.warning("Ground truth file not found.")
            except Exception as e:
                st.error(f"Error loading ground truth: {e}")
                st.warning("Ground truth information temporarily unavailable.")
            
            # Show business scenario context (not RAG chunks)
            st.markdown("### 📋 Business Scenario Context")
            if 'context' in question_data and question_data['context']:
                st.markdown(f"**Industry:** {question_data['domain'].title()}")
                st.markdown(f"**Business Question:** {question_data['question']}")
                st.markdown("**What to evaluate:** Look for specific data points, actionable insights, and business relevance in the AI responses below.")
            else:
                st.markdown(f"**Industry:** {question_data['domain'].title()}")
                st.markdown(f"**Analysis Focus:** {question_data['question']}")
                st.markdown("**Dataset:** AI responses use 8 chunks of curated business data with statistical analysis.")
        
        # Randomize responses for blind evaluation
        if 'response_order' not in st.session_state:
            responses = question_data['llm_responses']
            provider_names = list(responses.keys())
            random.shuffle(provider_names)
            st.session_state.response_order = provider_names
        
        responses = question_data['llm_responses']
        # Dynamically create response labels based on available responses
        response_labels = [chr(65 + i) for i in range(len(st.session_state.response_order))]  # A, B, C, D, etc.
        
        # Display responses in a 2x2 grid
        st.markdown("### Compare the Responses")
        st.markdown("*Please read all responses carefully and rank them from best to worst.*")
        
        # Show enhanced coverage information
        if is_enhanced:
            st.info("🚀 **Enhanced RAG System**: LLM responses use 8 chunks (vs 5), 150 rows per chunk (vs 200), dynamic guidance, and cross-chunk analysis.")
        else:
            st.markdown("**Note:** LLM responses use 40% dataset coverage + ground truth guidance for decision making.")
        
        # Add some spacing
        st.markdown("")
        
        # Create a responsive grid for responses
        if len(response_labels) >= 4:
            # 2x2 grid for 4+ responses
            # First row: Response A and B
            col1, col2 = st.columns(2)
            with col1:
                provider = st.session_state.response_order[0]
                response_data = responses[provider]
                
                # Get enhanced metadata if available
                metadata = response_data.get('metadata', {})
                chunks_used = metadata.get('chunks_used', 5)
                enhanced_coverage = metadata.get('enhanced_coverage', False)
                
                st.markdown("""
                <div style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f8f9fa;">
                <h4 style="color: #1f77b4; margin-bottom: 10px;">📄 Response A</h4>
                """, unsafe_allow_html=True)
                st.markdown(f"*{response_data['response']}*")
                st.caption(f"📏 Length: {len(response_data['response'])} characters")
                
                if enhanced_coverage:
                    st.caption(f"🚀 Enhanced: {chunks_used} chunks + Dynamic Guidance")
                else:
                    st.caption(f"🔍 40% Dataset + Ground Truth Guidance")
                    
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                provider = st.session_state.response_order[1]
                response_data = responses[provider]
                
                metadata = response_data.get('metadata', {})
                chunks_used = metadata.get('chunks_used', 5)
                enhanced_coverage = metadata.get('enhanced_coverage', False)
                
                st.markdown("""
                <div style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f8f9fa;">
                <h4 style="color: #ff7f0e; margin-bottom: 10px;">📄 Response B</h4>
                """, unsafe_allow_html=True)
                st.markdown(f"*{response_data['response']}*")
                st.caption(f"📏 Length: {len(response_data['response'])} characters")
                
                if enhanced_coverage:
                    st.caption(f"🚀 Enhanced: {chunks_used} chunks + Dynamic Guidance")
                else:
                    st.caption(f"🔍 40% Dataset + Ground Truth Guidance")
                    
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Second row: Response C and D (if available)
            col3, col4 = st.columns(2)
            with col3:
                provider = st.session_state.response_order[2]
                response_data = responses[provider]
                
                metadata = response_data.get('metadata', {})
                chunks_used = metadata.get('chunks_used', 5)
                enhanced_coverage = metadata.get('enhanced_coverage', False)
                
                st.markdown("""
                <div style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f8f9fa;">
                <h4 style="color: #2ca02c; margin-bottom: 10px;">📄 Response C</h4>
                """, unsafe_allow_html=True)
                st.markdown(f"*{response_data['response']}*")
                st.caption(f"📏 Length: {len(response_data['response'])} characters")
                
                if enhanced_coverage:
                    st.caption(f"🚀 Enhanced: {chunks_used} chunks + Dynamic Guidance")
                else:
                    st.caption(f"🔍 40% Dataset + Ground Truth Guidance")
                    
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col4:
                if len(response_labels) > 3:
                    provider = st.session_state.response_order[3]
                    response_data = responses[provider]
                    
                    metadata = response_data.get('metadata', {})
                    chunks_used = metadata.get('chunks_used', 5)
                    enhanced_coverage = metadata.get('enhanced_coverage', False)
                    
                    st.markdown("""
                    <div style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f8f9fa;">
                    <h4 style="color: #d62728; margin-bottom: 10px;">📄 Response D</h4>
                    """, unsafe_allow_html=True)
                    st.markdown(f"*{response_data['response']}*")
                    st.caption(f"📏 Length: {len(response_data['response'])} characters")
                    
                    if enhanced_coverage:
                        st.caption(f"🚀 Enhanced: {chunks_used} chunks + Dynamic Guidance")
                    else:
                        st.caption(f"🔍 40% Dataset + Ground Truth Guidance")
                        
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.empty()  # Placeholder for missing 4th response
                    
        else:
            # Handle 3 responses with 2+1 layout
            col1, col2 = st.columns(2)
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
            
            # First row: Response A and B
            for i, col in enumerate([col1, col2]):
                if i < len(response_labels):
                    with col:
                        provider = st.session_state.response_order[i]
                        response_data = responses[provider]
                        
                        metadata = response_data.get('metadata', {})
                        chunks_used = metadata.get('chunks_used', 5)
                        enhanced_coverage = metadata.get('enhanced_coverage', False)
                        
                        st.markdown(f"""
                        <div style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f8f9fa;">
                        <h4 style="color: {colors[i]}; margin-bottom: 10px;">📄 Response {response_labels[i]}</h4>
                        """, unsafe_allow_html=True)
                        st.markdown(f"*{response_data['response']}*")
                        st.caption(f"📏 Length: {len(response_data['response'])} characters")
                        
                        if enhanced_coverage:
                            st.caption(f"🚀 Enhanced: {chunks_used} chunks + Dynamic Guidance")
                        else:
                            st.caption(f"🔍 40% Dataset + Ground Truth Guidance")
                            
                        st.markdown("</div>", unsafe_allow_html=True)
            
            # Second row: Response C (centered)
            if len(response_labels) == 3:
                col_left, col_center, col_right = st.columns([1, 2, 1])
                with col_center:
                    provider = st.session_state.response_order[2]
                    response_data = responses[provider]
                    
                    metadata = response_data.get('metadata', {})
                    chunks_used = metadata.get('chunks_used', 5)
                    enhanced_coverage = metadata.get('enhanced_coverage', False)
                    
                    st.markdown(f"""
                    <div style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f8f9fa;">
                    <h4 style="color: {colors[2]}; margin-bottom: 10px;">📄 Response C</h4>
                    """, unsafe_allow_html=True)
                    st.markdown(f"*{response_data['response']}*")
                    st.caption(f"📏 Length: {len(response_data['response'])} characters")
                    
                    if enhanced_coverage:
                        st.caption(f"🚀 Enhanced: {chunks_used} chunks + Dynamic Guidance")
                    else:
                        st.caption(f"🔍 40% Dataset + Ground Truth Guidance")
                        
                    st.markdown("</div>", unsafe_allow_html=True)
        
        # Ranking interface
        st.markdown("### Your Ranking")
        st.markdown(f"*Please rank the responses from 1 (best) to {len(response_labels)} (worst).*")
        
        with st.form("ranking_form"):
            # Create ranking sliders for each response
            rankings = {}
            for i, label in enumerate(response_labels):
                rankings[label] = st.slider(
                    f"Rank Response {label}",
                    min_value=1, 
                    max_value=len(response_labels), 
                    value=i+1,
                    key=f"rank_{label}",
                    help=f"1 = Best, {len(response_labels)} = Worst"
                )
            
            confidence = st.slider(
                "How confident are you in your ranking?",
                min_value=1, max_value=5, value=3,
                help="1 = Not confident, 5 = Very confident"
            )
            
            comment = st.text_area(
                "Optional: Please explain your ranking",
                max_chars=500,
                help="Why did you rank the responses this way? What criteria did you use?"
            )
            
            submitted = st.form_submit_button("Submit Ranking", type="primary")
            
            if submitted:
                # Validate rankings (must be unique)
                rank_values = list(rankings.values())
                expected_ranks = list(range(1, len(response_labels) + 1))
                
                if len(set(rank_values)) != len(response_labels):
                    st.error("⚠️ Please ensure each response has a unique ranking!")
                elif sorted(rank_values) != expected_ranks:
                    st.error(f"⚠️ Please use ranks 1 through {len(response_labels)} exactly once!")
                else:
                    # Validate comment
                    comment_valid, comment_error = validator.validate_comment(comment)
                    if not comment_valid:
                        st.error(f"❌ {comment_error}")
                        return
                    
                    # Find the best ranked response (rank 1)
                    best_rank = 1
                    best_response_label = None
                    for label, rank in rankings.items():
                        if rank == best_rank:
                            best_response_label = label
                            break
                    
                    # Map to actual provider
                    best_index = response_labels.index(best_response_label)
                    actual_provider = st.session_state.response_order[best_index]
                    
                    # Save evaluation with ranking data
                    evaluation_entry = {
                        'question_id': st.session_state.current_question_id,
                        'question': question_data['question'],
                        'domain': question_data['domain'],
                        'rankings': rankings,
                        'best_response': best_response_label,
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
                        selected_response=best_response_label,
                        has_comment=bool(comment.strip())
                    )
                    
                    # Save to feedback logger
                    feedback_logger.log_evaluation(
                        user_id=st.session_state.tester_email,
                        evaluation_data=evaluation_entry
                    )
                    
                    # Show feedback
                    st.success(f"✅ Thank you! You ranked **Response {best_response_label}** as best")
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
            if st.button("🏠 Back to Home", use_container_width=True):
                st.session_state.step = 0
                st.rerun()
        
        with col2:
            if st.button("📊 View Progress", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
                
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")
        secure_logger.log_security_event(
            event_type="evaluation_error",
            severity="MEDIUM",
            details={"error": str(e), "user_email": st.session_state.get('tester_email', 'unknown')}
        )

def show_progress_interface(feedback_logger: FeedbackLogger):
    """Show evaluation progress and statistics."""
    
    st.title("📊 Evaluation Progress")
    
    # Get user's evaluation data
    if 'evaluation_progress' not in st.session_state:
        st.session_state.evaluation_progress = []
    
    evaluations = st.session_state.evaluation_progress
    
    if not evaluations:
        st.info("No evaluations completed yet. Start your evaluation to see progress here!")
        if st.button("🚀 Start Evaluation", type="primary"):
            st.session_state.step = 1
            st.rerun()
        return
    
    # Calculate statistics
    total_questions = st.session_state.get('total_questions', 10)
    completed = len(evaluations)
    remaining = total_questions - completed
    progress_percent = (completed / total_questions) * 100
    
    # Progress overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Questions Completed", completed, f"{remaining} remaining")
    with col2:
        st.metric("Progress", f"{progress_percent:.1f}%")
    with col3:
        st.metric("Best Provider", get_most_selected_provider(evaluations))
    
    # Progress bar
    st.progress(progress_percent / 100, text=f"Overall Progress: {completed}/{total_questions}")
    
    # Domain breakdown
    retail_evaluations = [e for e in evaluations if e['domain'] == 'retail']
    finance_evaluations = [e for e in evaluations if e['domain'] == 'finance']
    
    col1, col2 = st.columns(2)
    with col1:
        retail_progress = len(retail_evaluations) / 5 * 100
        st.metric("🛍️ Retail", f"{len(retail_evaluations)}/5", f"{retail_progress:.1f}%")
        st.progress(retail_progress / 100)
    
    with col2:
        finance_progress = len(finance_evaluations) / 5 * 100
        st.metric("💰 Finance", f"{len(finance_evaluations)}/5", f"{finance_progress:.1f}%")
        st.progress(finance_progress / 100)
    
    # Ranking statistics
    st.markdown("### 📈 Ranking Statistics")
    
    # Provider preference analysis
    provider_rankings = analyze_provider_rankings(evaluations)
    
    if provider_rankings:
        st.markdown("#### Provider Performance (Based on Rankings)")
        
        # Create a simple bar chart
        providers = list(provider_rankings.keys())
        avg_ranks = list(provider_rankings.values())
        
        # Lower rank is better, so invert for display
        performance_scores = [4 - rank for rank in avg_ranks]
        
        # Display as metrics
        cols = st.columns(len(providers))
        for i, (provider, score) in enumerate(zip(providers, performance_scores)):
            with cols[i]:
                st.metric(
                    provider.title(), 
                    f"{score:.1f}/3", 
                    f"Avg Rank: {avg_ranks[i]:.1f}"
                )
    
    # Recent evaluations
    st.markdown("### 📝 Recent Evaluations")
    
    for i, eval_data in enumerate(evaluations[-5:], 1):  # Show last 5
        with st.expander(f"Question {i}: {eval_data['question'][:50]}...", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Domain:** {eval_data['domain'].title()}")
                st.write(f"**Best Response:** {eval_data['best_response']}")
                st.write(f"**Provider:** {eval_data['actual_provider'].title()}")
            with col2:
                st.write(f"**Confidence:** {eval_data['confidence']}/5")
                if eval_data.get('comment'):
                    st.write(f"**Comment:** {eval_data['comment'][:100]}...")
    
    # Navigation
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.step = 0
            st.rerun()
    
    with col2:
        if remaining > 0:
            if st.button("🚀 Continue Evaluation", type="primary", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        else:
            if st.button("🏆 View Final Results", type="primary", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
    
    with col3:
        if st.button("📊 Export Data", use_container_width=True):
            export_evaluation_data(evaluations)

def get_most_selected_provider(evaluations):
    """Get the most frequently selected provider."""
    if not evaluations:
        return "None"
    
    provider_counts = {}
    for eval_data in evaluations:
        provider = eval_data['actual_provider']
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
    
    if provider_counts:
        return max(provider_counts, key=provider_counts.get).title()
    return "None"

def analyze_provider_rankings(evaluations):
    """Analyze provider performance based on rankings."""
    if not evaluations:
        return {}
    
    provider_ranks = {}
    provider_counts = {}
    
    for eval_data in evaluations:
        if 'rankings' not in eval_data:
            continue
            
        rankings = eval_data['rankings']
        response_order = eval_data.get('response_order', [])
        
        # Map response labels to providers
        for label, rank in rankings.items():
            if label in response_order:
                try:
                    label_index = response_order.index(label)
                    if label_index < len(st.session_state.response_order):
                        provider = st.session_state.response_order[label_index]
                        provider_ranks[provider] = provider_ranks.get(provider, []) + [rank]
                        provider_counts[provider] = provider_counts.get(provider, 0) + 1
                except (IndexError, KeyError):
                    continue
    
    # Calculate average ranks
    avg_ranks = {}
    for provider, ranks in provider_ranks.items():
        if ranks:
            avg_ranks[provider] = sum(ranks) / len(ranks)
    
    return avg_ranks

def export_evaluation_data(evaluations):
    """Export evaluation data to JSON."""
    import json
    from datetime import datetime
    
    export_data = {
        'export_timestamp': datetime.now().isoformat(),
        'user_email': st.session_state.get('tester_email', 'unknown'),
        'session_id': st.session_state.get('session_id', 'unknown'),
        'evaluations': evaluations
    }
    
    # Create download button
    st.download_button(
        label="📥 Download Evaluation Data",
        data=json.dumps(export_data, indent=2),
        file_name=f"evaluation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

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