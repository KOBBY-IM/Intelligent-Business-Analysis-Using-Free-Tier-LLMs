"""
Export Results Page

This page provides functionality to export evaluation results, user feedback,
and statistical analysis in various formats.
"""

import streamlit as st
import pandas as pd
import json
import csv
import os
import zipfile
import io
from pathlib import Path
from datetime import datetime
import sys
import smtplib
from email.message import EmailMessage

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from utils.feedback_logger import FeedbackLogger
from ui.components.admin_auth import show_inline_admin_login, show_admin_header

# Page configuration
st.set_page_config(
    page_title="Export Results",
    page_icon="📤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check admin authentication - use inline login if not authenticated
if not show_inline_admin_login("Export Results"):
    st.stop()

# Show admin header with logout option
show_admin_header("Export Results")

def load_export_data():
    """Load all available data for export"""
    data = {}
    
    # Load user feedback
    feedback_file = Path("data/results/user_feedback.json")
    if feedback_file.exists():
        try:
            with open(feedback_file, 'r') as f:
                content = f.read().strip()
                if content:
                    data['user_feedback'] = json.loads(content)
                else:
                    data['user_feedback'] = []
        except (json.JSONDecodeError, IOError) as e:
            st.warning(f"Could not load user feedback file: {e}")
            data['user_feedback'] = []
    
    # Load blind responses
    blind_file = Path("data/blind_responses.json")
    if blind_file.exists():
        try:
            with open(blind_file, 'r') as f:
                content = f.read().strip()
                if content:
                    data['blind_responses'] = json.loads(content)
                else:
                    data['blind_responses'] = {}
        except (json.JSONDecodeError, IOError) as e:
            st.warning(f"Could not load blind responses file: {e}")
            data['blind_responses'] = {}
    
    # Load ground truth
    ground_truth_file = Path("data/ground_truth_answers.json")
    if ground_truth_file.exists():
        try:
            with open(ground_truth_file, 'r') as f:
                content = f.read().strip()
                if content:
                    data['ground_truth'] = json.loads(content)
                else:
                    data['ground_truth'] = {}
        except (json.JSONDecodeError, IOError) as e:
            st.warning(f"Could not load ground truth file: {e}")
            data['ground_truth'] = {}
    
    # Load evaluation results
    eval_results_dir = Path("data/evaluation_results")
    if eval_results_dir.exists():
        eval_files = list(eval_results_dir.glob("*.json"))
        data['evaluation_results'] = []
        for file in eval_files:
            try:
                with open(file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        continue
                    result = json.loads(content)
                    data['evaluation_results'].append(result)
            except (json.JSONDecodeError, IOError) as e:
                st.warning(f"Skipping invalid JSON file {file.name}: {e}")
                continue
    
    return data

def create_user_feedback_csv(feedback_data):
    """Create CSV from user feedback data"""
    if not feedback_data:
        return None
    
    rows = []
    for feedback in feedback_data:
        row = {
            'timestamp': feedback.get('timestamp', ''),
            'session_id': feedback.get('session_id', ''),
            'industry': feedback.get('industry', ''),
            'prompt': feedback.get('prompt', ''),
            'selected_label': feedback.get('selected_label', ''),
            'selected_response_id': feedback.get('selected_response_id', ''),
            'comment': feedback.get('comment', ''),
            'step': feedback.get('step', '')
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

def create_evaluation_summary_csv(feedback_data, blind_data):
    """Create evaluation summary CSV with full model ranking for each feedback entry."""
    if not feedback_data or not blind_data:
        return None
    
    rows = []
    for feedback in feedback_data:
        industry = feedback.get('domain', '') or feedback.get('industry', '')
        ranking_model_order = feedback.get('ranking_model_order', [])
        comment = feedback.get('comment', '')
        timestamp = feedback.get('timestamp', '')
        session_id = feedback.get('session_id', '')
        tester_email = feedback.get('tester_email', '')
        question_text = feedback.get('question_text', '')
        question_idx = feedback.get('question_idx', '')
        # Build row with ranking as comma-separated string
        row = {
            'timestamp': timestamp,
            'session_id': session_id,
            'tester_email': tester_email,
            'industry': industry,
            'question_idx': question_idx,
            'question_text': question_text,
            'ranking_model_order': ', '.join(ranking_model_order),
            'comment': comment
        }
        # Add columns for each rank
        for i, model in enumerate(ranking_model_order):
            row[f'Rank_{i+1}'] = model
        rows.append(row)
    return pd.DataFrame(rows)

def create_research_report(data):
    """Create a comprehensive research report"""
    report = {
        'report_metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_evaluations': len(data.get('user_feedback', [])),
            'industries_covered': list(set(f.get('industry', '') for f in data.get('user_feedback', []))),
            'models_evaluated': set()
        },
        'evaluation_summary': {},
        'industry_analysis': {},
        'model_performance': {},
        'user_feedback_analysis': {}
    }
    
    # Extract model information
    if 'blind_responses' in data:
        for industry, industry_data in data['blind_responses'].items():
            for response in industry_data['responses']:
                report['report_metadata']['models_evaluated'].add(response.get('model', ''))
    
    # Convert set to list for JSON serialization
    report['report_metadata']['models_evaluated'] = list(report['report_metadata']['models_evaluated'])
    
    return report

def main():
    """Main export function"""
    
    # Page header
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='font-size: 2.5rem; color: #1f77b4; margin-bottom: 1rem;'>
                📤 Export Results
            </h1>
            <h2 style='font-size: 1.5rem; color: #666; margin-bottom: 2rem;'>
                Download evaluation data and research reports
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Load data
    data = load_export_data()
    
    if not data:
        st.warning("No data available for export. Please complete some evaluations first.")
        return
    
    # Data overview
    st.markdown("## 📊 Available Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        feedback_count = len(data.get('user_feedback', []))
        st.metric("User Feedback", feedback_count)
    
    with col2:
        industries = len(set(f.get('industry', '') for f in data.get('user_feedback', [])))
        st.metric("Industries", industries)
    
    with col3:
        eval_count = len(data.get('evaluation_results', []))
        st.metric("Evaluation Results", eval_count)
    
    with col4:
        models = len(data.get('report_metadata', {}).get('models_evaluated', []))
        st.metric("Models", models)
    
    st.markdown("---")
    
    # Export options
    st.markdown("## 📤 Export Options")
    
    # User Feedback Export
    st.markdown("### 👥 User Feedback Data")
    
    if 'user_feedback' in data and data['user_feedback']:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export User Feedback (CSV)", use_container_width=True):
                df = create_user_feedback_csv(data['user_feedback'])
                if df is not None:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"user_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("📄 Export User Feedback (JSON)", use_container_width=True):
                json_data = json.dumps(data['user_feedback'], indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"user_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    else:
        st.info("No user feedback data available.")
    
    # Evaluation Summary Export
    st.markdown("### 📈 Evaluation Summary")
    
    if 'user_feedback' in data and 'blind_responses' in data:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Evaluation Summary (CSV)", use_container_width=True):
                df = create_evaluation_summary_csv(data['user_feedback'], data['blind_responses'])
                if df is not None:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("📄 Export Evaluation Summary (JSON)", use_container_width=True):
                summary_data = {
                    'user_feedback': data['user_feedback'],
                    'blind_responses': data['blind_responses']
                }
                json_data = json.dumps(summary_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    else:
        st.info("No evaluation summary data available.")
    
    # Research Report Export
    st.markdown("### 📋 Research Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Generate Research Report (JSON)", use_container_width=True):
            report = create_research_report(data)
            json_data = json.dumps(report, indent=2)
            st.download_button(
                label="Download Report",
                data=json_data,
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Generate Research Report (CSV)", use_container_width=True):
            # Create comprehensive CSV report
            if 'user_feedback' in data and 'blind_responses' in data:
                df = create_evaluation_summary_csv(data['user_feedback'], data['blind_responses'])
                if df is not None:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download Report",
                        data=csv_data,
                        file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    # Raw Data Export
    st.markdown("### 🔧 Raw Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📦 Export All Data (JSON)", use_container_width=True):
            json_data = json.dumps(data, indent=2)
            st.download_button(
                label="Download All Data",
                data=json_data,
                file_name=f"complete_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("🗂️ Export Individual Files", use_container_width=True):
            st.info("Individual file export functionality will be implemented in the next iteration.")
    
    st.markdown("---")
    
    # Export settings
    st.markdown("## ⚙️ Export Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_timestamps = st.checkbox("Include timestamps", value=True)
        include_comments = st.checkbox("Include user comments", value=True)
    
    with col2:
        anonymize_data = st.checkbox("Anonymize data", value=True)
        include_metrics = st.checkbox("Include performance metrics", value=True)
    
    st.markdown("---")
    
    # Export history
    st.markdown("## 📚 Export History")
    st.info("Export history tracking will be implemented in the next iteration.")

    st.title("📥 Export All Results")

    # --- Download All Results as ZIP ---
    def get_all_results_zip():
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            # Add registration data
            reg_path = Path("data/tester_registrations.json")
            if reg_path.exists():
                zf.write(reg_path, arcname="tester_registrations.json")
            # Add user feedback/results
            results_dir = Path("data/results/")
            if results_dir.exists():
                for file in results_dir.glob("*.json"):
                    zf.write(file, arcname=f"results/{file.name}")
            # Add metrics/exports if any
            for extra in ["data/blind_responses.json", "data/rag_datasets.json"]:
                extra_path = Path(extra)
                if extra_path.exists():
                    zf.write(extra_path, arcname=extra_path.name)
        zip_buffer.seek(0)
        return zip_buffer

    st.markdown("### Download all research data as a ZIP archive:")
    if st.button("Download All Results as ZIP", use_container_width=True):
        zip_buffer = get_all_results_zip()
        st.download_button(
            label="Download ZIP",
            data=zip_buffer,
            file_name="all_blind_test_results.zip",
            mime="application/zip"
        )

    # --- Email Notification Logic ---
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

    # --- Trigger email notification if new results detected ---
    results_dir = Path("data/results/")
    if results_dir.exists():
        result_files = list(results_dir.glob("*.json"))
        if result_files:
            # Only show the button to send notification if admin wants
            if st.button("Send Admin Email Notification", use_container_width=True):
                sent = send_admin_email(
                    subject="New Blind Test Results Submitted",
                    body="New results have been submitted to the LLM Blind Evaluation system. Please log in to the admin dashboard to review."
                )
                if sent:
                    st.success("Admin notification email sent!")
                else:
                    st.error("Failed to send admin notification email. Check SMTP settings in .env.")

if __name__ == "__main__":
    main() 