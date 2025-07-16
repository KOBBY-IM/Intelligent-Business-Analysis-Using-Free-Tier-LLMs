


































































































































































import streamlit as st
import pandas as pd
import tempfile
import os
import json
import sys
from pathlib import Path
from typing import List

# Add src to path if not already there
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import the CSV RAG pipeline and admin auth
from rag.csv_rag_pipeline import CSVBlindTestGenerator
from ui.components.admin_auth import show_inline_admin_login, show_admin_header

st.set_page_config(
    page_title="Batch RAG Testing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check admin authentication - use inline login if not authenticated
if not show_inline_admin_login("Batch RAG Testing"):
    st.stop()

# Show admin header with logout option
show_admin_header("Batch RAG Testing")

st.title("📊 Batch RAG Testing with Uploaded CSV Files")
st.markdown("""
Upload one or more CSV files and enter your own business questions. The system will use Retrieval-Augmented Generation (RAG) to generate data-driven responses for each question based on your uploaded data.
""")

# --- File Upload ---
uploaded_files = st.file_uploader(
    "Upload one or more CSV files:",
    type=["csv"],
    accept_multiple_files=True,
    help="You can upload multiple CSV files. Each will be included in the RAG pipeline."
)

# --- User Question Input ---
st.markdown("### 📝 Enter Your Business Questions")
num_questions = st.number_input(
    "How many questions do you want to test?",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

user_questions = []
for i in range(num_questions):
    q = st.text_area(
        f"Question {i+1}",
        placeholder="Enter your business question here...",
        key=f"user_question_{i}"
    )
    user_questions.append(q.strip())

# --- Run Batch RAG ---
if st.button("Run Batch RAG", type="primary", use_container_width=True):
    # Validate input
    valid_questions = [q for q in user_questions if q]
    if not uploaded_files:
        st.error("Please upload at least one CSV file.")
    elif not valid_questions:
        st.error("Please enter at least one business question.")
    else:
        st.info("Processing uploaded files and generating responses. This may take a moment...")
        # Save uploaded files to temp directory
        temp_dir = tempfile.TemporaryDirectory()
        csv_paths = []
        for file in uploaded_files:
            file_path = os.path.join(temp_dir.name, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            csv_paths.append(file_path)
        
        # Prepare questions in the expected format
        questions_dict = {"user": []}
        for idx, q in enumerate(valid_questions):
            questions_dict["user"].append({
                "id": f"user_q{idx+1}",
                "prompt": q,
                "context": "(User-provided question)",
            })
        
        # Run CSV RAG pipeline
        try:
            generator = CSVBlindTestGenerator(csv_paths)
            results = generator.generate_all_responses(questions_dict)
            st.success("Batch RAG completed!")
            
            # Display results
            for q_idx, question in enumerate(results["user"]):
                st.markdown(f"#### Question {q_idx+1}: {question['prompt']}")
                st.markdown("---")
                
                # Display responses in card format (2 per row)
                responses = question["responses"]
                for i in range(0, len(responses), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(responses):
                            resp = responses[i + j]
                            # Check if there's an error
                            has_error = resp.get('error') or resp['content'].startswith('ERROR:')
                            
                            with cols[j]:
                                if has_error:
                                    # Error card
                                    st.markdown(f"""
                                    <div style="background-color: #f8d7da; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 10px 0; border: 1px solid #dc3545;">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #f0c6c7;">
                                            <h3 style="color: #721c24; margin: 0; font-size: 1.4rem; font-weight: 600;">❌ {resp['model']}</h3>
                                            <span style="background-color: #dc3545; color: white; padding: 5px 12px; border-radius: 20px; font-weight: 600; font-size: 0.9rem;">FAILED</span>
                                        </div>
                                        <div style="line-height: 1.7; font-size: 1.1rem; color: #721c24;">
                                            <strong>Error:</strong> {resp['content']}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    # Success card
                                    st.markdown(f"""
                                    <div style="background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 10px 0; border: 1px solid #e0e0e0;">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #f0f0f0;">
                                            <h3 style="color: #28a745; margin: 0; font-size: 1.4rem; font-weight: 600;">✅ {resp['model']}</h3>
                                            <span style="background-color: #28a745; color: white; padding: 5px 12px; border-radius: 20px; font-weight: 600; font-size: 0.9rem;">SUCCESS</span>
                                        </div>
                                        <div style="line-height: 1.7; font-size: 1.1rem; color: #333; text-align: justify; margin-bottom: 15px;">
                                            {resp['content']}
                                        </div>
                                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8;">
                                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.95rem; color: #495057;">
                                                <div><strong>Relevance:</strong> {resp['metrics']['relevance']:.2f}</div>
                                                <div><strong>Accuracy:</strong> {resp['metrics']['accuracy']:.2f}</div>
                                                <div><strong>Coherence:</strong> {resp['metrics']['coherence']:.2f}</div>
                                                <div><strong>Tokens:</strong> {resp['metrics']['token_count']}</div>
                                                <div><strong>Latency:</strong> {resp['metrics']['latency']:.2f}s</div>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # RAG context expandable section
                                    with st.expander("📖 Show RAG Context", expanded=False):
                                        st.code(resp['rag_context_used'], language='text')
                
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Download option
            st.markdown("---")
            st.markdown("### 📥 Download Results")
            results_json = json.dumps(results, indent=2)
            st.download_button(
                label="Download Results as JSON",
                data=results_json,
                file_name="batch_rag_results.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"Error during batch RAG processing: {e}")
        finally:
            temp_dir.cleanup() 