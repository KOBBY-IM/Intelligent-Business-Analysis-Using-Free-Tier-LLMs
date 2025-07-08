import pathlib

import streamlit as st

# Set page config
st.set_page_config(
    page_title="LLM Business Intelligence Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Fluent UI CSS
css_path = pathlib.Path(__file__).parent / "fluent_theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Large header and subheader
st.markdown(
    """
<div style='padding: 2rem 0 0.5rem 0;'>
  <h1 style='font-size:2.8rem; font-weight:700; margin-bottom:0.2em;'>
    LLM Business Intelligence Comparison Dashboard
  </h1>
  <h3 style='font-weight:400; color:#444; margin-top:0;'>
    Compare, evaluate, and benchmark free-tier LLMs (Groq, Gemini, Hugging Face) for decision support in retail, finance, and healthcare. Explore metrics, visualize results, and export insights—all in a Fluent-inspired, user-friendly interface.
  </h3>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("LLM BI Comparison")
st.sidebar.markdown(
    """
**Navigation**
- LLM Comparison
- Metrics
- Export Results
"""
)

# Main tabs
tabs = st.tabs(["LLM Comparison", "Metrics", "Export"])

with tabs[0]:
    st.header("LLM Comparison")
    st.info(
        "Compare responses from Groq, Gemini, and Hugging Face LLMs across industries."
    )
    st.write("(UI for blind evaluation and response display will go here.)")

with tabs[1]:
    st.header("Evaluation Metrics")
    st.info(
        "View relevance, accuracy, coherence, token count, and latency for each model."
    )
    st.write("(Metrics visualization and charts will go here.)")

with tabs[2]:
    st.header("Export Results")
    st.info("Export evaluation results and user preferences as CSV or JSON.")
    st.write("(Export/download buttons will go here.)")
