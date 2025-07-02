# Intelligent Business Analysis Using Free-Tier LLMs

A production-quality MSc research project implementing a RAG-enabled evaluation system comparing Groq, Gemini, and Hugging Face free-tier LLMs across retail, finance, and healthcare domains.

## Features
- Secure, configuration-driven architecture
- Async LLM provider interface (Groq, Gemini, Hugging Face)
- RAG pipeline with FAISS/Chroma
- Streamlit UI (Microsoft Fluent UI style)
- Blind user studies and exportable results

## Setup
1. Clone the repo
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your API keys
4. Run initial tests and scripts as described in `/scripts/`

## Directory Structure
- `src/` - Core source code
- `config/` - YAML configuration files
- `data/` - Datasets (gitignored)
- `docs/` - Documentation and data schema
- `tests/` - Test suite
- `scripts/` - Setup and utility scripts

## License
MIT 