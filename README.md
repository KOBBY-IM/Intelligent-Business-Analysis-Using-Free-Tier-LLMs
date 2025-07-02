# Intelligent Business Analysis Using Free-Tier LLMs

A comparative framework for multi-industry decision support using free-tier LLM providers (Groq, Gemini, Hugging Face) with RAG-enabled evaluation systems.

## Project Status: Day 1 Complete ✅

### Completed Tasks
- ✅ Project structure scaffolded
- ✅ Configuration files created (YAML-based)
- ✅ Environment setup with virtual environment
- ✅ API test scripts for all three LLM providers
- ✅ Shopping trends dataset (3,900 rows) added and analyzed
- ✅ Data processing pipeline implemented
- ✅ Business questions generated based on actual dataset
- ✅ Documentation updated with dataset schema

### Dataset Overview
- **File**: `data/shopping_trends.csv`
- **Size**: 3,900 rows × 19 columns
- **Categories**: Clothing (44.5%), Accessories (31.8%), Footwear (15.4%), Outerwear (8.3%)
- **Quality**: No missing values, clean data structure

## Quick Start

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### Setup
```bash
# Clone repository
git clone <repository-url>
cd Intelligent-Business-Analysis-Using-Free-Tier-LLMs

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### API Keys Required
Add your API keys to `.env`:
```bash
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_gemini_key
HUGGINGFACE_API_KEY=your_hf_key
```

### Test LLM Providers
```bash
# Test individual providers
python src/llm_providers/test_groq.py
python src/llm_providers/test_gemini.py
python src/llm_providers/test_huggingface.py
```

### Data Processing
```bash
# Analyze and clean shopping trends dataset
python src/data_processing/clean_retail_data.py
```

## Project Structure
```
project_root/
├── config/                 # YAML configuration files
├── data/                   # Datasets and results (gitignored)
├── docs/                   # Documentation
├── scripts/                # Setup and utility scripts
├── src/                    # Source code
│   ├── config/            # Configuration management
│   ├── data_processing/   # Data cleaning and analysis
│   ├── evaluation/        # LLM evaluation metrics
│   ├── llm_providers/     # LLM API integrations
│   ├── rag/              # RAG pipeline components
│   ├── security/         # Security and validation
│   ├── ui/               # Streamlit interface
│   └── utils/            # Utility functions
└── tests/                # Test suite
```

## Next Steps (Week 1 - MVP 1)
1. **Secure LLM Evaluation Engine**
   - Implement BaseLLMProvider abstract class
   - Build RAG pipeline (retriever.py, generator.py)
   - Create evaluation metrics (scorer.py)
   - Add CLI interface for batch evaluation

2. **Week 2 - Streamlit Dashboard**
   - Interactive UI with Microsoft Fluent UI styling
   - Blind user testing interface
   - User preference logging

3. **Week 3 - Statistical Analysis**
   - Full-scale evaluation across all models
   - Statistical analysis and reporting
   - Export functionality

## Configuration
All settings are externalized to YAML files:
- `config/app_config.yaml` - Application settings
- `config/llm_config.yaml` - LLM provider configurations
- `config/evaluation_config.yaml` - Evaluation parameters
- `config/logging_config.yaml` - Logging settings

## Security Features
- Environment variable-based API key management
- Input validation and sanitization
- Rate limiting and quota monitoring
- Secure logging (no PII leakage)

## Research Methodology
- Reproducible ML/AI practices
- Statistical validation (ANOVA, confidence intervals)
- Blinded user evaluations
- Exportable results format

## Contributing
This is an MSc research project. Please follow the established coding standards and security practices outlined in `.cursorrules`.

## License
Academic research project - see project documentation for details. 