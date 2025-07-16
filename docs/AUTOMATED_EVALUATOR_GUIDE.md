# Automated LLM Evaluator Guide

## Overview

The Automated LLM Evaluator is a comprehensive testing system that runs LLM performance evaluations every 2 minutes for 1 hour using RAG (Retrieval-Augmented Generation) with your datasets. This generates real performance data for your Model Performance Dashboard.

## Quick Start

### 1. Test the System (Recommended First)
```bash
# Run a 2-minute test to verify everything works
python scripts/test_automated_eval.py
```

### 2. Run Full Evaluation
```bash
# Run the complete 1-hour evaluation
python scripts/run_automated_eval.py
```

### 3. Custom Configuration
```bash
# Run with custom duration and interval
python scripts/run_automated_eval.py --duration 30 --interval 60

# Skip confirmation prompt
python scripts/run_automated_eval.py --no-confirm

# Show configuration without running
python scripts/run_automated_eval.py --dry-run
```

## 📋 What It Does

### Test Schedule
- **Duration**: 60 minutes (configurable)
- **Interval**: 2 minutes between tests (configurable)
- **Total Tests**: 30 evaluations (60 min ÷ 2 min)
- **Providers**: All available LLM providers (Groq, Gemini, Hugging Face, OpenRouter)

### Each Test Includes
1. **Random Dataset Selection**: Picks from your available datasets
2. **RAG Processing**: Uses your RAG pipeline with the selected dataset
3. **Multi-Provider Testing**: Tests all available LLM providers
4. **Performance Metrics**: Latency, token count, quality scores
5. **Error Handling**: Continues testing even if some providers fail

### Generated Data
- **Raw Responses**: Complete LLM responses with context
- **Performance Metrics**: Latency, token usage, quality scores
- **Error Tracking**: Failed requests and error reasons
- **Statistical Analysis**: Provider comparisons and trends

## 📁 Data Sources

### Datasets
The evaluator automatically loads:
- **CSV Files**: `data/*.csv` (e.g., `shopping_trends.csv`)
- **JSON Files**: `data/*.json` (structured data)
- **Sample Data**: Generated if no datasets found

### Questions
- **Custom Questions**: `data/business_questions.yaml`
- **Default Questions**: Auto-generated business analysis questions
- **Industry-Specific**: Retail, finance, healthcare contexts

## ⚙️ Configuration

### Main Configuration File
`config/automated_eval_config.yaml`

```yaml
# Test Schedule
test_schedule:
  duration_minutes: 60
  interval_seconds: 120
  max_concurrent_tests: 1

# RAG Settings
rag_settings:
  max_context_length: 2000
  chunk_size: 500
  chunk_overlap: 50
  max_results: 5
  similarity_threshold: 0.7

# Provider Settings
providers:
  groq:
    enabled: true
    models: ["mixtral-8x7b-32768", "llama2-70b-4096"]
    max_retries: 3
    timeout_seconds: 30
```

### Environment Variables
Ensure your `.env` file has API keys:
```bash
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_gemini_key
HUGGINGFACE_API_KEY=your_hf_key
OPENROUTER_API_KEY=your_openrouter_key
```

## 📊 Output Files

### Results Directory
- `data/evaluation_results/automated_eval_final_YYYYMMDD_HHMMSS.json`
- `data/results/automated_eval_final_YYYYMMDD_HHMMSS.json`
- `data/evaluation_results/automated_eval_summary_YYYYMMDD_HHMMSS.json`

### Data Structure
```json
{
  "test_id": "auto_test_001",
  "timestamp": "2025-07-15T10:30:00",
  "provider": "Groq",
  "model": "mixtral-8x7b-32768",
  "industry": "retail",
  "dataset": "shopping_trends",
  "question": "What are the top 5 products by sales?",
  "response": "Based on the data...",
  "latency": 1.234,
  "token_count": 150,
  "quality_score": 0.85,
  "relevance_score": 0.88,
  "coherence_score": 0.82,
  "accuracy_score": 0.87,
  "error": null
}
```

## 📈 Dashboard Integration

### Model Performance Dashboard
After running the evaluator:
1. **Start Streamlit**: `python start_streamlit.py`
2. **Navigate**: Go to "4 Model Performance Dashboard"
3. **View Results**: All automated test data will be available
4. **Analyze**: Statistical comparisons, trends, provider performance

### Available Analytics
- **Performance Comparison**: Latency, quality, token usage
- **Statistical Analysis**: ANOVA tests, significance testing
- **Industry Breakdown**: Performance by business domain
- **Trend Analysis**: Performance over time
- **Export Options**: PDF reports, CSV data, JSON statistics

## 🔧 Troubleshooting

### Common Issues

#### 1. No Providers Available
```
❌ No LLM providers available!
```
**Solution**: Check your API keys in `.env` file

#### 2. Dataset Not Found
```
❌ No datasets found, will use sample data
```
**Solution**: Add CSV/JSON files to `data/` directory

#### 3. RAG Pipeline Errors
```
❌ Error in RAG pipeline
```
**Solution**: Check dataset format and RAG configuration

#### 4. API Quota Exceeded
```
❌ API quota exceeded
```
**Solution**: Check API usage limits and billing

### Debug Mode
```bash
# Run with detailed logging
python scripts/run_automated_eval.py --config config/automated_eval_config.yaml
```

## 📋 Best Practices

### Before Running
1. **Check API Keys**: Ensure all providers are configured
2. **Verify Datasets**: Add business datasets to `data/` directory
3. **Test First**: Run quick test to verify setup
4. **Monitor Quotas**: Check API usage limits

### During Execution
1. **Monitor Progress**: Watch console output for progress
2. **Check Logs**: Review logs for any issues
3. **Don't Interrupt**: Let the full cycle complete
4. **Save Results**: Results are auto-saved every 5 tests

### After Completion
1. **Review Summary**: Check the generated summary report
2. **View Dashboard**: Analyze results in Model Performance Dashboard
3. **Export Data**: Download reports for thesis documentation
4. **Backup Results**: Save important results for future reference

## 🎯 Use Cases

### Academic Research
- **Performance Benchmarking**: Compare LLM providers systematically
- **Statistical Analysis**: Generate data for research papers
- **Reproducible Results**: Consistent testing methodology
- **Thesis Documentation**: Comprehensive performance data

### Business Analysis
- **Provider Selection**: Data-driven LLM provider choice
- **Cost Analysis**: Performance vs. cost optimization
- **Quality Assessment**: Systematic quality evaluation
- **Trend Monitoring**: Performance tracking over time

### Development Testing
- **System Validation**: Verify RAG pipeline performance
- **Error Detection**: Identify and fix system issues
- **Performance Optimization**: Optimize for speed and quality
- **Regression Testing**: Ensure system stability

## 📞 Support

### Logs Location
- **Application Logs**: `logs/` directory
- **Evaluation Logs**: Console output during execution
- **Error Logs**: Detailed error information in console

### Common Commands
```bash
# Quick test
python scripts/test_automated_eval.py

# Full evaluation
python scripts/run_automated_eval.py

# Custom duration
python scripts/run_automated_eval.py --duration 30

# View dashboard
python start_streamlit.py
```

### File Structure
```
scripts/
├── automated_llm_evaluator.py    # Main evaluator
├── run_automated_eval.py         # Launcher script
└── test_automated_eval.py        # Quick test

config/
└── automated_eval_config.yaml    # Configuration

data/
├── *.csv                         # Datasets
├── *.json                        # Structured data
└── evaluation_results/           # Output files
```

---

**Ready to run your automated LLM evaluation? Start with the quick test to verify everything works!** 🚀 