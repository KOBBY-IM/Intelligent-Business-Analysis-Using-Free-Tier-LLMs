# LLM Evaluation System Guide

## Overview

The LLM Evaluation System is a comprehensive framework for objectively measuring and comparing the performance of Large Language Models (LLMs) across multiple dimensions. It provides automated evaluation metrics, statistical analysis, and detailed reporting for research-grade LLM assessment.

## System Architecture

### Core Components

1. **Evaluation Metrics (`src/evaluation/metrics.py`)**
   - Relevance scoring (TF-IDF cosine similarity)
   - Factual accuracy assessment (F1-score based)
   - Response time and token efficiency metrics
   - Coherence and completeness scoring
   - Overall weighted score calculation

2. **Ground Truth Management (`src/evaluation/ground_truth.py`)**
   - Structured ground truth answers for business questions
   - Domain-specific evaluation (retail, finance, healthcare)
   - Difficulty levels (easy, medium, hard)
   - Metadata and key points tracking

3. **Statistical Analysis (`src/evaluation/statistical_analysis.py`)**
   - Descriptive statistics and confidence intervals
   - T-tests for pairwise comparisons
   - ANOVA for multi-group analysis
   - Effect size calculations (Cohen's d)
   - Provider rankings and performance analysis

4. **Evaluation Engine (`src/evaluation/evaluator.py`)**
   - Batch evaluation orchestration
   - Multi-provider testing
   - Results storage and retrieval
   - Comprehensive reporting

## Installation and Setup

### Prerequisites

```bash
# Install required dependencies
pip install nltk scikit-learn scipy matplotlib seaborn

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Configuration

1. **Environment Variables** (`.env` file):
   ```bash
   # API Keys
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_API_KEY=your_google_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   
   # Evaluation Settings
   EVALUATION_RESULTS_DIR=data/evaluation_results
   GROUND_TRUTH_FILE=data/ground_truth_answers.json
   ```

2. **Ground Truth Data** (`data/ground_truth_answers.json`):
   - Pre-configured with 7 business questions
   - Covers retail, finance, and healthcare domains
   - Includes difficulty levels and metadata

## Usage

### Quick Evaluation (Testing)

```bash
# Run quick evaluation with limited scope
python scripts/run_full_evaluation.py --mode quick
```

### Full Evaluation

```bash
# Run full evaluation across all providers and questions
python scripts/run_full_evaluation.py --mode full

# Filter by specific domains
python scripts/run_full_evaluation.py --mode full --domains retail finance

# Filter by difficulty levels
python scripts/run_full_evaluation.py --mode full --difficulties medium hard

# Filter by specific providers
python scripts/run_full_evaluation.py --mode full --providers groq gemini
```

### Programmatic Usage

```python
from src.evaluation.evaluator import LLMEvaluator

# Initialize evaluator
evaluator = LLMEvaluator()

# Run batch evaluation
batch_result = evaluator.run_batch_evaluation(
    question_ids=["retail_001", "retail_002"],
    provider_names=["groq", "gemini"],
    model_names=["llama3-8b-8192", "gemini-pro"]
)

# Access results
print(f"Total evaluations: {len(batch_result.results)}")
print(f"Best provider: {batch_result.summary_stats['overall_stats']['best_provider']}")
```

## Evaluation Metrics

### 1. Relevance Score (25% weight)
- **Method**: TF-IDF cosine similarity between query/context and response
- **Range**: 0.0 - 1.0
- **Interpretation**: Higher scores indicate better query relevance

### 2. Factual Accuracy (25% weight)
- **Method**: F1-score comparing extracted facts with ground truth
- **Range**: 0.0 - 1.0
- **Interpretation**: Higher scores indicate more accurate factual claims

### 3. Coherence Score (20% weight)
- **Method**: Sentence similarity and logical connector analysis
- **Range**: 0.0 - 1.0
- **Interpretation**: Higher scores indicate better logical flow

### 4. Completeness Score (15% weight)
- **Method**: Query term coverage and response adequacy
- **Range**: 0.0 - 1.0
- **Interpretation**: Higher scores indicate more comprehensive responses

### 5. Token Efficiency (10% weight)
- **Method**: Information density per token consumed
- **Range**: 0.0 - 1.0
- **Interpretation**: Higher scores indicate more efficient token usage

### 6. Time Efficiency (5% weight)
- **Method**: Characters generated per second
- **Range**: 0.0 - 1.0
- **Interpretation**: Higher scores indicate faster response generation

### Overall Score
- **Method**: Weighted average of all metrics
- **Range**: 0.0 - 1.0
- **Formula**: `0.25*relevance + 0.25*factual + 0.20*coherence + 0.15*completeness + 0.10*token_efficiency + 0.05*time_efficiency`

## Statistical Analysis

### Descriptive Statistics
- Mean, median, standard deviation
- Confidence intervals (95% default)
- Min/max values and quartiles

### Inferential Statistics
- **T-tests**: Pairwise provider comparisons
- **ANOVA**: Multi-group significance testing
- **Effect sizes**: Cohen's d for practical significance
- **Correlation analysis**: Pearson and Spearman correlations

### Provider Rankings
- Performance-based ranking
- Percentile scores
- Statistical significance indicators

## Output Files

### Results Structure
```
data/evaluation_results/
├── eval_YYYYMMDD_HHMMSS_results.json    # Detailed results
├── eval_YYYYMMDD_HHMMSS_summary.txt     # Human-readable summary
└── logs/
    └── evaluation.log                   # Evaluation logs
```

### JSON Results Format
```json
{
  "evaluation_id": "eval_20250705_002103",
  "timestamp": "2025-07-05T00:21:05.049457",
  "total_questions": 1,
  "total_providers": 1,
  "results": [
    {
      "question_id": "retail_001",
      "provider_name": "groq",
      "model_name": "llama3-8b-8192",
      "query": "What are the top performing product categories?",
      "response": "...",
      "ground_truth": "...",
      "metrics": {
        "relevance_score": 0.1307,
        "factual_accuracy": 0.1053,
        "response_time_ms": 687.06,
        "token_efficiency": 1.0,
        "coherence_score": 0.2062,
        "completeness_score": 0.775,
        "overall_score": 0.3665,
        "confidence_interval": [0, 0.7636]
      }
    }
  ],
  "summary_stats": {
    "total_evaluations": 1,
    "providers_tested": 1,
    "provider_stats": {...},
    "overall_stats": {...}
  },
  "statistical_analysis": {
    "anova_result": {...},
    "rankings": {...},
    "pairwise_comparisons": {...},
    "effect_sizes": {...}
  }
}
```

## Ground Truth Management

### Adding New Questions

```python
from src.evaluation.ground_truth import GroundTruthManager, GroundTruthAnswer

manager = GroundTruthManager()

new_answer = GroundTruthAnswer(
    question_id="retail_006",
    question="What is the customer retention rate?",
    answer="The customer retention rate is 78% based on repeat purchases...",
    category="customer_analysis",
    difficulty="medium",
    key_points=["Retention rate: 78%", "Repeat purchases", "Customer loyalty"],
    factual_claims=["Retention rate: 78%"],
    expected_length="medium",
    domain="retail",
    metadata={"data_source": "customer_data.csv", "analysis_type": "retention_analysis"}
)

manager.add_answer(new_answer)
```

### Question Categories
- **Sales Analysis**: Revenue, sales trends, product performance
- **Customer Analysis**: Segments, behavior, satisfaction
- **Trend Analysis**: Time series, seasonal patterns
- **Product Analysis**: Performance, comparisons, rankings
- **Data Overview**: Dataset structure and characteristics
- **Financial Analysis**: KPIs, metrics, ratios
- **Patient Analysis**: Healthcare-specific metrics

## Best Practices

### 1. Evaluation Design
- Use consistent question formats across providers
- Include diverse difficulty levels
- Balance domain representation
- Ensure ground truth quality and accuracy

### 2. Statistical Rigor
- Collect sufficient sample sizes (n ≥ 30 per group)
- Report confidence intervals
- Use appropriate significance levels (α = 0.05)
- Consider effect sizes for practical significance

### 3. Result Interpretation
- Consider context and domain specificity
- Account for API rate limits and costs
- Validate metrics against human judgment
- Document evaluation conditions and limitations

### 4. Performance Optimization
- Use batch processing for efficiency
- Implement proper error handling
- Monitor API usage and costs
- Cache results for reproducibility

## Troubleshooting

### Common Issues

1. **NLTK Data Missing**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

2. **API Rate Limits**
   - Implement exponential backoff
   - Use rate limiting in provider configurations
   - Monitor API usage

3. **Memory Issues**
   - Process results in batches
   - Clear caches between evaluations
   - Use streaming for large datasets

4. **JSON Serialization Errors**
   - Ensure all data types are JSON-serializable
   - Convert numpy types to Python types
   - Handle None values appropriately

### Debug Mode

```bash
# Enable debug logging
python scripts/run_full_evaluation.py --mode quick --log-level DEBUG
```

## Research Applications

### Academic Research
- Comparative LLM performance studies
- Domain-specific evaluation
- Cost-effectiveness analysis
- Reproducible evaluation protocols

### Industry Applications
- Vendor selection and comparison
- Performance monitoring
- Quality assurance
- ROI analysis

### Custom Extensions
- Domain-specific metrics
- Custom ground truth datasets
- Additional statistical tests
- Visualization and reporting

## Contributing

### Adding New Metrics
1. Implement metric class in `src/evaluation/metrics.py`
2. Add to `EvaluationMetricsCalculator`
3. Update weights in overall score calculation
4. Add tests and documentation

### Adding New Providers
1. Implement provider class inheriting from `BaseProvider`
2. Add to `ProviderManager`
3. Update configuration files
4. Test with evaluation system

### Adding New Statistical Tests
1. Implement test in `src/evaluation/statistical_analysis.py`
2. Add to evaluation pipeline
3. Update reporting functions
4. Document methodology

## License and Citation

This evaluation system is part of the MSc research project "Intelligent Business Analysis Using Free-Tier LLMs: A Comparative Framework for Multi-Industry Decision Support".

For academic use, please cite:
```
[Your Citation Here]
```

## Support

For questions, issues, or contributions:
1. Check the troubleshooting section
2. Review the logs in `logs/evaluation.log`
3. Create an issue with detailed error information
4. Contact the development team

---

*Last updated: July 2025* 