# Technical Methodology

## Overview

This document outlines the technical methodology employed in the LLM Business Analysis System, including the evaluation framework, statistical analysis methods, and research design principles.

## Table of Contents

1. [Research Design](#research-design)
2. [Evaluation Framework](#evaluation-framework)
3. [Statistical Analysis](#statistical-analysis)
4. [Data Collection](#data-collection)
5. [Quality Assurance](#quality-assurance)
6. [Reproducibility](#reproducibility)

---

## Research Design

### Research Questions

1. **Primary Question**: How do free-tier LLM providers compare in terms of response quality, speed, and cost-effectiveness for business analysis tasks?

2. **Secondary Questions**:
   - Which provider offers the best balance of speed and accuracy?
   - How do different business domains affect LLM performance?
   - What is the relationship between model complexity and response quality?
   - How reliable are free-tier LLMs for production business applications?

### Research Hypotheses

**H1**: Groq's optimized inference will provide faster response times compared to other providers.

**H2**: Gemini models will show higher factual accuracy due to Google's training data quality.

**H3**: OpenRouter's diverse model selection will provide better domain-specific performance.

**H4**: There will be significant performance differences across business domains (retail, finance, healthcare).

### Experimental Design

#### Independent Variables
- **LLM Provider**: Groq, Gemini, OpenRouter
- **Model**: Specific models within each provider
- **Business Domain**: Retail, Finance, Healthcare
- **Question Difficulty**: Easy, Medium, Hard
- **Question Type**: Factual, Analytical, Creative

#### Dependent Variables
- **Response Quality**: Relevance, accuracy, coherence scores
- **Performance**: Response time, token usage
- **Cost**: API calls, computational resources
- **User Satisfaction**: Preference ratings, usability scores

#### Control Variables
- **Question Set**: Standardized questions across all providers
- **Evaluation Criteria**: Consistent scoring methodology
- **Environment**: Same hardware, network conditions
- **Timing**: Concurrent testing to minimize external factors

---

## Evaluation Framework

### Multi-Dimensional Evaluation Model

The evaluation framework employs a comprehensive multi-dimensional approach:

```python
class EvaluationMetrics:
    """Multi-dimensional evaluation metrics"""
    
    def __init__(self):
        self.dimensions = {
            'relevance': RelevanceEvaluator(),
            'accuracy': AccuracyEvaluator(),
            'coherence': CoherenceEvaluator(),
            'completeness': CompletenessEvaluator(),
            'usefulness': UsefulnessEvaluator()
        }
    
    def evaluate_response(self, response: str, ground_truth: str, 
                         context: str = None) -> Dict[str, float]:
        """Evaluate response across all dimensions"""
        
        scores = {}
        for dimension, evaluator in self.dimensions.items():
            scores[dimension] = evaluator.score(response, ground_truth, context)
        
        # Calculate overall score
        scores['overall'] = self.calculate_overall_score(scores)
        
        return scores
```

### Evaluation Dimensions

#### 1. Relevance (0-1 scale)
**Definition**: How well the response addresses the specific question asked.

**Evaluation Method**:
```python
class RelevanceEvaluator:
    def score(self, response: str, question: str, context: str = None) -> float:
        # Use semantic similarity with question
        question_embedding = self.get_embedding(question)
        response_embedding = self.get_embedding(response)
        
        similarity = cosine_similarity(question_embedding, response_embedding)
        
        # Apply domain-specific relevance rules
        domain_score = self.apply_domain_rules(response, context)
        
        return (similarity + domain_score) / 2
```

#### 2. Factual Accuracy (0-1 scale)
**Definition**: The correctness of factual information provided in the response.

**Evaluation Method**:
```python
class AccuracyEvaluator:
    def score(self, response: str, ground_truth: str) -> float:
        # Extract factual claims from response
        claims = self.extract_claims(response)
        
        # Compare with ground truth
        accuracy_scores = []
        for claim in claims:
            truth_score = self.verify_claim(claim, ground_truth)
            accuracy_scores.append(truth_score)
        
        return np.mean(accuracy_scores) if accuracy_scores else 0.0
```

#### 3. Coherence (0-1 scale)
**Definition**: The logical flow and readability of the response.

**Evaluation Method**:
```python
class CoherenceEvaluator:
    def score(self, response: str) -> float:
        # Evaluate sentence structure
        structure_score = self.evaluate_structure(response)
        
        # Evaluate logical flow
        flow_score = self.evaluate_logical_flow(response)
        
        # Evaluate grammar and readability
        readability_score = self.evaluate_readability(response)
        
        return (structure_score + flow_score + readability_score) / 3
```

#### 4. Completeness (0-1 scale)
**Definition**: How thoroughly the response addresses all aspects of the question.

**Evaluation Method**:
```python
class CompletenessEvaluator:
    def score(self, response: str, question: str, expected_aspects: List[str]) -> float:
        # Identify expected aspects from question
        aspects = self.identify_question_aspects(question)
        
        # Check coverage of each aspect
        coverage_scores = []
        for aspect in aspects:
            coverage = self.check_aspect_coverage(response, aspect)
            coverage_scores.append(coverage)
        
        return np.mean(coverage_scores) if coverage_scores else 0.0
```

#### 5. Usefulness (0-1 scale)
**Definition**: The practical value and actionable insights provided.

**Evaluation Method**:
```python
class UsefulnessEvaluator:
    def score(self, response: str, business_context: str) -> float:
        # Check for actionable insights
        actionability = self.check_actionability(response)
        
        # Check for specific recommendations
        specificity = self.check_specificity(response)
        
        # Check for business relevance
        relevance = self.check_business_relevance(response, business_context)
        
        return (actionability + specificity + relevance) / 3
```

### Ground Truth System

#### Ground Truth Database
```python
class GroundTruthManager:
    """Manages ground truth data for evaluation"""
    
    def __init__(self):
        self.ground_truth_data = {
            'retail': {
                'easy': self.load_retail_easy_questions(),
                'medium': self.load_retail_medium_questions(),
                'hard': self.load_retail_hard_questions()
            },
            'finance': {
                'easy': self.load_finance_easy_questions(),
                'medium': self.load_finance_medium_questions(),
                'hard': self.load_finance_hard_questions()
            },
            'healthcare': {
                'easy': self.load_healthcare_easy_questions(),
                'medium': self.load_healthcare_medium_questions(),
                'hard': self.load_healthcare_hard_questions()
            }
        }
    
    def get_question_set(self, domain: str, difficulty: str) -> List[GroundTruthItem]:
        """Get question set for specific domain and difficulty"""
        return self.ground_truth_data[domain][difficulty]
    
    def get_question_by_id(self, question_id: str) -> Optional[GroundTruthItem]:
        """Get specific question by ID"""
        for domain in self.ground_truth_data.values():
            for difficulty in domain.values():
                for question in difficulty:
                    if question.id == question_id:
                        return question
        return None
```

#### Ground Truth Item Structure
```python
@dataclass
class GroundTruthItem:
    id: str
    question: str
    expected_answer: str
    key_points: List[str]
    domain: str
    difficulty: str
    question_type: str
    metadata: Dict[str, Any]
```

---

## Statistical Analysis

### Statistical Framework

#### 1. Descriptive Statistics
```python
class DescriptiveStatistics:
    """Calculate descriptive statistics for evaluation results"""
    
    def calculate_summary_stats(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics"""
        
        # Extract metrics
        relevance_scores = [r.metrics.relevance_score for r in results]
        accuracy_scores = [r.metrics.factual_accuracy for r in results]
        coherence_scores = [r.metrics.coherence_score for r in results]
        overall_scores = [r.metrics.overall_score for r in results]
        response_times = [r.response_time_ms for r in results]
        
        return {
            'relevance': self.calculate_stats(relevance_scores),
            'accuracy': self.calculate_stats(accuracy_scores),
            'coherence': self.calculate_stats(coherence_scores),
            'overall': self.calculate_stats(overall_scores),
            'response_time': self.calculate_stats(response_times)
        }
    
    def calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of values"""
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75)
        }
```

#### 2. Inferential Statistics

**ANOVA Analysis**
```python
class ANOVAAnalyzer:
    """Perform ANOVA analysis to compare provider performance"""
    
    def compare_providers(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compare performance across providers using ANOVA"""
        
        # Group results by provider
        provider_groups = {}
        for result in results:
            if result.provider_name not in provider_groups:
                provider_groups[result.provider_name] = []
            provider_groups[result.provider_name].append(result.metrics.overall_score)
        
        # Perform one-way ANOVA
        from scipy.stats import f_oneway
        
        groups = list(provider_groups.values())
        f_stat, p_value = f_oneway(*groups)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': self.calculate_effect_size(groups)
        }
```

**Post-hoc Analysis**
```python
class PostHocAnalyzer:
    """Perform post-hoc analysis for pairwise comparisons"""
    
    def tukey_hsd(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """Perform Tukey's HSD test for pairwise comparisons"""
        
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        
        # Prepare data
        providers = [r.provider_name for r in results]
        scores = [r.metrics.overall_score for r in results]
        
        # Perform Tukey's HSD
        tukey_result = pairwise_tukeyhsd(scores, providers)
        
        return pd.DataFrame(tukey_result._results_table.data[1:],
                           columns=tukey_result._results_table.data[0])
```

#### 3. Effect Size Analysis
```python
class EffectSizeAnalyzer:
    """Calculate effect sizes for statistical comparisons"""
    
    def cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def eta_squared(self, f_stat: float, df_between: int, df_within: int) -> float:
        """Calculate eta-squared effect size from F-statistic"""
        
        return (f_stat * df_between) / (f_stat * df_between + df_within)
```

### Confidence Intervals

```python
class ConfidenceIntervalCalculator:
    """Calculate confidence intervals for evaluation metrics"""
    
    def calculate_ci(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for a list of values"""
        
        from scipy.stats import t
        
        n = len(values)
        mean = np.mean(values)
        std_err = np.std(values, ddof=1) / np.sqrt(n)
        
        # Calculate t-critical value
        t_critical = t.ppf((1 + confidence) / 2, df=n-1)
        
        margin_of_error = t_critical * std_err
        
        return (mean - margin_of_error, mean + margin_of_error)
    
    def calculate_bootstrap_ci(self, values: List[float], confidence: float = 0.95, 
                              n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        
        return (np.percentile(bootstrap_means, lower_percentile),
                np.percentile(bootstrap_means, upper_percentile))
```

---

## Data Collection

### Data Collection Protocol

#### 1. Question Generation
```python
class QuestionGenerator:
    """Generate standardized questions for evaluation"""
    
    def __init__(self):
        self.question_templates = {
            'retail': {
                'easy': [
                    "What is the definition of {concept}?",
                    "How does {process} work?",
                    "What are the benefits of {strategy}?"
                ],
                'medium': [
                    "Compare {concept1} and {concept2} in retail.",
                    "Analyze the impact of {factor} on {outcome}.",
                    "What strategies can improve {metric}?"
                ],
                'hard': [
                    "Develop a comprehensive strategy for {challenge}.",
                    "Evaluate the effectiveness of {approach} in {context}.",
                    "Propose innovative solutions for {problem}."
                ]
            }
        }
    
    def generate_questions(self, domain: str, difficulty: str, 
                          count: int = 10) -> List[str]:
        """Generate questions for specific domain and difficulty"""
        
        templates = self.question_templates[domain][difficulty]
        questions = []
        
        for i in range(count):
            template = templates[i % len(templates)]
            question = self.fill_template(template, domain, difficulty)
            questions.append(question)
        
        return questions
```

#### 2. Response Collection
```python
class ResponseCollector:
    """Collect and store LLM responses"""
    
    def __init__(self):
        self.provider_manager = ProviderManager()
        self.results = []
    
    def collect_responses(self, questions: List[str], providers: List[str], 
                         models: List[str]) -> List[EvaluationResult]:
        """Collect responses from all providers for all questions"""
        
        results = []
        
        for question in questions:
            for provider_name in providers:
                for model_name in models:
                    try:
                        # Generate response
                        provider = self.provider_manager.get_provider(provider_name)
                        response = provider.generate_response(
                            query=question,
                            model=model_name
                        )
                        
                        # Create evaluation result
                        result = EvaluationResult(
                            question_id=question.id,
                            provider_name=provider_name,
                            model_name=model_name,
                            response=response.text,
                            response_time_ms=response.latency_ms,
                            tokens_used=response.tokens_used,
                            timestamp=datetime.now()
                        )
                        
                        results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Error collecting response: {e}")
                        continue
        
        return results
```

#### 3. Data Validation
```python
class DataValidator:
    """Validate collected data for quality and completeness"""
    
    def validate_response_data(self, results: List[EvaluationResult]) -> ValidationReport:
        """Validate response data quality"""
        
        report = ValidationReport()
        
        for result in results:
            # Check response completeness
            if not result.response or len(result.response.strip()) < 10:
                report.add_issue("incomplete_response", result)
            
            # Check response time validity
            if result.response_time_ms < 0 or result.response_time_ms > 60000:
                report.add_issue("invalid_response_time", result)
            
            # Check token usage validity
            if result.tokens_used and result.tokens_used < 0:
                report.add_issue("invalid_token_usage", result)
        
        return report
```

---

## Quality Assurance

### Quality Control Measures

#### 1. Automated Quality Checks
```python
class QualityController:
    """Implement quality control measures"""
    
    def check_response_quality(self, response: str) -> QualityReport:
        """Check response quality automatically"""
        
        report = QualityReport()
        
        # Check for empty or very short responses
        if not response or len(response.strip()) < 10:
            report.add_issue("response_too_short", severity="high")
        
        # Check for repetitive content
        if self.detect_repetition(response):
            report.add_issue("repetitive_content", severity="medium")
        
        # Check for inappropriate content
        if self.detect_inappropriate_content(response):
            report.add_issue("inappropriate_content", severity="high")
        
        # Check for coherent structure
        if not self.check_coherence(response):
            report.add_issue("poor_coherence", severity="medium")
        
        return report
```

#### 2. Human Validation
```python
class HumanValidator:
    """Implement human validation for quality assurance"""
    
    def validate_sample_responses(self, results: List[EvaluationResult], 
                                 sample_size: int = 50) -> ValidationReport:
        """Validate a sample of responses with human reviewers"""
        
        # Select random sample
        sample = random.sample(results, min(sample_size, len(results)))
        
        report = ValidationReport()
        
        for result in sample:
            # Human evaluation
            human_score = self.get_human_evaluation(result)
            
            # Compare with automated score
            automated_score = result.metrics.overall_score
            difference = abs(human_score - automated_score)
            
            if difference > 0.2:  # Significant difference
                report.add_issue("score_discrepancy", result, 
                               details=f"Human: {human_score}, Auto: {automated_score}")
        
        return report
```

#### 3. Inter-rater Reliability
```python
class InterRaterReliability:
    """Calculate inter-rater reliability for human evaluations"""
    
    def calculate_krippendorff_alpha(self, ratings: List[List[float]]) -> float:
        """Calculate Krippendorff's alpha for inter-rater reliability"""
        
        from krippendorff import alpha
        
        # Convert ratings to format expected by krippendorff
        reliability_data = []
        for i in range(len(ratings[0])):
            reliability_data.append([ratings[j][i] for j in range(len(ratings))])
        
        return alpha(reliability_data)
    
    def calculate_fleiss_kappa(self, ratings: List[List[int]]) -> float:
        """Calculate Fleiss' kappa for categorical ratings"""
        
        from statsmodels.stats.inter_rater import fleiss_kappa
        
        return fleiss_kappa(ratings)
```

---

## Reproducibility

### Reproducibility Framework

#### 1. Version Control
```python
class ReproducibilityManager:
    """Manage reproducibility of experiments"""
    
    def __init__(self):
        self.experiment_config = {}
        self.results_metadata = {}
    
    def save_experiment_config(self, config: Dict[str, Any], 
                              experiment_id: str) -> None:
        """Save experiment configuration for reproducibility"""
        
        config_data = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'git_commit': self.get_git_commit(),
            'python_version': sys.version,
            'dependencies': self.get_dependencies()
        }
        
        # Save to file
        config_path = f"experiments/{experiment_id}/config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_experiment_config(self, experiment_id: str) -> Dict[str, Any]:
        """Load experiment configuration for reproduction"""
        
        config_path = f"experiments/{experiment_id}/config.json"
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return config_data
```

#### 2. Random Seed Management
```python
class SeedManager:
    """Manage random seeds for reproducible results"""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.seed_counter = 0
    
    def get_seed(self, component: str) -> int:
        """Get deterministic seed for component"""
        
        # Use hash of component name for deterministic but different seeds
        component_hash = hash(component) % 10000
        return self.base_seed + component_hash + self.seed_counter
    
    def set_seed(self, component: str) -> None:
        """Set random seed for component"""
        
        seed = self.get_seed(component)
        random.seed(seed)
        np.random.seed(seed)
        
        # Set PyTorch seed if available
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
    
    def increment_counter(self) -> None:
        """Increment seed counter for different runs"""
        
        self.seed_counter += 1
```

#### 3. Results Archiving
```python
class ResultsArchiver:
    """Archive experiment results for long-term storage"""
    
    def archive_results(self, results: List[EvaluationResult], 
                       experiment_id: str) -> str:
        """Archive results with metadata"""
        
        # Create archive directory
        archive_dir = f"archives/{experiment_id}"
        os.makedirs(archive_dir, exist_ok=True)
        
        # Save results
        results_path = f"{archive_dir}/results.json"
        self.save_results(results, results_path)
        
        # Save metadata
        metadata = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'providers': list(set(r.provider_name for r in results)),
            'models': list(set(r.model_name for r in results)),
            'domains': list(set(r.domain for r in results)),
            'difficulties': list(set(r.difficulty for r in results))
        }
        
        metadata_path = f"{archive_dir}/metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create checksum for integrity
        checksum = self.calculate_checksum(archive_dir)
        
        with open(f"{archive_dir}/checksum.txt", 'w') as f:
            f.write(checksum)
        
        return archive_dir
    
    def verify_archive(self, archive_dir: str) -> bool:
        """Verify archive integrity"""
        
        checksum_path = f"{archive_dir}/checksum.txt"
        
        if not os.path.exists(checksum_path):
            return False
        
        with open(checksum_path, 'r') as f:
            stored_checksum = f.read().strip()
        
        current_checksum = self.calculate_checksum(archive_dir)
        
        return stored_checksum == current_checksum
```

### Reporting Standards

#### 1. Statistical Reporting
```python
class StatisticalReporter:
    """Generate standardized statistical reports"""
    
    def generate_report(self, results: List[EvaluationResult]) -> str:
        """Generate comprehensive statistical report"""
        
        report = []
        report.append("# Statistical Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Descriptive statistics
        report.append("## Descriptive Statistics")
        desc_stats = self.calculate_descriptive_statistics(results)
        report.extend(self.format_descriptive_stats(desc_stats))
        
        # Inferential statistics
        report.append("## Inferential Statistics")
        inf_stats = self.calculate_inferential_statistics(results)
        report.extend(self.format_inferential_stats(inf_stats))
        
        # Effect sizes
        report.append("## Effect Sizes")
        effect_sizes = self.calculate_effect_sizes(results)
        report.extend(self.format_effect_sizes(effect_sizes))
        
        # Confidence intervals
        report.append("## Confidence Intervals")
        confidence_intervals = self.calculate_confidence_intervals(results)
        report.extend(self.format_confidence_intervals(confidence_intervals))
        
        return "\n".join(report)
```

#### 2. Visualization Standards
```python
class VisualizationGenerator:
    """Generate standardized visualizations"""
    
    def create_performance_comparison(self, results: List[EvaluationResult]) -> plt.Figure:
        """Create standardized performance comparison chart"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall scores by provider
        self.plot_overall_scores(results, axes[0, 0])
        
        # Response times by provider
        self.plot_response_times(results, axes[0, 1])
        
        # Scores by domain
        self.plot_domain_scores(results, axes[1, 0])
        
        # Scores by difficulty
        self.plot_difficulty_scores(results, axes[1, 1])
        
        plt.tight_layout()
        return fig
    
    def plot_overall_scores(self, results: List[EvaluationResult], ax: plt.Axes) -> None:
        """Plot overall scores by provider"""
        
        providers = list(set(r.provider_name for r in results))
        scores_by_provider = {}
        
        for provider in providers:
            provider_results = [r for r in results if r.provider_name == provider]
            scores = [r.metrics.overall_score for r in provider_results]
            scores_by_provider[provider] = scores
        
        # Create box plot
        ax.boxplot(scores_by_provider.values(), labels=scores_by_provider.keys())
        ax.set_title("Overall Scores by Provider")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
```

This technical methodology provides a comprehensive framework for conducting rigorous, reproducible research on LLM performance comparison, ensuring scientific validity and practical relevance for business applications. 