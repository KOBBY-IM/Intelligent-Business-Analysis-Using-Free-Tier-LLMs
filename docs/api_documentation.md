# API Documentation

## Overview

This document provides comprehensive API documentation for the LLM Business Analysis System, covering all major components and their interfaces.

## Table of Contents

1. [LLM Providers API](#llm-providers-api)
2. [Evaluation System API](#evaluation-system-api)
3. [RAG Pipeline API](#rag-pipeline-api)
4. [Data Processing API](#data-processing-api)
5. [Configuration API](#configuration-api)
6. [Security API](#security-api)

---

## LLM Providers API

### Base Provider Interface

All LLM providers implement the `BaseProvider` interface:

```python
class BaseProvider:
    def __init__(self, name: str, models: List[str])
    def list_models(self) -> List[str]
    def health_check(self) -> bool
    def generate_response(self, query: str, context: str = "", model: Optional[str] = None, **kwargs) -> LLMResponse
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]
```

### LLMResponse Object

```python
@dataclass
class LLMResponse:
    success: bool
    text: str
    model: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
```

### Provider Manager

```python
class ProviderManager:
    def __init__(self)
    def get_provider(self, name: str) -> BaseProvider
    def get_provider_names(self) -> List[str]
    def get_all_providers(self) -> Dict[str, BaseProvider]
    def health_check_all(self) -> Dict[str, bool]
```

### Available Providers

#### Groq Provider
- **Models**: `llama3-8b-8192`, `llama-3.1-8b-instant`, `qwen-qwq-32b`
- **API Key**: `GROQ_API_KEY`
- **Base URL**: `https://api.groq.com/openai/v1`

#### Gemini Provider
- **Models**: `gemini-1.5-flash`, `gemma-3-12b-it`
- **API Key**: `GOOGLE_API_KEY`
- **Base URL**: `https://generativelanguage.googleapis.com/v1beta/models`

#### OpenRouter Provider
- **Models**: `mistralai/mistral-7b-instruct`, `deepseek/deepseek-r1-0528-qwen3-8b`, `openrouter/cypher-alpha`, `qwen/qwen3-14b`
- **API Key**: `OPENROUTER_API_KEY`
- **Base URL**: `https://openrouter.ai/api/v1/chat/completions`

---

## Evaluation System API

### LLM Evaluator

```python
class LLMEvaluator:
    def __init__(self)
    def evaluate_single_response(self, question_id: str, provider_name: str, model_name: str, 
                               response: str, response_time_ms: float, tokens_used: int) -> EvaluationResult
    def run_batch_evaluation(self, provider_names: List[str], domains: List[str], 
                           difficulties: List[str]) -> BatchEvaluationResult
    def get_evaluation_history(self) -> List[Dict[str, Any]]
    def export_results(self, format: str = "json") -> str
```

### Evaluation Result

```python
@dataclass
class EvaluationResult:
    question_id: str
    provider_name: str
    model_name: str
    response: str
    response_time_ms: float
    tokens_used: int
    metrics: EvaluationMetrics
    timestamp: datetime
```

### Evaluation Metrics

```python
@dataclass
class EvaluationMetrics:
    relevance_score: float
    factual_accuracy: float
    coherence_score: float
    overall_score: float
    confidence_interval: Optional[Tuple[float, float]] = None
```

### Ground Truth System

```python
class GroundTruthManager:
    def __init__(self)
    def load_ground_truth(self, domain: str, difficulty: str) -> List[GroundTruthItem]
    def get_question_by_id(self, question_id: str) -> Optional[GroundTruthItem]
    def calculate_similarity(self, response: str, ground_truth: str) -> float
```

---

## RAG Pipeline API

### RAG Pipeline

```python
class RAGPipeline:
    def __init__(self, vector_store: VectorStore, llm_provider: BaseProvider, 
                 top_k: int = 5, max_context_length: int = 2000)
    def add_documents(self, documents: List[str])
    def query(self, question: str, **kwargs) -> RAGResponse
    def get_context(self, question: str, top_k: int = 5) -> List[str]
```

### Vector Store Interface

```python
class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[str])
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]
```

### FAISS Vector Store

```python
class FAISSVectorStore(VectorStore):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 300, 
                 chunk_overlap: int = 50)
    def add_documents(self, documents: List[str])
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]
    def save(self, path: str)
    def load(self, path: str)
```

### RAG Response

```python
@dataclass
class RAGResponse:
    answer: str
    context: List[str]
    sources: List[str]
    confidence: float
    response_time_ms: float
```

---

## Data Processing API

### Data Cleaner

```python
class DataCleaner:
    def __init__(self)
    def clean_text(self, text: str) -> str
    def remove_special_characters(self, text: str) -> str
    def normalize_whitespace(self, text: str) -> str
    def validate_data(self, data: pd.DataFrame) -> ValidationResult
```

### Retail Data Processor

```python
class RetailDataProcessor:
    def __init__(self)
    def process_sales_data(self, data: pd.DataFrame) -> pd.DataFrame
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame
    def generate_insights(self, data: pd.DataFrame) -> List[str]
```

---

## Configuration API

### Configuration Manager

```python
class ConfigurationManager:
    def __init__(self, config_dir: str = "config")
    def load_config(self, config_name: str) -> Dict[str, Any]
    def save_config(self, config_name: str, config: Dict[str, Any])
    def validate_config(self, config: Dict[str, Any]) -> bool
    def get_default_config(self, config_type: str) -> Dict[str, Any]
```

### Available Configurations

#### App Configuration (`app_config.yaml`)
```yaml
app:
  name: "LLM Business Analysis System"
  version: "1.0.0"
  debug: false
  log_level: "INFO"

ui:
  theme: "light"
  auto_refresh: false
  max_upload_size_mb: 10

security:
  allowed_hosts: ["localhost", "127.0.0.1"]
  rate_limit_per_minute: 60
  query_limit_per_hour: 100
```

#### LLM Configuration (`llm_config.yaml`)
```yaml
providers:
  groq:
    api_key_env: "GROQ_API_KEY"
    base_url: "https://api.groq.com/openai/v1"
    timeout: 30
    
  gemini:
    api_key_env: "GOOGLE_API_KEY"
    base_url: "https://generativelanguage.googleapis.com/v1beta/models"
    timeout: 30
    
  openrouter:
    api_key_env: "OPENROUTER_API_KEY"
    base_url: "https://openrouter.ai/api/v1/chat/completions"
    timeout: 30

generation:
  default_max_tokens: 1000
  default_temperature: 0.7
  default_top_p: 0.9
```

#### Evaluation Configuration (`evaluation_config.yaml`)
```yaml
evaluation:
  domains: ["retail", "finance", "healthcare"]
  difficulties: ["easy", "medium", "hard"]
  max_questions_per_domain: 10
  evaluation_metrics:
    - "relevance"
    - "factual_accuracy"
    - "coherence"
    - "overall_score"

ground_truth:
  data_path: "data/ground_truth"
  similarity_threshold: 0.7
```

---

## Security API

### Input Validator

```python
class InputValidator:
    def __init__(self)
    def validate_query(self, query: str) -> ValidationResult
    def validate_file_upload(self, file) -> ValidationResult
    def sanitize_input(self, text: str) -> str
    def check_rate_limit(self, user_id: str) -> bool
```

### Security Manager

```python
class SecurityManager:
    def __init__(self)
    def validate_api_key(self, provider: str, api_key: str) -> bool
    def check_permissions(self, user_id: str, action: str) -> bool
    def log_security_event(self, event_type: str, details: Dict[str, Any])
    def encrypt_sensitive_data(self, data: str) -> str
    def decrypt_sensitive_data(self, encrypted_data: str) -> str
```

---

## Error Handling

### Custom Exceptions

```python
class LLMTimeoutError(Exception):
    """Raised when LLM request times out"""
    pass

class RateLimitException(Exception):
    """Raised when API rate limit is exceeded"""
    pass

class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass
```

### Error Response Format

```python
{
    "error": {
        "type": "error_type",
        "message": "Human readable error message",
        "details": "Additional error details",
        "timestamp": "2024-01-01T00:00:00Z",
        "request_id": "unique_request_id"
    }
}
```

---

## Usage Examples

### Basic LLM Query

```python
from src.llm_providers import ProviderManager

# Initialize provider manager
manager = ProviderManager()
provider = manager.get_provider("groq")

# Generate response
response = provider.generate_response(
    query="What is machine learning?",
    model="llama3-8b-8192",
    max_tokens=300,
    temperature=0.7
)

if response.success:
    print(f"Response: {response.text}")
    print(f"Tokens used: {response.tokens_used}")
    print(f"Latency: {response.latency_ms}ms")
else:
    print(f"Error: {response.error}")
```

### Evaluation Example

```python
from src.evaluation import LLMEvaluator

# Initialize evaluator
evaluator = LLMEvaluator()

# Evaluate single response
result = evaluator.evaluate_single_response(
    question_id="retail_001",
    provider_name="groq",
    model_name="llama3-8b-8192",
    response="Machine learning is a subset of AI...",
    response_time_ms=1500,
    tokens_used=45
)

print(f"Overall score: {result.metrics.overall_score}")
```

### RAG Pipeline Example

```python
from src.rag import RAGPipeline, FAISSVectorStore
from src.llm_providers import ProviderManager

# Initialize components
vector_store = FAISSVectorStore()
provider = ProviderManager().get_provider("groq")
pipeline = RAGPipeline(vector_store, provider)

# Add documents
documents = ["Document 1 content...", "Document 2 content..."]
pipeline.add_documents(documents)

# Query
response = pipeline.query("What is the main topic?")
print(f"Answer: {response.answer}")
print(f"Sources: {response.sources}")
```

---

## Rate Limits and Quotas

### Provider-Specific Limits

| Provider | Rate Limit | Quota | Notes |
|----------|------------|-------|-------|
| Groq | 500 requests/min | 1000 requests/day | Free tier |
| Gemini | 60 requests/min | 1500 requests/day | Free tier |
| OpenRouter | 100 requests/min | 5000 requests/day | Free tier |

### System Limits

- **Query Length**: Maximum 5000 characters
- **File Upload**: Maximum 10MB
- **Response Tokens**: Maximum 1000 tokens
- **Concurrent Requests**: Maximum 5 per user

---

## Performance Metrics

### Response Time Benchmarks

| Provider | Model | Avg Response Time | 95th Percentile |
|----------|-------|-------------------|-----------------|
| Groq | llama3-8b-8192 | 800ms | 1200ms |
| Gemini | gemini-1.5-flash | 1200ms | 1800ms |
| OpenRouter | mistral-7b-instruct | 1500ms | 2200ms |

### Accuracy Metrics

| Provider | Model | Relevance Score | Factual Accuracy | Coherence |
|----------|-------|----------------|------------------|-----------|
| Groq | llama3-8b-8192 | 0.85 | 0.82 | 0.88 |
| Gemini | gemini-1.5-flash | 0.87 | 0.85 | 0.90 |
| OpenRouter | mistral-7b-instruct | 0.83 | 0.80 | 0.86 |

---

## Version History

### v1.0.0 (Current)
- Initial release with 3 LLM providers
- Complete evaluation system
- RAG pipeline implementation
- Streamlit UI
- Comprehensive documentation

### Future Versions
- v1.1.0: Additional providers and models
- v1.2.0: Advanced evaluation metrics
- v1.3.0: Real-time collaboration features 