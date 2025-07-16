# System Architecture Documentation

## Overview

The LLM Business Analysis System is a comprehensive evaluation framework designed to compare free-tier LLM providers across multiple business domains. The system provides a unified interface for testing, evaluating, and comparing LLM responses with a focus on reproducibility and user-centered design.

## Architecture Principles

### 1. **Modular Design**
- Each component is self-contained with clear interfaces
- Loose coupling between modules
- Easy to extend and maintain

### 2. **Security First**
- All credentials managed via environment variables
- Input validation and sanitization
- Rate limiting and quota management

### 3. **Configuration-Driven**
- External configuration for all tunable parameters
- Environment-specific configurations
- Runtime configuration validation

### 4. **Research-Grade Quality**
- Reproducible evaluation metrics
- Statistical validation
- Comprehensive logging and monitoring

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI Layer                       │
├─────────────────────────────────────────────────────────────────┤
│  Dashboard  │  Single Query  │  Comparison  │  Evaluation  │ RAG │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  Provider Manager  │  Evaluation Engine  │  RAG Pipeline  │     │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      Service Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  LLM Providers  │  Vector Store  │  Data Processing  │ Security │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      Infrastructure Layer                       │
├─────────────────────────────────────────────────────────────────┤
│  Configuration  │  Logging  │  Storage  │  External APIs  │     │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. **UI Layer (Streamlit)**

#### Structure
```
src/ui/
├── main_app.py              # Main application entry point
├── pages/                   # Page components
│   ├── single_query.py      # Single query testing
│   ├── comparison.py        # Side-by-side comparison
│   ├── evaluation.py        # Automated evaluation
│   └── rag_demo.py          # RAG pipeline demo
├── components/              # Reusable UI components
│   ├── sidebar.py           # Navigation sidebar
│   └── formatting.py        # UI formatting utilities
└── utils/                   # UI utilities
    └── formatting.py        # Response formatting
```

#### Key Features
- **Multi-page Navigation**: Dashboard, Single Query, Comparison, Evaluation, RAG Demo
- **Dynamic Provider Loading**: Automatically loads available providers and models
- **Real-time Metrics**: Displays response times, token usage, and evaluation scores
- **Export Functionality**: CSV/JSON export of results
- **Responsive Design**: Works on desktop and mobile devices

### 2. **Application Layer**

#### Provider Manager
```python
class ProviderManager:
    """Centralized provider management and coordination"""
    
    def __init__(self):
        self.providers = {
            'groq': GroqProvider(),
            'gemini': GeminiProvider(),
            'openrouter': OpenRouterProvider()
        }
    
    def get_provider(self, name: str) -> BaseProvider
    def health_check_all(self) -> Dict[str, bool]
    def get_provider_names(self) -> List[str]
```

#### Evaluation Engine
```python
class LLMEvaluator:
    """Automated evaluation and scoring system"""
    
    def __init__(self):
        self.ground_truth = GroundTruthManager()
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_single_response(self, ...) -> EvaluationResult
    def run_batch_evaluation(self, ...) -> BatchEvaluationResult
    def calculate_statistics(self, results: List[EvaluationResult]) -> Dict
```

#### RAG Pipeline
```python
class RAGPipeline:
    """Retrieval-Augmented Generation pipeline"""
    
    def __init__(self, vector_store: VectorStore, llm_provider: BaseProvider):
        self.vector_store = vector_store
        self.llm_provider = llm_provider
    
    def add_documents(self, documents: List[str])
    def query(self, question: str) -> RAGResponse
```

### 3. **Service Layer**

#### LLM Providers
```python
class BaseProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    @abstractmethod
    def generate_response(self, query: str, **kwargs) -> LLMResponse
    
    @abstractmethod
    def health_check(self) -> bool
    
    @abstractmethod
    def list_models(self) -> List[str]
```

**Implemented Providers:**
- **GroqProvider**: Fast inference with Llama models
- **GeminiProvider**: Google's Gemini and Gemma models
- **OpenRouterProvider**: Access to multiple open-source models

#### Vector Store
```python
class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for document retrieval"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
    
    def add_documents(self, documents: List[str])
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]
```

#### Data Processing
```python
class DataCleaner:
    """Data cleaning and preprocessing utilities"""
    
    def clean_text(self, text: str) -> str
    def validate_data(self, data: pd.DataFrame) -> ValidationResult

class RetailDataProcessor:
    """Domain-specific data processing for retail"""
    
    def process_sales_data(self, data: pd.DataFrame) -> pd.DataFrame
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame
```

### 4. **Infrastructure Layer**

#### Configuration Management
```python
class ConfigurationManager:
    """Centralized configuration management"""
    
    def load_config(self, config_name: str) -> Dict[str, Any]
    def validate_config(self, config: Dict[str, Any]) -> bool
    def get_default_config(self, config_type: str) -> Dict[str, Any]
```

**Configuration Files:**
- `app_config.yaml`: Application settings
- `llm_config.yaml`: LLM provider configurations
- `evaluation_config.yaml`: Evaluation parameters
- `logging_config.yaml`: Logging configuration

#### Security Layer
```python
class SecurityManager:
    """Security and access control"""
    
    def validate_api_key(self, provider: str, api_key: str) -> bool
    def check_rate_limit(self, user_id: str) -> bool
    def sanitize_input(self, text: str) -> str
```

## Data Flow Architecture

### 1. **Single Query Flow**
```
User Input → Input Validation → Provider Selection → 
LLM API Call → Response Processing → Metrics Calculation → 
UI Display → Result Storage
```

### 2. **Comparison Flow**
```
User Query → Multi-Provider Selection → Parallel API Calls → 
Response Collection → Evaluation → Statistical Analysis → 
Side-by-Side Display → Export Options
```

### 3. **Evaluation Flow**
```
Configuration → Question Selection → Batch Processing → 
Response Generation → Automated Scoring → Statistical Analysis → 
Results Aggregation → Report Generation
```

### 4. **RAG Flow**
```
Document Upload → Text Processing → Embedding Generation → 
Vector Storage → Query Processing → Context Retrieval → 
LLM Generation → Response Synthesis
```

## Security Architecture

### 1. **Authentication & Authorization**
- API key validation for each provider
- Rate limiting per user and provider
- Input sanitization and validation

### 2. **Data Protection**
- No sensitive data in logs
- Encrypted storage for API keys
- Secure file upload validation

### 3. **Network Security**
- HTTPS for all external API calls
- Request timeout management
- Error handling without information leakage

## Performance Architecture

### 1. **Response Time Optimization**
- Async I/O for API calls
- Connection pooling
- Caching of embeddings and results

### 2. **Scalability Considerations**
- Stateless design for horizontal scaling
- Efficient vector search with FAISS
- Lazy loading of heavy components

### 3. **Resource Management**
- Memory-efficient document processing
- Automatic cleanup of temporary files
- Configurable timeouts and limits

## Monitoring & Logging

### 1. **Structured Logging**
```python
import structlog

logger = structlog.get_logger()
logger.info("api_request", 
           provider="groq", 
           model="llama3-8b-8192", 
           response_time_ms=850)
```

### 2. **Metrics Collection**
- Response times per provider/model
- Token usage statistics
- Error rates and types
- User interaction patterns

### 3. **Health Monitoring**
- Provider availability checks
- API quota monitoring
- System resource usage

## Deployment Architecture

### 1. **Development Environment**
```
Local Development:
├── Python 3.9+ virtual environment
├── Streamlit development server
├── Local file storage
└── Environment variables for API keys
```

### 2. **Production Environment**
```
Production Deployment:
├── Docker containerization
├── Environment-specific configurations
├── Persistent storage for data
├── Monitoring and alerting
└── Backup and recovery procedures
```

### 3. **Configuration Management**
```
Configuration Hierarchy:
├── Default configurations (code)
├── Environment-specific overrides
├── User-specific settings
└── Runtime configuration validation
```

## Error Handling Architecture

### 1. **Exception Hierarchy**
```python
class LLMTimeoutError(Exception): pass
class RateLimitException(Exception): pass
class DataValidationError(Exception): pass
class ConfigurationError(Exception): pass
```

### 2. **Error Recovery**
- Automatic retry with exponential backoff
- Fallback to alternative providers
- Graceful degradation of features
- User-friendly error messages

### 3. **Error Reporting**
- Structured error logging
- Error categorization and analysis
- Performance impact tracking

## Testing Architecture

### 1. **Unit Testing**
- Individual component testing
- Mock external dependencies
- Automated test execution

### 2. **Integration Testing**
- End-to-end workflow testing
- Provider integration testing
- UI component testing

### 3. **Performance Testing**
- Load testing with multiple concurrent users
- Response time benchmarking
- Resource usage monitoring

## Future Architecture Considerations

### 1. **Scalability Enhancements**
- Microservices architecture
- Message queue for async processing
- Distributed vector storage

### 2. **Advanced Features**
- Real-time collaboration
- Advanced analytics dashboard
- Custom model fine-tuning

### 3. **Integration Capabilities**
- REST API for external integrations
- Webhook support for notifications
- Third-party tool integrations

## Technology Stack

### **Backend Technologies**
- **Python 3.9+**: Core application language
- **Streamlit**: Web application framework
- **FAISS**: Vector similarity search
- **SentenceTransformers**: Text embeddings
- **Pandas**: Data processing and analysis
- **Pydantic**: Data validation and settings

### **External APIs**
- **Groq API**: Fast LLM inference
- **Google Generative AI**: Gemini and Gemma models
- **OpenRouter API**: Multiple open-source models

### **Development Tools**
- **pytest**: Testing framework
- **structlog**: Structured logging
- **python-dotenv**: Environment management
- **PyYAML**: Configuration management

### **Data Storage**
- **FAISS**: Vector database
- **CSV/JSON**: Results and configuration storage
- **File system**: Document storage

This architecture provides a solid foundation for the LLM Business Analysis System, ensuring scalability, maintainability, and extensibility for future enhancements. 