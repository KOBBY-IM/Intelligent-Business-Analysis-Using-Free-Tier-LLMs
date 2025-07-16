# Unified LLM Provider Architecture

This document describes the refactored, unified architecture for LLM providers in the project.

## Overview

The code has been refactored to ensure uniformity across all LLM providers through a modular, object-oriented design. This provides:

- **Consistent Interface**: All providers implement the same base interface
- **Easy Extensibility**: Adding new providers is straightforward
- **Centralized Management**: Single point of control for all providers
- **Type Safety**: Full type annotations and validation
- **Error Handling**: Consistent error handling across all providers

## Architecture Components

### 1. Base Provider (`BaseLLMProvider`)

**File**: `src/llm_providers/base_provider.py`

The abstract base class that defines the interface for all LLM providers:

```python
class BaseLLMProvider(ABC):
    def __init__(self, provider_name: str, api_key_env: str)
    def generate_response(self, model_name: str, prompt: str, **kwargs) -> Tuple[bool, str, Optional[str]]
    def test_model(self, model_name: str, test_prompt: str) -> TestResult
    def test_all_models(self, test_prompt: str) -> Dict[str, TestResult]
```

**Key Features**:
- Abstract interface for all providers
- Common functionality (API key validation, testing, etc.)
- Consistent error handling and response format
- Latency measurement for performance tracking

### 2. Data Classes

**File**: `src/llm_providers/base_provider.py`

```python
@dataclass
class ModelConfig:
    name: str
    description: str
    max_tokens: int = 100
    temperature: float = 0.7

@dataclass
class TestResult:
    model_name: str
    success: bool
    response_text: str
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    latency_ms: Optional[float] = None
```

### 3. Provider Implementations

Each provider implements the `BaseLLMProvider` interface:

#### Groq Provider (`GroqProvider`)
**File**: `src/llm_providers/groq_provider.py`
- Models: llama3-8b-8192, llama-3.1-8b-instant, qwen-qwq-32b
- API: OpenAI-compatible format

#### Gemini Provider (`GeminiProvider`)
**File**: `src/llm_providers/gemini_provider.py`
- Models: gemini-1.5-flash, gemma-3-12b-it
- API: Google Generative AI format

#### OpenRouter Provider (`OpenRouterProvider`)
**File**: `src/llm_providers/openrouter_provider.py`
- Models: mistralai/mistral-7b-instruct, deepseek/deepseek-r1-0528-qwen3-8b:free, openrouter/cypher-alpha:free
- API: OpenAI-compatible format with custom headers

### 4. Provider Manager (`ProviderManager`)

**File**: `src/llm_providers/provider_manager.py`

Central manager for all providers:

```python
class ProviderManager:
    def __init__(self)
    def get_provider(self, provider_name: str) -> Optional[BaseLLMProvider]
    def test_all_providers(self, test_prompt: str) -> Dict[str, Dict[str, TestResult]]
    def generate_response(self, provider_name: str, model_name: str, prompt: str, **kwargs) -> Tuple[bool, str, Optional[str]]
    def add_custom_model(self, provider_name: str, model_name: str, description: str, **kwargs)
```

## Usage Examples

### Basic Usage

```python
from llm_providers import ProviderManager

# Initialize manager
manager = ProviderManager()

# Generate response
success, response, error = manager.generate_response(
    "groq", "llama3-8b-8192", "Hello, how are you?"
)

if success:
    print(f"Response: {response}")
else:
    print(f"Error: {error}")
```

### Testing All Providers

```python
# Test all providers
results = manager.test_all_providers("Hello, how are you?")

# Print summary
manager.print_comprehensive_summary(results)
```

### Adding Custom Models

```python
# Add custom model to a provider
manager.add_custom_model(
    provider_name="groq",
    model_name="my-custom-model",
    description="A custom model for testing",
    max_tokens=150,
    temperature=0.5
)
```

## Benefits of the New Architecture

### 1. **Consistency**
- All providers use the same interface
- Consistent error handling and response formats
- Uniform testing and validation

### 2. **Maintainability**
- Single base class to maintain
- Clear separation of concerns
- Easy to add new providers

### 3. **Extensibility**
- Simple to add new models to existing providers
- Easy to add new providers
- Custom model support

### 4. **Type Safety**
- Full type annotations
- Data classes for structured data
- IDE support and error detection

### 5. **Performance Monitoring**
- Built-in latency measurement
- Consistent performance tracking
- Easy to add metrics

## Migration from Old System

The old individual test files have been replaced with:

- **Old**: `src/llm_providers/test_groq.py`
- **New**: `src/llm_providers/groq_provider.py` + `ProviderManager`

### Key Changes:
1. **Unified Interface**: All providers now use the same methods
2. **Centralized Testing**: Single manager handles all testing
3. **Better Error Handling**: Consistent error messages and formats
4. **Performance Tracking**: Built-in latency measurement
5. **Type Safety**: Full type annotations throughout

## Testing

### Individual Provider Testing
```bash
python scripts/test_unified_providers.py
```

### Example Usage
```bash
python scripts/example_usage.py
```

### Legacy Testing (Still Available)
```bash
python src/llm_providers/test_groq.py
python src/llm_providers/test_gemini.py
python src/llm_providers/test_openrouter.py
```

## Configuration

Models are configured in each provider's `_setup_default_models()` method:

```python
def _setup_default_models(self):
    models = [
        ModelConfig(
            name="model-name",
            description="Model description",
            max_tokens=100,
            temperature=0.7
        ),
        # ... more models
    ]
    
    for model in models:
        self.add_model(model)
```

## Future Enhancements

1. **Configuration Files**: Move model configurations to YAML files
2. **Caching**: Add response caching for performance
3. **Rate Limiting**: Implement rate limiting per provider
4. **Metrics**: Add comprehensive metrics collection
5. **Async Support**: Add async/await support for better performance
6. **Streaming**: Add support for streaming responses

## File Structure

```
src/llm_providers/
├── __init__.py              # Module exports
├── base_provider.py         # Base class and data structures
├── groq_provider.py         # Groq implementation
├── gemini_provider.py       # Gemini implementation
├── openrouter_provider.py   # OpenRouter implementation
├── provider_manager.py      # Central manager
└── test_*.py               # Legacy test files (deprecated)

scripts/
├── test_unified_providers.py  # New unified testing
└── example_usage.py           # Usage examples
``` 