# Comprehensive Testing Guide

## Overview

This document provides a comprehensive guide to testing the LLM Evaluation System. The testing framework includes unit tests, integration tests, performance tests, and security tests to ensure system reliability, performance, and security.

## Test Structure

### Test Categories

1. **Unit Tests** (`tests/test_*.py`)
   - Individual component testing
   - Mock external dependencies
   - Fast execution
   - High coverage

2. **Integration Tests** (`tests/test_integration.py`)
   - End-to-end workflow testing
   - Component interaction testing
   - Real system behavior validation

3. **Performance Tests** (`tests/test_performance.py`)
   - Response time testing
   - Memory usage monitoring
   - Scalability validation
   - Load testing

4. **Security Tests** (`tests/test_security.py`)
   - Input validation testing
   - Vulnerability scanning
   - Data protection validation
   - Access control testing

### Test Configuration

The test suite uses `pytest` with the following configuration:

```yaml
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    security: Security tests
    slow: Slow running tests
```

## Running Tests

### Quick Start

```bash
# Install dependencies
make install-dev

# Run all tests
make test

# Run comprehensive test suite
make test-all
```

### Individual Test Categories

```bash
# Unit tests only
make test-unit

# Integration tests only
make test-integration

# Performance tests only
make test-performance

# Security tests only
make test-security
```

### Advanced Testing

```bash
# Run with coverage
make coverage

# Generate detailed report
make report

# Run with specific markers
pytest -m "unit and not slow"

# Run specific test file
pytest tests/test_llm_providers.py -v

# Run with parallel execution
pytest -n auto tests/
```

## Test Fixtures and Utilities

### Common Fixtures

The test suite provides several fixtures in `tests/conftest.py`:

```python
@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "llm_providers": {...},
        "evaluation": {...},
        "rag": {...}
    }

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    return {
        "GROQ_API_KEY": "test_key",
        "GOOGLE_API_KEY": "test_key",
        # ...
    }

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    # Automatically cleaned up after tests
```

### Test Utilities

```python
class TestUtils:
    @staticmethod
    def assert_response_structure(response: Dict[str, Any]) -> None:
        """Assert that response has the expected structure."""
        
    @staticmethod
    def assert_metrics_structure(metrics: Dict[str, Any]) -> None:
        """Assert that metrics have the expected structure."""
```

## Writing Tests

### Unit Test Example

```python
class TestGroqProvider:
    """Test the Groq LLM provider."""
    
    @pytest.fixture
    def groq_provider(self, mock_env_vars):
        """Create a GroqProvider instance for testing."""
        return GroqProvider()
    
    def test_groq_provider_initialization(self, groq_provider):
        """Test GroqProvider initialization."""
        assert groq_provider is not None
        assert hasattr(groq_provider, 'api_key')
        assert hasattr(groq_provider, 'model')
    
    @pytest.mark.asyncio
    async def test_groq_generate_response_success(self, groq_provider, mock_llm_response):
        """Test successful response generation from Groq."""
        with patch('llm_providers.groq_provider.groq.Groq') as mock_groq:
            # Setup mock
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content=mock_llm_response["response"]))],
                usage=Mock(total_tokens=mock_llm_response["tokens_used"])
            )
            mock_groq.return_value = mock_client
            
            # Test
            response = await groq_provider.generate_response(
                query="Test query",
                context="Test context"
            )
            
            # Assertions
            assert response["response"] == mock_llm_response["response"]
            assert response["tokens_used"] == mock_llm_response["tokens_used"]
            assert "latency" in response
```

### Integration Test Example

```python
@pytest.mark.asyncio
async def test_complete_evaluation_workflow(self, full_system_setup, sample_queries, sample_contexts):
    """Test the complete evaluation workflow from query to results."""
    provider_manager = full_system_setup["provider_manager"]
    evaluator = full_system_setup["evaluator"]
    
    # Mock all LLM providers
    mock_response = {
        "response": "This is a comprehensive response about the query.",
        "tokens_used": 150,
        "latency": 2.5
    }
    
    with patch.object(provider_manager.providers["groq"], 'generate_response', return_value=mock_response):
        # 1. Generate responses from all providers
        responses = await provider_manager.generate_response_from_all_providers(query, context)
        
        # 2. Evaluate responses
        evaluation_results = await evaluator.evaluate_multiple_responses(
            query=query,
            context=context,
            responses=list(responses.values())
        )
        
        # 3. Verify results
        assert isinstance(evaluation_results, list)
        assert len(evaluation_results) == len(responses)
```

## Performance Testing

### Response Time Testing

```python
def test_provider_response_time(self, mock_env_vars, performance_metrics):
    """Test response time performance of providers."""
    import time
    
    providers = [GroqProvider(), GeminiProvider(), OpenRouterProvider()]
    
    for provider in providers:
        with patch.object(provider, 'generate_response') as mock_generate:
            mock_generate.return_value = {
                "response": "Test response",
                "tokens_used": 100,
                "latency": 1.5
            }
            
            start_time = time.time()
            response = await provider.generate_response("Test query", "Test context")
            end_time = time.time()
            
            actual_latency = end_time - start_time
            performance_metrics["response_times"].append(actual_latency)
            
            # Response should be fast (under 1 second for mocked calls)
            assert actual_latency < 1.0
```

### Memory Usage Testing

```python
def test_provider_memory_usage(self, mock_env_vars):
    """Test memory usage of providers."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Create multiple provider instances
    providers = []
    for _ in range(10):
        providers.extend([
            GroqProvider(),
            GeminiProvider(),
            OpenRouterProvider()
        ])
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (less than 100MB)
    assert memory_increase < 100 * 1024 * 1024  # 100MB in bytes
```

## Security Testing

### Input Validation Testing

```python
def test_provider_input_sanitization(self, mock_env_vars, malicious_inputs):
    """Test that providers sanitize malicious inputs."""
    providers = [GroqProvider(), GeminiProvider(), OpenRouterProvider()]
    
    for provider in providers:
        for malicious_input in malicious_inputs:
            # Providers should handle malicious inputs gracefully
            assert hasattr(provider, 'sanitize_input')
            sanitized = provider.sanitize_input(malicious_input)
            assert sanitized != malicious_input  # Input should be sanitized
```

### API Key Protection Testing

```python
def test_provider_api_key_protection(self, mock_env_vars):
    """Test that API keys are properly protected."""
    providers = [GroqProvider(), GeminiProvider(), OpenRouterProvider()]
    
    for provider in providers:
        # API keys should not be exposed in string representations
        provider_str = str(provider)
        assert "test_groq_key" not in provider_str
        assert "test_google_key" not in provider_str
        assert "test_openrouter_key" not in provider_str
```

## Test Coverage

### Coverage Requirements

- **Unit Tests**: >90% line coverage
- **Integration Tests**: >80% integration coverage
- **Performance Tests**: All critical paths covered
- **Security Tests**: All security-critical functions covered

### Coverage Reports

```bash
# Generate coverage report
make coverage

# View coverage in browser
open data/reports/coverage/index.html
```

### Coverage Configuration

```ini
# .coveragerc
[run]
source = src
omit = 
    */tests/*
    */__pycache__/*
    */venv/*
    */.venv/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        make install-dev
    
    - name: Run tests
      run: |
        make test-all
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Test Data Management

### Test Data Fixtures

```python
@pytest.fixture
def sample_queries():
    """Sample business queries for testing."""
    return [
        "What are the key performance indicators for retail sales?",
        "How can we improve customer retention in the finance sector?",
        "What are the compliance requirements for healthcare data?",
        # ...
    ]

@pytest.fixture
def sample_documents():
    """Sample documents for RAG testing."""
    return [
        "Retail KPIs include sales per square foot, customer lifetime value, and inventory turnover.",
        "Financial risk management involves credit risk, market risk, and operational risk assessment.",
        # ...
    ]
```

### Test Data Cleanup

```python
@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)  # Cleanup after test
```

## Debugging Tests

### Common Issues

1. **Import Errors**
   ```bash
   # Add src to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Mock Issues**
   ```python
   # Use proper mock paths
   with patch('module.path.to.function') as mock_func:
       # Mock setup
   ```

3. **Async Test Issues**
   ```python
   @pytest.mark.asyncio
   async def test_async_function():
       result = await async_function()
       assert result is not None
   ```

### Debug Commands

```bash
# Run with debug output
pytest -v -s tests/

# Run single test with debug
pytest -v -s tests/test_file.py::test_function

# Run with pdb
pytest --pdb tests/

# Run with coverage and debug
pytest -v -s --cov=src --cov-report=term-missing tests/
```

## Test Maintenance

### Regular Tasks

1. **Update Test Data**
   - Review and update sample data quarterly
   - Ensure test data reflects current business requirements

2. **Review Test Coverage**
   - Monitor coverage reports monthly
   - Add tests for new features

3. **Performance Monitoring**
   - Track test execution times
   - Optimize slow tests

4. **Security Updates**
   - Update security test cases
   - Review for new vulnerabilities

### Test Documentation

- Keep test documentation updated
- Document test data sources
- Maintain test environment setup guides

## Best Practices

### Test Writing

1. **Descriptive Names**
   ```python
   def test_groq_provider_handles_api_timeout_gracefully():
       # Test implementation
   ```

2. **Arrange-Act-Assert Pattern**
   ```python
   def test_function():
       # Arrange
       input_data = prepare_test_data()
       
       # Act
       result = function_under_test(input_data)
       
       # Assert
       assert result == expected_output
   ```

3. **Test Isolation**
   ```python
   @pytest.fixture(autouse=True)
   def setup_test_environment():
       # Setup
       yield
       # Cleanup
   ```

### Test Organization

1. **Group Related Tests**
   ```python
   class TestGroqProvider:
       def test_initialization(self): ...
       def test_response_generation(self): ...
       def test_error_handling(self): ...
   ```

2. **Use Appropriate Markers**
   ```python
   @pytest.mark.unit
   @pytest.mark.slow
   def test_complex_calculation(): ...
   ```

3. **Maintain Test Data**
   - Keep test data in fixtures
   - Use realistic test scenarios
   - Document data sources

### Performance Considerations

1. **Fast Unit Tests**
   - Mock external dependencies
   - Use in-memory databases
   - Avoid file I/O

2. **Efficient Integration Tests**
   - Reuse test data
   - Parallel execution where possible
   - Clean up resources

3. **Realistic Performance Tests**
   - Use production-like data volumes
   - Test under realistic load
   - Monitor resource usage

## Troubleshooting

### Common Problems

1. **Test Failures**
   ```bash
   # Run with verbose output
   pytest -v tests/
   
   # Check for specific errors
   pytest --tb=long tests/
   ```

2. **Coverage Issues**
   ```bash
   # Check coverage configuration
   coverage run --source=src -m pytest tests/
   coverage report --show-missing
   ```

3. **Performance Issues**
   ```bash
   # Profile test execution
   pytest --durations=10 tests/
   ```

### Getting Help

1. Check the test logs in `data/test_results/`
2. Review the test report in `data/reports/`
3. Consult the project documentation
4. Check GitHub Issues for known problems

## Conclusion

This comprehensive testing framework ensures the LLM Evaluation System is reliable, performant, and secure. Regular testing and maintenance will help maintain high code quality and system stability.

For questions or issues, please refer to the project documentation or create an issue in the project repository. 