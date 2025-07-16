# LLM Model Configuration

This document outlines the current model configuration for all LLM providers in the project.

## Overview

The project currently supports three LLM providers with multiple models each:

- **Groq**: 3 models
- **Gemini**: 2 models  
- **OpenRouter**: 3 models

## Provider Details

### Groq Provider

**API Endpoint**: `https://api.groq.com/openai/v1/chat/completions`  
**Environment Variable**: `GROQ_API_KEY`

#### Models:
1. **llama3-8b-8192**
   - Description: Original Llama 3 8B model
   - Status: ✅ Working
   - Response Quality: Good

2. **llama-3.1-8b-instant**
   - Description: New Llama 3.1 8B Instant model
   - Status: ✅ Working
   - Response Quality: Good

3. **qwen-qwq-32b**
   - Description: New Qwen QWQ 32B model
   - Status: ✅ Working
   - Response Quality: Good (shows thinking process)

### Gemini Provider

**API Endpoint**: `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`  
**Environment Variable**: `GOOGLE_API_KEY`

#### Models:
1. **gemini-1.5-flash**
   - Description: Original Gemini 1.5 Flash model
   - Status: ✅ Working
   - Response Quality: Excellent

2. **gemma-3-12b-it**
   - Description: New Gemma 3 12B Instruction Tuned model
   - Status: ✅ Working
   - Response Quality: Good

### OpenRouter Provider

**API Endpoint**: `https://openrouter.ai/api/v1/chat/completions`  
**Environment Variable**: `OPENROUTER_API_KEY`

#### Models:
1. **mistralai/mistral-7b-instruct**
   - Description: Original Mistral 7B Instruct model
   - Status: ✅ Working
   - Response Quality: Good

2. **deepseek/deepseek-r1-0528-qwen3-8b:free**
   - Description: DeepSeek R1 0528 Qwen3 8B Free model
   - Status: ✅ Working
   - Response Quality: Minimal response (may need investigation)

3. **openrouter/cypher-alpha:free**
   - Description: OpenRouter Cypher Alpha Free model
   - Status: ✅ Working
   - Response Quality: Good

## Configuration Files

### Main Configuration
- **File**: `config/llm_config.yaml`
- **Purpose**: Central configuration for all LLM providers and models

### Test Scripts
- **Groq**: `src/llm_providers/test_groq.py`
- **Gemini**: `src/llm_providers/test_gemini.py`
- **OpenRouter**: `src/llm_providers/test_openrouter.py`
- **Comprehensive**: `scripts/test_all_models.py`

## Testing Results

Last comprehensive test run: All 7 models passed successfully.

### Test Summary:
- **Groq**: 3/3 models working
- **Gemini**: 2/2 models working  
- **OpenRouter**: 3/3 models working
- **Overall**: 8/8 models working

## Notes

1. **Gemma Model**: Initially tried `gemma2-9b-it` but it was not available. Switched to `gemma-3-12b-it` which is available through the Gemini API.

2. **DeepSeek Model**: The `deepseek/deepseek-r1-0528-qwen3-8b:free` model returns minimal responses. This may be normal behavior or could indicate a configuration issue.

3. **Model Availability**: All models are confirmed to be available and accessible with the current API keys.

4. **Qwen Model**: The `qwen-qwq-32b` model shows its thinking process in responses, which can be useful for understanding the model's reasoning.

## Next Steps

1. **Response Quality Analysis**: Conduct detailed analysis of response quality across all models
2. **Performance Benchmarking**: Measure latency and token usage for each model
3. **Business Context Testing**: Test models with industry-specific queries
4. **Integration Testing**: Integrate models into the RAG pipeline

## Environment Setup

Ensure the following environment variables are set in your `.env` file:

```bash
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
``` 