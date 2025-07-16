# Secrets Setup Guide

This guide explains how to set up API keys and secrets for both Streamlit Cloud deployment and local development.

## 📁 TOML Files Overview

We've created three template files for different use cases:

1. **`streamlit_secrets_template.toml`** - For Streamlit Cloud deployment
2. **`local_secrets_template.toml`** - For local development 
3. **`.streamlit/secrets.toml.example`** - Example file in the standard Streamlit location

## 🚀 Streamlit Cloud Setup

### Step 1: Copy Template Content
Open `streamlit_secrets_template.toml` and copy all the content.

### Step 2: Add to Streamlit Cloud
1. Go to your Streamlit Cloud app dashboard
2. Click on your app → **Settings** → **Secrets**
3. Paste the template content
4. Replace all placeholder values with your actual API keys

### Step 3: Get Your API Keys
- **Groq**: https://console.groq.com/ → API Keys
- **Google**: Google Cloud Console → Enable Generative AI API → Credentials
- **OpenRouter**: https://openrouter.ai/ → Account Settings → API Keys

### Step 4: Generate Security Keys
For `SECRET_KEY` and `DATA_ENCRYPTION_KEY`, use random strings:
```python
import secrets
print(secrets.token_urlsafe(32))  # Run this twice for two different keys
```

## 💻 Local Development Setup

### Step 1: Create Local Secrets File
```bash
# Make sure .streamlit directory exists
mkdir -p .streamlit

# Copy the local template
cp local_secrets_template.toml .streamlit/secrets.toml
```

### Step 2: Add Your API Keys
Edit `.streamlit/secrets.toml` and replace placeholder values with your actual keys.

### Step 3: Verify .gitignore
Make sure your `.gitignore` includes:
```
.streamlit/secrets.toml
*.toml
!*.toml.example
```

## 🔧 Configuration Options

### API Keys (Required)
```toml
GROQ_API_KEY = "gsk_..."
GOOGLE_API_KEY = "AIza..."
OPENROUTER_API_KEY = "sk-or-v1-..."
```

### Security Settings
```toml
SECRET_KEY = "random_string_for_session_security"
DATA_ENCRYPTION_KEY = "random_string_for_data_encryption"
ALLOWED_HOSTS = "localhost,127.0.0.1"
UPLOAD_MAX_SIZE_MB = "10"
API_RATE_LIMIT_PER_MINUTE = "60"
USER_QUERY_LIMIT_PER_HOUR = "100"
```

### Environment Configuration
```toml
ENVIRONMENT = "production"  # or "development"
LOG_LEVEL = "INFO"          # or "DEBUG"
DEBUG_MODE = "false"        # or "true"
```

### Research Settings
```toml
STUDY_NAME = "LLM Business Intelligence Comparison"
CONSENT_VERSION = "1.0"
IRB_APPROVAL = "true"
```

## 🔒 Security Best Practices

1. **Never commit actual secrets** to git
2. **Use different keys** for development and production
3. **Rotate API keys** regularly
4. **Monitor API usage** to detect unauthorized access
5. **Use environment-specific settings**

## 🧪 Testing Your Setup

### Local Testing
```bash
streamlit run streamlit_app.py
```

### Verify API Keys Work
The app will show warnings if any API keys are missing or invalid.

## 🆘 Troubleshooting

### Common Issues
1. **Missing quotes**: All TOML values must be in quotes
2. **Invalid API keys**: Check key format and permissions
3. **Rate limits**: Adjust `API_RATE_LIMIT_PER_MINUTE` if needed
4. **File permissions**: Ensure `.streamlit/secrets.toml` is readable

### Error Messages
- `"API key not found"` → Check key names match exactly
- `"Invalid API key format"` → Verify key prefix (gsk_, AIza, sk-or-v1-)
- `"Rate limit exceeded"` → Increase rate limit or wait

## 📝 Template Files Usage

| File | Purpose | Location |
|------|---------|----------|
| `streamlit_secrets_template.toml` | Copy to Streamlit Cloud | Cloud secrets management |
| `local_secrets_template.toml` | Copy to `.streamlit/secrets.toml` | Local development |
| `.streamlit/secrets.toml.example` | Reference example | Version control safe | 