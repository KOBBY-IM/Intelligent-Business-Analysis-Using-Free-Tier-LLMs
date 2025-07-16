# Streamlit Cloud Environment Setup

To deploy this application on Streamlit Cloud, you need to set up the following environment variables in the Streamlit Cloud secrets management:

## Required API Keys

Go to your Streamlit Cloud app settings and add these to the "Secrets" section:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GOOGLE_API_KEY = "your_google_api_key_here"  
OPENROUTER_API_KEY = "your_openrouter_api_key_here"

# Security Settings
SECRET_KEY = "your_secret_key_for_session_management"
DATA_ENCRYPTION_KEY = "your_encryption_key_for_data_protection"
ALLOWED_HOSTS = "localhost,127.0.0.1"
UPLOAD_MAX_SIZE_MB = "10"
API_RATE_LIMIT_PER_MINUTE = "60"
USER_QUERY_LIMIT_PER_HOUR = "100"

# Configuration
ENVIRONMENT = "production"
LOG_LEVEL = "INFO"
```

## Getting API Keys

1. **Groq API Key**: Sign up at https://console.groq.com/ and get your API key
2. **Google API Key**: Create a project in Google Cloud Console and enable the Generative AI API
3. **OpenRouter API Key**: Sign up at https://openrouter.ai/ and get your API key

## Note

- Make sure to keep your API keys secure and never commit them to the repository
- The application will work with console logging on Streamlit Cloud due to read-only file system
- User feedback will be stored in session state rather than files on Streamlit Cloud 