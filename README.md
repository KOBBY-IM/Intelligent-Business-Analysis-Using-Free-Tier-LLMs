# LLM Blind Evaluation System

A production-ready web application for conducting blind evaluations of free-tier Large Language Models (LLMs) across different business domains. This system enables researchers and organizations to compare the performance of Groq, Gemini, and OpenRouter models through user-friendly blind testing.

## Features

- **Blind Evaluation**: Users evaluate responses without knowing which model generated them
- **Multi-Provider Support**: Integrates Groq, Gemini (Google), and OpenRouter APIs
- **Business Domain Focus**: Evaluation questions tailored for retail and finance industries
- **RAG Integration**: Retrieval-Augmented Generation using uploaded CSV data
- **Admin Dashboard**: Comprehensive metrics and export functionality
- **Security-First**: Input validation, rate limiting, and secure credential management
- **Production Ready**: Structured logging, error handling, and deployment scripts

## Quick Start

### Prerequisites

- Python 3.9 or higher
- API keys for supported LLM providers
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Intelligent-Business-Analysis-Using-Free-Tier-LLMs
   ```

2. **Set up environment**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your API keys and configuration
   nano .env
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Deploy and start**
   ```bash
   python scripts/deploy.py
   ```

The application will be available at `http://localhost:8501` (or the next available port)

## Starting the Application

You have several options to start the application:

### Option 1: Python Launcher (Recommended)
```bash
python start_streamlit.py
```
- Automatically finds available ports (8501-8510)
- Kills existing Streamlit processes if needed
- Provides clear startup feedback

### Option 2: Shell Script Launcher
```bash
./launch.sh
```
- Bash script alternative
- Same port detection and cleanup features
- Works well for Unix/Linux systems

### Option 3: Process Manager
```bash
# Check status of Streamlit processes
python scripts/manage_streamlit.py status

# Stop all Streamlit processes
python scripts/manage_streamlit.py stop

# Start using the main launcher
python scripts/manage_streamlit.py start
```

### Option 4: Direct Streamlit Command
```bash
streamlit run src/ui/main.py --server.port 8501
```
- Manual port specification required
- No automatic conflict resolution

## Environment Configuration

Create a `.env` file in the project root with the following variables:

### Required API Keys
```bash
# LLM Provider API Keys
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Security Settings
ADMIN_PASSWORD=your_secure_admin_password_here
```

### Optional Configuration
```bash
# Application Settings
ALLOWED_HOSTS=localhost,127.0.0.1
UPLOAD_MAX_SIZE_MB=10
API_RATE_LIMIT_PER_MINUTE=60

# Logging Settings
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/app.log

# Streamlit Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## Getting API Keys

### Groq API Key
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to API Keys section
4. Generate a new API key

### Google Generative AI Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the generated key

### OpenRouter API Key
1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up for an account
3. Go to the API Keys section
4. Generate a new API key

## Application Structure

```
├── src/
│   ├── ui/                 # Streamlit user interface
│   │   ├── main.py        # Main application entry point
│   │   ├── pages/         # Additional pages
│   │   └── components/    # Reusable UI components
│   ├── llm_providers/     # LLM API integrations
│   ├── rag/              # Retrieval-Augmented Generation
│   ├── evaluation/       # Evaluation metrics and scoring
│   ├── security/         # Security utilities
│   └── utils/            # Helper utilities
├── config/               # Configuration files
├── data/                # Data storage (gitignored)
├── docs/                # Documentation
├── scripts/             # Deployment and utility scripts
└── tests/               # Test suite
```

## Usage

### For Researchers/Evaluators

1. **Start a New Evaluation**
   - Visit the main page
   - Enter your email and consent to participate
   - Complete the blind evaluation questions

2. **Evaluation Process**
   - View randomized responses from different models
   - Select the best response for each question
   - Optionally provide comments
   - Submit your evaluation

### For Administrators

1. **Access Admin Panel**
   - Navigate to Admin Login page
   - Enter admin password (set in .env)

2. **View Metrics Dashboard**
   - Monitor evaluation statistics
   - View model performance comparisons
   - Track user engagement metrics

3. **Export Results**
   - Download evaluation data as CSV
   - Generate comprehensive reports
   - Access raw feedback data

### RAG Testing (Admin Only)

1. **Upload CSV Data**
   - Use the Batch RAG Test page
   - Upload business data files
   - Enter custom business questions

2. **Generate Responses**
   - System uses RAG to generate contextual responses
   - Compare model performance on your data
   - Export results for analysis

## Security Features

- **Input Validation**: All user inputs are sanitized and validated
- **Rate Limiting**: Prevents API abuse and ensures fair usage
- **Secure Logging**: PII is automatically redacted from logs
- **Environment-based Configuration**: No hardcoded secrets
- **Admin Authentication**: Protected admin functions
- **File Upload Restrictions**: Limited file types and sizes

## API Reference

### Supported Models

**Groq:**
- `llama3-8b-8192`: Fast Llama 3 model
- `gemma-3-12b-it`: Instruction-tuned Gemma model

**Google Generative AI:**
- `gemini-1.5-flash`: Fast Gemini model
- `gemma-3-12b-it`: Google's Gemma model

**OpenRouter:**
- `mistralai/mistral-7b-instruct`: Mistral instruction model
- `deepseek/deepseek-r1-0528-qwen3-8b`: DeepSeek reasoning model

### Configuration Files

- `config/app_config.yaml`: Application settings
- `config/llm_config.yaml`: LLM provider configuration
- `config/evaluation_config.yaml`: Evaluation parameters
- `config/logging_config.yaml`: Logging configuration

## Deployment

### Development Deployment

```bash
python scripts/deploy.py --check-only
```

### Production Deployment

```bash
# Run all checks and start application
python scripts/deploy.py --host 0.0.0.0 --port 8501

# Run checks only (without starting)
python scripts/deploy.py --check-only
```

### Docker Deployment (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["python", "scripts/deploy.py"]
```

## Monitoring and Maintenance

### Logs
- Application logs: `logs/app.log`
- Evaluation data: `data/results/`
- User feedback: `data/results/user_feedback.json`

### Health Checks
```bash
# Run deployment checks
python scripts/deploy.py --check-only

# Check API connectivity
python -c "from llm_providers.provider_manager import ProviderManager; pm = ProviderManager(); print('API health check passed')"
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'xxx'**
```bash
pip install -r requirements.txt
```

**API Key Not Found Error**
- Ensure `.env` file exists and contains required API keys
- Check API key format and validity
- Verify environment variables are loaded

**Permission Denied Errors**
```bash
chmod +x scripts/deploy.py
```

**Streamlit Won't Start**
- Check if port 8501 is available
- Verify Python version (3.9+ required)
- Check firewall settings

### Support

For technical issues:
1. Check the logs in `logs/app.log`
2. Verify environment configuration
3. Run deployment checks: `python scripts/deploy.py --check-only`

## Research and Academic Use

This system is designed for academic research and evaluation purposes. When using this system for research:

1. **Citation**: Please cite this work in your publications
2. **Data Privacy**: User data is anonymized by default
3. **Ethical Use**: Follow your institution's guidelines for human subject research
4. **Reproducibility**: Export evaluation data for analysis and replication

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built for academic research in LLM evaluation
- Supports free-tier models to enable accessible research
- Designed with security and privacy best practices 