# Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the LLM Business Analysis System in various environments, from local development to production deployment.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Production Deployment](#production-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Monitoring and Maintenance](#monitoring-and-maintenance)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.9 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB free disk space
- **Network**: Internet connection for API access

### Required Accounts and API Keys

1. **Groq Account**
   - Sign up at [https://console.groq.com](https://console.groq.com)
   - Generate API key from dashboard
   - Free tier: 1000 requests/day

2. **Google Cloud Account**
   - Sign up at [https://console.cloud.google.com](https://console.cloud.google.com)
   - Enable Generative AI API
   - Generate API key
   - Free tier: 1500 requests/day

3. **OpenRouter Account**
   - Sign up at [https://openrouter.ai](https://openrouter.ai)
   - Generate API key
   - Free tier: 5000 requests/day

---

## Local Development Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/llm-business-analysis.git
cd llm-business-analysis
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Step 4: Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env
```

**Required Environment Variables:**
```bash
# API Keys
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Security
SECRET_KEY=your_secret_key_here
DATA_ENCRYPTION_KEY=your_encryption_key_here

# Application Settings
ALLOWED_HOSTS=localhost,127.0.0.1
UPLOAD_MAX_SIZE_MB=10
API_RATE_LIMIT_PER_MINUTE=60
USER_QUERY_LIMIT_PER_HOUR=100
```

### Step 5: Initialize Data Directories

```bash
# Create necessary directories
mkdir -p data/results
mkdir -p data/vector_store
mkdir -p logs
```

### Step 6: Run the Application

```bash
# Start the Streamlit application
python run_streamlit.py
```

The application will be available at `http://localhost:8501`

---

## Production Deployment

### Option 1: Docker Deployment

#### Step 1: Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start application
CMD ["streamlit", "run", "src/ui/main_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Step 2: Create Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  llm-analysis:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - SECRET_KEY=${SECRET_KEY}
      - DATA_ENCRYPTION_KEY=${DATA_ENCRYPTION_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

#### Step 3: Deploy with Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 2: Cloud Deployment (AWS)

#### Step 1: Prepare AWS Environment

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure
```

#### Step 2: Create EC2 Instance

```bash
# Create security group
aws ec2 create-security-group \
    --group-name llm-analysis-sg \
    --description "Security group for LLM Analysis System"

# Add rules
aws ec2 authorize-security-group-ingress \
    --group-name llm-analysis-sg \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-name llm-analysis-sg \
    --protocol tcp \
    --port 8501 \
    --cidr 0.0.0.0/0

# Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --count 1 \
    --instance-type t3.medium \
    --key-name your-key-pair \
    --security-groups llm-analysis-sg
```

#### Step 3: Deploy Application

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Clone repository
git clone https://github.com/your-username/llm-business-analysis.git
cd llm-business-analysis

# Setup application (follow local development steps 2-6)
```

### Option 3: Serverless Deployment (AWS Lambda + API Gateway)

#### Step 1: Create Lambda Function

```python
# lambda_function.py
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from llm_providers import ProviderManager
from evaluation import LLMEvaluator

def lambda_handler(event, context):
    """Lambda handler for LLM analysis requests"""
    
    try:
        # Parse request
        body = json.loads(event['body'])
        action = body.get('action')
        
        # Initialize components
        provider_manager = ProviderManager()
        evaluator = LLMEvaluator()
        
        if action == 'query':
            # Handle single query
            provider_name = body['provider']
            model = body['model']
            query = body['query']
            
            provider = provider_manager.get_provider(provider_name)
            response = provider.generate_response(query, model=model)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'success': response.success,
                    'text': response.text,
                    'latency_ms': response.latency_ms,
                    'tokens_used': response.tokens_used
                })
            }
        
        elif action == 'evaluate':
            # Handle evaluation
            # Implementation details...
            pass
        
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid action'})
            }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

#### Step 2: Deploy Lambda Function

```bash
# Create deployment package
pip install -r requirements.txt -t package/
cp -r src/ package/
cd package
zip -r ../lambda-deployment.zip .

# Deploy to AWS Lambda
aws lambda create-function \
    --function-name llm-analysis \
    --runtime python3.9 \
    --role arn:aws:iam::your-account:role/lambda-execution-role \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://lambda-deployment.zip
```

---

## Environment Configuration

### Development Environment

```bash
# .env.development
DEBUG=true
LOG_LEVEL=DEBUG
ALLOWED_HOSTS=localhost,127.0.0.1
API_RATE_LIMIT_PER_MINUTE=1000
USER_QUERY_LIMIT_PER_HOUR=10000
```

### Staging Environment

```bash
# .env.staging
DEBUG=false
LOG_LEVEL=INFO
ALLOWED_HOSTS=staging.yourdomain.com
API_RATE_LIMIT_PER_MINUTE=100
USER_QUERY_LIMIT_PER_HOUR=1000
```

### Production Environment

```bash
# .env.production
DEBUG=false
LOG_LEVEL=WARNING
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
API_RATE_LIMIT_PER_MINUTE=60
USER_QUERY_LIMIT_PER_HOUR=100
```

### Configuration Management

```python
# config/environment.py
import os
from typing import Dict, Any

class EnvironmentConfig:
    """Environment-specific configuration management"""
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        env = os.getenv('ENVIRONMENT', 'development')
        
        configs = {
            'development': {
                'debug': True,
                'log_level': 'DEBUG',
                'allowed_hosts': ['localhost', '127.0.0.1'],
                'rate_limit': 1000,
                'query_limit': 10000
            },
            'staging': {
                'debug': False,
                'log_level': 'INFO',
                'allowed_hosts': ['staging.yourdomain.com'],
                'rate_limit': 100,
                'query_limit': 1000
            },
            'production': {
                'debug': False,
                'log_level': 'WARNING',
                'allowed_hosts': ['yourdomain.com', 'www.yourdomain.com'],
                'rate_limit': 60,
                'query_limit': 100
            }
        }
        
        return configs.get(env, configs['development'])
```

---

## Monitoring and Maintenance

### Health Monitoring

#### Application Health Checks

```python
# health_check.py
import requests
import time
from typing import Dict, Any

def check_application_health() -> Dict[str, Any]:
    """Check application health status"""
    
    health_status = {
        'timestamp': time.time(),
        'status': 'healthy',
        'checks': {}
    }
    
    try:
        # Check Streamlit application
        response = requests.get('http://localhost:8501/_stcore/health', timeout=5)
        health_status['checks']['streamlit'] = {
            'status': 'healthy' if response.status_code == 200 else 'unhealthy',
            'response_time': response.elapsed.total_seconds()
        }
        
        # Check provider APIs
        from src.llm_providers import ProviderManager
        provider_manager = ProviderManager()
        
        for provider_name, provider in provider_manager.providers.items():
            try:
                healthy = provider.health_check()
                health_status['checks'][f'provider_{provider_name}'] = {
                    'status': 'healthy' if healthy else 'unhealthy'
                }
            except Exception as e:
                health_status['checks'][f'provider_{provider_name}'] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Overall status
        all_healthy = all(
            check['status'] == 'healthy' 
            for check in health_status['checks'].values()
        )
        health_status['status'] = 'healthy' if all_healthy else 'unhealthy'
        
    except Exception as e:
        health_status['status'] = 'error'
        health_status['error'] = str(e)
    
    return health_status
```

#### Monitoring Dashboard

```python
# monitoring_dashboard.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def show_monitoring_dashboard():
    """Display system monitoring dashboard"""
    
    st.header("📊 System Monitoring")
    
    # Health status
    health_status = check_application_health()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_color = "green" if health_status['status'] == 'healthy' else "red"
        st.metric(
            "System Status",
            health_status['status'].upper(),
            delta=None,
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Uptime",
            "99.9%",
            delta="+0.1%",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "Active Users",
            "12",
            delta="+3",
            delta_color="normal"
        )
    
    # Health checks details
    st.subheader("🔍 Health Checks")
    
    for check_name, check_status in health_status['checks'].items():
        status_icon = "✅" if check_status['status'] == 'healthy' else "❌"
        st.write(f"{status_icon} {check_name}: {check_status['status']}")
    
    # Performance metrics
    st.subheader("📈 Performance Metrics")
    
    # Mock performance data
    performance_data = {
        'Metric': ['Avg Response Time', 'Requests/min', 'Error Rate', 'Token Usage'],
        'Value': ['850ms', '45', '0.2%', '2.3M'],
        'Trend': ['↘️', '↗️', '↘️', '↗️']
    }
    
    df = pd.DataFrame(performance_data)
    st.dataframe(df, use_container_width=True)
```

### Logging and Alerting

#### Structured Logging

```python
# logging_config.py
import structlog
import logging
from typing import Dict, Any

def setup_logging(level: str = "INFO") -> None:
    """Setup structured logging configuration"""
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

def log_api_request(provider: str, model: str, response_time: float, 
                   success: bool, error: str = None) -> None:
    """Log API request details"""
    
    logger = structlog.get_logger()
    
    log_data = {
        "event": "api_request",
        "provider": provider,
        "model": model,
        "response_time_ms": response_time,
        "success": success
    }
    
    if error:
        log_data["error"] = error
    
    if success:
        logger.info(**log_data)
    else:
        logger.error(**log_data)
```

#### Alerting System

```python
# alerting.py
import smtplib
from email.mime.text import MIMEText
from typing import List, Dict, Any

class AlertManager:
    """Manage system alerts and notifications"""
    
    def __init__(self, smtp_server: str, smtp_port: int, 
                 username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
    
    def send_alert(self, subject: str, message: str, 
                   recipients: List[str]) -> bool:
        """Send alert email"""
        
        try:
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = self.username
            msg['To'] = ', '.join(recipients)
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False
    
    def check_and_alert(self, health_status: Dict[str, Any]) -> None:
        """Check health status and send alerts if needed"""
        
        if health_status['status'] != 'healthy':
            subject = "🚨 LLM Analysis System Alert"
            message = f"""
            System Status: {health_status['status']}
            Timestamp: {health_status['timestamp']}
            
            Failed Checks:
            """
            
            for check_name, check_status in health_status['checks'].items():
                if check_status['status'] != 'healthy':
                    message += f"- {check_name}: {check_status['status']}\n"
            
            self.send_alert(subject, message, ['admin@yourdomain.com'])
```

---

## Troubleshooting

### Common Issues

#### 1. API Key Issues

**Problem**: "API key not found" or "Invalid API key"

**Solution**:
```bash
# Check environment variables
echo $GROQ_API_KEY
echo $GOOGLE_API_KEY
echo $OPENROUTER_API_KEY

# Verify API keys are valid
curl -H "Authorization: Bearer $GROQ_API_KEY" \
     https://api.groq.com/openai/v1/models
```

#### 2. Port Already in Use

**Problem**: "Port 8501 is already in use"

**Solution**:
```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different port
streamlit run src/ui/main_app.py --server.port=8502
```

#### 3. Memory Issues

**Problem**: "Out of memory" errors

**Solution**:
```bash
# Check memory usage
free -h

# Increase swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. Dependency Issues

**Problem**: Import errors or missing packages

**Solution**:
```bash
# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Check Python version
python --version

# Verify virtual environment
which python
```

### Performance Optimization

#### 1. Response Time Optimization

```python
# Enable connection pooling
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)
```

#### 2. Memory Optimization

```python
# Use generators for large datasets
def process_large_dataset(data):
    for item in data:
        yield process_item(item)

# Clear cache periodically
import gc
gc.collect()
```

#### 3. Caching

```python
# Implement response caching
import hashlib
import json
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_api_call(query: str, provider: str, model: str):
    # Implementation
    pass
```

### Backup and Recovery

#### 1. Data Backup

```bash
#!/bin/bash
# backup.sh

# Create backup directory
BACKUP_DIR="/backup/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup data directory
cp -r data/ $BACKUP_DIR/

# Backup configuration
cp -r config/ $BACKUP_DIR/

# Backup logs
cp -r logs/ $BACKUP_DIR/

# Compress backup
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR

# Clean up
rm -rf $BACKUP_DIR

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

#### 2. Recovery Procedure

```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: ./restore.sh <backup_file>"
    exit 1
fi

# Stop application
docker-compose down

# Extract backup
tar -xzf $BACKUP_FILE

# Restore data
cp -r backup_*/data/ ./
cp -r backup_*/config/ ./
cp -r backup_*/logs/ ./

# Start application
docker-compose up -d

echo "Recovery completed"
```

---

## Security Considerations

### 1. API Key Security

- Store API keys in environment variables
- Use secret management services in production
- Rotate keys regularly
- Monitor API usage for anomalies

### 2. Network Security

- Use HTTPS for all external communications
- Implement proper firewall rules
- Use VPN for remote access
- Monitor network traffic

### 3. Application Security

- Validate all user inputs
- Implement rate limiting
- Use secure session management
- Regular security updates

### 4. Data Security

- Encrypt sensitive data at rest
- Implement proper access controls
- Regular security audits
- Compliance with data protection regulations

This deployment guide provides comprehensive instructions for deploying the LLM Business Analysis System in various environments while ensuring security, performance, and maintainability. 