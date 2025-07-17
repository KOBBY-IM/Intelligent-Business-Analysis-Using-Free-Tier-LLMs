# 🔧 Local Development Guide

## 🎯 Overview

This guide sets up a local development environment that **exactly mirrors Streamlit Cloud** functionality, ensuring we never lose work and can develop safely offline.

## 🚀 Quick Start

### Option 1: Simple Start (Recommended)
```bash
# Start local development server
python run_local.py
```

### Option 2: Direct Streamlit
```bash
# Run Streamlit directly
streamlit run src/ui/main.py --server.port=8501
```

## 📋 Pre-Requirements Check

Ensure these files exist (the script will check automatically):

```
✅ Required Files:
   📄 data/enhanced_blind_responses.json (888KB)
   📄 data/ground_truth_answers.json (25KB)
   📄 data/evaluation_questions.yaml (8KB)
   📄 data/shopping_trends.csv (retail data)
   📄 data/Tesla_stock_data.csv (finance data)
   📄 src/ui/main.py (main application)
```

## 🌐 Local Features (Same as Streamlit Cloud)

### ✅ **Enhanced RAG System**
- **8-chunk coverage** (vs 5) = 60% more data
- **Dynamic ground truth guidance**
- **Cross-chunk pattern analysis**
- **Enhanced statistical analysis**

### ✅ **LLM Providers**
- **Groq**: llama3-8b-8192
- **Gemini**: gemma-3-12b-it  
- **OpenRouter**: mistralai/mistral-7b-instruct + deepseek models

### ✅ **UI Features**
- **Side-by-side response comparison**
- **Dynamic ranking interface** (handles 3 or 4 responses)
- **Ground truth business answers** (not RAG chunks)
- **User registration and feedback**
- **Progress tracking**
- **Secure logging**

### ✅ **Data Features**
- **Enhanced blind responses** (80 total responses)
- **Expert ground truth answers**
- **Business scenario context**
- **Real-time evaluation tracking**

## 🛠️ Development Workflow

### 1. File Synchronization Strategy

To avoid losing work, use this approach:

```bash
# Work locally, commit frequently
git add -A
git commit -m "feat: describe your changes"
git push origin master

# Streamlit Cloud auto-deploys from master branch
```

### 2. Testing Before Deployment

```bash
# Test locally first
python run_local.py

# Verify all features work:
# ✅ User registration
# ✅ Response display (3 responses side-by-side)
# ✅ Ground truth shows business answers
# ✅ Ranking interface works
# ✅ Progress tracking
# ✅ Data persistence

# Then deploy
git push origin master
```

### 3. Safe Development Process

1. **Always develop locally first**
2. **Test thoroughly on localhost:8501**
3. **Commit changes incrementally**
4. **Push to trigger Streamlit Cloud deployment**
5. **Monitor Streamlit Cloud for successful deployment**

## 📁 Local Directory Structure

```
Project Root/
├── run_local.py              # 🚀 Quick start script
├── src/ui/main.py            # 🎯 Main application
├── data/                     # 📊 All data files
│   ├── enhanced_blind_responses.json
│   ├── ground_truth_answers.json
│   ├── evaluation_questions.yaml
│   ├── shopping_trends.csv
│   └── Tesla_stock_data.csv
├── config/                   # ⚙️ Configuration files
├── scripts/                  # 🔧 Utility scripts
└── logs/                     # 📝 Local logs (created automatically)
```

## 🔄 Development vs Production

### Local Development Benefits:
- **Fast iteration** - No deployment delays
- **Full debugging** - Access to logs and errors
- **Offline work** - No internet dependency
- **Safe testing** - Test breaking changes safely
- **File preservation** - Git ensures no work is lost

### Streamlit Cloud Benefits:
- **Public access** - Share with users/testers
- **Production environment** - Real deployment conditions
- **Automatic deployment** - Git push = instant deployment
- **No local setup** - Works anywhere

## 🎛️ Environment Variables (Optional)

Create `.env` file for full functionality:

```bash
# API Keys (for live LLM testing)
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here
OPENROUTER_API_KEY=your_openrouter_key_here

# Development Settings
STREAMLIT_SERVER_PORT=8501
DEVELOPMENT_MODE=true
```

**Note**: The app works fully without API keys using pre-generated responses.

## 🐛 Troubleshooting

### Common Issues:

**❌ "Module not found" errors:**
```bash
# Install dependencies
pip install -r requirements.txt
```

**❌ "File not found" errors:**
```bash
# Ensure you're in the project root
cd /path/to/Intelligent-Business-Analysis-Using-Free-Tier-LLMs
python run_local.py
```

**❌ "Port already in use":**
```bash
# Kill existing Streamlit processes
pkill -f streamlit
# Or use different port
streamlit run src/ui/main.py --server.port=8502
```

**❌ Data files missing:**
```bash
# Check git status
git status

# Pull latest changes
git pull origin master

# Verify files exist
ls -la data/
```

## 📊 Development Checklist

Before making changes:
- [ ] ✅ Local server starts successfully
- [ ] ✅ All data files present and loaded
- [ ] ✅ 3 responses display side-by-side
- [ ] ✅ Ground truth shows business answers
- [ ] ✅ Ranking interface works for 3 responses
- [ ] ✅ User can register and provide feedback
- [ ] ✅ Progress tracking functions

After making changes:
- [ ] ✅ Test all modified functionality locally
- [ ] ✅ Verify no regressions in existing features
- [ ] ✅ Commit changes with descriptive message
- [ ] ✅ Push to trigger Streamlit Cloud deployment
- [ ] ✅ Monitor Streamlit Cloud for successful deployment

## 🔄 Sync Strategy

### Daily Workflow:
1. **Morning**: `git pull origin master` (get latest changes)
2. **Development**: Work locally, test frequently
3. **Commits**: Small, frequent commits with clear messages
4. **Evening**: `git push origin master` (deploy to Streamlit Cloud)

### Emergency Recovery:
If you lose local changes:
```bash
# Get latest from Streamlit Cloud deployment
git pull origin master --force

# Check what files are present
git status
ls -la data/
```

## 🌟 Advantages of This Setup

1. **🔒 Never Lose Work**: Git ensures everything is backed up
2. **⚡ Fast Development**: Local testing is instant
3. **🎯 Feature Parity**: Local = Streamlit Cloud exactly
4. **🛡️ Safe Testing**: Test breaking changes locally first
5. **📱 Easy Sharing**: Push to Git = instant public deployment
6. **🔧 Full Control**: Access to all files, logs, and debugging
7. **💾 Data Persistence**: All data files are version controlled
8. **🚀 Streamlined Workflow**: Simple commands for everything

## 🎉 Ready to Develop!

Your local environment now has **all the same features** as Streamlit Cloud:
- Enhanced RAG system with 8-chunk coverage
- 3 LLM providers with side-by-side comparison
- Ground truth business answers for context
- Dynamic ranking interface
- User registration and feedback system
- Progress tracking and data persistence

**Start developing**: `python run_local.py` and visit http://localhost:8501 