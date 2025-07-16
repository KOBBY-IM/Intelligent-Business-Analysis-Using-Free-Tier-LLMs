# Streamlit Cloud Deployment Guide

## 🚀 Deploying LLM Evaluation System on Streamlit Cloud

This guide will help you deploy your LLM evaluation research system on Streamlit Cloud for public access.

---

## 📋 Prerequisites

1. **Streamlit Cloud Account** (free tier available)
2. **GitHub Repository** with your code
3. **Generated Fixed Responses** (see Step 1 below)

---

## 🔧 Step 1: Prepare Fixed Responses (Local)

Before deploying, you need to generate the fixed blind responses locally:

```bash
# Clone your repository locally
git clone <your-repo-url>
cd Intelligent-Business-Analysis-Using-Free-Tier-LLMs

# Install dependencies
pip install -r requirements.txt

# Generate fixed blind responses
python3 scripts/generate_fixed_blind_responses.py
```

This creates `data/fixed_blind_responses.json` which ensures all users see the same LLM responses.

---

## 📁 Step 2: Prepare Repository Structure

Ensure your repository has this structure:

```
your-repo/
├── streamlit_app.py              # Main entry point
├── requirements.txt              # Dependencies
├── .streamlit/
│   └── config.toml              # Streamlit config
├── src/                         # Source code
├── config/                      # Configuration files
├── data/
│   ├── fixed_blind_responses.json  # Generated responses
│   ├── ground_truth_answers.json   # Ground truth data
│   ├── evaluation_questions.yaml   # Questions
│   ├── shopping_trends.csv         # Retail dataset
│   └── Tesla_stock_data.csv        # Finance dataset
└── README.md
```

---

## 🔐 Step 3: Configure Environment Variables

In Streamlit Cloud, add these secrets:

### Required API Keys:
```
GROQ_API_KEY = your_groq_api_key
GOOGLE_API_KEY = your_google_api_key
HUGGINGFACE_API_KEY = your_huggingface_api_key
```

### Optional Configuration:
```
SECRET_KEY = your_secret_key_for_sessions
DATA_ENCRYPTION_KEY = your_encryption_key
```

**How to add secrets:**
1. Go to your Streamlit Cloud app
2. Click "Settings" → "Secrets"
3. Add each key-value pair

---

## 🌐 Step 4: Deploy to Streamlit Cloud

### Option A: Deploy from GitHub

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Connect to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app.py`
   - Click "Deploy"

### Option B: Manual Upload

1. **Upload files to Streamlit Cloud:**
   - Go to your Streamlit Cloud app
   - Click "Files" → "Upload files"
   - Upload the entire `data/` directory

2. **Redeploy the app**

---

## ⚙️ Step 5: Configuration Files

### Environment Variables (.env)
Create a `.env` file locally (don't commit to Git):

```env
# API Keys
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key

# Security
SECRET_KEY=your_secret_key
DATA_ENCRYPTION_KEY=your_encryption_key

# App Configuration
ALLOWED_HOSTS=localhost,127.0.0.1,*.streamlit.app
UPLOAD_MAX_SIZE_MB=10
API_RATE_LIMIT_PER_MINUTE=60
```

### Streamlit Config (.streamlit/config.toml)
Already configured for production deployment.

---

## 🧪 Step 6: Test Your Deployment

1. **Check the app loads correctly**
2. **Verify fixed responses are available**
3. **Test blind evaluation flow**
4. **Check ground truth display**

---

## 📊 Step 7: Monitor and Maintain

### Monitoring:
- **App performance** in Streamlit Cloud dashboard
- **User feedback** in `data/user_feedback.json`
- **Error logs** in Streamlit Cloud logs

### Maintenance:
- **Update fixed responses** periodically
- **Monitor API usage** and rate limits
- **Backup user feedback** data

---

## 🔍 Troubleshooting

### Common Issues:

1. **"Fixed responses not found"**
   - Ensure `data/fixed_blind_responses.json` is uploaded
   - Check file path in Streamlit Cloud

2. **"API key not found"**
   - Verify secrets are set correctly in Streamlit Cloud
   - Check environment variable names

3. **"Import errors"**
   - Ensure all dependencies are in `requirements.txt`
   - Check Python version compatibility

4. **"Rate limit exceeded"**
   - Monitor API usage
   - Consider upgrading API tiers

---

## 📈 Advanced Features

### Custom Domain (Optional):
- Upgrade to Streamlit Cloud Pro
- Configure custom domain

### Analytics (Optional):
- Add Google Analytics
- Track user engagement

### Backup Strategy:
- Regular backups of user feedback
- Version control for fixed responses

---

## 🎯 Success Metrics

Monitor these metrics for your research:

1. **User Participation:**
   - Number of completed evaluations
   - User retention rate

2. **Data Quality:**
   - Response completion rate
   - Feedback quality

3. **System Performance:**
   - App uptime
   - Response times

---

## 📞 Support

For issues:
1. Check Streamlit Cloud logs
2. Review this documentation
3. Check GitHub issues
4. Contact Streamlit support

---

**Your LLM evaluation system is now ready for public research deployment on Streamlit Cloud!** 🚀 