# 🚀 Streamlit Cloud Deployment Checklist

## Pre-Deployment Checklist

### ✅ 1. Generate Fixed Responses (Local)
```bash
# Run this locally first
python3 scripts/generate_fixed_blind_responses.py
```
- [ ] `data/fixed_blind_responses.json` exists
- [ ] File contains responses for all questions
- [ ] All LLM providers are represented

### ✅ 2. Repository Structure
- [ ] `streamlit_app.py` exists (main entry point)
- [ ] `requirements.txt` is updated
- [ ] `.streamlit/config.toml` is configured
- [ ] `.gitignore` excludes sensitive files
- [ ] Core data files are included:
  - [ ] `data/ground_truth_answers.json`
  - [ ] `data/evaluation_questions.yaml`
  - [ ] `data/shopping_trends.csv`
  - [ ] `data/Tesla_stock_data.csv`
  - [ ] `data/fixed_blind_responses.json`

### ✅ 3. Environment Variables (Streamlit Cloud Secrets)
Add these in Streamlit Cloud → Settings → Secrets:

```toml
GROQ_API_KEY = "your_groq_api_key"
GOOGLE_API_KEY = "your_google_api_key"
HUGGINGFACE_API_KEY = "your_huggingface_api_key"
SECRET_KEY = "your_secret_key"
DATA_ENCRYPTION_KEY = "your_encryption_key"
```

### ✅ 4. Code Verification
- [ ] All imports work correctly
- [ ] Fixed responses loading works
- [ ] UI components render properly
- [ ] No hardcoded paths or secrets

### ✅ 5. Dependencies
- [ ] `requirements.txt` includes all packages
- [ ] No conflicting package versions
- [ ] All packages are compatible with Streamlit Cloud

---

## Deployment Steps

### 🔧 Step 1: Prepare Repository
```bash
# Commit all changes
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### 🌐 Step 2: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Configure:
   - **Repository**: Your GitHub repo
   - **Branch**: main
   - **Main file path**: `streamlit_app.py`
5. Click "Deploy"

### ⚙️ Step 3: Configure Secrets
1. Go to your deployed app
2. Click "Settings" → "Secrets"
3. Add all required API keys
4. Save and redeploy

### 📁 Step 4: Upload Data Files (if needed)
1. Go to "Files" → "Upload files"
2. Upload the `data/` directory
3. Ensure all core files are present

---

## Post-Deployment Verification

### ✅ 6. App Functionality
- [ ] App loads without errors
- [ ] Fixed responses are accessible
- [ ] Blind evaluation UI works
- [ ] Ground truth display works
- [ ] User feedback collection works

### ✅ 7. API Integration
- [ ] LLM providers respond correctly
- [ ] Rate limits are respected
- [ ] Error handling works
- [ ] No API key leaks in logs

### ✅ 8. User Experience
- [ ] UI is responsive
- [ ] Navigation works smoothly
- [ ] Forms submit correctly
- [ ] Data is saved properly

### ✅ 9. Security
- [ ] No sensitive data in logs
- [ ] API keys are secure
- [ ] User data is protected
- [ ] Rate limiting works

---

## Monitoring Checklist

### 📊 Daily Monitoring
- [ ] App uptime
- [ ] User participation
- [ ] API usage
- [ ] Error rates

### 📈 Weekly Monitoring
- [ ] User feedback analysis
- [ ] Performance metrics
- [ ] Data quality checks
- [ ] System health

### 🔄 Monthly Maintenance
- [ ] Update fixed responses
- [ ] Review API usage
- [ ] Backup user data
- [ ] Performance optimization

---

## Troubleshooting

### Common Issues:

**❌ "Fixed responses not found"**
- Solution: Upload `data/fixed_blind_responses.json` to Streamlit Cloud

**❌ "API key not found"**
- Solution: Check Streamlit Cloud secrets configuration

**❌ "Import errors"**
- Solution: Verify `requirements.txt` and package compatibility

**❌ "Rate limit exceeded"**
- Solution: Monitor API usage and implement better rate limiting

---

## Success Criteria

### 🎯 Deployment Success
- [ ] App deploys without errors
- [ ] All features work correctly
- [ ] Users can participate in evaluations
- [ ] Data collection is working

### 📊 Research Success
- [ ] Sufficient user participation
- [ ] Quality feedback data
- [ ] Statistical significance achieved
- [ ] Research objectives met

---

## Contact & Support

- **Streamlit Cloud Issues**: [Streamlit Community](https://discuss.streamlit.io/)
- **API Issues**: Contact respective providers
- **Research Questions**: Your academic advisor

---

**🎓 Your LLM evaluation system is ready for research deployment!** 