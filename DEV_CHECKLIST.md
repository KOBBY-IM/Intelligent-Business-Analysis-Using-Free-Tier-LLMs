# 🔧 Development Checklist

## 🚀 Before Starting Development

- [ ] ✅ Run `python run_local.py` to start local server
- [ ] ✅ Verify app loads at http://localhost:8501
- [ ] ✅ Check all data files are present:
  - [ ] `data/enhanced_blind_responses.json` (888KB)
  - [ ] `data/ground_truth_answers.json` (25KB)
  - [ ] `data/evaluation_questions.yaml`
  - [ ] `data/shopping_trends.csv`
  - [ ] `data/Tesla_stock_data.csv`

## 🧪 Feature Testing Checklist

### ✅ User Registration
- [ ] Registration form displays correctly
- [ ] Email validation works
- [ ] User can register successfully
- [ ] Returning users are recognized

### ✅ Response Display  
- [ ] **3 responses show side-by-side** (A, B, C layout)
- [ ] Response cards have proper styling
- [ ] Enhanced RAG indicators show (8 chunks, dynamic guidance)
- [ ] Response lengths display correctly

### ✅ Ground Truth & Context
- [ ] **Ground truth shows business answers** (not RAG chunks)
- [ ] Expert answers load from `ground_truth_answers.json`
- [ ] Key data points and factual claims display
- [ ] Business scenario context is clear

### ✅ Ranking Interface
- [ ] **Dynamic ranking for 3 responses** (ranks 1-3)
- [ ] Sliders work correctly
- [ ] Confidence slider (1-5) functions
- [ ] Comment box accepts input
- [ ] Form validation prevents duplicate ranks

### ✅ Progress Tracking
- [ ] Progress bar shows completion status
- [ ] Question counter updates correctly
- [ ] User feedback is saved
- [ ] Navigation works (home, progress)

### ✅ Data Persistence
- [ ] User evaluations save to JSON
- [ ] Feedback logger records events
- [ ] Session state maintains across pages
- [ ] No data loss on refresh

## 📊 Enhanced RAG Verification

- [ ] ✅ Enhanced responses file loads correctly
- [ ] ✅ 8-chunk coverage indicators show
- [ ] ✅ Dynamic guidance mentions appear
- [ ] ✅ Cross-chunk analysis descriptions work
- [ ] ✅ Enhanced vs standard RAG labeling correct

## 🔄 Development Workflow

### 1. Make Changes
- [ ] Test changes locally first
- [ ] Verify no regressions in existing features
- [ ] Check console for errors

### 2. Commit & Deploy
```bash
# Add changes
git add -A

# Commit with descriptive message
git commit -m "feat: describe what you changed"

# Push to trigger Streamlit Cloud deployment
git push origin master
```

### 3. Verify Deployment
- [ ] Check Streamlit Cloud app updates
- [ ] Test same features on cloud deployment
- [ ] Verify no differences between local and cloud

## 🛡️ Data Safety Protocol

### Daily Backup
```bash
# Create backup before major changes
python scripts/sync_local_cloud.py
# Choose option 1: Create backup
```

### Before Major Changes
- [ ] Create git branch for experimental features
- [ ] Backup current data files
- [ ] Test thoroughly on localhost first

### Emergency Recovery
```bash
# If local files get corrupted
git pull origin master --force

# Verify files are restored
python run_local.py
```

## 🎯 Pre-Deployment Checklist

Before pushing to Streamlit Cloud:

- [ ] ✅ All features tested locally
- [ ] ✅ No console errors or warnings
- [ ] ✅ Data files are correct versions
- [ ] ✅ Git status is clean
- [ ] ✅ Commit message is descriptive
- [ ] ✅ Changed files are appropriate

## 🚨 Red Flags (STOP Development)

**Stop and investigate if any of these occur:**

- ❌ Local server won't start
- ❌ Data files are missing or corrupted  
- ❌ Responses don't display side-by-side
- ❌ Ground truth shows RAG chunks instead of business answers
- ❌ Ranking interface doesn't adapt to response count
- ❌ User feedback doesn't save
- ❌ Console shows critical errors

## 🏆 Success Criteria

**Local environment is ready when:**

✅ All checkboxes above are checked  
✅ App functions identically to Streamlit Cloud  
✅ No data loss risk  
✅ Fast development iteration possible  
✅ Easy deployment to cloud  

## 🔧 Quick Commands

```bash
# Start local development
python run_local.py

# Check git status
git status

# Quick commit and deploy
git add -A && git commit -m "quick update" && git push origin master

# Create backup
python scripts/sync_local_cloud.py

# Kill local server (if needed)
pkill -f streamlit
```

## 📱 Development Tips

1. **Always test locally first** - Never push untested changes
2. **Commit frequently** - Small commits are easier to debug
3. **Use descriptive commit messages** - Help future you understand changes
4. **Backup before major changes** - Better safe than sorry
5. **Monitor Streamlit Cloud deployment** - Ensure changes deploy correctly 