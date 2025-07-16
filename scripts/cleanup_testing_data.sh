#!/bin/bash
# Cleanup Testing Data Script
# Safely removes all testing results and user feedback to prepare for clean research deployment

echo "🧹 Cleaning Up Testing Data for Fresh Research Deployment"
echo "=========================================================="

# Configuration
BACKUP_DIR="data/cleanup_backup_$(date +%Y%m%d_%H%M%S)"
CLEAN_FILES=(
    "data/pre_generated_blind_responses.json"
    "data/user_feedback.json"
    "data/enhanced_user_feedback.json"
    "data/tester_registrations.json"
)

CLEAN_DIRS=(
    "data/results"
    "data/test_results"
    "data/response_cache"
    "data/evaluation_results"
    "data/reports"
    "data/continuous_evaluation"
)

PRESERVE_FILES=(
    "data/ground_truth_answers.json"
    "data/evaluation_questions.yaml"
    "data/shopping_trends.csv"
    "data/Tesla_stock_data.csv"
)

echo "📊 Files to clean:"
for file in "${CLEAN_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   - $file"
    fi
done

echo ""
echo "📁 Directories to clean:"
for dir in "${CLEAN_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "   - $dir"
    fi
done

echo ""
echo "💾 Files to preserve:"
for file in "${PRESERVE_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   - $file"
    fi
done

echo ""
echo "⚠️  This will remove all testing results and user feedback."
echo "   Core datasets and ground truth will be preserved."
echo ""

# Ask for confirmation
read -p "Do you want to proceed with cleanup? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled."
    exit 1
fi

# Create backup directory
echo "📦 Creating backup of current data..."
mkdir -p "$BACKUP_DIR"

# Backup files before deletion
echo "💾 Backing up files..."
for file in "${CLEAN_FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_DIR/"
        echo "   - Backed up: $file"
    fi
done

# Backup directories before deletion
echo "💾 Backing up directories..."
for dir in "${CLEAN_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        cp -r "$dir" "$BACKUP_DIR/"
        echo "   - Backed up: $dir"
    fi
done

# Clean files
echo "🗑️  Removing testing files..."
for file in "${CLEAN_FILES[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "   - Removed: $file"
    fi
done

# Clean directories
echo "🗑️  Removing testing directories..."
for dir in "${CLEAN_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
        echo "   - Removed: $dir"
    fi
done

# Recreate necessary directories
echo "📁 Recreating necessary directories..."
mkdir -p data/results
mkdir -p data/continuous_evaluation
mkdir -p data/response_cache
mkdir -p data/evaluation_results
mkdir -p data/reports
mkdir -p logs

echo ""
echo "✅ Cleanup completed successfully!"
echo "📦 Backup saved to: $BACKUP_DIR"
echo ""
echo "🎯 Ready for clean research deployment!"
echo "   - All testing results cleared"
echo "   - User feedback removed"
echo "   - Core datasets preserved"
echo "   - Ground truth maintained"
echo ""
echo "🚀 You can now start fresh with:"
echo "   ./scripts/run_research_evaluation.sh" 