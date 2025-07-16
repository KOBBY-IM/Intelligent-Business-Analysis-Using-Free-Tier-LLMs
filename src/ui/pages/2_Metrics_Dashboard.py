"""
Metrics Dashboard Page

This page displays comprehensive metrics and evaluation results for the LLM comparison system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from scipy.stats import kendalltau
from scipy.stats import rankdata
from scipy.stats import spearmanr
from scipy.stats import entropy
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from ui.components.admin_auth import show_inline_admin_login, show_admin_header

# Page configuration
st.set_page_config(
    page_title="LLM Blind Test Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check admin authentication - use inline login if not authenticated
if not show_inline_admin_login("Metrics Dashboard"):
    st.stop()

# Show admin header with logout option
show_admin_header("Metrics Dashboard")

def load_blind_feedback():
    feedback_file = Path("data/results/user_feedback.json")
    if feedback_file.exists():
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
    else:
        feedback_data = []
    return feedback_data

def load_blind_responses():
    blind_file = Path("data/blind_responses.json")
    if blind_file.exists():
        with open(blind_file, 'r') as f:
            blind_data = json.load(f)
    else:
        blind_data = {}
    return blind_data

def get_completed_testers(feedback_data):
    user_responses = defaultdict(list)
    for entry in feedback_data:
        user = entry.get('tester_email')
        if user:
            user_responses[user].append(entry)
    completed_users = [u for u, responses in user_responses.items() if len(responses) >= 12]
    return completed_users, user_responses

def build_choice_dataframe(feedback_data):
    rows = []
    for entry in feedback_data:
        # Use ranking_model_order if available
        ranking = entry.get('ranking_model_order', [])
        top_model = ranking[0] if ranking else None
        row = {
            'tester_email': entry.get('tester_email', ''),
            'industry': entry.get('domain', '') or entry.get('industry', ''),
            'top_model': top_model,
            'ranking': ranking,
            'blind_map': entry.get('blind_map', {}),
            'prompt': entry.get('question_text', entry.get('prompt', '')),
            'session_id': entry.get('session_id', ''),
            'timestamp': entry.get('timestamp', ''),
            'comment': entry.get('comment', ''),
            'comment_length': len(entry.get('comment', '')),
            'question_idx': entry.get('question_idx', None)
        }
        rows.append(row)
    return pd.DataFrame(rows)

def map_labels_to_models(df, blind_data):
    model_names = []
    providers = []
    for idx, row in df.iterrows():
        industry = row['industry']
        label = row['top_model'] # Changed from selected_label to top_model
        blind_map = row['blind_map']
        response_id = blind_map.get(label)
        model = 'unknown'
        provider = 'unknown'
        if response_id and industry in blind_data:
            for q in blind_data[industry]:
                for resp in q['responses']:
                    if resp['id'] == response_id:
                        model = resp.get('model', 'unknown')
                        provider = resp.get('provider', 'unknown')
                        break
        model_names.append(f"{provider} | {model}")
        providers.append(provider)
    df['selected_model'] = model_names
    df['provider'] = providers
    return df

def calculate_statistical_significance(df):
    """Calculate statistical significance tests"""
    results = {}
    
    # Chi-square test for industry vs model preferences
    if len(df['industry'].unique()) > 1 and len(df['top_model'].unique()) > 1: # Changed from selected_model to top_model
        contingency_table = pd.crosstab(df['industry'], df['top_model']) # Changed from selected_model to top_model
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        results['chi_square'] = {
            'chi2': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05
        }
    
    # Calculate confidence intervals for model performance
    model_ci = {}
    for model in df['top_model'].unique(): # Changed from selected_model to top_model
        model_selections = len(df[df['top_model'] == model]) # Changed from selected_model to top_model
        total_selections = len(df)
        proportion = model_selections / total_selections
        
        # 95% confidence interval for proportion
        z_score = 1.96  # 95% CI
        margin_error = z_score * np.sqrt(proportion * (1 - proportion) / total_selections)
        ci_lower = max(0, proportion - margin_error)
        ci_upper = min(1, proportion + margin_error)
        
        model_ci[model] = {
            'proportion': proportion,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'margin_error': margin_error
        }
    
    results['confidence_intervals'] = model_ci
    return results

def calculate_inter_rater_agreement(df):
    """Calculate inter-rater agreement metrics"""
    # Create user-question matrix
    user_question_matrix = df.pivot_table(
        index='tester_email', 
        columns=['industry', 'prompt'], 
        values='top_model', # Changed from selected_label to top_model
        aggfunc='first'
    )
    
    # Calculate agreement rate for each question
    agreement_rates = {}
    for col in user_question_matrix.columns:
        responses = user_question_matrix[col].dropna()
        if len(responses) > 1:
            most_common = responses.mode()
            if len(most_common) > 0:
                agreement_count = sum(responses == most_common.iloc[0])
                agreement_rate = agreement_count / len(responses)
                agreement_rates[col] = agreement_rate
    
    return agreement_rates

def analyze_user_behavior(df):
    """Analyze individual user behavior patterns"""
    user_stats = {}
    
    for user in df['tester_email'].unique():
        user_data = df[df['tester_email'] == user]
        
        # User consistency: How often does user pick same models?
        model_counts = user_data['top_model'].value_counts() # Changed from selected_model to top_model
        consistency_score = model_counts.iloc[0] / len(user_data) if len(model_counts) > 0 else 0
        
        # Response quality indicators
        avg_comment_length = user_data['comment_length'].mean()
        has_comments = sum(user_data['comment_length'] > 0)
        
        # Industry preference difference
        industry_preferences = {}
        for industry in user_data['industry'].unique():
            industry_data = user_data[user_data['industry'] == industry]
            top_model = industry_data['top_model'].mode() # Changed from selected_model to top_model
            if len(top_model) > 0:
                industry_preferences[industry] = top_model.iloc[0]
        
        user_stats[user] = {
            'total_responses': len(user_data),
            'consistency_score': consistency_score,
            'most_selected_model': model_counts.index[0] if len(model_counts) > 0 else 'none',
            'avg_comment_length': avg_comment_length,
            'response_rate_with_comments': has_comments / len(user_data),
            'industry_preferences': industry_preferences
        }
    
    return user_stats

def detect_outliers_and_bias(df, user_stats):
    """Detect outliers and potential bias in user responses"""
    outliers = {}
    
    # Users with extreme consistency (potential bots or biased users)
    consistency_scores = [stats['consistency_score'] for stats in user_stats.values()]
    q75, q25 = np.percentile(consistency_scores, [75, 25])
    iqr = q75 - q25
    upper_threshold = q75 + 1.5 * iqr
    
    outliers['high_consistency'] = [
        user for user, stats in user_stats.items() 
        if stats['consistency_score'] > upper_threshold
    ]
    
    # Users with unusually short/long comment patterns
    comment_lengths = [stats['avg_comment_length'] for stats in user_stats.values()]
    if comment_lengths:
        q75_comment, q25_comment = np.percentile(comment_lengths, [75, 25])
        iqr_comment = q75_comment - q25_comment
        comment_upper = q75_comment + 1.5 * iqr_comment
        comment_lower = max(0, q25_comment - 1.5 * iqr_comment)
        
        outliers['extreme_comment_patterns'] = [
            user for user, stats in user_stats.items() 
            if stats['avg_comment_length'] > comment_upper or stats['avg_comment_length'] < comment_lower
        ]
    
    return outliers

def generate_academic_summary(df, stats_results, user_stats, outliers):
    """Generate academic-style summary statistics"""
    summary = {
        'study_overview': {
            'total_participants': len(df['tester_email'].unique()),
            'total_responses': len(df),
            'study_period': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'industries_tested': list(df['industry'].unique()),
            'models_evaluated': list(df['top_model'].unique()) # Changed from selected_model to top_model
        },
        'statistical_results': stats_results,
        'quality_metrics': {
            'response_completion_rate': len(df) / (len(df['tester_email'].unique()) * 12),  # 12 expected responses per user
            'average_comment_rate': np.mean([s['response_rate_with_comments'] for s in user_stats.values()]),
            'outliers_detected': len(outliers.get('high_consistency', [])) + len(outliers.get('extreme_comment_patterns', [])),
            'data_quality_score': 1 - (len(outliers.get('high_consistency', [])) / len(df['tester_email'].unique()))
        }
    }
    return summary

def create_pdf_report(summary_data, df):
    """Generate PDF report for academic use"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("LLM Blind Test Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Study Overview
    story.append(Paragraph("Study Overview", styles['Heading1']))
    overview = summary_data['study_overview']
    for key, value in overview.items():
        story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Statistical Results
    if 'chi_square' in summary_data['statistical_results']:
        story.append(Paragraph("Statistical Significance", styles['Heading1']))
        chi2_data = summary_data['statistical_results']['chi_square']
        story.append(Paragraph(f"Chi-square test: χ² = {chi2_data['chi2']:.4f}, p = {chi2_data['p_value']:.4f}", styles['Normal']))
        story.append(Paragraph(f"Statistically significant: {chi2_data['significant']}", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- Advanced Ranking Stats ---
def compute_kendalls_w(rank_matrix):
    # rank_matrix: rows = users, cols = models (ranks)
    # W = 12 * S / (m^2 * (n^3 - n)), S = sum of squared deviations from mean rank
    n, m = rank_matrix.shape
    mean_ranks = np.mean(rank_matrix, axis=1, keepdims=True)
    S = np.sum((rank_matrix - mean_ranks) ** 2)
    W = 12 * S / (m ** 2 * (n ** 3 - n)) if n > 1 else np.nan
    return W

def plot_rank_correlation_heatmap(df):
    # Build user x model rank matrix
    user_model_ranks = defaultdict(dict)
    for _, row in df.iterrows():
        user = row['tester_email']
        for rank, model in enumerate(row['ranking']):
            user_model_ranks[user][model] = rank + 1
    users = list(user_model_ranks.keys())
    models = sorted({m for ranks in user_model_ranks.values() for m in ranks})
    rank_matrix = np.full((len(users), len(models)), np.nan)
    for i, user in enumerate(users):
        for j, model in enumerate(models):
            if model in user_model_ranks[user]:
                rank_matrix[i, j] = user_model_ranks[user][model]
    # Spearman correlation between models
    corr = np.full((len(models), len(models)), np.nan)
    for i in range(len(models)):
        for j in range(len(models)):
            valid = ~np.isnan(rank_matrix[:, i]) & ~np.isnan(rank_matrix[:, j])
            if np.sum(valid) > 1:
                corr[i, j] = spearmanr(rank_matrix[valid, i], rank_matrix[valid, j]).correlation
    st.markdown('#### 🔗 Rank Correlation Heatmap (Spearman)')
    fig = px.imshow(corr, x=models, y=models, color_continuous_scale='RdBu', zmin=-1, zmax=1, text_auto=True, title='Model Rank Correlation Heatmap')
    st.plotly_chart(fig, use_container_width=True)

# Rank transition matrix

def plot_rank_transition_matrix(df):
    rank_rows = []
    for _, row in df.iterrows():
        for rank, model in enumerate(row['ranking']):
            rank_rows.append({'model': model, 'rank': rank+1})
    rank_df = pd.DataFrame(rank_rows)
    matrix = pd.crosstab(rank_df['model'], rank_df['rank'])
    st.markdown('#### 🔄 Rank Transition Matrix')
    st.dataframe(matrix)
    fig = px.imshow(matrix, text_auto=True, aspect='auto', labels={'x': 'Rank', 'y': 'Model', 'color': 'Count'}, title='Model Rank Transition Matrix')
    st.plotly_chart(fig, use_container_width=True)

# Rank entropy per model

def plot_rank_entropy(df):
    rank_rows = []
    for _, row in df.iterrows():
        for rank, model in enumerate(row['ranking']):
            rank_rows.append({'model': model, 'rank': rank+1})
    rank_df = pd.DataFrame(rank_rows)
    entropies = {}
    for model in rank_df['model'].unique():
        counts = rank_df[rank_df['model'] == model]['rank'].value_counts().sort_index()
        probs = counts / counts.sum()
        entropies[model] = entropy(probs, base=2)
    ent_df = pd.DataFrame({'Model': list(entropies.keys()), 'Rank Entropy': list(entropies.values())}).sort_values('Rank Entropy')
    st.markdown('#### 📉 Rank Entropy per Model (Lower = More Consistent)')
    fig = px.bar(ent_df, x='Model', y='Rank Entropy', title='Rank Entropy per Model')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(ent_df)

# User consistency score (average Kendall's tau to group median ranking)
def compute_user_consistency(df):
    # For each question, compute group median ranking, then average tau for each user
    user_scores = {}
    questions = df.groupby(['industry', 'question_idx'])
    for user in df['tester_email'].unique():
        taus = []
        for (industry, qidx), group in questions:
            user_row = group[group['tester_email'] == user]
            if user_row.empty or len(group) < 2:
                continue
            # Build model list
            models = group.iloc[0]['ranking']
            # Build rank matrix
            rank_matrix = np.array([rankdata([r['ranking'].index(m) for m in models]) for _, r in group.iterrows()])
            median_ranking = np.median(rank_matrix, axis=0)
            user_ranking = rankdata([user_row.iloc[0]['ranking'].index(m) for m in models])
            tau, _ = kendalltau(user_ranking, median_ranking)
            if not np.isnan(tau):
                taus.append(tau)
        user_scores[user] = np.mean(taus) if taus else np.nan
    return user_scores

def plot_user_consistency(df):
    user_scores = compute_user_consistency(df)
    st.markdown('#### 👤 User Consistency Score (Kendall’s Tau to Group Median)')
    score_df = pd.DataFrame({'User': list(user_scores.keys()), 'Consistency': list(user_scores.values())})
    fig = px.bar(score_df, x='User', y='Consistency', title='User Consistency Score (Kendall’s Tau)')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(score_df)

def main():
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='font-size: 2.5rem; color: #1f77b4; margin-bottom: 1rem;'>
            📊 LLM Blind Test Analytics - Research Edition
        </h1>
        <h2 style='font-size: 1.5rem; color: #666; margin-bottom: 2rem;'>
            Comprehensive Statistical Analysis & Academic Reporting
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # Load and process data
    feedback_data = load_blind_feedback()
    blind_data = load_blind_responses()
    completed_users, user_responses = get_completed_testers(feedback_data)
    if len(completed_users) < 20:
        st.warning(f"Analysis will be available once at least 20 users have completed the evaluation. Current: {len(completed_users)}/20")
        st.stop()

    df = build_choice_dataframe(feedback_data)
    if df.empty:
        st.info("No blind test data available yet. Complete some evaluations to see analytics.")
        return

    df = map_labels_to_models(df, blind_data)

    # Sidebar filters for interactive analysis
    st.sidebar.title("🔍 Interactive Filters")
    
    # Date range filter
    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]
    
    # Industry filter
    industries = st.sidebar.multiselect(
        "Select Industries",
        options=df['industry'].unique(),
        default=df['industry'].unique()
    )
    df = df[df['industry'].isin(industries)]
    
    # Provider filter
    providers = st.sidebar.multiselect(
        "Select Providers",
        options=df['provider'].unique(),
        default=df['provider'].unique()
    )
    df = df[df['provider'].isin(providers)]

    # Calculate advanced statistics
    stats_results = calculate_statistical_significance(df)
    agreement_rates = calculate_inter_rater_agreement(df)
    user_stats = analyze_user_behavior(df)
    outliers = detect_outliers_and_bias(df, user_stats)
    academic_summary = generate_academic_summary(df, stats_results, user_stats, outliers)

    # Display sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## 📈 Study Overview")
        overview = academic_summary['study_overview']
        st.metric("Total Participants", overview['total_participants'])
        st.metric("Total Responses", overview['total_responses'])
        st.metric("Data Quality Score", f"{academic_summary['quality_metrics']['data_quality_score']:.2%}")
    
    with col2:
        st.markdown("## 🧪 Statistical Significance")
        if 'chi_square' in stats_results:
            chi2_data = stats_results['chi_square']
            st.metric("Chi-square p-value", f"{chi2_data['p_value']:.4f}")
            significance = "✅ Significant" if chi2_data['significant'] else "❌ Not Significant"
            st.metric("Statistical Significance (p<0.05)", significance)

    # Basic visualizations (from previous version)
    st.markdown("## 🏷️ User Choice Distribution")
    col1, col2 = st.columns(2)
    with col1:
        label_counts = df['top_model'].value_counts().sort_index() # Changed from selected_label to top_model
        fig = px.bar(
            x=label_counts.index,
            y=label_counts.values,
            labels={'x': 'Response Label', 'y': 'Selection Count'},
            title="Frequency of Each Label Selected"
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        industry_counts = df['industry'].value_counts()
        fig = px.pie(
            values=industry_counts.values,
            names=industry_counts.index,
            title="Selections by Industry"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance with confidence intervals
    st.markdown("## 📊 Model Performance with Confidence Intervals")
    if 'confidence_intervals' in stats_results:
        ci_data = []
        for model, ci_info in stats_results['confidence_intervals'].items():
            ci_data.append({
                'Model': model,
                'Selection Rate': f"{ci_info['proportion']:.1%}",
                '95% CI Lower': f"{ci_info['ci_lower']:.1%}",
                '95% CI Upper': f"{ci_info['ci_upper']:.1%}",
                'Margin of Error': f"±{ci_info['margin_error']:.1%}"
            })
        ci_df = pd.DataFrame(ci_data)
        st.dataframe(ci_df, use_container_width=True, hide_index=True)

    # User behavior analysis
    st.markdown("## 👥 User Behavior Patterns")
    
    col1, col2 = st.columns(2)
    with col1:
        consistency_scores = [stats['consistency_score'] for stats in user_stats.values()]
        fig = px.histogram(
            x=consistency_scores,
            nbins=20,
            title="User Consistency Distribution",
            labels={'x': 'Consistency Score', 'y': 'Number of Users'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        comment_rates = [stats['response_rate_with_comments'] for stats in user_stats.values()]
        fig = px.histogram(
            x=comment_rates,
            nbins=20,
            title="Comment Participation Rate",
            labels={'x': 'Comment Rate', 'y': 'Number of Users'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Quality assurance metrics
    st.markdown("## 🔍 Quality Assurance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("High Consistency Outliers", len(outliers.get('high_consistency', [])))
        if outliers.get('high_consistency'):
            with st.expander("View High Consistency Users"):
                for user in outliers['high_consistency']:
                    st.write(f"• {user} (Consistency: {user_stats[user]['consistency_score']:.1%})")
    
    with col2:
        st.metric("Comment Pattern Outliers", len(outliers.get('extreme_comment_patterns', [])))
        if outliers.get('extreme_comment_patterns'):
            with st.expander("View Comment Pattern Outliers"):
                for user in outliers['extreme_comment_patterns']:
                    st.write(f"• {user} (Avg Comment Length: {user_stats[user]['avg_comment_length']:.1f})")
    
    with col3:
        completion_rate = academic_summary['quality_metrics']['response_completion_rate']
        st.metric("Response Completion Rate", f"{completion_rate:.1%}")

    # Inter-rater agreement
    if agreement_rates:
        st.markdown("## 🤝 Inter-Rater Agreement")
        agreement_df = pd.DataFrame([
            {'Question': f"{q[0]} - {q[1][:50]}...", 'Agreement Rate': f"{rate:.1%}"}
            for q, rate in agreement_rates.items()
        ])
        st.dataframe(agreement_df, use_container_width=True, hide_index=True)

    # Advanced ranking analysis
    st.markdown("## 📊 Advanced Ranking Analysis")
    plot_average_rank(df)
    plot_rank_distribution(df)

    # Industry-specific analysis (from previous version)
    st.markdown("## 🏭 Model Performance by Industry")
    col1, col2 = st.columns(2)
    
    with col1:
        retail_df = df[df['industry'] == 'retail']
        if not retail_df.empty:
            retail_model_counts = retail_df['top_model'].value_counts() # Changed from selected_model to top_model
            fig = px.bar(
                x=retail_model_counts.index,
                y=retail_model_counts.values,
                labels={'x': 'Provider | Model', 'y': 'Times Selected'},
                title="🛍️ Retail Industry - Model Selection"
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        finance_df = df[df['industry'] == 'finance']
        if not finance_df.empty:
            finance_model_counts = finance_df['top_model'].value_counts() # Changed from selected_model to top_model
            fig = px.bar(
                x=finance_model_counts.index,
                y=finance_model_counts.values,
                labels={'x': 'Provider | Model', 'y': 'Times Selected'},
                title="💰 Finance Industry - Model Selection"
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    # Academic reporting and export
    st.markdown("## 📤 Academic Export & Reporting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate PDF Report"):
            pdf_buffer = create_pdf_report(academic_summary, df)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer.getvalue(),
                file_name=f"llm_blind_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
    
    with col2:
        if st.button("📊 Export Statistical Data"):
            export_data = {
                'summary': academic_summary,
                'raw_selections': df.to_dict('records'),
                'user_statistics': user_stats,
                'outliers': outliers
            }
            st.download_button(
                label="Download Statistical Data (JSON)",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"llm_statistical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📋 Export Raw Data"):
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download Raw Data (CSV)",
                data=csv_data,
                file_name=f"llm_blind_test_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    # Detailed data tables
    st.markdown("## 📋 Detailed Analysis Tables")
    
    tab1, tab2, tab3 = st.tabs(["User Statistics", "Model Performance", "Raw Data"])
    
    with tab1:
        user_stats_df = pd.DataFrame([
            {
                'User': user,
                'Total Responses': stats['total_responses'],
                'Consistency Score': f"{stats['consistency_score']:.1%}",
                'Most Selected Model': stats['most_selected_model'],
                'Avg Comment Length': f"{stats['avg_comment_length']:.1f}",
                'Comment Rate': f"{stats['response_rate_with_comments']:.1%}"
            }
            for user, stats in user_stats.items()
        ])
        st.dataframe(user_stats_df, use_container_width=True, hide_index=True)
    
    with tab2:
        model_performance = df.groupby('top_model').agg({ # Changed from selected_model to top_model
            'tester_email': 'count',
            'industry': lambda x: list(x.unique()),
            'comment_length': 'mean'
        }).round(2)
        model_performance.columns = ['Times Selected', 'Industries', 'Avg Comment Length']
        st.dataframe(model_performance, use_container_width=True)
    
    with tab3:
        st.dataframe(df[['tester_email', 'industry', 'top_model', 'prompt', 'timestamp', 'comment_length']], use_container_width=True, hide_index=True)

    # --- Advanced Ranking Statistics ---
    st.markdown('## 🧮 Advanced Ranking Statistics')
    # Compute Kendall’s W for all questions
    question_groups = df.groupby(['industry', 'question_idx'])
    W_scores = []
    for (industry, qidx), group in question_groups:
        if len(group) > 1:
            models = group.iloc[0]['ranking']
            rank_matrix = np.array([[r['ranking'].index(m) + 1 for m in models] for _, r in group.iterrows()])
            W = compute_kendalls_w(rank_matrix)
            W_scores.append({'industry': industry, 'question_idx': qidx, 'kendalls_W': W})
    W_df = pd.DataFrame(W_scores)
    st.markdown('#### 🏆 Kendall’s W (Agreement) per Question')
    st.dataframe(W_df)
    if not W_df.empty:
        st.markdown(f"**Overall Median Kendall’s W:** {W_df['kendalls_W'].median():.3f}")

    plot_rank_correlation_heatmap(df)
    plot_rank_transition_matrix(df)
    plot_rank_entropy(df)
    plot_user_consistency(df)

if __name__ == "__main__":
    main() 