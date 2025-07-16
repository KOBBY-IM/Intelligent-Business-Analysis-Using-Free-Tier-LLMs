"""
Model Performance Dashboard

Advanced analytics dashboard for LLM model and provider performance metrics.
Analyzes technical metrics like latency, token counts, quality scores, and provider comparisons.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Page configuration
st.set_page_config(
    page_title="Model Performance Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .performance-header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .provider-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: bold;
        margin: 0.125rem;
    }
    .groq-badge { background-color: #e8f5e8; color: #28a745; }
    .gemini-badge { background-color: #fff3cd; color: #856404; }
    .huggingface-badge { background-color: #e6f3ff; color: #0056b3; }
    .openrouter-badge { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

def load_evaluation_data():
    """Load evaluation data from various sources"""
    data_sources = []
    
    # Load from evaluation results
    eval_dir = Path("data/evaluation_results")
    if eval_dir.exists():
        for file in eval_dir.glob("eval_*_results.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        data_sources.extend(data)
                    else:
                        data_sources.append(data)
            except Exception as e:
                st.error(f"Error loading {file}: {e}")
    
    # Load from results directory
    results_dir = Path("data/results")
    if results_dir.exists():
        for file in results_dir.glob("*.json"):
            if "feedback" not in file.name.lower():
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            data_sources.extend(data)
                        elif isinstance(data, dict) and 'responses' in data:
                            data_sources.extend(data['responses'])
                        else:
                            data_sources.append(data)
                except Exception as e:
                    st.error(f"Error loading {file}: {e}")
    
    return data_sources

def standardize_provider_name(provider):
    """Standardize provider names for consistency"""
    provider_mapping = {
        'groq': 'Groq',
        'gemini': 'Gemini',
        'huggingface': 'Hugging Face',
        'openrouter': 'OpenRouter',
        'google': 'Gemini',
        'hf': 'Hugging Face'
    }
    return provider_mapping.get(provider.lower(), provider.title())

def extract_metrics_from_data(data):
    """Extract performance metrics from evaluation data"""
    metrics = []
    
    for item in data:
        if isinstance(item, dict):
            # Extract basic info
            provider = item.get('provider', item.get('model_provider', 'Unknown'))
            model = item.get('model', item.get('model_name', 'Unknown'))
            
            # Extract performance metrics
            metric_entry = {
                'provider': standardize_provider_name(provider),
                'model': model,
                'timestamp': item.get('timestamp', datetime.now().isoformat()),
                'industry': item.get('industry', item.get('context', 'Unknown')),
                'question': item.get('question', item.get('query', '')),
                'response': item.get('response', item.get('answer', '')),
                'latency': item.get('latency', item.get('response_time', 0)),
                'token_count': item.get('token_count', len(str(item.get('response', '')).split())),
                'quality_score': item.get('quality_score', item.get('score', 0)),
                'relevance_score': item.get('relevance_score', 0),
                'coherence_score': item.get('coherence_score', 0),
                'accuracy_score': item.get('accuracy_score', 0),
                'error': item.get('error', None)
            }
            
            # Extract nested metrics if available
            if 'metrics' in item:
                metrics_data = item['metrics']
                metric_entry.update({
                    'latency': metrics_data.get('latency', metric_entry['latency']),
                    'token_count': metrics_data.get('token_count', metric_entry['token_count']),
                    'quality_score': metrics_data.get('quality_score', metric_entry['quality_score']),
                    'relevance_score': metrics_data.get('relevance_score', metric_entry['relevance_score']),
                    'coherence_score': metrics_data.get('coherence_score', metric_entry['coherence_score']),
                    'accuracy_score': metrics_data.get('accuracy_score', metric_entry['accuracy_score'])
                })
            
            metrics.append(metric_entry)
    
    return pd.DataFrame(metrics)

def calculate_performance_statistics(df):
    """Calculate comprehensive performance statistics"""
    if df.empty:
        return {}
    
    stats_by_provider = {}
    
    for provider in df['provider'].unique():
        provider_data = df[df['provider'] == provider]
        
        stats_by_provider[provider] = {
            'total_responses': len(provider_data),
            'avg_latency': provider_data['latency'].mean(),
            'std_latency': provider_data['latency'].std(),
            'avg_tokens': provider_data['token_count'].mean(),
            'std_tokens': provider_data['token_count'].std(),
            'avg_quality': provider_data['quality_score'].mean(),
            'std_quality': provider_data['quality_score'].std(),
            'avg_relevance': provider_data['relevance_score'].mean(),
            'avg_coherence': provider_data['coherence_score'].mean(),
            'avg_accuracy': provider_data['accuracy_score'].mean(),
            'error_rate': (provider_data['error'].notna().sum() / len(provider_data)) * 100,
            'success_rate': ((provider_data['error'].isna().sum() / len(provider_data)) * 100)
        }
    
    return stats_by_provider

def create_performance_comparison_chart(df):
    """Create comprehensive performance comparison charts"""
    if df.empty:
        return go.Figure()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Latency', 'Token Count Distribution', 'Quality Scores', 'Success Rate'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    providers = df['provider'].unique()
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # 1. Average Latency
    avg_latency = df.groupby('provider')['latency'].mean()
    fig.add_trace(
        go.Bar(x=avg_latency.index, y=avg_latency.values, 
               name='Avg Latency', marker_color=colors[0]),
        row=1, col=1
    )
    
    # 2. Token Count Distribution
    for i, provider in enumerate(providers):
        provider_data = df[df['provider'] == provider]['token_count']
        fig.add_trace(
            go.Box(y=provider_data, name=provider, marker_color=colors[i % len(colors)]),
            row=1, col=2
        )
    
    # 3. Quality Scores
    quality_metrics = ['quality_score', 'relevance_score', 'coherence_score', 'accuracy_score']
    for i, metric in enumerate(quality_metrics):
        avg_scores = df.groupby('provider')[metric].mean()
        fig.add_trace(
            go.Scatter(x=avg_scores.index, y=avg_scores.values, 
                      mode='lines+markers', name=metric.replace('_', ' ').title()),
            row=2, col=1
        )
    
    # 4. Success Rate
    success_rates = df.groupby('provider').apply(
        lambda x: (x['error'].isna().sum() / len(x)) * 100
    )
    fig.add_trace(
        go.Bar(x=success_rates.index, y=success_rates.values, 
               name='Success Rate %', marker_color=colors[2]),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Model Performance Comparison")
    return fig

def create_latency_analysis(df):
    """Create detailed latency analysis"""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Latency Distribution by Provider', 'Latency vs Quality Score'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Latency distribution
    providers = df['provider'].unique()
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, provider in enumerate(providers):
        provider_data = df[df['provider'] == provider]
        fig.add_trace(
            go.Histogram(x=provider_data['latency'], name=provider, 
                        opacity=0.7, marker_color=colors[i % len(colors)]),
            row=1, col=1
        )
    
    # Latency vs Quality scatter
    fig.add_trace(
        go.Scatter(x=df['latency'], y=df['quality_score'], 
                  mode='markers', 
                  marker=dict(size=8, opacity=0.6, color=df['quality_score'],
                             colorscale='Viridis', showscale=True),
                  text=df['provider'], name='Quality vs Latency'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Latency Performance Analysis")
    return fig

def create_industry_performance_matrix(df):
    """Create performance matrix by industry and provider"""
    if df.empty:
        return go.Figure()
    
    # Create pivot table for heatmap
    pivot_data = df.groupby(['industry', 'provider'])['quality_score'].mean().unstack(fill_value=0)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlBu_r',
        text=np.round(pivot_data.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Quality Score Heatmap by Industry and Provider",
        xaxis_title="Provider",
        yaxis_title="Industry",
        height=400
    )
    
    return fig

def perform_statistical_analysis(df):
    """Perform statistical significance tests"""
    if df.empty or len(df['provider'].unique()) < 2:
        return {}
    
    results = {}
    
    # ANOVA for quality scores across providers
    try:
        provider_groups = [df[df['provider'] == p]['quality_score'].dropna() 
                          for p in df['provider'].unique()]
        f_stat, p_value = f_oneway(*provider_groups)
        results['quality_anova'] = {'f_statistic': f_stat, 'p_value': p_value}
    except:
        results['quality_anova'] = {'f_statistic': 0, 'p_value': 1}
    
    # ANOVA for latency across providers
    try:
        latency_groups = [df[df['provider'] == p]['latency'].dropna() 
                         for p in df['provider'].unique()]
        f_stat, p_value = f_oneway(*latency_groups)
        results['latency_anova'] = {'f_statistic': f_stat, 'p_value': p_value}
    except:
        results['latency_anova'] = {'f_statistic': 0, 'p_value': 1}
    
    # Pairwise t-tests for quality
    providers = df['provider'].unique()
    pairwise_tests = {}
    for i in range(len(providers)):
        for j in range(i+1, len(providers)):
            p1, p2 = providers[i], providers[j]
            try:
                group1 = df[df['provider'] == p1]['quality_score'].dropna()
                group2 = df[df['provider'] == p2]['quality_score'].dropna()
                t_stat, p_val = ttest_ind(group1, group2)
                pairwise_tests[f"{p1}_vs_{p2}"] = {'t_statistic': t_stat, 'p_value': p_val}
            except:
                pairwise_tests[f"{p1}_vs_{p2}"] = {'t_statistic': 0, 'p_value': 1}
    
    results['pairwise_quality_tests'] = pairwise_tests
    
    return results

def export_performance_report(df, stats, filename="model_performance_report.pdf"):
    """Export comprehensive performance report as PDF"""
    buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Model Performance Analysis Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Summary statistics
    summary_text = f"""
    <b>Executive Summary</b><br/>
    Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>
    Total Evaluations: {len(df)}<br/>
    Providers Analyzed: {', '.join(df['provider'].unique())}<br/>
    Industries Covered: {', '.join(df['industry'].unique())}<br/>
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Performance statistics table
    if stats:
        table_data = [['Provider', 'Avg Latency', 'Avg Quality', 'Success Rate', 'Total Responses']]
        for provider, data in stats.items():
            table_data.append([
                provider,
                f"{data['avg_latency']:.2f}s",
                f"{data['avg_quality']:.2f}",
                f"{data['success_rate']:.1f}%",
                str(data['total_responses'])
            ])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Main Dashboard
def main():
    st.markdown('<h1 class="performance-header">⚡ Model Performance Analytics</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading performance data..."):
        evaluation_data = load_evaluation_data()
    
    if not evaluation_data:
        st.warning("No performance data available. Please run some evaluations first.")
        st.info("Use the RAG pipeline or batch evaluation tools to generate performance data.")
        return
    
    # Process data
    df = extract_metrics_from_data(evaluation_data)
    
    if df.empty:
        st.warning("No valid performance metrics found in the data.")
        return
    
    st.success(f"Loaded {len(df)} performance evaluations from {len(df['provider'].unique())} providers")
    
    # Calculate statistics
    stats = calculate_performance_statistics(df)
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Provider filter
    selected_providers = st.sidebar.multiselect(
        "Select Providers",
        options=df['provider'].unique(),
        default=df['provider'].unique()
    )
    
    # Industry filter
    selected_industries = st.sidebar.multiselect(
        "Select Industries",
        options=df['industry'].unique(),
        default=df['industry'].unique()
    )
    
    # Date range filter
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['timestamp'].dt.date >= start_date) & 
                   (df['timestamp'].dt.date <= end_date)]
    
    # Apply filters
    df_filtered = df[
        (df['provider'].isin(selected_providers)) & 
        (df['industry'].isin(selected_industries))
    ]
    
    # Recalculate stats for filtered data
    stats_filtered = calculate_performance_statistics(df_filtered)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "⚡ Performance", "🏭 Industry Analysis", 
        "📈 Statistical Analysis", "📄 Export", "🩺 Health Checks"
    ])
    
    with tab1:
        st.subheader("Performance Overview")
        
        if stats_filtered:
            # Key metrics cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_evaluations = sum(s['total_responses'] for s in stats_filtered.values())
                st.metric("Total Evaluations", total_evaluations)
            
            with col2:
                avg_latency = np.mean([s['avg_latency'] for s in stats_filtered.values()])
                st.metric("Avg Latency", f"{avg_latency:.2f}s")
            
            with col3:
                avg_quality = np.mean([s['avg_quality'] for s in stats_filtered.values()])
                st.metric("Avg Quality Score", f"{avg_quality:.2f}")
            
            with col4:
                avg_success = np.mean([s['success_rate'] for s in stats_filtered.values()])
                st.metric("Avg Success Rate", f"{avg_success:.1f}%")
            
            # Provider comparison table
            st.subheader("Provider Performance Summary")
            
            summary_data = []
            for provider, data in stats_filtered.items():
                summary_data.append({
                    'Provider': provider,
                    'Responses': data['total_responses'],
                    'Avg Latency (s)': f"{data['avg_latency']:.2f}",
                    'Avg Quality': f"{data['avg_quality']:.2f}",
                    'Avg Relevance': f"{data['avg_relevance']:.2f}",
                    'Avg Coherence': f"{data['avg_coherence']:.2f}",
                    'Success Rate (%)': f"{data['success_rate']:.1f}",
                    'Error Rate (%)': f"{data['error_rate']:.1f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No data available for the selected filters.")
    
    with tab2:
        st.subheader("Detailed Performance Analysis")
        
        # Performance comparison chart
        perf_chart = create_performance_comparison_chart(df_filtered)
        st.plotly_chart(perf_chart, use_container_width=True)
        
        # Latency analysis
        latency_chart = create_latency_analysis(df_filtered)
        st.plotly_chart(latency_chart, use_container_width=True)
        
        # Token usage analysis
        st.subheader("Token Usage Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            token_dist = px.box(df_filtered, x='provider', y='token_count', 
                               title="Token Count Distribution by Provider")
            st.plotly_chart(token_dist, use_container_width=True)
        
        with col2:
            token_vs_quality = px.scatter(df_filtered, x='token_count', y='quality_score', 
                                        color='provider', title="Token Count vs Quality Score")
            st.plotly_chart(token_vs_quality, use_container_width=True)
    
    with tab3:
        st.subheader("Industry-Specific Analysis")
        
        # Industry performance heatmap
        industry_heatmap = create_industry_performance_matrix(df_filtered)
        st.plotly_chart(industry_heatmap, use_container_width=True)
        
        # Industry breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            industry_perf = df_filtered.groupby('industry')['quality_score'].mean().reset_index()
            industry_chart = px.bar(industry_perf, x='industry', y='quality_score',
                                   title="Average Quality Score by Industry")
            st.plotly_chart(industry_chart, use_container_width=True)
        
        with col2:
            industry_latency = df_filtered.groupby('industry')['latency'].mean().reset_index()
            latency_chart = px.bar(industry_latency, x='industry', y='latency',
                                  title="Average Latency by Industry")
            st.plotly_chart(latency_chart, use_container_width=True)
    
    with tab4:
        st.subheader("Statistical Significance Analysis")
        
        # Perform statistical tests
        stat_results = perform_statistical_analysis(df_filtered)
        
        if stat_results:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ANOVA Results")
                
                # Quality ANOVA
                quality_anova = stat_results.get('quality_anova', {})
                st.metric("Quality Score F-statistic", f"{quality_anova.get('f_statistic', 0):.4f}")
                st.metric("Quality Score p-value", f"{quality_anova.get('p_value', 1):.4f}")
                
                if quality_anova.get('p_value', 1) < 0.05:
                    st.success("✅ Significant difference in quality scores between providers")
                else:
                    st.info("ℹ️ No significant difference in quality scores between providers")
                
                # Latency ANOVA
                latency_anova = stat_results.get('latency_anova', {})
                st.metric("Latency F-statistic", f"{latency_anova.get('f_statistic', 0):.4f}")
                st.metric("Latency p-value", f"{latency_anova.get('p_value', 1):.4f}")
                
                if latency_anova.get('p_value', 1) < 0.05:
                    st.success("✅ Significant difference in latency between providers")
                else:
                    st.info("ℹ️ No significant difference in latency between providers")
            
            with col2:
                st.subheader("Pairwise Comparisons")
                
                pairwise_tests = stat_results.get('pairwise_quality_tests', {})
                if pairwise_tests:
                    for comparison, test_result in pairwise_tests.items():
                        p_val = test_result.get('p_value', 1)
                        providers = comparison.replace('_vs_', ' vs ')
                        
                        if p_val < 0.05:
                            st.success(f"✅ {providers}: p = {p_val:.4f}")
                        else:
                            st.info(f"ℹ️ {providers}: p = {p_val:.4f}")
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        
        numeric_cols = ['latency', 'token_count', 'quality_score', 'relevance_score', 
                       'coherence_score', 'accuracy_score']
        available_cols = [col for col in numeric_cols if col in df_filtered.columns]
        
        if len(available_cols) > 1:
            corr_matrix = df_filtered[available_cols].corr()
            
            fig_corr = px.imshow(corr_matrix, 
                               text_auto=True, 
                               aspect="auto",
                               title="Performance Metrics Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab5:
        st.subheader("Export Performance Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Export Raw Data")
            
            # CSV export
            csv_data = df_filtered.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV Data",
                data=csv_data,
                file_name=f"model_performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # JSON export
            json_data = df_filtered.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Download JSON Data",
                data=json_data,
                file_name=f"model_performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            st.subheader("📄 Generate Report")
            
            if st.button("Generate Performance Report"):
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_buffer = export_performance_report(df_filtered, stats_filtered)
                        
                        st.download_button(
                            label="📄 Download Performance Report (PDF)",
                            data=pdf_buffer,
                            file_name=f"model_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                        st.success("✅ Report generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating report: {e}")
        
        # Summary statistics for export
        st.subheader("📈 Summary Statistics")
        
        if stats_filtered:
            stats_json = json.dumps(stats_filtered, indent=2)
            st.download_button(
                label="📊 Download Summary Statistics (JSON)",
                data=stats_json,
                file_name=f"performance_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    with tab6:
        st.subheader("LLM Provider Health Checks & Rate Limits")
        health_log_file = Path("data/evaluation_results/llm_health_checks.json")
        if not health_log_file.exists():
            st.info("No health check data available.")
        else:
            with open(health_log_file, "r") as f:
                health_data = json.load(f)
            if not health_data:
                st.info("No health check data available.")
            else:
                health_df = pd.DataFrame(health_data)
                
                # Check if this is model-level data
                has_models = 'model' in health_df.columns
                
                st.write(f"Total health checks: {len(health_df)}")
                if has_models:
                    st.write(f"Models checked: {len(health_df)}")
                    st.write(f"Providers checked: {health_df['provider'].unique().tolist()}")
                else:
                    st.write(f"Providers checked: {health_df['provider'].unique().tolist()}")
                
                # Status counts
                if has_models:
                    status_counts = health_df.groupby(['provider', 'status']).size().unstack(fill_value=0)
                    st.write("### Health Check Status Counts (by Provider)")
                    st.dataframe(status_counts)
                    
                    # Model-level status
                    st.write("### Model-Level Status")
                    for provider in health_df['provider'].unique():
                        provider_models = health_df[health_df['provider'] == provider]
                        st.write(f"**{provider.upper()}:**")
                        for _, row in provider_models.iterrows():
                            status_icon = "✅" if row['status'] == 'ok' else "❌"
                            st.write(f"{status_icon} {row['model']}: {row['status']}")
                else:
                    status_counts = health_df.groupby(['provider', 'status']).size().unstack(fill_value=0)
                    st.write("### Health Check Status Counts")
                    st.dataframe(status_counts)
                
                # Error analysis
                if 'error' in health_df.columns:
                    RATE_LIMIT_KEYWORDS = ["rate limit", "quota", "429", "too many requests", "exceeded"]
                    OVERLOADED_KEYWORDS = ["overloaded", "503", "unavailable"]
                    DEPRECATED_KEYWORDS = ["decommissioned", "not found", "404"]
                    
                    health_df['rate_limit'] = health_df['error'].apply(lambda x: any(keyword in str(x).lower() for keyword in RATE_LIMIT_KEYWORDS))
                    health_df['overloaded'] = health_df['error'].apply(lambda x: any(keyword in str(x).lower() for keyword in OVERLOADED_KEYWORDS))
                    health_df['deprecated'] = health_df['error'].apply(lambda x: any(keyword in str(x).lower() for keyword in DEPRECATED_KEYWORDS))
                    
                    rate_limited = health_df[health_df['rate_limit']]
                    overloaded = health_df[health_df['overloaded']]
                    deprecated = health_df[health_df['deprecated']]
                    
                    if not rate_limited.empty:
                        st.write("### 🚨 Rate Limited Models")
                        for _, row in rate_limited.iterrows():
                            model_name = row.get('model', row['provider'])
                            st.write(f"- {row['provider']}/{model_name}")
                    
                    if not overloaded.empty:
                        st.write("### ⚠️ Overloaded Models")
                        for _, row in overloaded.iterrows():
                            model_name = row.get('model', row['provider'])
                            st.write(f"- {row['provider']}/{model_name}")
                    
                    if not deprecated.empty:
                        st.write("### 🗑️ Deprecated Models")
                        for _, row in deprecated.iterrows():
                            model_name = row.get('model', row['provider'])
                            st.write(f"- {row['provider']}/{model_name}")
                
                # Latency stats
                if 'latency' in health_df.columns:
                    if has_models:
                        latency_stats = health_df[health_df['status'] == 'ok'].groupby(['provider', 'model'])['latency'].agg(['mean', 'max', 'min']).round(3)
                        st.write("### Latency Stats by Model (seconds)")
                        st.dataframe(latency_stats)
                        
                        # Fastest models
                        working_models = health_df[health_df['status'] == 'ok']
                        if not working_models.empty:
                            fastest_models = working_models.nsmallest(3, 'latency')[['provider', 'model', 'latency']]
                            st.write("### ⚡ Fastest Models")
                            for _, row in fastest_models.iterrows():
                                st.write(f"- {row['provider']}/{row['model']}: {row['latency']}s")
                    else:
                        latency_stats = health_df[health_df['status'] == 'ok'].groupby('provider')['latency'].agg(['mean', 'max', 'min', 'count'])
                        st.write("### Latency Stats (seconds)")
                        st.dataframe(latency_stats)
                
                # Working models summary
                if has_models:
                    working_models = health_df[health_df['status'] == 'ok']
                    total_models = len(health_df)
                    working_count = len(working_models)
                    
                    st.write("### 📈 Working Models Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Models", total_models)
                    with col2:
                        st.metric("Working", working_count)
                    with col3:
                        st.metric("Success Rate", f"{(working_count/total_models)*100:.1f}%")
                    
                    # Provider breakdown
                    provider_stats = health_df.groupby('provider').agg({
                        'status': lambda x: (x == 'ok').sum(),
                        'model': 'count'
                    }).rename(columns={'status': 'working', 'model': 'total'})
                    provider_stats['success_rate'] = (provider_stats['working'] / provider_stats['total'] * 100).round(1)
                    
                    st.write("### 📋 Provider Success Rates")
                    st.dataframe(provider_stats)
                
                # Error timeline
                if 'timestamp' in health_df.columns:
                    health_df['timestamp'] = pd.to_datetime(health_df['timestamp'])
                    error_timeline = health_df[health_df['status'] != 'ok'].groupby([pd.Grouper(key='timestamp', freq='10T'), 'provider']).size().unstack(fill_value=0)
                    if not error_timeline.empty:
                        st.write("### Error Timeline (10-min intervals)")
                        st.dataframe(error_timeline)

if __name__ == "__main__":
    main() 