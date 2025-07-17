#!/usr/bin/env python3
"""
Enhanced CSV-based RAG Pipeline
Improved with better data coverage, chunk quality, and dynamic guidance.
"""

import pandas as pd
import numpy as np
import faiss
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import json
import re

# Import LLM providers
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from llm_providers.provider_manager import ProviderManager
from evaluation.ground_truth_generator import GroundTruthGenerator

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class CSVChunk:
    """A chunk of CSV data with enhanced metadata"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    statistical_summary: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    outlier_info: Dict[str, Any]

class EnhancedCSVRAGPipeline:
    """Enhanced CSV-based RAG pipeline with improved coverage and quality"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize enhanced RAG pipeline
        
        Args:
            embedding_model: Name of the sentence transformer model to use
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunks = []
        self.index = None
        self.chunk_embeddings = None
        self.dataset_stats = {}
        
    def load_csv_data(self, csv_path: str, chunk_size: int = 150) -> List[CSVChunk]:
        """
        Load and chunk CSV data with enhanced processing
        
        Args:
            csv_path: Path to CSV file
            chunk_size: Number of rows per chunk (reduced for better granularity)
            
        Returns:
            List of enhanced CSV chunks
        """
        logger.info(f"Loading CSV data from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Calculate dataset statistics
            self.dataset_stats[csv_path] = self._calculate_dataset_statistics(df)
            
            chunks = []
            total_chunks = (len(df) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(df), chunk_size):
                chunk_start = i
                chunk_end = min(i + chunk_size, len(df))
                df_chunk = df.iloc[chunk_start:chunk_end]
                
                # Create enhanced chunk summary
                chunk_content = self._create_enhanced_chunk_summary(df_chunk, chunk_start, csv_path)
                
                # Calculate statistical summary
                statistical_summary = self._calculate_chunk_statistics(df_chunk)
                
                # Perform trend analysis
                trend_analysis = self._analyze_chunk_trends(df_chunk, chunk_start)
                
                # Detect outliers
                outlier_info = self._detect_chunk_outliers(df_chunk)
                
                chunk = CSVChunk(
                    content=chunk_content,
                    metadata={
                        'file_path': csv_path,
                        'start_row': chunk_start,
                        'end_row': chunk_end - 1,
                        'chunk_size': len(df_chunk),
                        'columns': list(df_chunk.columns)
                    },
                    chunk_id=f"{Path(csv_path).stem}_{chunk_start}_{chunk_end}",
                    statistical_summary=statistical_summary,
                    trend_analysis=trend_analysis,
                    outlier_info=outlier_info
                )
                
                chunks.append(chunk)
                
                if len(chunks) % 10 == 0:
                    logger.info(f"Processed {len(chunks)}/{total_chunks} chunks")
            
            logger.info(f"Created {len(chunks)} chunks from {csv_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return []
    
    def _calculate_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive dataset statistics"""
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical column statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            stats['categorical_summary'] = {}
            for col in categorical_cols:
                stats['categorical_summary'][col] = {
                    'unique_values': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict()
                }
        
        return stats
    
    def _create_enhanced_chunk_summary(self, df_chunk: pd.DataFrame, start_row: int, file_path: str) -> str:
        """Create comprehensive chunk summary with enhanced details"""
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"=== CHUNK SUMMARY ===")
        summary_parts.append(f"Rows {start_row}-{start_row + len(df_chunk) - 1} ({len(df_chunk)} records)")
        summary_parts.append(f"Source: {Path(file_path).name}")
        summary_parts.append("")
        
        # Enhanced statistical summary
        numeric_cols = df_chunk.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary_parts.append("=== NUMERIC STATISTICS ===")
            for col in numeric_cols:
                if col in df_chunk.columns:
                    mean_val = df_chunk[col].mean()
                    median_val = df_chunk[col].median()
                    std_val = df_chunk[col].std()
                    min_val = df_chunk[col].min()
                    max_val = df_chunk[col].max()
                    
                    summary_parts.append(f"{col}:")
                    summary_parts.append(f"  Mean: {mean_val:.2f}, Median: {median_val:.2f}")
                    summary_parts.append(f"  Std: {std_val:.2f}, Range: {min_val:.2f}-{max_val:.2f}")
            summary_parts.append("")
        
        # Enhanced categorical analysis
        categorical_cols = ['Category', 'Gender', 'Season', 'Payment Method', 'Subscription Status']
        for col in categorical_cols:
            if col in df_chunk.columns:
                value_counts = df_chunk[col].value_counts()
                summary_parts.append(f"=== {col.upper()} DISTRIBUTION ===")
                for val, count in value_counts.head(5).items():
                    percentage = (count / len(df_chunk)) * 100
                    summary_parts.append(f"  {val}: {count} ({percentage:.1f}%)")
                summary_parts.append("")
        
        # Trend analysis
        if 'Date' in df_chunk.columns or any('date' in col.lower() for col in df_chunk.columns):
            summary_parts.append("=== TREND ANALYSIS ===")
            date_col = 'Date' if 'Date' in df_chunk.columns else [col for col in df_chunk.columns if 'date' in col.lower()][0]
            if pd.api.types.is_datetime64_any_dtype(df_chunk[date_col]):
                df_sorted = df_chunk.sort_values(date_col)
                summary_parts.append(f"Date range: {df_sorted[date_col].min()} to {df_sorted[date_col].max()}")
                
                # Analyze trends in numeric columns
                for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    if col != date_col:
                        correlation = df_chunk[date_col].astype(np.int64).corr(df_chunk[col])
                        summary_parts.append(f"{col} trend: {'Increasing' if correlation > 0.1 else 'Decreasing' if correlation < -0.1 else 'Stable'} (r={correlation:.3f})")
            summary_parts.append("")
        
        # Sample records with more detail
        summary_parts.append("=== SAMPLE RECORDS ===")
        sample_data = df_chunk.head(3)
        for i, (_, row) in enumerate(sample_data.iterrows()):
            record_info = []
            for col in df_chunk.columns[:5]:  # Show first 5 columns
                if pd.notna(row[col]):
                    record_info.append(f"{col}: {row[col]}")
            summary_parts.append(f"Record {i+1}: {', '.join(record_info)}")
        
        return "\n".join(summary_parts)
    
    def _calculate_chunk_statistics(self, df_chunk: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed statistics for a chunk"""
        stats = {
            'row_count': len(df_chunk),
            'column_count': len(df_chunk.columns),
            'missing_data': df_chunk.isnull().sum().to_dict()
        }
        
        # Numeric statistics
        numeric_cols = df_chunk.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_stats'] = {}
            for col in numeric_cols:
                stats['numeric_stats'][col] = {
                    'mean': float(df_chunk[col].mean()),
                    'median': float(df_chunk[col].median()),
                    'std': float(df_chunk[col].std()),
                    'min': float(df_chunk[col].min()),
                    'max': float(df_chunk[col].max()),
                    'q25': float(df_chunk[col].quantile(0.25)),
                    'q75': float(df_chunk[col].quantile(0.75))
                }
        
        return stats
    
    def _analyze_chunk_trends(self, df_chunk: pd.DataFrame, start_row: int) -> Dict[str, Any]:
        """Analyze trends within a chunk"""
        trends = {
            'has_temporal_data': False,
            'trend_direction': {},
            'seasonal_patterns': {},
            'correlations': {}
        }
        
        # Check for temporal data
        date_cols = [col for col in df_chunk.columns if 'date' in col.lower() or col.lower() == 'date']
        if date_cols and pd.api.types.is_datetime64_any_dtype(df_chunk[date_cols[0]]):
            trends['has_temporal_data'] = True
            date_col = date_cols[0]
            
            # Analyze trends in numeric columns
            numeric_cols = df_chunk.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != date_col:
                    correlation = df_chunk[date_col].astype(np.int64).corr(df_chunk[col])
                    trends['trend_direction'][col] = {
                        'correlation': float(correlation),
                        'direction': 'increasing' if correlation > 0.1 else 'decreasing' if correlation < -0.1 else 'stable'
                    }
        
        # Analyze seasonal patterns for retail data
        if 'Season' in df_chunk.columns:
            season_counts = df_chunk['Season'].value_counts()
            trends['seasonal_patterns'] = season_counts.to_dict()
        
        return trends
    
    def _detect_chunk_outliers(self, df_chunk: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numeric columns"""
        outliers = {}
        
        numeric_cols = df_chunk.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df_chunk[col].quantile(0.25)
            Q3 = df_chunk[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df_chunk[col] < lower_bound) | (df_chunk[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': float((outlier_count / len(df_chunk)) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_values': df_chunk[col][outlier_mask].tolist()
                }
        
        return outliers
    
    def build_index(self, csv_files: List[str], chunk_size: int = 150):
        """
        Build enhanced FAISS index from multiple CSV files
        
        Args:
            csv_files: List of CSV file paths
            chunk_size: Number of rows per chunk (enhanced granularity)
        """
        all_chunks = []
        
        # Load all CSV files
        for csv_file in csv_files:
            chunks = self.load_csv_data(csv_file, chunk_size)
            all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        
        # Create embeddings
        logger.info("Creating embeddings...")
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings.astype('float32'))
        self.chunk_embeddings = embeddings
        
        logger.info(f"Built enhanced index with {len(all_chunks)} chunks and {dimension} dimensions")
        logger.info(f"Dataset statistics: {self.dataset_stats}")
    
    def query(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Query the enhanced RAG system with increased coverage
        
        Args:
            query: User query
            top_k: Number of top results to return (increased from 5 to 8)
            
        Returns:
            List of relevant chunks with enhanced metadata
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            chunk = self.chunks[idx]
            results.append({
                "chunk": chunk,
                "score": float(score),
                "rank": i + 1
            })
        
        return results
    
    def _get_dynamic_guidance(self, query: str, question_id: str = None) -> Dict[str, Any]:
        """Generate dynamic ground truth guidance based on query type"""
        guidance = {
            'key_points': [],
            'factual_claims': [],
            'expected_length': 'medium',
            'analysis_focus': [],
            'data_requirements': []
        }
        
        query_lower = query.lower()
        
        # Determine query type and provide appropriate guidance
        if any(word in query_lower for word in ['trend', 'pattern', 'over time', 'seasonal']):
            guidance['analysis_focus'].extend([
                'Temporal analysis and trend identification',
                'Seasonal pattern detection',
                'Growth rate calculations'
            ])
            guidance['expected_length'] = 'long'
            
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            guidance['analysis_focus'].extend([
                'Comparative analysis between groups',
                'Statistical significance testing',
                'Effect size calculations'
            ])
            guidance['expected_length'] = 'medium'
            
        elif any(word in query_lower for word in ['top', 'best', 'highest', 'most']):
            guidance['analysis_focus'].extend([
                'Ranking and prioritization',
                'Performance benchmarking',
                'Market share analysis'
            ])
            guidance['expected_length'] = 'medium'
            
        elif any(word in query_lower for word in ['risk', 'volatility', 'uncertainty']):
            guidance['analysis_focus'].extend([
                'Risk factor identification',
                'Volatility analysis',
                'Confidence interval calculations'
            ])
            guidance['expected_length'] = 'long'
            
        elif any(word in query_lower for word in ['customer', 'user', 'demographic']):
            guidance['analysis_focus'].extend([
                'Customer segmentation analysis',
                'Demographic profiling',
                'Behavioral pattern identification'
            ])
            guidance['expected_length'] = 'medium'
        
        # Add domain-specific guidance
        if question_id and question_id.startswith('retail'):
            guidance['data_requirements'].extend([
                'Product category analysis',
                'Sales performance metrics',
                'Customer behavior patterns'
            ])
        elif question_id and question_id.startswith('finance'):
            guidance['data_requirements'].extend([
                'Financial performance indicators',
                'Risk assessment metrics',
                'Market trend analysis'
            ])
        
        return guidance
    
    def generate_context(self, query: str, top_k: int = 8, question_id: str = None, ground_truth_guidance: Dict = None) -> str:
        """
        Generate enhanced context string with improved coverage and dynamic guidance
        
        Args:
            query: User query
            top_k: Number of top results to return (increased coverage)
            question_id: Optional question ID for ground truth lookup
            ground_truth_guidance: Optional ground truth information for LLM guidance
            
        Returns:
            Enhanced formatted context string
        """
        # Get relevant chunks with increased coverage
        results = self.query(query, top_k)
        
        if not results:
            return "No relevant data found."
        
        # Get dynamic guidance
        dynamic_guidance = self._get_dynamic_guidance(query, question_id)
        
        # Build enhanced context
        context_parts = []
        
        # Add enhanced ground truth guidance
        if ground_truth_guidance:
            context_parts.append("=== ENHANCED GROUND TRUTH GUIDANCE ===")
            context_parts.append(f"Question: {query}")
            context_parts.append("")
            context_parts.append("Key insights to consider:")
            for point in ground_truth_guidance.get('key_points', []):
                context_parts.append(f"• {point}")
            context_parts.append("")
            context_parts.append("Factual claims to verify:")
            for claim in ground_truth_guidance.get('factual_claims', []):
                context_parts.append(f"• {claim}")
            context_parts.append("")
            context_parts.append("Expected analysis depth: " + ground_truth_guidance.get('expected_length', 'medium'))
            context_parts.append("")
        
        # Add dynamic guidance
        context_parts.append("=== DYNAMIC ANALYSIS GUIDANCE ===")
        context_parts.append("Analysis focus areas:")
        for focus in dynamic_guidance['analysis_focus']:
            context_parts.append(f"• {focus}")
        context_parts.append("")
        context_parts.append("Data requirements:")
        for req in dynamic_guidance['data_requirements']:
            context_parts.append(f"• {req}")
        context_parts.append("")
        
        # Add dataset coverage information
        total_chunks = len(self.chunks)
        coverage_percentage = (top_k / total_chunks) * 100
        context_parts.append(f"=== ENHANCED DATASET INFORMATION ({coverage_percentage:.1f}% Coverage) ===")
        context_parts.append(f"Total dataset chunks: {total_chunks}")
        context_parts.append(f"Retrieved chunks: {len(results)}")
        context_parts.append("")
        
        # Add enhanced retrieved data chunks
        for i, result in enumerate(results):
            chunk = result["chunk"]
            score = result["score"]
            
            context_parts.append(f"[Enhanced Data Chunk {i+1}] (Relevance: {score:.3f})")
            context_parts.append(f"Source: {chunk.metadata['file_path']}")
            context_parts.append(f"Rows: {chunk.metadata['start_row']}-{chunk.metadata['end_row']}")
            context_parts.append(f"Chunk ID: {chunk.chunk_id}")
            context_parts.append("")
            
            # Add statistical summary
            if chunk.statistical_summary:
                context_parts.append("Statistical Summary:")
                for col, stats in chunk.statistical_summary.get('numeric_stats', {}).items():
                    context_parts.append(f"  {col}: μ={stats['mean']:.2f}, σ={stats['std']:.2f}, range=[{stats['min']:.2f}-{stats['max']:.2f}]")
                context_parts.append("")
            
            # Add trend analysis
            if chunk.trend_analysis.get('has_temporal_data'):
                context_parts.append("Trend Analysis:")
                for col, trend in chunk.trend_analysis.get('trend_direction', {}).items():
                    context_parts.append(f"  {col}: {trend['direction']} (r={trend['correlation']:.3f})")
                context_parts.append("")
            
            # Add outlier information
            if chunk.outlier_info:
                context_parts.append("Outlier Detection:")
                for col, outlier in chunk.outlier_info.items():
                    context_parts.append(f"  {col}: {outlier['count']} outliers ({outlier['percentage']:.1f}%)")
                context_parts.append("")
            
            context_parts.append(chunk.content)
            context_parts.append("")
        
        # Add cross-chunk analysis instructions
        context_parts.append("=== CROSS-CHUNK ANALYSIS INSTRUCTIONS ===")
        context_parts.append("Use the enhanced dataset information above to provide a comprehensive business analysis.")
        context_parts.append("Consider patterns across multiple chunks and identify overarching trends.")
        context_parts.append("Focus on accuracy, relevance, and actionable insights for the business question.")
        context_parts.append("")
        context_parts.append("Key analysis requirements:")
        context_parts.append("• Synthesize information across multiple data chunks")
        context_parts.append("• Identify patterns and trends in the data")
        context_parts.append("• Provide specific, actionable business insights")
        context_parts.append("• Support conclusions with data evidence")
        
        context = "\n".join(context_parts)
        
        logger.info(f"Generated enhanced context with {len(results)} chunks and dynamic guidance")
        return context


class EnhancedCSVBlindTestGenerator:
    """Enhanced blind test generator with improved RAG capabilities"""
    
    def __init__(self, csv_files: List[str]):
        """
        Initialize enhanced CSV blind test generator
        
        Args:
            csv_files: List of CSV file paths to use for RAG
        """
        self.csv_files = csv_files
        self.rag_pipeline = EnhancedCSVRAGPipeline()
        self.provider_manager = ProviderManager()
        
    def setup_rag(self):
        """Set up the enhanced RAG pipeline with CSV data"""
        logger.info("Setting up enhanced CSV-based RAG pipeline...")
        self.rag_pipeline.build_index(self.csv_files, chunk_size=150)  # Smaller chunks for better granularity
        logger.info("Enhanced RAG pipeline setup complete!")
    
    def generate_response_with_rag(self, question: Dict, model_name: str, ground_truth_guidance: Dict = None) -> Dict[str, Any]:
        """
        Generate enhanced response using RAG with improved coverage
        
        Args:
            question: Question dictionary with 'question' and 'question_id' keys
            model_name: Name of the LLM model to use
            ground_truth_guidance: Optional ground truth information for guidance
            
        Returns:
            Dictionary with enhanced response and metadata
        """
        try:
            # Get the question text and ID
            query = question.get('question', '')
            question_id = question.get('question_id', '')
            
            if not query:
                return {
                    'response': 'No question provided.',
                    'rag_context': '',
                    'model_name': model_name,
                    'error': 'No question provided'
                }
            
            # Generate enhanced context with increased coverage
            rag_context = self.rag_pipeline.generate_context(
                query=query,
                top_k=8,  # Increased from 5 to 8 for better coverage
                question_id=question_id,
                ground_truth_guidance=ground_truth_guidance
            )
            
            # Get LLM provider
            provider = self.provider_manager.get_provider_for_model(model_name)
            if not provider:
                return {
                    'response': f'Model {model_name} not available.',
                    'rag_context': rag_context,
                    'model_name': model_name,
                    'error': f'Model {model_name} not available'
                }
            
            # Generate response with increased token limit using provider manager
            start_time = time.time()
            llm_response = self.provider_manager.generate_response(
                provider_name=provider.provider_name,
                query=query,
                context=rag_context,
                model=model_name,
                max_tokens=750  # Increased from 500 to 750 for more detailed responses
            )
            response_time = time.time() - start_time
            
            # Check if response is successful
            if not llm_response.get('success', False):
                return {
                    'response': f"Error: {llm_response.get('error', 'Unknown error')}",
                    'rag_context': rag_context,
                    'model_name': model_name,
                    'error': llm_response.get('error', 'Unknown error')
                }
            
            return {
                'response': llm_response.get('response', ''),
                'rag_context': rag_context,
                'model_name': model_name,
                'response_time_ms': response_time * 1000,
                'tokens_used': llm_response.get('token_count', 0),
                'ground_truth_guidance_used': ground_truth_guidance is not None,
                'enhanced_coverage': True,
                'chunks_used': 8  # Track number of chunks used
            }
            
        except Exception as e:
            logger.error(f"Error generating enhanced response for {model_name}: {e}")
            return {
                'response': f'Error generating response: {str(e)}',
                'rag_context': '',
                'model_name': model_name,
                'error': str(e)
            }
    
    def _generate_fallback_response(self, prompt: str, rag_context: str) -> str:
        """Generate a fallback response when LLM providers fail"""
        return f"Based on the provided data and business scenario, I recommend implementing data-driven strategies that leverage the insights from the analysis. The available information suggests focusing on key performance indicators and optimizing business processes for better outcomes."
    
    def generate_all_responses(self, questions: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Generate responses for all questions using RAG with real LLMs
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            Dictionary with industry -> list of questions with responses
        """
        # Set up RAG pipeline
        self.setup_rag()
        
        # Models to use (these should match the models available in your providers)
        models = [
            "llama3-8b-8192",
            "gemini-1.5-flash", 
            "mistralai/mistral-7b-instruct",
            "qwen-qwq-32b",
            "gemma-3-12b-it",
            "deepseek/deepseek-r1-0528-qwen3-8b"
        ]
        
        results = {}
        
        for industry, industry_questions in questions.items():
            logger.info(f"Generating responses for {industry} industry...")
            industry_results = []
            
            for question in industry_questions:
                responses = []
                
                for i, model in enumerate(models):
                    logger.info(f"Generating response for {model}...")
                    response = self.generate_response_with_rag(question, model)
                    response["id"] = f"{question['id']}_response_{i+1}"
                    responses.append(response)
                
                question_data = {
                    "prompt": question["prompt"],
                    "context": question["context"],
                    "responses": responses
                }
                industry_results.append(question_data)
            
            results[industry] = industry_results
        
        return results


def main():
    """Test the CSV RAG pipeline with real LLMs"""
    # Test with the available CSV files
    csv_files = [
        "data/shopping_trends.csv",
        "data/uploaded_data.csv"
    ]
    
    # Load questions
    with open("data/rag_datasets.json", 'r') as f:
        questions = json.load(f)
    
    # Generate responses
    generator = EnhancedCSVBlindTestGenerator(csv_files)
    results = generator.generate_all_responses(questions)
    
    # Save results
    with open("data/blind_responses_csv_rag.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ CSV RAG responses generated successfully with real LLMs!")


if __name__ == "__main__":
    main() 