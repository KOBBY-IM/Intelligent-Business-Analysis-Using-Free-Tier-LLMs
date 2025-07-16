#!/usr/bin/env python3
"""
CSV-based RAG Pipeline for Blind Evaluation
Loads CSV datasets, creates embeddings, and generates responses using actual LLM providers
"""

import pandas as pd
import numpy as np
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Please install sentence-transformers and faiss-cpu: pip install sentence-transformers faiss-cpu")
    raise

# Import LLM providers
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from llm_providers.provider_manager import ProviderManager
from evaluation.ground_truth_generator import GroundTruthGenerator

logger = logging.getLogger(__name__)


@dataclass
class CSVChunk:
    """A chunk of CSV data with metadata"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str


class CSVRAGPipeline:
    """RAG pipeline that works with CSV datasets"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize CSV RAG pipeline
        
        Args:
            embedding_model: Sentence transformer model to use
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []
        self.chunk_embeddings = None
        
    def load_csv_data(self, csv_path: str, chunk_size: int = 100) -> List[CSVChunk]:
        """
        Load CSV data and create chunks
        
        Args:
            csv_path: Path to CSV file
            chunk_size: Number of rows per chunk
            
        Returns:
            List of CSVChunk objects
        """
        df = pd.read_csv(csv_path)
        chunks = []
        
        # Create chunks of data
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]
            
            # Create a summary of the chunk
            chunk_summary = self._create_chunk_summary(chunk_df, i)
            
            chunk = CSVChunk(
                content=chunk_summary,
                metadata={
                    "start_row": i,
                    "end_row": min(i + chunk_size, len(df)),
                    "columns": list(df.columns),
                    "file_path": csv_path
                },
                chunk_id=f"{Path(csv_path).stem}_chunk_{i//chunk_size}"
            )
            chunks.append(chunk)
            
        logger.info(f"Created {len(chunks)} chunks from {csv_path}")
        return chunks
    
    def _create_chunk_summary(self, df_chunk: pd.DataFrame, start_row: int) -> str:
        """Create a condensed text summary of a DataFrame chunk for optimal coverage"""
        summary_parts = []
        
        # Basic info (condensed)
        summary_parts.append(f"Rows {start_row}-{start_row + len(df_chunk) - 1} ({len(df_chunk)} records)")
        
        # Key statistics only (most important numeric columns)
        numeric_cols = df_chunk.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_summary = []
            # Focus on most business-relevant columns
            key_cols = ['Purchase Amount (USD)', 'Age', 'Review Rating', 'Previous Purchases']
            for col in key_cols:
                if col in numeric_cols:
                    mean_val = df_chunk[col].mean()
                    stats_summary.append(f"{col.replace(' (USD)', '').replace('Purchase Amount', 'Spend')}: μ={mean_val:.1f}")
            if stats_summary:
                summary_parts.append("Stats: " + ", ".join(stats_summary))
        
        # Top categorical distributions (condensed)
        categorical_cols = ['Category', 'Gender', 'Season', 'Payment Method', 'Subscription Status']
        for col in categorical_cols:
            if col in df_chunk.columns:
                top_values = df_chunk[col].value_counts().head(3)
                if len(top_values) > 0:
                    val_summary = ", ".join([f"{val}({count})" for val, count in top_values.items()])
                    summary_parts.append(f"{col}: {val_summary}")
        
        # Sample records (very condensed - only 1-2 key examples)
        sample_data = df_chunk.head(2)
        for i, (_, row) in enumerate(sample_data.iterrows()):
            key_info = []
            if 'Category' in row:
                key_info.append(str(row['Category']))
            if 'Purchase Amount (USD)' in row:
                key_info.append(f"${row['Purchase Amount (USD)']:.0f}")
            if 'Age' in row:
                key_info.append(f"Age{row['Age']}")
            if 'Gender' in row:
                key_info.append(str(row['Gender']))
            
            if key_info:
                summary_parts.append(f"Ex{i+1}: {', '.join(key_info)}")
        
        return "\n".join(summary_parts)
    
    def build_index(self, csv_files: List[str], chunk_size: int = 100):
        """
        Build FAISS index from multiple CSV files
        
        Args:
            csv_files: List of CSV file paths
            chunk_size: Number of rows per chunk
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
        
        logger.info(f"Built index with {len(all_chunks)} chunks and {dimension} dimensions")
    
    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the RAG system
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of relevant chunks with scores
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
    
    def generate_context(self, query: str, top_k: int = 5, question_id: str = None, ground_truth_guidance: Dict = None) -> str:
        """
        Generate context string from query results with optional ground truth guidance
        
        Args:
            query: User query
            top_k: Number of top results to return
            question_id: Optional question ID for ground truth lookup
            ground_truth_guidance: Optional ground truth information for LLM guidance
            
        Returns:
            Formatted context string
        """
        # Get relevant chunks
        results = self.query(query, top_k)
        
        if not results:
            return "No relevant data found."
        
        # Build context from retrieved chunks
        context_parts = []
        
        # Add ground truth guidance if provided
        if ground_truth_guidance:
            context_parts.append("=== GROUND TRUTH GUIDANCE ===")
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
            context_parts.append("=== AVAILABLE DATASET INFORMATION (40% Coverage) ===")
        
        # Add retrieved data chunks
        for i, result in enumerate(results):
            chunk = result["chunk"]
            score = result["score"]
            
            context_parts.append(f"[Data Chunk {i+1}] (Relevance: {score:.3f})")
            context_parts.append(f"Source: {chunk.metadata['file_path']}")
            context_parts.append(f"Rows: {chunk.metadata['start_row']}-{chunk.metadata['end_row']}")
            context_parts.append("")
            context_parts.append(chunk.content)
            context_parts.append("")
        
        # Add instructions for LLM
        if ground_truth_guidance:
            context_parts.append("=== INSTRUCTIONS ===")
            context_parts.append("Use the available dataset information (40% coverage) along with the ground truth insights above to provide a comprehensive business analysis.")
            context_parts.append("The ground truth serves as a reference for what information should be included and verified.")
            context_parts.append("Focus on accuracy and relevance to the business question.")
        
        context = "\n".join(context_parts)
        
        logger.info(f"Generated context with {len(results)} chunks and ground truth guidance")
        return context


class CSVBlindTestGenerator:
    """Generate blind test responses using CSV-based RAG with real LLM providers"""
    
    def __init__(self, csv_files: List[str]):
        """
        Initialize CSV blind test generator
        
        Args:
            csv_files: List of CSV file paths to use for RAG
        """
        self.csv_files = csv_files
        self.rag_pipeline = CSVRAGPipeline()
        self.provider_manager = ProviderManager()
        
    def setup_rag(self):
        """Set up the RAG pipeline with CSV data"""
        logger.info("Setting up CSV-based RAG pipeline...")
        self.rag_pipeline.build_index(self.csv_files, chunk_size=200)
        logger.info("RAG pipeline setup complete!")
    
    def generate_response_with_rag(self, question: Dict, model_name: str, ground_truth_guidance: Dict = None) -> Dict[str, Any]:
        """
        Generate response using RAG with optional ground truth guidance
        
        Args:
            question: Question dictionary with 'question' and 'question_id' keys
            model_name: Name of the LLM model to use
            ground_truth_guidance: Optional ground truth information for guidance
            
        Returns:
            Dictionary with response and metadata
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
            
            # Generate context with ground truth guidance
            rag_context = self.rag_pipeline.generate_context(
                query=query,
                top_k=5,
                question_id=question_id,
                ground_truth_guidance=ground_truth_guidance
            )
            
            # Get LLM provider
            provider = self.provider_manager.get_provider(model_name)
            if not provider:
                return {
                    'response': f'Model {model_name} not available.',
                    'rag_context': rag_context,
                    'model_name': model_name,
                    'error': f'Model {model_name} not available'
                }
            
            # Generate response
            start_time = time.time()
            llm_response = provider.generate_response(
                query=query,
                context=rag_context,
                max_tokens=500
            )
            response_time = time.time() - start_time
            
            return {
                'response': llm_response.response,
                'rag_context': rag_context,
                'model_name': model_name,
                'response_time_ms': response_time * 1000,
                'tokens_used': llm_response.tokens_used,
                'ground_truth_guidance_used': ground_truth_guidance is not None
            }
            
        except Exception as e:
            logger.error(f"Error generating response for {model_name}: {e}")
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
    generator = CSVBlindTestGenerator(csv_files)
    results = generator.generate_all_responses(questions)
    
    # Save results
    with open("data/blind_responses_csv_rag.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ CSV RAG responses generated successfully with real LLMs!")


if __name__ == "__main__":
    main() 