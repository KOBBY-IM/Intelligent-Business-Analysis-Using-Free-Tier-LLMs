#!/usr/bin/env python3
"""
Document Retrieval Logic
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .vector_store import DocumentChunk, FAISSVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of a document retrieval operation"""

    query: str
    retrieved_chunks: List[DocumentChunk]
    similarity_scores: List[float]
    total_chunks_searched: int
    retrieval_time_ms: float
    metadata: Dict[str, Any]


class DocumentRetriever:
    """Document retriever with similarity search capabilities"""

    def __init__(self, vector_store: FAISSVectorStore, top_k: int = 5):
        """
        Initialize document retriever

        Args:
            vector_store: FAISS vector store instance
            top_k: Number of top results to retrieve
        """
        self.vector_store = vector_store
        self.top_k = top_k
        logger.info(f"Initialized document retriever with top_k={top_k}")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search query
            top_k: Number of results to return (overrides default)

        Returns:
            RetrievalResult with retrieved chunks and metadata
        """
        import time

        start_time = time.time()

        # Use provided top_k or default
        k = top_k if top_k is not None else self.top_k

        # Perform similarity search
        search_results = self.vector_store.search(query, k=k)

        # Extract chunks and scores
        chunks = [chunk for chunk, score in search_results]
        scores = [score for chunk, score in search_results]

        # Calculate retrieval time
        end_time = time.time()
        retrieval_time_ms = (end_time - start_time) * 1000

        # Create result
        result = RetrievalResult(
            query=query,
            retrieved_chunks=chunks,
            similarity_scores=scores,
            total_chunks_searched=len(self.vector_store.chunks),
            retrieval_time_ms=retrieval_time_ms,
            metadata={"top_k": k, "vector_store_stats": self.vector_store.get_stats()},
        )

        logger.info(
            f"Retrieved {len(chunks)} chunks for query '{query[:50]}...' in {retrieval_time_ms:.2f}ms"
        )
        return result

    def format_context(
        self, retrieval_result: RetrievalResult, max_context_length: int = 2000
    ) -> str:
        """
        Format retrieved chunks into context for LLM

        Args:
            retrieval_result: Result from retrieve() method
            max_context_length: Maximum context length in characters

        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        for i, (chunk, score) in enumerate(
            zip(retrieval_result.retrieved_chunks, retrieval_result.similarity_scores)
        ):
            # Add chunk with metadata
            chunk_text = f"[Document {i+1}] (Similarity: {score:.3f})\n{chunk.text}\n"

            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_text) > max_context_length:
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        context = "\n".join(context_parts)

        if not context:
            context = "No relevant documents found."

        logger.info(
            f"Formatted context with {len(context_parts)} chunks ({len(context)} characters)"
        )
        return context

    def get_relevant_sources(self, retrieval_result: RetrievalResult) -> List[str]:
        """
        Get list of relevant source documents

        Args:
            retrieval_result: Result from retrieve() method

        Returns:
            List of unique source document names
        """
        sources = set()
        for chunk in retrieval_result.retrieved_chunks:
            sources.add(chunk.source_document)

        return list(sources)

    def analyze_retrieval_quality(
        self, retrieval_result: RetrievalResult
    ) -> Dict[str, Any]:
        """
        Analyze the quality of retrieval results

        Args:
            retrieval_result: Result from retrieve() method

        Returns:
            Dictionary with quality metrics
        """
        if not retrieval_result.retrieved_chunks:
            return {
                "avg_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "coverage": 0.0,
                "diversity": 0.0,
            }

        scores = retrieval_result.similarity_scores

        # Basic statistics
        avg_similarity = sum(scores) / len(scores)
        min_similarity = min(scores)
        max_similarity = max(scores)

        # Coverage (percentage of total chunks searched)
        coverage = (
            len(retrieval_result.retrieved_chunks)
            / retrieval_result.total_chunks_searched
        )

        # Diversity (number of unique sources)
        unique_sources = len(
            set(chunk.source_document for chunk in retrieval_result.retrieved_chunks)
        )
        diversity = unique_sources / len(retrieval_result.retrieved_chunks)

        quality_metrics = {
            "avg_similarity": avg_similarity,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity,
            "coverage": coverage,
            "diversity": diversity,
            "num_chunks": len(retrieval_result.retrieved_chunks),
            "num_sources": unique_sources,
        }

        logger.info(
            f"Retrieval quality - Avg similarity: {avg_similarity:.3f}, "
            f"Diversity: {diversity:.3f}, Coverage: {coverage:.3f}"
        )

        return quality_metrics
