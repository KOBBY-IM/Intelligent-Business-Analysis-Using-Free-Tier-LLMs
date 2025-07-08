#!/usr/bin/env python3
"""
Complete RAG Pipeline Implementation with Unified LLM Providers
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .retrieval import DocumentRetriever, RetrievalResult
from .vector_store import FAISSVectorStore

try:
    from ..llm_providers.base_provider import BaseProvider, LLMResponse
except ImportError:
    # Fallback for when running as script
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from llm_providers.base_provider import BaseProvider, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Complete RAG pipeline result"""

    query: str
    response: str
    context_used: str
    retrieval_result: RetrievalResult
    llm_response: LLMResponse
    total_time_ms: float
    pipeline_metrics: Dict[str, Any]


class RAGPipeline:
    """Complete RAG pipeline with vector store, retrieval, and LLM generation"""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        llm_provider: BaseProvider,
        top_k: int = 5,
        max_context_length: int = 2000,
    ):
        """
        Initialize RAG pipeline

        Args:
            vector_store: FAISS vector store instance
            llm_provider: Unified LLM provider instance
            top_k: Number of documents to retrieve
            max_context_length: Maximum context length for LLM
        """
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.retriever = DocumentRetriever(vector_store, top_k=top_k)
        self.max_context_length = max_context_length

        logger.info(
            f"Initialized RAG pipeline with top_k={top_k}, max_context_length={max_context_length}"
        )

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        model: Optional[str] = None,
    ) -> RAGResult:
        """
        Execute complete RAG pipeline

        Args:
            query: User query
            top_k: Number of documents to retrieve (overrides default)
            max_tokens: Maximum tokens for LLM response
            temperature: Sampling temperature for LLM
            model: Specific model to use (optional)

        Returns:
            RAGResult with complete pipeline output
        """
        start_time = time.time()

        # Step 1: Document Retrieval
        logger.info(f"Starting RAG pipeline for query: {query[:50]}...")

        retrieval_start = time.time()
        retrieval_result = self.retriever.retrieve(query, top_k=top_k)
        retrieval_time = (time.time() - retrieval_start) * 1000

        # Step 2: Context Formatting
        context_start = time.time()
        context = self.retriever.format_context(
            retrieval_result, self.max_context_length
        )
        context_time = (time.time() - context_start) * 1000

        # Step 3: LLM Generation
        llm_start = time.time()
        llm_response = self.llm_provider.generate_response(
            query=query,
            context=context,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        llm_time = (time.time() - llm_start) * 1000

        # Calculate total time
        total_time = (time.time() - start_time) * 1000

        # Compile metrics
        pipeline_metrics = {
            "retrieval_time_ms": retrieval_time,
            "context_formatting_time_ms": context_time,
            "llm_generation_time_ms": llm_time,
            "total_time_ms": total_time,
            "retrieval_quality": self.retriever.analyze_retrieval_quality(
                retrieval_result
            ),
            "context_length": len(context),
            "num_chunks_retrieved": len(retrieval_result.retrieved_chunks),
            "llm_tokens_used": llm_response.tokens_used,
            "llm_latency_ms": llm_response.latency_ms,
        }

        # Create result
        result = RAGResult(
            query=query,
            response=(
                llm_response.text
                if llm_response.success
                else f"Error: {llm_response.error}"
            ),
            context_used=context,
            retrieval_result=retrieval_result,
            llm_response=llm_response,
            total_time_ms=total_time,
            pipeline_metrics=pipeline_metrics,
        )

        logger.info(
            f"RAG pipeline completed in {total_time:.2f}ms - "
            f"Retrieval: {retrieval_time:.2f}ms, LLM: {llm_time:.2f}ms"
        )

        return result

    def batch_query(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        model: Optional[str] = None,
    ) -> List[RAGResult]:
        """
        Execute RAG pipeline for multiple queries

        Args:
            queries: List of user queries
            top_k: Number of documents to retrieve
            max_tokens: Maximum tokens for LLM response
            temperature: Sampling temperature for LLM
            model: Specific model to use (optional)

        Returns:
            List of RAGResult objects
        """
        results = []

        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")

            try:
                result = self.query(query, top_k, max_tokens, temperature, model)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query {i+1}: {e}")
                # Create error result
                error_result = RAGResult(
                    query=query,
                    response=f"Error: {str(e)}",
                    context_used="",
                    retrieval_result=None,
                    llm_response=None,
                    total_time_ms=0,
                    pipeline_metrics={"error": str(e)},
                )
                results.append(error_result)

        return results

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        vector_store_stats = self.vector_store.get_stats()

        stats = {
            "vector_store": vector_store_stats,
            "retriever": {
                "top_k": self.retriever.top_k,
                "max_context_length": self.max_context_length,
            },
            "llm_provider": {
                "provider_name": self.llm_provider.get_provider_name(),
                "available_models": self.llm_provider.list_models(),
                "health_status": self.llm_provider.get_last_health(),
            },
        }

        return stats

    def save_pipeline(self, directory: str):
        """Save pipeline state"""
        os.makedirs(directory, exist_ok=True)

        # Save vector store
        vector_store_dir = os.path.join(directory, "vector_store")
        self.vector_store.save(vector_store_dir)

        # Save pipeline configuration
        config = {
            "top_k": self.retriever.top_k,
            "max_context_length": self.max_context_length,
            "llm_provider": self.llm_provider.get_provider_name(),
        }

        config_path = os.path.join(directory, "pipeline_config.json")
        import json

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Pipeline saved to {directory}")

    def load_pipeline(self, directory: str):
        """Load pipeline state"""
        # Load vector store
        vector_store_dir = os.path.join(directory, "vector_store")
        self.vector_store.load(vector_store_dir)

        # Load pipeline configuration
        config_path = os.path.join(directory, "pipeline_config.json")
        if os.path.exists(config_path):
            import json

            with open(config_path, "r") as f:
                config = json.load(f)

            self.retriever.top_k = config.get("top_k", self.retriever.top_k)
            self.max_context_length = config.get(
                "max_context_length", self.max_context_length
            )

        logger.info(f"Pipeline loaded from {directory}")

    def test_pipeline(self) -> Dict[str, Any]:
        """Test pipeline functionality"""
        test_query = "What is the main topic of the documents?"

        try:
            result = self.query(test_query, max_tokens=50)

            return {
                "success": result.llm_response.success,
                "response_time_ms": result.total_time_ms,
                "retrieval_quality": result.pipeline_metrics["retrieval_quality"],
                "test_response": (
                    result.response[:100] + "..."
                    if len(result.response) > 100
                    else result.response
                ),
            }
        except Exception as e:
            logger.error(f"Pipeline test failed: {e}")
            return {"success": False, "error": str(e)}
