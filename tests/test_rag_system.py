"""
Comprehensive unit tests for the RAG system.
"""

import asyncio
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag.pipeline import RAGPipeline
from rag.retrieval import DocumentRetriever
from rag.vector_store import FAISSVectorStore


class TestVectorStore:
    """Test the VectorStore class."""

    @pytest.fixture
    def vector_store(self, temp_data_dir):
        """Create a VectorStore instance for testing."""
        return FAISSVectorStore(data_dir=temp_data_dir)

    def test_vector_store_initialization(self, vector_store):
        """Test VectorStore initialization."""
        assert vector_store is not None
        assert hasattr(vector_store, "add_documents")
        assert hasattr(vector_store, "search")
        assert hasattr(vector_store, "save")
        assert hasattr(vector_store, "load")

    def test_add_documents(self, vector_store, sample_documents):
        """Test adding documents to the vector store."""
        # Test adding single document
        vector_store.add_documents([sample_documents[0]])
        assert len(vector_store.documents) == 1

        # Test adding multiple documents
        vector_store.add_documents(sample_documents[1:])
        assert len(vector_store.documents) == len(sample_documents)

    def test_search_documents(self, vector_store, sample_documents):
        """Test searching documents in the vector store."""
        # Add documents first
        vector_store.add_documents(sample_documents)

        # Test search
        query = "retail KPIs"
        results = vector_store.search(query, top_k=3)

        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(isinstance(result, dict) for result in results)
        assert all("content" in result for result in results)
        assert all("score" in result for result in results)

    def test_save_and_load(self, vector_store, sample_documents, temp_data_dir):
        """Test saving and loading the vector store."""
        # Add documents
        vector_store.add_documents(sample_documents)

        # Save
        vector_store.save()

        # Create new instance and load
        new_vector_store = FAISSVectorStore(data_dir=temp_data_dir)
        new_vector_store.load()

        # Verify documents are loaded
        assert len(new_vector_store.documents) == len(sample_documents)

        # Test search still works
        query = "retail KPIs"
        results = new_vector_store.search(query, top_k=3)
        assert len(results) > 0

    def test_document_chunking(self, vector_store):
        """Test document chunking functionality."""
        long_document = "This is a very long document. " * 100

        chunks = vector_store.chunk_document(
            long_document, chunk_size=100, chunk_overlap=20
        )

        assert isinstance(chunks, list)
        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)

    def test_embedding_generation(self, vector_store):
        """Test embedding generation."""
        text = "This is a test document for embedding generation."

        embedding = vector_store.generate_embedding(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] > 0  # Should have some dimension

    def test_similarity_calculation(self, vector_store):
        """Test similarity calculation between embeddings."""
        text1 = "Retail KPIs include sales per square foot."
        text2 = "Retail performance indicators include sales metrics."
        text3 = "Healthcare compliance requires HIPAA adherence."

        embedding1 = vector_store.generate_embedding(text1)
        embedding2 = vector_store.generate_embedding(text2)
        embedding3 = vector_store.generate_embedding(text3)

        # Similar texts should have higher similarity
        similarity_12 = vector_store.calculate_similarity(embedding1, embedding2)
        similarity_13 = vector_store.calculate_similarity(embedding1, embedding3)

        assert 0.0 <= similarity_12 <= 1.0
        assert 0.0 <= similarity_13 <= 1.0
        assert similarity_12 > similarity_13  # Similar topics should be more similar

    def test_empty_vector_store(self, vector_store):
        """Test behavior with empty vector store."""
        # Search in empty store
        results = vector_store.search("test query")
        assert results == []

        # Save empty store
        vector_store.save()
        assert os.path.exists(os.path.join(vector_store.data_dir, "vector_store.pkl"))

    def test_invalid_queries(self, vector_store, sample_documents):
        """Test behavior with invalid queries."""
        vector_store.add_documents(sample_documents)

        # Empty query
        results = vector_store.search("")
        assert results == []

        # Very long query
        long_query = "test " * 1000
        results = vector_store.search(long_query)
        assert isinstance(results, list)

    def test_top_k_parameter(self, vector_store, sample_documents):
        """Test top_k parameter in search."""
        vector_store.add_documents(sample_documents)

        query = "retail"

        # Test different top_k values
        for top_k in [1, 3, 5, 10]:
            results = vector_store.search(query, top_k=top_k)
            assert len(results) <= top_k


class TestDocumentRetriever:
    """Test the DocumentRetriever class."""

    @pytest.fixture
    def document_retriever(self, temp_data_dir):
        """Create a DocumentRetriever instance for testing."""
        return DocumentRetriever(data_dir=temp_data_dir)

    def test_document_retriever_initialization(self, document_retriever):
        """Test DocumentRetriever initialization."""
        assert document_retriever is not None
        assert hasattr(document_retriever, "retrieve_documents")
        assert hasattr(document_retriever, "add_documents")

    def test_add_documents(self, document_retriever, sample_documents):
        """Test adding documents to the retriever."""
        document_retriever.add_documents(sample_documents)

        assert len(document_retriever.vector_store.documents) == len(sample_documents)

    @pytest.mark.asyncio
    async def test_retrieve_documents(self, document_retriever, sample_documents):
        """Test document retrieval."""
        # Add documents
        document_retriever.add_documents(sample_documents)

        # Test retrieval
        query = "retail KPIs"
        results = await document_retriever.retrieve_documents(query, top_k=3)

        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(isinstance(result, dict) for result in results)
        assert all("content" in result for result in results)
        assert all("score" in result for result in results)

    def test_document_preprocessing(self, document_retriever):
        """Test document preprocessing."""
        raw_document = "This is a TEST document with UPPERCASE and punctuation!!!"

        processed = document_retriever.preprocess_document(raw_document)

        assert isinstance(processed, str)
        assert "test" in processed.lower()
        assert len(processed) > 0

    def test_document_filtering(self, document_retriever):
        """Test document filtering."""
        documents = [
            "Retail KPIs include sales per square foot.",
            "Financial risk management involves credit risk.",
            "Healthcare compliance requires HIPAA adherence.",
            "This is an irrelevant document about cooking.",
        ]

        # Filter by relevance to retail
        filtered = document_retriever.filter_documents(documents, "retail")

        assert len(filtered) < len(documents)
        assert any("retail" in doc.lower() for doc in filtered)

    def test_document_ranking(self, document_retriever, sample_documents):
        """Test document ranking."""
        document_retriever.add_documents(sample_documents)

        query = "retail performance"
        ranked_docs = document_retriever.rank_documents(query, sample_documents)

        assert isinstance(ranked_docs, list)
        assert len(ranked_docs) == len(sample_documents)
        assert all("score" in doc for doc in ranked_docs)

        # Check that documents are sorted by score (descending)
        scores = [doc["score"] for doc in ranked_docs]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_retrieve_with_context(self, document_retriever, sample_documents):
        """Test document retrieval with context."""
        document_retriever.add_documents(sample_documents)

        query = "KPIs"
        context = "retail"

        results = await document_retriever.retrieve_documents_with_context(
            query=query, context=context, top_k=3
        )

        assert isinstance(results, list)
        assert len(results) <= 3

        # Results should be relevant to the context
        for result in results:
            assert (
                "retail" in result["content"].lower()
                or "sales" in result["content"].lower()
            )

    def test_document_metadata(self, document_retriever):
        """Test document metadata handling."""
        documents_with_metadata = [
            {
                "content": "Retail KPIs include sales per square foot.",
                "metadata": {"source": "retail_guide", "date": "2024-01-01"},
            },
            {
                "content": "Financial risk management involves credit risk.",
                "metadata": {"source": "finance_guide", "date": "2024-01-02"},
            },
        ]

        document_retriever.add_documents_with_metadata(documents_with_metadata)

        # Verify metadata is preserved
        assert len(document_retriever.vector_store.documents) == 2
        assert hasattr(document_retriever.vector_store.documents[0], "metadata")


class TestRAGPipeline:
    """Test the RAGPipeline class."""

    @pytest.fixture
    def rag_pipeline(self, temp_data_dir):
        """Create a RAGPipeline instance for testing."""
        return RAGPipeline(data_dir=temp_data_dir)

    def test_rag_pipeline_initialization(self, rag_pipeline):
        """Test RAGPipeline initialization."""
        assert rag_pipeline is not None
        assert hasattr(rag_pipeline, "process_query")
        assert hasattr(rag_pipeline, "add_documents")
        assert hasattr(rag_pipeline, "save_pipeline")
        assert hasattr(rag_pipeline, "load_pipeline")

    @pytest.mark.asyncio
    async def test_process_query(self, rag_pipeline, sample_documents):
        """Test query processing through the RAG pipeline."""
        # Add documents to the pipeline
        rag_pipeline.add_documents(sample_documents)

        # Process a query
        query = "What are retail KPIs?"
        context = "retail"

        result = await rag_pipeline.process_query(query, context)

        assert isinstance(result, dict)
        assert "query" in result
        assert "context" in result
        assert "retrieved_documents" in result
        assert "generated_response" in result
        assert "metadata" in result

    def test_add_documents(self, rag_pipeline, sample_documents):
        """Test adding documents to the RAG pipeline."""
        rag_pipeline.add_documents(sample_documents)

        assert len(rag_pipeline.retriever.vector_store.documents) == len(
            sample_documents
        )

    def test_save_and_load_pipeline(
        self, rag_pipeline, sample_documents, temp_data_dir
    ):
        """Test saving and loading the RAG pipeline."""
        # Add documents
        rag_pipeline.add_documents(sample_documents)

        # Save pipeline
        rag_pipeline.save_pipeline()

        # Create new pipeline and load
        new_pipeline = RAGPipeline(data_dir=temp_data_dir)
        new_pipeline.load_pipeline()

        # Verify documents are loaded
        assert len(new_pipeline.retriever.vector_store.documents) == len(
            sample_documents
        )

    @pytest.mark.asyncio
    async def test_pipeline_with_llm_integration(
        self, rag_pipeline, sample_documents, mock_llm_response
    ):
        """Test RAG pipeline with LLM integration."""
        # Add documents
        rag_pipeline.add_documents(sample_documents)

        # Mock LLM provider
        with patch.object(rag_pipeline, "llm_provider") as mock_llm:
            mock_llm.generate_response.return_value = mock_llm_response

            # Process query
            query = "What are retail KPIs?"
            context = "retail"

            result = await rag_pipeline.process_query(query, context)

            assert "generated_response" in result
            assert result["generated_response"] == mock_llm_response["response"]

    def test_document_chunking_in_pipeline(self, rag_pipeline):
        """Test document chunking in the RAG pipeline."""
        long_document = "This is a very long document about retail KPIs. " * 50

        chunks = rag_pipeline.chunk_document(long_document)

        assert isinstance(chunks, list)
        assert len(chunks) > 1
        assert all(len(chunk) <= rag_pipeline.chunk_size for chunk in chunks)

    def test_context_enhancement(self, rag_pipeline, sample_documents):
        """Test context enhancement in the RAG pipeline."""
        rag_pipeline.add_documents(sample_documents)

        query = "KPIs"
        base_context = "retail"

        enhanced_context = rag_pipeline.enhance_context(query, base_context)

        assert isinstance(enhanced_context, str)
        assert len(enhanced_context) > len(base_context)
        assert "retail" in enhanced_context.lower()

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, rag_pipeline):
        """Test error handling in the RAG pipeline."""
        # Test with empty query
        with pytest.raises(ValueError):
            await rag_pipeline.process_query("", "retail")

        # Test with invalid context
        with pytest.raises(ValueError):
            await rag_pipeline.process_query("test query", "invalid_context")

    def test_pipeline_configuration(self, rag_pipeline):
        """Test RAG pipeline configuration."""
        # Test default configuration
        assert rag_pipeline.chunk_size > 0
        assert rag_pipeline.chunk_overlap >= 0
        assert rag_pipeline.top_k > 0

        # Test configuration update
        new_config = {"chunk_size": 500, "chunk_overlap": 100, "top_k": 10}

        rag_pipeline.update_configuration(new_config)

        assert rag_pipeline.chunk_size == 500
        assert rag_pipeline.chunk_overlap == 100
        assert rag_pipeline.top_k == 10


class TestRAGIntegration:
    """Integration tests for the RAG system."""

    @pytest.mark.asyncio
    async def test_full_rag_pipeline(
        self, temp_data_dir, sample_documents, mock_llm_response
    ):
        """Test the complete RAG pipeline."""
        # Create pipeline
        pipeline = RAGPipeline(data_dir=temp_data_dir)

        # Add documents
        pipeline.add_documents(sample_documents)

        # Mock LLM provider
        with patch.object(pipeline, "llm_provider") as mock_llm:
            mock_llm.generate_response.return_value = mock_llm_response

            # Process query
            query = "What are retail KPIs?"
            context = "retail"

            result = await pipeline.process_query(query, context)

            # Verify result structure
            assert isinstance(result, dict)
            assert "query" in result
            assert "context" in result
            assert "retrieved_documents" in result
            assert "generated_response" in result
            assert "metadata" in result

            # Verify retrieved documents
            assert len(result["retrieved_documents"]) > 0
            assert all("content" in doc for doc in result["retrieved_documents"])
            assert all("score" in doc for doc in result["retrieved_documents"])

            # Verify generated response
            assert result["generated_response"] == mock_llm_response["response"]

    def test_vector_store_persistence(self, temp_data_dir, sample_documents):
        """Test vector store persistence across sessions."""
        # Create first vector store
        vector_store1 = FAISSVectorStore(data_dir=temp_data_dir)
        vector_store1.add_documents(sample_documents)
        vector_store1.save()

        # Create second vector store and load
        vector_store2 = FAISSVectorStore(data_dir=temp_data_dir)
        vector_store2.load()

        # Verify documents are loaded
        assert len(vector_store2.documents) == len(sample_documents)

        # Test search functionality
        query = "retail KPIs"
        results1 = vector_store1.search(query, top_k=3)
        results2 = vector_store2.search(query, top_k=3)

        # Results should be identical
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1["content"] == r2["content"]
            assert abs(r1["score"] - r2["score"]) < 1e-6

    def test_document_retrieval_consistency(self, temp_data_dir, sample_documents):
        """Test that document retrieval is consistent."""
        retriever = DocumentRetriever(data_dir=temp_data_dir)
        retriever.add_documents(sample_documents)

        query = "retail KPIs"

        # Test multiple retrievals
        results1 = asyncio.run(retriever.retrieve_documents(query, top_k=3))
        results2 = asyncio.run(retriever.retrieve_documents(query, top_k=3))

        # Results should be identical
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1["content"] == r2["content"]
            assert abs(r1["score"] - r2["score"]) < 1e-6


class TestRAGPerformance:
    """Performance tests for the RAG system."""

    def test_vector_store_performance(self, temp_data_dir):
        """Test vector store performance with large datasets."""
        import time

        vector_store = FAISSVectorStore(data_dir=temp_data_dir)

        # Create large dataset
        large_documents = [
            f"This is document number {i} about various topics." for i in range(1000)
        ]

        # Test adding documents
        start_time = time.time()
        vector_store.add_documents(large_documents)
        add_time = time.time() - start_time

        # Adding should be reasonably fast (under 30 seconds)
        assert add_time < 30.0

        # Test search performance
        start_time = time.time()
        results = vector_store.search("document", top_k=10)
        search_time = time.time() - start_time

        # Search should be fast (under 5 seconds)
        assert search_time < 5.0
        assert len(results) == 10

    def test_document_retrieval_performance(self, temp_data_dir):
        """Test document retrieval performance."""
        import time

        retriever = DocumentRetriever(data_dir=temp_data_dir)

        # Add large dataset
        large_documents = [
            f"This is document number {i} about various topics." for i in range(500)
        ]
        retriever.add_documents(large_documents)

        # Test retrieval performance
        start_time = time.time()
        results = asyncio.run(retriever.retrieve_documents("document", top_k=10))
        retrieval_time = time.time() - start_time

        # Retrieval should be fast (under 3 seconds)
        assert retrieval_time < 3.0
        assert len(results) == 10

    def test_pipeline_performance(self, temp_data_dir, mock_llm_response):
        """Test RAG pipeline performance."""
        import time

        pipeline = RAGPipeline(data_dir=temp_data_dir)

        # Add documents
        documents = [
            f"This is document number {i} about various topics." for i in range(100)
        ]
        pipeline.add_documents(documents)

        # Mock LLM provider
        with patch.object(pipeline, "llm_provider") as mock_llm:
            mock_llm.generate_response.return_value = mock_llm_response

            # Test pipeline performance
            start_time = time.time()
            result = asyncio.run(pipeline.process_query("document", "general"))
            pipeline_time = time.time() - start_time

            # Pipeline should be reasonably fast (under 10 seconds)
            assert pipeline_time < 10.0
            assert "generated_response" in result


class TestRAGSecurity:
    """Security tests for the RAG system."""

    def test_input_sanitization(self, temp_data_dir, malicious_inputs):
        """Test that malicious inputs are properly sanitized."""
        vector_store = FAISSVectorStore(data_dir=temp_data_dir)

        for malicious_input in malicious_inputs:
            # Test that malicious input doesn't cause issues
            try:
                embedding = vector_store.generate_embedding(malicious_input)
                assert isinstance(embedding, np.ndarray)
            except Exception as e:
                # If exception is raised, it should be handled gracefully
                assert "security" in str(e).lower() or "invalid" in str(e).lower()

    def test_file_path_validation(self, temp_data_dir):
        """Test that file paths are properly validated."""
        vector_store = FAISSVectorStore(data_dir=temp_data_dir)

        # Test with valid path
        valid_path = os.path.join(temp_data_dir, "test.pkl")
        assert vector_store.validate_file_path(valid_path) is True

        # Test with malicious path
        malicious_path = "../../../etc/passwd"
        assert vector_store.validate_file_path(malicious_path) is False

    def test_document_content_validation(self, temp_data_dir):
        """Test that document content is properly validated."""
        vector_store = FAISSVectorStore(data_dir=temp_data_dir)

        # Test valid document
        valid_doc = "This is a valid document about retail KPIs."
        assert vector_store.validate_document(valid_doc) is True

        # Test invalid documents
        invalid_docs = [
            "",  # Empty document
            "a" * 10000,  # Too long document
            None,  # None document
        ]

        for invalid_doc in invalid_docs:
            assert vector_store.validate_document(invalid_doc) is False

    def test_embedding_security(self, temp_data_dir):
        """Test that embeddings are generated securely."""
        vector_store = FAISSVectorStore(data_dir=temp_data_dir)

        # Test with normal text
        normal_text = "This is normal text for embedding."
        embedding = vector_store.generate_embedding(normal_text)
        assert isinstance(embedding, np.ndarray)
        assert not np.any(np.isnan(embedding))  # No NaN values
        assert not np.any(np.isinf(embedding))  # No infinite values

        # Test with potentially problematic text
        problematic_text = (
            "This text contains special characters: <script>alert('xss')</script>"
        )
        embedding = vector_store.generate_embedding(problematic_text)
        assert isinstance(embedding, np.ndarray)
        assert not np.any(np.isnan(embedding))
        assert not np.any(np.isinf(embedding))
