#!/usr/bin/env python3
"""
Unit tests for RAG pipeline components with unified LLM providers
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_providers.base_provider import LLMResponse
from llm_providers.groq_provider import GroqProvider
from rag.pipeline import RAGPipeline, RAGResult
from rag.retrieval import DocumentRetriever, RetrievalResult
from rag.vector_store import DocumentChunk, FAISSVectorStore


class TestFAISSVectorStore(unittest.TestCase):
    """Test FAISS vector store functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.vector_store = FAISSVectorStore(
            model_name="all-MiniLM-L6-v2", chunk_size=200, chunk_overlap=50
        )

        # Test documents
        self.test_documents = [
            {
                "text": "This is a test document about retail sales. It contains information about customer purchases and product categories.",
                "source": "test_doc_1",
                "metadata": {"category": "retail", "type": "sales"},
            },
            {
                "text": "Another document about e-commerce trends. Online shopping has increased significantly in recent years.",
                "source": "test_doc_2",
                "metadata": {"category": "ecommerce", "type": "trends"},
            },
        ]

    def test_chunk_creation(self):
        """Test document chunking"""
        text = (
            "This is a long document that should be split into multiple chunks. " * 20
        )
        chunks = self.vector_store._create_chunks(text, "test_doc")

        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(len(chunks[0].text.split()), self.vector_store.chunk_size)

    def test_document_addition(self):
        """Test adding documents to vector store"""
        self.vector_store.add_documents(self.test_documents)

        self.assertEqual(len(self.vector_store.chunks), 2)
        self.assertIsNotNone(self.vector_store.chunk_embeddings)
        self.assertIsNotNone(self.vector_store.index)

    def test_search_functionality(self):
        """Test search functionality"""
        self.vector_store.add_documents(self.test_documents)

        results = self.vector_store.search("retail sales", k=2)

        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 2)

        # Check that results have chunks and scores
        for chunk, score in results:
            self.assertIsInstance(chunk, DocumentChunk)
            self.assertIsInstance(score, float)

    def test_save_and_load(self):
        """Test saving and loading vector store"""
        self.vector_store.add_documents(self.test_documents)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save
            self.vector_store.save(temp_dir)

            # Create new instance and load
            new_store = FAISSVectorStore()
            new_store.load(temp_dir)

            # Verify data is loaded correctly
            self.assertEqual(len(new_store.chunks), len(self.vector_store.chunks))
            self.assertIsNotNone(new_store.index)

    def test_get_stats(self):
        """Test getting vector store statistics"""
        self.vector_store.add_documents(self.test_documents)
        stats = self.vector_store.get_stats()

        self.assertIn("total_chunks", stats)
        self.assertIn("embedding_dimension", stats)
        self.assertEqual(stats["total_chunks"], 2)


class TestDocumentRetriever(unittest.TestCase):
    """Test document retriever functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.vector_store = FAISSVectorStore(chunk_size=200)
        self.retriever = DocumentRetriever(self.vector_store, top_k=3)

        # Add test documents
        test_docs = [
            {
                "text": "Retail sales analysis shows strong performance in electronics category.",
                "source": "sales_report",
                "metadata": {"quarter": "Q1"},
            }
        ]
        self.vector_store.add_documents(test_docs)

    def test_retrieve_documents(self):
        """Test document retrieval"""
        result = self.retriever.retrieve("electronics sales")

        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(result.query, "electronics sales")
        self.assertGreater(len(result.retrieved_chunks), 0)
        self.assertGreater(result.retrieval_time_ms, 0)

    def test_format_context(self):
        """Test context formatting"""
        result = self.retriever.retrieve("electronics")
        context = self.retriever.format_context(result, max_context_length=1000)

        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)

    def test_retrieval_quality_analysis(self):
        """Test retrieval quality analysis"""
        result = self.retriever.retrieve("electronics")
        quality = self.retriever.analyze_retrieval_quality(result)

        self.assertIn("avg_similarity", quality)
        self.assertIn("diversity", quality)
        self.assertIn("coverage", quality)


class TestGroqProvider(unittest.TestCase):
    """Test Groq provider functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock the API key for testing
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
            self.provider = GroqProvider()

    @patch("requests.post")
    def test_generate_response_success(self, mock_post):
        """Test successful response generation"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"total_tokens": 10},
        }
        mock_post.return_value = mock_response

        result = self.provider.generate_response("test query")

        self.assertTrue(result.success)
        self.assertEqual(result.text, "Test response")
        self.assertEqual(result.tokens_used, 10)

    @patch("requests.post")
    def test_generate_response_with_context(self, mock_post):
        """Test response generation with context"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Context-aware response"}}]
        }
        mock_post.return_value = mock_response

        result = self.provider.generate_response("test query", context="test context")

        self.assertTrue(result.success)
        self.assertIn("Context-aware response", result.text)

    @patch("requests.post")
    def test_generate_response_error(self, mock_post):
        """Test error handling"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_post.return_value = mock_response

        result = self.provider.generate_response("test query")

        self.assertFalse(result.success)
        self.assertIn("Error", result.error)


class TestRAGPipeline(unittest.TestCase):
    """Test RAG pipeline functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.vector_store = FAISSVectorStore(chunk_size=200)

        # Add test documents
        test_docs = [
            {
                "text": "Retail sales analysis shows strong performance in electronics category.",
                "source": "sales_report",
                "metadata": {"quarter": "Q1"},
            }
        ]
        self.vector_store.add_documents(test_docs)

        # Mock Groq provider
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
            self.llm_provider = GroqProvider()

        self.pipeline = RAGPipeline(
            vector_store=self.vector_store,
            llm_provider=self.llm_provider,
            top_k=3,
            max_context_length=1000,
        )

    @patch.object(GroqProvider, "generate_response")
    def test_pipeline_query(self, mock_generate):
        """Test pipeline query functionality"""
        # Mock LLM response
        mock_response = LLMResponse(
            success=True,
            text="Test response",
            model="llama3-8b-8192",
            tokens_used=10,
            latency_ms=100,
        )
        mock_generate.return_value = mock_response

        result = self.pipeline.query("test query")

        self.assertIsInstance(result, RAGResult)
        self.assertEqual(result.query, "test query")
        self.assertEqual(result.response, "Test response")
        self.assertGreater(result.total_time_ms, 0)

    def test_pipeline_stats(self):
        """Test pipeline statistics"""
        stats = self.pipeline.get_pipeline_stats()

        self.assertIn("vector_store", stats)
        self.assertIn("retriever", stats)
        self.assertIn("llm_provider", stats)

    def test_pipeline_save_and_load(self):
        """Test pipeline save and load functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save pipeline
            self.pipeline.save_pipeline(temp_dir)

            # Create new pipeline and load
            new_pipeline = RAGPipeline(
                vector_store=FAISSVectorStore(), llm_provider=self.llm_provider
            )
            new_pipeline.load_pipeline(temp_dir)

            # Verify data is loaded
            self.assertGreater(len(new_pipeline.vector_store.chunks), 0)

    @patch.object(GroqProvider, "generate_response")
    def test_pipeline_test(self, mock_generate):
        """Test pipeline test functionality"""
        # Mock LLM response
        mock_response = LLMResponse(
            success=True,
            text="Test response",
            model="llama3-8b-8192",
            tokens_used=10,
            latency_ms=100,
        )
        mock_generate.return_value = mock_response

        test_result = self.pipeline.test_pipeline()

        self.assertIn("success", test_result)
        self.assertTrue(test_result["success"])


if __name__ == "__main__":
    unittest.main()
