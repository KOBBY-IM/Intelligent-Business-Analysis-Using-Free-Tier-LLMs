#!/usr/bin/env python3
"""
FAISS Vector Store Implementation
"""

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of text with metadata"""

    text: str
    chunk_id: str
    source_document: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any]


class FAISSVectorStore:
    """FAISS-based vector store for document retrieval"""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        index_type: str = "flat",
    ):
        """
        Initialize FAISS vector store

        Args:
            model_name: Sentence transformer model name
            chunk_size: Number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            index_type: FAISS index type ('flat', 'ivf', 'hnsw')
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_type = index_type

        # Initialize sentence transformer
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.chunk_embeddings: Optional[np.ndarray] = None

        logger.info(
            f"Initialized FAISS vector store with embedding dimension: {self.embedding_dim}"
        )

    def _create_chunks(
        self, text: str, source_document: str, metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        """
        Create chunks from text with overlap

        Args:
            text: Input text to chunk
            source_document: Source document name
            metadata: Additional metadata

        Returns:
            List of document chunks
        """
        if metadata is None:
            metadata = {}

        # Simple tokenization (split by words)
        words = text.split()
        chunks = []

        if len(words) <= self.chunk_size:
            # Single chunk for short text
            chunk_text = text
            chunk = DocumentChunk(
                text=chunk_text,
                chunk_id=f"{source_document}_chunk_0",
                source_document=source_document,
                start_index=0,
                end_index=len(words),
                metadata=metadata.copy(),
            )
            chunks.append(chunk)
        else:
            # Multiple chunks with overlap
            start = 0
            chunk_id = 0

            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)

                chunk = DocumentChunk(
                    text=chunk_text,
                    chunk_id=f"{source_document}_chunk_{chunk_id}",
                    source_document=source_document,
                    start_index=start,
                    end_index=end,
                    metadata=metadata.copy(),
                )
                chunks.append(chunk)

                # Move start position with overlap
                start = end - self.chunk_overlap
                chunk_id += 1

                # Ensure we don't get stuck in infinite loop
                if start >= len(words) - 1:
                    break

        logger.info(f"Created {len(chunks)} chunks from {source_document}")
        return chunks

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector store

        Args:
            documents: List of documents with 'text' and 'metadata' keys
        """
        logger.info(f"Adding {len(documents)} documents to vector store")

        # Create chunks from all documents
        all_chunks = []
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            source_document = doc.get("source", "unknown")

            if text.strip():
                chunks = self._create_chunks(text, source_document, metadata)
                all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks created from documents")
            return

        # Generate embeddings for all chunks
        chunk_texts = [chunk.text for chunk in all_chunks]
        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks")

        embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True)

        # Store chunks and embeddings
        self.chunks.extend(all_chunks)

        if self.chunk_embeddings is None:
            self.chunk_embeddings = embeddings
        else:
            self.chunk_embeddings = np.vstack([self.chunk_embeddings, embeddings])

        # Create or update FAISS index
        self._build_index()

        logger.info(f"Successfully added {len(all_chunks)} chunks to vector store")

    def _build_index(self):
        """Build FAISS index from embeddings"""
        if self.chunk_embeddings is None or len(self.chunk_embeddings) == 0:
            logger.warning("No embeddings available to build index")
            return

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.chunk_embeddings)

        # Create index based on type
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "ivf":
            nlist = min(100, len(self.chunk_embeddings) // 10)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.train(self.chunk_embeddings)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.index.hnsw.efConstruction = 200
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        # Add vectors to index
        self.index.add(self.chunk_embeddings.astype("float32"))

        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar documents

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.index is None or len(self.chunks) == 0:
            logger.warning("No documents indexed for search")
            return []

        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search index
        scores, indices = self.index.search(query_embedding.astype("float32"), k)

        # Return results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append((chunk, float(score)))

        logger.info(
            f"Search returned {len(results)} results for query: {query[:50]}..."
        )
        return results

    def save(self, directory: str):
        """Save vector store to directory"""
        os.makedirs(directory, exist_ok=True)

        # Save index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, "faiss_index.bin"))

        # Save chunks and embeddings
        with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

        if self.chunk_embeddings is not None:
            np.save(os.path.join(directory, "embeddings.npy"), self.chunk_embeddings)

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
        }

        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved vector store to {directory}")

    def load(self, directory: str):
        """Load vector store from directory"""
        # Load metadata
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        self.model_name = metadata["model_name"]
        self.chunk_size = metadata["chunk_size"]
        self.chunk_overlap = metadata["chunk_overlap"]
        self.index_type = metadata["index_type"]
        self.embedding_dim = metadata["embedding_dim"]

        # Load embedding model
        self.embedding_model = SentenceTransformer(self.model_name)

        # Load chunks
        with open(os.path.join(directory, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)

        # Load embeddings
        embeddings_path = os.path.join(directory, "embeddings.npy")
        if os.path.exists(embeddings_path):
            self.chunk_embeddings = np.load(embeddings_path)

        # Load index
        index_path = os.path.join(directory, "faiss_index.bin")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)

        logger.info(
            f"Loaded vector store from {directory} with {len(self.chunks)} chunks"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        stats = {
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "index_type": self.index_type,
        }

        if self.index is not None:
            stats["index_size"] = self.index.ntotal

        if self.chunk_embeddings is not None:
            stats["embeddings_shape"] = self.chunk_embeddings.shape

        return stats
