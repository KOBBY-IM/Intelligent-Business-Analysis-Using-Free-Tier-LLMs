#!/usr/bin/env python3
"""
RAG Pipeline Implementation
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Standard library imports
from abc import ABC, abstractmethod

# Try relative import first, fallback to direct import
try:
    from llm_providers.base_provider import BaseProvider, LLMResponse
except ImportError:
    from llm_providers.base_provider import BaseProvider, LLMResponse

logger = logging.getLogger(__name__)


class RAGPipeline(ABC):
    """Abstract base class for RAG pipelines"""
    
    @abstractmethod
    def build_index(self, data_sources: List[str], **kwargs) -> None:
        """Build the retrieval index from data sources"""
        pass
    
    @abstractmethod
    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the index and return relevant chunks"""
        pass
    
    @abstractmethod
    def generate_context(self, query: str, top_k: int = 5) -> str:
        """Generate context string from query results"""
        pass
