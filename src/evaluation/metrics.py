#!/usr/bin/env python3
"""
Comprehensive evaluation metrics for LLM performance assessment
"""

import time
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics"""
    relevance_score: float
    factual_accuracy: float
    response_time_ms: float
    token_efficiency: float
    coherence_score: float
    completeness_score: float
    overall_score: float
    confidence_interval: Tuple[float, float]

class RelevanceScorer:
    """Calculate relevance of LLM response to query and context"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
    
    def calculate_relevance(self, query: str, response: str, context: str = "") -> float:
        """
        Calculate relevance score using TF-IDF cosine similarity
        
        Args:
            query: Original user query
            response: LLM response
            context: Retrieved context (optional)
            
        Returns:
            Relevance score between 0 and 1
        """
        try:
            # Combine query and context for comparison
            reference_text = query
            if context:
                reference_text = f"{query} {context}"
            
            # Vectorize texts
            texts = [reference_text, response]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.0

class FactualAccuracyScorer:
    """Assess factual accuracy of responses against ground truth"""
    
    def __init__(self):
        self.key_phrase_patterns = [
            r'\b\d+(?:\.\d+)?%?\b',  # Numbers and percentages
            r'\b\d{4}\b',  # Years
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # Currency amounts
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'  # Proper nouns
        ]
    
    def extract_facts(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        facts = []
        
        # Extract key phrases
        for pattern in self.key_phrase_patterns:
            matches = re.findall(pattern, text)
            facts.extend(matches)
        
        # Extract sentences with factual content (simplified approach)
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and any(word in sentence.lower() for word in ['is', 'are', 'was', 'were', 'has', 'have', 'had']):
                facts.append(sentence)
        
        return facts
    
    def calculate_factual_accuracy(self, response: str, ground_truth: str) -> float:
        """
        Calculate factual accuracy by comparing extracted facts
        
        Args:
            response: LLM response
            ground_truth: Reference answer
            
        Returns:
            Factual accuracy score between 0 and 1
        """
        try:
            response_facts = set(self.extract_facts(response))
            truth_facts = set(self.extract_facts(ground_truth))
            
            if not truth_facts:
                return 0.0
            
            # Calculate precision and recall
            correct_facts = response_facts.intersection(truth_facts)
            precision = len(correct_facts) / len(response_facts) if response_facts else 0.0
            recall = len(correct_facts) / len(truth_facts)
            
            # F1 score
            if precision + recall == 0:
                return 0.0
            
            f1_score = 2 * (precision * recall) / (precision + recall)
            return f1_score
            
        except Exception as e:
            logger.error(f"Error calculating factual accuracy: {e}")
            return 0.0

class TokenEfficiencyScorer:
    """Calculate token efficiency metrics"""
    
    def calculate_token_efficiency(self, response: str, tokens_used: int, query: str) -> float:
        """
        Calculate token efficiency (information density)
        
        Args:
            response: LLM response
            tokens_used: Number of tokens consumed
            query: Original query
            
        Returns:
            Token efficiency score (higher is better)
        """
        try:
            if tokens_used == 0:
                return 0.0
            
            # Calculate information density (simplified approach)
            word_count = len(response.split())
            query_word_count = len(query.split())
            
            # Efficiency = (response words - query words) / tokens used
            # This measures how much new information we get per token
            efficiency = (word_count - query_word_count) / tokens_used
            
            # Normalize to 0-1 scale (assuming reasonable efficiency range)
            normalized_efficiency = min(max(efficiency / 0.5, 0.0), 1.0)
            
            return normalized_efficiency
            
        except Exception as e:
            logger.error(f"Error calculating token efficiency: {e}")
            return 0.0

class CoherenceScorer:
    """Assess coherence and fluency of responses"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def calculate_coherence(self, response: str) -> float:
        """
        Calculate coherence score based on sentence flow and structure
        
        Args:
            response: LLM response
            
        Returns:
            Coherence score between 0 and 1
        """
        try:
            # Simplified sentence splitting
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            if len(sentences) < 2:
                return 1.0  # Single sentence is coherent by default
            
            # Calculate sentence similarity scores
            similarities = []
            for i in range(len(sentences) - 1):
                sim = SequenceMatcher(None, sentences[i], sentences[i + 1]).ratio()
                similarities.append(sim)
            
            # Average similarity (higher is better for coherence)
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            # Check for logical connectors
            connectors = ['however', 'therefore', 'furthermore', 'moreover', 'additionally', 
                        'consequently', 'thus', 'hence', 'as a result', 'in conclusion']
            connector_count = sum(1 for connector in connectors 
                                if connector in response.lower())
            
            # Normalize connector score
            connector_score = min(connector_count / 3, 1.0)
            
            # Combine similarity and connector scores
            coherence_score = 0.7 * avg_similarity + 0.3 * connector_score
            
            return coherence_score
            
        except Exception as e:
            logger.error(f"Error calculating coherence: {e}")
            return 0.0

class CompletenessScorer:
    """Assess completeness of responses"""
    
    def calculate_completeness(self, response: str, query: str, context: str = "") -> float:
        """
        Calculate completeness score based on query coverage
        
        Args:
            response: LLM response
            query: Original query
            context: Retrieved context (optional)
            
        Returns:
            Completeness score between 0 and 1
        """
        try:
            # Extract key terms from query (simplified approach)
            query_words = set(word.lower() for word in query.split() 
                            if word.lower() not in stopwords.words('english'))
            
            # Check if response addresses key query terms
            response_lower = response.lower()
            addressed_terms = sum(1 for term in query_words if term in response_lower)
            
            if not query_words:
                return 1.0
            
            term_coverage = addressed_terms / len(query_words)
            
            # Check response length adequacy
            word_count = len(response.split())
            min_words = 10  # Minimum expected response length
            length_score = min(word_count / min_words, 1.0)
            
            # Check for conclusion or summary
            conclusion_indicators = ['in conclusion', 'to summarize', 'overall', 'therefore', 'thus']
            has_conclusion = any(indicator in response_lower for indicator in conclusion_indicators)
            conclusion_score = 1.0 if has_conclusion else 0.5
            
            # Combine scores
            completeness_score = 0.5 * term_coverage + 0.3 * length_score + 0.2 * conclusion_score
            
            return completeness_score
            
        except Exception as e:
            logger.error(f"Error calculating completeness: {e}")
            return 0.0

class ResponseTimeScorer:
    """Calculate response time efficiency"""
    
    def calculate_time_efficiency(self, response_time_ms: float, response_length: int) -> float:
        """
        Calculate time efficiency score
        
        Args:
            response_time_ms: Response time in milliseconds
            response_length: Length of response in characters
            
        Returns:
            Time efficiency score (higher is better)
        """
        try:
            if response_time_ms <= 0 or response_length <= 0:
                return 0.0
            
            # Calculate characters per second
            chars_per_second = (response_length * 1000) / response_time_ms
            
            # Normalize to 0-1 scale (assuming reasonable range)
            # 1000 chars/second is considered excellent
            normalized_efficiency = min(chars_per_second / 1000, 1.0)
            
            return normalized_efficiency
            
        except Exception as e:
            logger.error(f"Error calculating time efficiency: {e}")
            return 0.0

class EvaluationMetricsCalculator:
    """Main class for calculating all evaluation metrics"""
    
    def __init__(self):
        self.relevance_scorer = RelevanceScorer()
        self.factual_scorer = FactualAccuracyScorer()
        self.token_scorer = TokenEfficiencyScorer()
        self.coherence_scorer = CoherenceScorer()
        self.completeness_scorer = CompletenessScorer()
        self.time_scorer = ResponseTimeScorer()
    
    def calculate_all_metrics(self, 
                            query: str,
                            response: str,
                            ground_truth: str,
                            response_time_ms: float,
                            tokens_used: int,
                            context: str = "") -> EvaluationMetrics:
        """
        Calculate all evaluation metrics for a response
        
        Args:
            query: Original user query
            response: LLM response
            ground_truth: Reference answer
            response_time_ms: Response time in milliseconds
            tokens_used: Number of tokens consumed
            context: Retrieved context (optional)
            
        Returns:
            EvaluationMetrics object with all scores
        """
        try:
            # Calculate individual metrics
            relevance = self.relevance_scorer.calculate_relevance(query, response, context)
            factual_accuracy = self.factual_scorer.calculate_factual_accuracy(response, ground_truth)
            token_efficiency = self.token_scorer.calculate_token_efficiency(response, tokens_used, query)
            coherence = self.coherence_scorer.calculate_coherence(response)
            completeness = self.completeness_scorer.calculate_completeness(response, query, context)
            time_efficiency = self.time_scorer.calculate_time_efficiency(response_time_ms, len(response))
            
            # Calculate overall score (weighted average)
            weights = {
                'relevance': 0.25,
                'factual_accuracy': 0.25,
                'coherence': 0.20,
                'completeness': 0.15,
                'token_efficiency': 0.10,
                'time_efficiency': 0.05
            }
            
            overall_score = (
                weights['relevance'] * relevance +
                weights['factual_accuracy'] * factual_accuracy +
                weights['coherence'] * coherence +
                weights['completeness'] * completeness +
                weights['token_efficiency'] * token_efficiency +
                weights['time_efficiency'] * time_efficiency
            )
            
            # Calculate confidence interval (simplified)
            scores = [relevance, factual_accuracy, coherence, completeness, token_efficiency, time_efficiency]
            std_dev = np.std(scores)
            confidence_interval = (max(0, overall_score - std_dev), min(1, overall_score + std_dev))
            
            return EvaluationMetrics(
                relevance_score=relevance,
                factual_accuracy=factual_accuracy,
                response_time_ms=response_time_ms,
                token_efficiency=token_efficiency,
                coherence_score=coherence,
                completeness_score=completeness,
                overall_score=overall_score,
                confidence_interval=confidence_interval
            )
            
        except Exception as e:
            logger.error(f"Error calculating evaluation metrics: {e}")
            # Return default metrics on error
            return EvaluationMetrics(
                relevance_score=0.0,
                factual_accuracy=0.0,
                response_time_ms=response_time_ms,
                token_efficiency=0.0,
                coherence_score=0.0,
                completeness_score=0.0,
                overall_score=0.0,
                confidence_interval=(0.0, 0.0)
            )
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all metrics"""
        return {
            'relevance_score': 'How well the response addresses the query and context',
            'factual_accuracy': 'Accuracy of factual claims compared to ground truth',
            'response_time_ms': 'Time taken to generate response in milliseconds',
            'token_efficiency': 'Information density per token consumed',
            'coherence_score': 'Logical flow and structure of the response',
            'completeness_score': 'How thoroughly the response covers the query',
            'overall_score': 'Weighted combination of all metrics'
        } 