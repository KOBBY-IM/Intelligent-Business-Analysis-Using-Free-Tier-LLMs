#!/usr/bin/env python3
"""
Blind Test Generator

Generates blind test responses for LLM evaluation using the RAG pipeline.
"""

import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Use direct imports instead of relative imports
from llm_providers.provider_manager import ProviderManager
from rag.csv_rag_pipeline import CSVRAGPipeline
from data_processing.question_sampler import QuestionSampler
from utils.structured_logger import setup_logger

logger = logging.getLogger(__name__)


@dataclass
class BlindTestResponse:
    """Response for blind testing"""
    id: str
    content: str
    model: str
    metrics: Dict[str, float]


@dataclass
class BlindTestQuestion:
    """Question for blind testing"""
    id: str
    prompt: str
    context: str
    industry: str
    category: str
    responses: List[BlindTestResponse]


class BlindTestGenerator:
    """Generates blind test responses using RAG pipeline"""

    def __init__(self, config_path: str = "config/llm_config.yaml"):
        """
        Initialize blind test generator
        
        Args:
            config_path: Path to LLM configuration file
        """
        self.provider_manager = ProviderManager(config_path)
        self.rag_pipelines = {}
        self.models = [
            "groq/mixtral-8x7b-32768",
            "gemini/gemini-1.5-flash", 
            "openrouter/mistral-7b-instruct",
            "groq/llama-3.1-8b-instant",
            "gemini/gemini-1.5-pro",
            "openrouter/llama-3.1-8b-instruct"
        ]
        
        logger.info(f"Initialized BlindTestGenerator with {len(self.models)} models")

    async def generate_responses_for_question(
        self, 
        question: Dict, 
        industry_context: str = ""
    ) -> List[BlindTestResponse]:
        """
        Generate responses from all models for a single question
        
        Args:
            question: Question dictionary with prompt and context
            industry_context: Additional industry-specific context
            
        Returns:
            List of BlindTestResponse objects
        """
        responses = []
        prompt = question["prompt"]
        context = f"{question['context']}\n\n{industry_context}".strip()
        
        # Create full prompt for LLM
        full_prompt = f"""
Business Scenario: {prompt}

Additional Context: {context}

Please provide a comprehensive business analysis and recommendations for this scenario. 
Focus on practical, actionable insights that would be valuable for business decision-making.
"""
        
        logger.info(f"Generating responses for question: {question['id']}")
        
        for i, model in enumerate(self.models):
            try:
                # Get provider for this model
                provider = self.provider_manager.get_provider_for_model(model)
                if not provider:
                    logger.warning(f"No provider found for model: {model}")
                    continue
                
                # Generate response
                start_time = time.time()
                llm_response = await provider.generate_response(
                    query=full_prompt,
                    context=context,
                    model=model,
                    max_tokens=500,
                    temperature=0.7
                )
                latency = (time.time() - start_time) * 1000
                
                if llm_response.success:
                    # Calculate metrics
                    metrics = {
                        "relevance": random.uniform(0.75, 0.95),  # Placeholder - should be calculated
                        "accuracy": random.uniform(0.70, 0.90),   # Placeholder - should be calculated  
                        "coherence": random.uniform(0.80, 0.95),  # Placeholder - should be calculated
                        "token_count": len(llm_response.text.split()),
                        "latency": latency / 1000  # Convert to seconds
                    }
                    
                    response = BlindTestResponse(
                        id=f"{question['id']}_response_{i+1}",
                        content=llm_response.text,
                        model=model,
                        metrics=metrics
                    )
                    responses.append(response)
                    
                    logger.info(f"Generated response {i+1}/{len(self.models)} for {question['id']}")
                else:
                    logger.error(f"Failed to generate response for {model}: {llm_response.error}")
                    
            except Exception as e:
                logger.error(f"Error generating response for {model}: {e}")
                continue
        
        logger.info(f"Generated {len(responses)} responses for question {question['id']}")
        return responses

    async def generate_all_blind_tests(self, dataset_path: str = "data/rag_datasets.json") -> Dict[str, List[BlindTestQuestion]]:
        """
        Generate blind test responses for all questions in the dataset
        
        Args:
            dataset_path: Path to the RAG dataset file
            
        Returns:
            Dictionary with industry -> list of BlindTestQuestion
        """
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        industry_contexts = {
            "retail": "This analysis should focus on retail industry best practices, customer behavior patterns, and operational efficiency.",
            "finance": "This analysis should focus on financial risk management, regulatory compliance, and investment optimization strategies.",
            "healthcare": "This analysis should focus on patient care quality, healthcare operations, and medical decision support."
        }
        
        results = {}
        
        for industry, questions in dataset.items():
            logger.info(f"Generating blind tests for {industry} industry")
            industry_questions = []
            
            for question in questions:
                responses = await self.generate_responses_for_question(
                    question, 
                    industry_contexts.get(industry, "")
                )
                
                blind_question = BlindTestQuestion(
                    id=question["id"],
                    prompt=question["prompt"],
                    context=question["context"],
                    industry=question["industry"],
                    category=question["category"],
                    responses=responses
                )
                industry_questions.append(blind_question)
            
            results[industry] = industry_questions
            
        return results

    def save_blind_tests(self, blind_tests: Dict[str, List[BlindTestQuestion]], output_path: str = "data/blind_responses_rag.json"):
        """
        Save generated blind tests to JSON file
        
        Args:
            blind_tests: Generated blind tests
            output_path: Output file path
        """
        output_data = {}
        
        for industry, questions in blind_tests.items():
            output_data[industry] = []
            
            for question in questions:
                question_data = {
                    "prompt": question.prompt,
                    "context": question.context,
                    "responses": [
                        {
                            "id": resp.id,
                            "content": resp.content,
                            "model": resp.model,
                            "metrics": resp.metrics
                        }
                        for resp in question.responses
                    ]
                }
                output_data[industry].append(question_data)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved blind tests to {output_path}")

    async def generate_and_save(self, dataset_path: str = "data/rag_datasets.json", output_path: str = "data/blind_responses_rag.json"):
        """
        Generate blind tests and save to file
        
        Args:
            dataset_path: Input dataset path
            output_path: Output file path
        """
        logger.info("Starting blind test generation...")
        
        blind_tests = await self.generate_all_blind_tests(dataset_path)
        self.save_blind_tests(blind_tests, output_path)
        
        logger.info("Blind test generation completed!")


async def main():
    """Main function for testing"""
    generator = BlindTestGenerator()
    await generator.generate_and_save()


if __name__ == "__main__":
    asyncio.run(main()) 