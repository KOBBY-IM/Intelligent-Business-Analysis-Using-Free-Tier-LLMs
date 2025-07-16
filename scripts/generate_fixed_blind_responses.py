#!/usr/bin/env python3
"""
Generate Fixed Blind Responses
Generates LLM responses once and saves them permanently for consistent blind evaluations.
All users will see the same responses for fair comparison.
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime
import time

# Add src to path for direct imports
project_root = Path(__file__).parent.parent
src_path = str(project_root / "src")
sys.path.insert(0, src_path)

from rag.csv_rag_pipeline import CSVRAGPipeline
from llm_providers.provider_manager import ProviderManager
from utils.question_sampler import QuestionSampler
from config.config_loader import ConfigLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_fixed_blind_responses():
    """
    Generate LLM responses once and save them permanently for blind evaluations.
    These responses will be used consistently across all blind evaluations.
    """
    print("🎯 Generating Fixed Blind Responses for Consistent Evaluation")
    print("=============================================================")
    
    # Initialize components with proper config
    config_loader = ConfigLoader(str(project_root / "config"))
    sampler = QuestionSampler()
    manager = ProviderManager(config_loader)
    rag_pipeline = CSVRAGPipeline()
    
    # Build RAG index with all datasets
    print("🔍 Building RAG index with datasets...")
    dataset_files = [
        "data/shopping_trends.csv",  # Retail dataset
        "data/Tesla_stock_data.csv"  # Finance dataset
    ]
    rag_pipeline.build_index(dataset_files, chunk_size=200)
    print("✅ RAG index built successfully")
    
    # Load all questions
    questions = sampler.get_all_questions()
    
    # Initialize results structure
    fixed_responses = {
        "generation_timestamp": datetime.now().isoformat(),
        "description": "Fixed LLM responses for consistent blind evaluations",
        "responses": {}
    }
    
    total_questions = len(questions)
    print(f"📋 Generating responses for {total_questions} questions...")
    
    for q_idx, question in enumerate(questions, 1):
        domain = question['domain']
        question_text = question['question']
        question_id = f"{domain}_{question['question_idx']}"
        
        print(f"\n   📋 Question {q_idx}/{total_questions}: {question_text[:60]}...")
        
        # Generate RAG context with ground truth summary
        rag_context = rag_pipeline.generate_context(question_text, top_k=5, question_id=question_id)
        print(f"   🔍 Generated RAG context: {len(rag_context)} characters")
        
        # Initialize question responses
        fixed_responses["responses"][question_id] = {
            "question": question_text,
            "domain": domain,
            "rag_context": rag_context,
            "llm_responses": {}
        }
        
        # Generate responses from all available providers
        providers = manager.get_available_providers()
        
        for provider_name in providers:
            try:
                print(f"   🤖 Generating response from {provider_name}...")
                
                # Generate response
                response_data = manager.generate_response(
                    provider_name=provider_name,
                    query=question_text,
                    context=rag_context
                )
                
                if response_data and response_data.get('response'):
                    # Store response with metadata
                    fixed_responses["responses"][question_id]["llm_responses"][provider_name] = {
                        "response": response_data['response'],
                        "metadata": {
                            "latency": response_data.get('latency', 0),
                            "token_count": response_data.get('token_count', 0),
                            "timestamp": datetime.now().isoformat(),
                            "model": response_data.get('model', 'unknown')
                        }
                    }
                    print(f"   ✅ {provider_name}: {len(response_data['response'])} chars")
                else:
                    print(f"   ❌ {provider_name}: No response generated")
                    
            except Exception as e:
                logger.error(f"Error generating response from {provider_name} for {question_id}: {str(e)}")
                print(f"   ❌ {provider_name}: Error - {str(e)}")
                continue
        
        # Save progress every 5 questions
        if q_idx % 5 == 0:
            save_fixed_responses(fixed_responses)
            print(f"   💾 Progress saved ({q_idx}/{total_questions})")
    
    # Final save
    save_fixed_responses(fixed_responses)
    
    print(f"\n✅ Fixed blind responses generated successfully!")
    print(f"📁 Saved to: data/fixed_blind_responses.json")
    print(f"📊 Total questions: {total_questions}")
    print(f"🤖 Providers used: {', '.join(providers)}")
    
    return fixed_responses

def save_fixed_responses(responses_data):
    """Save fixed responses to file."""
    output_file = Path("data/fixed_blind_responses.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(responses_data, f, indent=2, ensure_ascii=False)
    
    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = Path(f"data/backups/fixed_responses_{timestamp}.json")
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(responses_data, f, indent=2, ensure_ascii=False)

def main():
    """Main function."""
    try:
        responses = generate_fixed_blind_responses()
        print("\n🎯 Fixed responses ready for blind evaluations!")
        print("   - All users will see the same responses")
        print("   - Consistent evaluation across all participants")
        print("   - Fair comparison between LLM providers")
        
    except KeyboardInterrupt:
        print("\n❌ Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 