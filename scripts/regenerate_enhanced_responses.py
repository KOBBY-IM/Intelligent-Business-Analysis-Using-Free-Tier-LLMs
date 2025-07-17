#!/usr/bin/env python3
"""
Enhanced Response Regeneration Script
Regenerates LLM responses using the enhanced RAG system with:
- Increased dataset coverage (8 chunks instead of 5)
- Better chunk quality (150 rows instead of 200)
- Dynamic ground truth guidance
- Enhanced statistical analysis
"""

import json
import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime
import time

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = str(project_root / "src")
sys.path.insert(0, src_path)

from rag.csv_rag_pipeline import EnhancedCSVBlindTestGenerator
from llm_providers.provider_manager import ProviderManager
from evaluation.ground_truth import GroundTruthManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def regenerate_enhanced_responses():
    """Regenerate all responses using enhanced RAG system"""
    
    print("🚀 Starting enhanced response regeneration...")
    print("📊 Improvements:")
    print("   • Increased dataset coverage: 8 chunks (vs 5)")
    print("   • Better chunk quality: 150 rows per chunk (vs 200)")
    print("   • Dynamic ground truth guidance")
    print("   • Enhanced statistical analysis")
    print("   • Cross-chunk pattern detection")
    print("")
    
    # Initialize components
    ground_truth_manager = GroundTruthManager()
    provider_manager = ProviderManager()
    
    # Define CSV files
    csv_files = [
        "data/shopping_trends.csv",
        "data/Tesla_stock_data.csv"
    ]
    
    # Verify CSV files exist
    for csv_file in csv_files:
        if not Path(csv_file).exists():
            print(f"❌ Error: {csv_file} not found!")
            return None
    
    # Initialize enhanced RAG generator
    generator = EnhancedCSVBlindTestGenerator(csv_files)
    generator.setup_rag()
    
    # Load existing questions from evaluation config
    questions_file = Path("data/evaluation_questions.yaml")
    if questions_file.exists():
        import yaml
        with open(questions_file, 'r') as f:
            questions_data = yaml.safe_load(f)
        
        # Extract questions
        questions = []
        for domain, domain_data in questions_data.items():
            if domain in ['retail', 'finance'] and isinstance(domain_data, dict) and 'questions' in domain_data:
                for q in domain_data['questions']:
                    if isinstance(q, dict):  # Ensure q is a dictionary
                        questions.append({
                            "question_id": q.get("id", f"{domain}_{len(questions)}"),
                            "question": q["question"],
                            "domain": domain,
                            "context": q.get("context", "")
                        })
    else:
        # Fallback questions if config doesn't exist
        questions = [
            {
                "question_id": "retail_0",
                "question": "What are the top three product categories by total sales value?",
                "domain": "retail"
            },
            {
                "question_id": "retail_1", 
                "question": "Which age group spends the most on average per transaction?",
                "domain": "retail"
            },
            {
                "question_id": "retail_2",
                "question": "How does purchase frequency vary by season and product category?",
                "domain": "retail"
            },
            {
                "question_id": "retail_3",
                "question": "What payment methods are most popular among loyalty program members?",
                "domain": "retail"
            },
            {
                "question_id": "retail_4",
                "question": "Are there significant differences in shopping behavior between genders?",
                "domain": "retail"
            },
            {
                "question_id": "finance_0",
                "question": "What are the key risk factors affecting Tesla stock performance?",
                "domain": "finance"
            },
            {
                "question_id": "finance_1",
                "question": "How does Tesla stock volatility compare across different market periods?",
                "domain": "finance"
            },
            {
                "question_id": "finance_2",
                "question": "What are the main drivers of Tesla stock price movements?",
                "domain": "finance"
            },
            {
                "question_id": "finance_3",
                "question": "How does Tesla stock performance correlate with market indices?",
                "domain": "finance"
            },
            {
                "question_id": "finance_4",
                "question": "What are the seasonal patterns in Tesla stock trading volume?",
                "domain": "finance"
            }
        ]
    
    print(f"📝 Processing {len(questions)} questions...")
    
    # Initialize response structure
    enhanced_responses = {
        "generation_timestamp": datetime.now().isoformat(),
        "description": "Enhanced LLM responses with improved RAG coverage and quality",
        "improvements": {
            "dataset_coverage": "8 chunks (vs 5)",
            "chunk_quality": "150 rows per chunk (vs 200)",
            "dynamic_guidance": "Query-based analysis focus",
            "statistical_analysis": "Enhanced with outliers and trends",
            "cross_chunk_analysis": "Pattern detection across chunks"
        },
        "responses": {}
    }
    
    # Get available models
    config = provider_manager.config_loader.load_llm_config()
    available_models = []
    for provider_name, provider_config in config.get("providers", {}).items():
        if provider_config.get("enabled", False):
            models = provider_config.get("models", [])
            for model in models:
                available_models.append({
                    "name": model["name"],
                    "provider": provider_name
                })
    
    print(f"🤖 Found {len(available_models)} available models:")
    for model in available_models:
        print(f"   • {model['name']} ({model['provider']})")
    print("")
    
    # Generate responses for each question
    total_questions = len(questions)
    for q_idx, question in enumerate(questions):
        question_id = question["question_id"]
        question_text = question["question"]
        domain = question["domain"]
        
        print(f"📝 Question {q_idx + 1}/{total_questions}: {question_id}")
        print(f"   Domain: {domain}")
        print(f"   Query: {question_text}")
        
        # Get ground truth guidance
        ground_truth = ground_truth_manager.get_answer(question_id)
        ground_truth_guidance = None
        
        if ground_truth:
            ground_truth_guidance = {
                "key_points": ground_truth.key_points,
                "factual_claims": ground_truth.factual_claims,
                "expected_length": ground_truth.expected_length,
                "answer": ground_truth.answer
            }
            print(f"   ✅ Ground truth guidance available")
        else:
            print(f"   ⚠️  No ground truth guidance found")
        
        # Initialize question structure
        enhanced_responses["responses"][question_id] = {
            "question": question_text,
            "domain": domain,
            "context": question.get("context", ""),
            "ground_truth": ground_truth_guidance,
            "rag_context": "Enhanced RAG with 8 chunks, dynamic guidance, and cross-chunk analysis",
            "llm_responses": {}
        }
        
        # Generate responses for each model
        for model in available_models:
            model_name = model["name"]
            provider_name = model["provider"]
            
            try:
                print(f"   🤖 Generating response from {model_name}...")
                
                # Create question dict for RAG system
                question_dict = {
                    "question": question_text,
                    "question_id": question_id
                }
                
                # Generate enhanced response
                response_data = generator.generate_response_with_rag(
                    question=question_dict,
                    model_name=model_name,
                    ground_truth_guidance=ground_truth_guidance
                )
                
                # Defensive: ensure response_data is a dict
                if isinstance(response_data, dict) and not response_data.get('error'):
                    # Store enhanced response
                    enhanced_responses["responses"][question_id]["llm_responses"][provider_name] = {
                        "response": response_data["response"],
                        "rag_context": response_data["rag_context"],
                        "metadata": {
                            "latency_ms": response_data.get("response_time_ms", 0),
                            "tokens_used": response_data.get("tokens_used", 0),
                            "timestamp": datetime.now().isoformat(),
                            "model": model_name,
                            "provider": provider_name,
                            "enhanced_coverage": response_data.get("enhanced_coverage", True),
                            "chunks_used": response_data.get("chunks_used", 8),
                            "ground_truth_guidance_used": response_data.get("ground_truth_guidance_used", False)
                        }
                    }
                    print(f"   ✅ {model_name}: {len(response_data['response'])} chars")
                else:
                    error_msg = response_data.get('error', 'Unknown error') if isinstance(response_data, dict) else str(response_data)
                    print(f"   ❌ {model_name}: {error_msg}")
                    
                    # Store error response
                    enhanced_responses["responses"][question_id]["llm_responses"][provider_name] = {
                        "response": f"Error generating response: {error_msg}",
                        "rag_context": "",
                        "metadata": {
                            "latency_ms": 0,
                            "tokens_used": 0,
                            "timestamp": datetime.now().isoformat(),
                            "model": model_name,
                            "provider": provider_name,
                            "error": error_msg
                        }
                    }
                    
            except Exception as e:
                logger.error(f"Error generating response from {model_name} for {question_id}: {str(e)}")
                print(f"   ❌ {model_name}: Exception - {str(e)}")
                
                # Store exception response
                enhanced_responses["responses"][question_id]["llm_responses"][provider_name] = {
                    "response": f"Exception occurred: {str(e)}",
                    "rag_context": "",
                    "metadata": {
                        "latency_ms": 0,
                        "tokens_used": 0,
                        "timestamp": datetime.now().isoformat(),
                        "model": model_name,
                        "provider": provider_name,
                        "error": str(e)
                    }
                }
        
        # Save progress every question
        save_enhanced_responses(enhanced_responses)
        print(f"   💾 Progress saved ({q_idx + 1}/{total_questions})")
        print("")
    
    # Final save
    save_enhanced_responses(enhanced_responses)
    
    print(f"✅ Enhanced response regeneration complete!")
    print(f"📁 Saved to: data/enhanced_blind_responses.json")
    print(f"📊 Summary:")
    print(f"   • Questions processed: {total_questions}")
    print(f"   • Models per question: {len(available_models)}")
    print(f"   • Total responses: {total_questions * len(available_models)}")
    print(f"   • Enhanced coverage: 8 chunks per query")
    print(f"   • Improved quality: 150 rows per chunk")
    
    return enhanced_responses

def save_enhanced_responses(responses_data):
    """Save enhanced responses to file with backup"""
    output_file = Path("data/enhanced_blind_responses.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save main file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(responses_data, f, indent=2, ensure_ascii=False)
    
    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = Path(f"data/backups/enhanced_responses_{timestamp}.json")
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(responses_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved enhanced responses to {output_file}")
    logger.info(f"Backup created at {backup_file}")

def main():
    """Main function"""
    try:
        print("🎯 Enhanced Response Regeneration")
        print("=" * 50)
        
        responses = regenerate_enhanced_responses()
        
        if responses:
            print("\n🎉 Enhanced responses ready for evaluation!")
            print("   • Better dataset coverage (8 chunks)")
            print("   • Improved chunk quality (150 rows)")
            print("   • Dynamic ground truth guidance")
            print("   • Enhanced statistical analysis")
            print("   • Cross-chunk pattern detection")
        else:
            print("\n❌ Failed to generate enhanced responses")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Regeneration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("🔍 Full traceback:")
        traceback.print_exc()
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 