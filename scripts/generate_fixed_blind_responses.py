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
    """Generate fixed blind responses for consistent evaluations"""
    
    print("🚀 Generating fixed blind responses...")
    
    # Initialize provider manager
    manager = ProviderManager()
    
    # Get all available models across all providers
    all_models = []
    config = manager.config_loader.load_llm_config()
    for provider_name, provider_config in config.get("providers", {}).items():
        if provider_config.get("enabled", False):
            models = provider_config.get("models", [])
            for model in models:
                all_models.append({
                    "name": model["name"],
                    "description": model["description"], 
                    "provider": provider_name
                })
    
    print(f"📊 Found {len(all_models)} models across all providers:")
    for model in all_models:
        print(f"   - {model['name']} ({model['provider']})")
    
    # Load questions
    questions = [
        {
            "id": "retail_0",
            "question": "What are the top three product categories by total sales value?",
            "domain": "retail",
            "rag_context": "Query: What are the top three product categories by total sales value?\n\nRelevant data insights:\n\n--- Chunk 1 (Relevance: 0.472) ---\nRows 3200-3399 (200 records)\nStats: Spend: μ=59.7, Age: μ=43.8, Review Rating: μ=3.7, Previous Purchases: μ=24.5\nCategory: Clothing(94), Accessories(63), Footwear(22)\nGender: Female(200)\nSeason: Fall(57), Winter(50), Summer(50)\nPayment Method: Credit Card(41), Debit Card(39), PayPal(35)\nSubscription Status: No(200)\nEx1: Clothing, $24, Age67, Female\nEx2: Clothing, $38, Age46, Female\n\n--- Chunk 2 (Relevance: 0.472) ---\nRows 3400-3599 (200 records)\nStats: Spend: μ=61.3, Age: μ=42.5, Review Rating: μ=3.9, Previous Purchases: μ=24.6\nCategory: Clothing(89), Accessories(58), Footwear(41)\nGender: Female(200)\nSeason: Spring(59), Summer(48), Fall(47)\nPayment Method: Cash(38), Bank Transfer(37), PayPal(35)\nSubscription Status: No(200)\nEx1: Accessories, $51, Age39, Female\nEx2: Accessories, $71, Age44, Female\n\n--- Chunk 3 (Relevance: 0.471) ---\nRows 2000-2199 (200 records)\nStats: Spend: μ=59.4, Age: μ=43.6, Review Rating: μ=3.7, Previous Purchases: μ=26.2\nCategory: Clothing(92), Accessories(64), Footwear(30)\nGender: Male(200)\nSeason: Fall(53), Winter(52), Spring(49)\nPayment Method: Credit Card(42), PayPal(35), Bank Transfer(33)\nSubscription Status: No(200)\nEx1: Accessories, $27, Age26, Male\nEx2: Accessories, $90, Age51, Male\n\n--- Chunk 4 (Relevance: 0.465) ---\nRows 2200-2399 (200 records)\nStats: Spend: μ=61.1, Age: μ=43.0, Review Rating: μ=3.8, Previous Purchases: μ=26.1\nCategory: Clothing(83), Accessories(63), Footwear(36)\nGender: Male(200)\nSeason: Winter(55), Fall(51), Spring(49)\nPayment Method: Credit Card(40), PayPal(36), Cash(34)\nSubscription Status: No(200)\nEx1: Accessories, $51, Age60, Male\nEx2: Clothing, $25, Age39, Male\n\n--- Chunk 5 (Relevance: 0.459) ---\nRows 1400-1599 (200 records)\nStats: Spend: μ=58.9, Age: μ=43.9, Review Rating: μ=3.8, Previous Purchases: μ=23.2\nCategory: Clothing(82), Accessories(70), Footwear(33)\nGender: Male(200)\nSeason: Summer(61), Spring(49), Winter(48)\nPayment Method: Venmo(44), PayPal(35), Cash(32)\nSubscription Status: No(200)\nEx1: Accessories, $23, Age36, Male\nEx2: Footwear, $59, Age23, Male"
        },
        {
            "id": "retail_1", 
            "question": "Which age group spends the most on average per transaction?",
            "domain": "retail",
            "rag_context": "Query: Which age group spends the most on average per transaction?\n\nRelevant data insights:\n\n--- Chunk 1 (Relevance: 0.461) ---\nRows 200-399 (200 records)\nStats: Spend: μ=59.0, Age: μ=44.5, Review Rating: μ=3.7, Previous Purchases: μ=24.5\nCategory: Clothing(88), Accessories(70), Footwear(29)\nGender: Male(200)\nSeason: Spring(57), Winter(50), Summer(47)\nPayment Method: Cash(39), Credit Card(38), Debit Card(33)\nSubscription Status: Yes(200)\nEx1: Clothing, $61, Age25, Male\nEx2: Clothing, $22, Age69, Male\n\n--- Chunk 2 (Relevance: 0.455) ---\nRows 0-199 (200 records)\nStats: Spend: μ=59.8, Age: μ=44.2, Review Rating: μ=3.8, Previous Purchases: μ=28.0\nCategory: Clothing(86), Accessories(61), Footwear(28)\nGender: Male(200)\nSeason: Summer(57), Fall(55), Spring(44)\nPayment Method: Debit Card(40), Bank Transfer(38), Credit Card(32)\nSubscription Status: Yes(200)\nEx1: Clothing, $53, Age55, Male\nEx2: Clothing, $64, Age19, Male\n\n--- Chunk 3 (Relevance: 0.454) ---\nRows 1000-1199 (200 records)\nStats: Spend: μ=59.1, Age: μ=44.7, Review Rating: μ=3.7, Previous Purchases: μ=26.6\nCategory: Clothing(89), Accessories(72), Footwear(22)\nGender: Male(200)\nSeason: Spring(52), Winter(50), Fall(49)\nPayment Method: Credit Card(40), Cash(40), Bank Transfer(31)\nSubscription Status: No(147), Yes(53)\nEx1: Clothing, $46, Age43, Male\nEx2: Clothing, $60, Age61, Male\n\n--- Chunk 4 (Relevance: 0.451) ---\nRows 3800-3899 (100 records)\nStats: Spend: μ=59.2, Age: μ=46.3, Review Rating: μ=3.7, Previous Purchases: μ=26.7\nCategory: Clothing(46), Accessories(32), Footwear(14)\nGender: Female(100)\nSeason: Spring(29), Summer(26), Winter(25)\nPayment Method: Credit Card(21), Venmo(19), Bank Transfer(18)\nSubscription Status: No(100)\nEx1: Clothing, $26, Age19, Female\nEx2: Clothing, $84, Age26, Female\n\n--- Chunk 5 (Relevance: 0.445) ---\nRows 1800-1999 (200 records)\nStats: Spend: μ=61.0, Age: μ=44.8, Review Rating: μ=3.8, Previous Purchases: μ=25.3\nCategory: Clothing(101), Accessories(61), Footwear(28)\nGender: Male(200)\nSeason: Summer(54), Winter(51), Spring(50)\nPayment Method: Debit Card(39), Credit Card(37), Venmo(34)\nSubscription Status: No(200)\nEx1: Outerwear, $58, Age22, Male\nEx2: Clothing, $57, Age33, Male"
        },
        {
            "id": "finance_0",
            "question": "What are the key risk factors affecting loan default rates?", 
            "domain": "finance",
            "rag_context": "Query: What are the key risk factors affecting loan default rates?\n\nRelevant data insights:\n\n--- Chunk 1 (Relevance: 0.482) ---\nRows 4800-4999 (200 records)\nStats: Open: μ=266.5, High: μ=269.2, Low: μ=263.8, Close: μ=266.5, Volume: μ=45.2M\nDate range: Various trading days\nTrend: Mixed with volatility patterns\nVolume patterns: High volume on price movements\nPrice action: Range-bound with breakout attempts\nEx1: Open: $245.12, Close: $248.95, Volume: 52.3M\nEx2: Open: $287.45, Close: $284.21, Volume: 38.7M\n\n--- Chunk 2 (Relevance: 0.475) ---\nRows 2400-2599 (200 records)\nStats: Open: μ=247.8, High: μ=250.4, Low: μ=245.2, Close: μ=247.8, Volume: μ=48.1M\nDate range: Various trading days\nTrend: Consolidation phase with lower volatility\nVolume patterns: Steady institutional flow\nPrice action: Tight range trading\nEx1: Open: $238.67, Close: $241.23, Volume: 41.2M\nEx2: Open: $256.89, Close: $254.44, Volume: 55.8M"
        }
    ]
    
    # Initialize response structure
    fixed_responses = {
        "generation_timestamp": datetime.now().isoformat(),
        "description": "Fixed LLM responses for consistent blind evaluations with individual model responses",
        "responses": {}
    }
    
    total_questions = len(questions)
    
    # Generate responses for each question
    for q_idx, question in enumerate(questions):
        question_id = question["id"]
        question_text = question["question"]
        rag_context = question["rag_context"]
        
        print(f"\n📝 Question {q_idx + 1}/{total_questions}: {question_id}")
        print(f"   Query: {question_text}")
        
        # Initialize question structure
        fixed_responses["responses"][question_id] = {
            "question": question_text,
            "domain": question["domain"],
            "rag_context": rag_context,
            "llm_responses": {}
        }
        
        # Generate responses for each individual model
        for model in all_models:
            model_name = model["name"]
            provider_name = model["provider"]
            model_key = f"{provider_name}_{model_name.replace('/', '_').replace('-', '_')}"
            
            try:
                print(f"   🤖 Generating response from {model_name} ({provider_name})...")
                
                # Get provider for this model
                provider = manager.get_provider_for_model(model_name)
                if not provider:
                    print(f"   ❌ {model_name}: No provider found")
                    continue
                
                # Generate response
                response_data = provider.generate_response(
                    query=question_text,
                    context=rag_context,
                    model=model_name
                )
                
                if response_data and response_data.success:
                    # Store response with metadata
                    fixed_responses["responses"][question_id]["llm_responses"][model_key] = {
                        "response": response_data.text,
                        "model_name": model_name,
                        "provider": provider_name,
                        "metadata": {
                            "latency": getattr(response_data, 'latency', 0),
                            "token_count": getattr(response_data, 'tokens_used', len(response_data.text.split())),
                            "timestamp": datetime.now().isoformat(),
                            "model": model_name
                        }
                    }
                    print(f"   ✅ {model_name}: {len(response_data.text)} chars")
                else:
                    print(f"   ❌ {model_name}: No response generated")
                    
            except Exception as e:
                logger.error(f"Error generating response from {model_name} for {question_id}: {str(e)}")
                print(f"   ❌ {model_name}: Error - {str(e)}")
                continue
        
        # Save progress every question
        save_fixed_responses(fixed_responses)
        print(f"   💾 Progress saved ({q_idx + 1}/{total_questions})")
    
    # Final save
    save_fixed_responses(fixed_responses)
    
    print(f"\n✅ Fixed blind responses generated successfully!")
    print(f"📁 Saved to: data/fixed_blind_responses.json")
    print(f"📊 Total questions: {total_questions}")
    print(f"🤖 Models per question: {len(all_models)}")
    
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