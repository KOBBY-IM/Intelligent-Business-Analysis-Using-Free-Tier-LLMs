#!/usr/bin/env python3
"""
Script to regenerate responses using RAG with ground truth guidance
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.csv_rag_pipeline import CSVBlindTestGenerator
from evaluation.ground_truth import GroundTruthManager

def regenerate_responses_with_guidance():
    """Regenerate all responses using RAG with ground truth guidance"""
    
    # Initialize ground truth manager
    ground_truth_manager = GroundTruthManager()
    
    # Define questions
    questions = [
        {
            "question_id": "retail_001",
            "question": "What are the top performing product categories?",
            "domain": "retail"
        },
        {
            "question_id": "retail_002", 
            "question": "How do sales vary by customer segment?",
            "domain": "retail"
        },
        {
            "question_id": "retail_003",
            "question": "What are the monthly sales trends?",
            "domain": "retail"
        },
        {
            "question_id": "retail_004",
            "question": "Which products generate the highest revenue?",
            "domain": "retail"
        },
        {
            "question_id": "retail_005",
            "question": "What is the overall structure of this retail dataset?",
            "domain": "retail"
        },
        {
            "question_id": "finance_001",
            "question": "What are the key financial performance indicators?",
            "domain": "finance"
        },
        {
            "question_id": "finance_002",
            "question": "How do customer acquisition costs vary by channel?",
            "domain": "finance"
        },
        {
            "question_id": "finance_003",
            "question": "What is the customer lifetime value analysis?",
            "domain": "finance"
        },
        {
            "question_id": "finance_004",
            "question": "What are the profit margin trends?",
            "domain": "finance"
        },
        {
            "question_id": "finance_005",
            "question": "How do revenue streams perform across different segments?",
            "domain": "finance"
        }
    ]
    
    # Initialize RAG generator
    csv_files = [
        "data/retail_data.csv",
        "data/finance_data.csv"
    ]
    
    # Check if CSV files exist
    for csv_file in csv_files:
        if not Path(csv_file).exists():
            print(f"Warning: {csv_file} not found. Creating placeholder data...")
            # Create placeholder CSV files if they don't exist
            create_placeholder_csv(csv_file)
    
    generator = CSVBlindTestGenerator(csv_files)
    generator.setup_rag()
    
    # Generate responses for each question
    all_responses = {}
    
    for question in questions:
        question_id = question["question_id"]
        print(f"Generating responses for {question_id}...")
        
        # Get ground truth guidance
        ground_truth = ground_truth_manager.get_answer(question_id)
        ground_truth_guidance = None
        
        if ground_truth:
            ground_truth_guidance = {
                "key_points": ground_truth.key_points,
                "factual_claims": ground_truth.factual_claims,
                "expected_length": ground_truth.expected_length,
                "category": ground_truth.category,
                "difficulty": ground_truth.difficulty
            }
        
        # Generate responses for each model
        question_responses = {}
        
        # Define models to use
        models = ["groq", "gemini", "openrouter_mistral", "openrouter_deepseek"]
        
        for model in models:
            try:
                response_data = generator.generate_response_with_rag(
                    question=question,
                    model_name=model,
                    ground_truth_guidance=ground_truth_guidance
                )
                
                question_responses[model] = {
                    "response": response_data["response"],
                    "rag_context": response_data["rag_context"],
                    "response_time_ms": response_data.get("response_time_ms", 0),
                    "tokens_used": response_data.get("tokens_used", 0),
                    "ground_truth_guidance_used": response_data.get("ground_truth_guidance_used", False)
                }
                
                print(f"  ✓ {model}: {len(response_data['response'])} chars")
                
            except Exception as e:
                print(f"  ✗ {model}: Error - {e}")
                question_responses[model] = {
                    "response": f"Error generating response: {str(e)}",
                    "rag_context": "",
                    "response_time_ms": 0,
                    "tokens_used": 0,
                    "ground_truth_guidance_used": False
                }
        
        all_responses[question_id] = {
            "question": question["question"],
            "domain": question["domain"],
            "ground_truth": ground_truth.answer if ground_truth else "Ground truth not available",
            "key_points": ground_truth.key_points if ground_truth else [],
            "factual_claims": ground_truth.factual_claims if ground_truth else [],
            "rag_context": "Generated using RAG with 40% dataset coverage + ground truth guidance",
            "llm_responses": question_responses
        }
    
    # Save to file
    output_file = "data/fixed_blind_responses_with_guidance.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "generated_at": "2024-07-17",
                "description": "Responses generated using RAG with ground truth guidance",
                "dataset_coverage": "40% for LLMs, 100% for ground truth",
                "ground_truth_guidance": "Included for all LLM responses"
            },
            "responses": all_responses
        }, f, indent=2)
    
    print(f"\n✅ Generated {len(all_responses)} questions with responses")
    print(f"📁 Saved to: {output_file}")
    
    return output_file

def create_placeholder_csv(file_path: str):
    """Create placeholder CSV files for testing"""
    import pandas as pd
    import numpy as np
    
    if "retail" in file_path:
        # Create retail data
        np.random.seed(42)
        n_records = 1000
        
        data = {
            'Date': pd.date_range('2024-01-01', periods=n_records, freq='D'),
            'Product_Name': np.random.choice(['Smartphone X1', 'Laptop Pro', 'Wireless Headphones', 'Tablet Air', 'Smart Watch'], n_records),
            'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books'], n_records),
            'Customer_Segment': np.random.choice(['Premium', 'Regular', 'Budget'], n_records),
            'Sales_Amount': np.random.uniform(20, 500, n_records),
            'Quantity': np.random.randint(1, 10, n_records),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
            'Payment_Method': np.random.choice(['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet'], n_records)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        
    elif "finance" in file_path:
        # Create finance data
        np.random.seed(42)
        n_records = 500
        
        data = {
            'Month': pd.date_range('2024-01-01', periods=n_records, freq='M'),
            'Revenue': np.random.uniform(50000, 200000, n_records),
            'Customer_Acquisition_Cost': np.random.uniform(30, 80, n_records),
            'Customer_Lifetime_Value': np.random.uniform(200, 500, n_records),
            'Profit_Margin': np.random.uniform(0.15, 0.35, n_records),
            'Channel': np.random.choice(['Organic', 'Paid Search', 'Social Media', 'Email', 'Direct'], n_records),
            'Segment': np.random.choice(['Enterprise', 'SMB', 'Individual'], n_records),
            'Region': np.random.choice(['North America', 'Europe', 'Asia Pacific', 'Latin America'], n_records)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    
    print(f"Created placeholder CSV: {file_path}")

if __name__ == "__main__":
    regenerate_responses_with_guidance() 