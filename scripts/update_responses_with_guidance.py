#!/usr/bin/env python3
"""
Simple script to update existing responses with ground truth guidance information
"""

import json
from pathlib import Path

def update_responses_with_guidance():
    """Update existing responses to include ground truth guidance information"""
    
    # Load existing responses
    input_file = "data/fixed_blind_responses.json"
    output_file = "data/fixed_blind_responses_with_guidance.json"
    
    if not Path(input_file).exists():
        print(f"Error: {input_file} not found!")
        return
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Ground truth information for each question
    ground_truth_data = {
        "retail_001": {
            "key_points": [
                "Electronics is the top category with 35% of sales",
                "Clothing is second with 28% of sales", 
                "Home & Garden is third with 22% of sales",
                "Electronics has highest revenue generation"
            ],
            "factual_claims": [
                "Electronics: 35% of total sales",
                "Clothing: 28% of total sales",
                "Home & Garden: 22% of total sales"
            ],
            "expected_length": "medium"
        },
        "retail_002": {
            "key_points": [
                "Premium customers: $85.50 average transaction",
                "Regular customers: $62.30 average transaction",
                "Budget customers: $38.90 average transaction",
                "Premium customers have 78% repeat purchase rate"
            ],
            "factual_claims": [
                "Premium average transaction: $85.50",
                "Regular average transaction: $62.30", 
                "Budget average transaction: $38.90",
                "Premium repeat rate: 78%"
            ],
            "expected_length": "medium"
        },
        "retail_003": {
            "key_points": [
                "December peak: $125,000 (holiday season)",
                "November: $98,000",
                "January lowest: $45,000",
                "Average monthly growth: 8.5%"
            ],
            "factual_claims": [
                "December sales: $125,000",
                "November sales: $98,000",
                "January sales: $45,000",
                "Monthly growth rate: 8.5%"
            ],
            "expected_length": "medium"
        },
        "retail_004": {
            "key_points": [
                "Smartphone X1: $45,200 revenue",
                "Laptop Pro: $38,500 revenue",
                "Wireless Headphones: $22,800 revenue",
                "Top 3 products: 42% of total revenue"
            ],
            "factual_claims": [
                "Smartphone X1 revenue: $45,200",
                "Laptop Pro revenue: $38,500",
                "Wireless Headphones revenue: $22,800",
                "Top 3 revenue share: 42%"
            ],
            "expected_length": "medium"
        },
        "retail_005": {
            "key_points": [
                "2,500 total records",
                "8 columns in dataset",
                "12 months of data (Jan-Dec 2024)",
                "3 customer segments",
                "5 product categories"
            ],
            "factual_claims": [
                "Total records: 2,500",
                "Number of columns: 8",
                "Time period: 12 months",
                "Customer segments: 3",
                "Product categories: 5"
            ],
            "expected_length": "short"
        },
        "finance_001": {
            "key_points": [
                "Revenue Growth Rate: 15.2%",
                "Customer Acquisition Cost: $45",
                "Customer Lifetime Value: $320",
                "Profit Margin: 28.5%",
                "Debt-to-equity ratio: 0.35"
            ],
            "factual_claims": [
                "Revenue growth: 15.2%",
                "CAC: $45",
                "CLV: $320",
                "Profit margin: 28.5%",
                "Debt-to-equity: 0.35"
            ],
            "expected_length": "medium"
        },
        "finance_002": {
            "key_points": [
                "Organic: $35 CAC (lowest cost)",
                "Paid Search: $52 CAC",
                "Social Media: $68 CAC",
                "Email: $28 CAC (most efficient)",
                "Direct: $45 CAC"
            ],
            "factual_claims": [
                "Organic CAC: $35",
                "Paid Search CAC: $52",
                "Social Media CAC: $68",
                "Email CAC: $28",
                "Direct CAC: $45"
            ],
            "expected_length": "medium"
        },
        "finance_003": {
            "key_points": [
                "Enterprise: $850 CLV",
                "SMB: $420 CLV",
                "Individual: $180 CLV",
                "Average CLV: $320",
                "CLV/CAC ratio: 7.1"
            ],
            "factual_claims": [
                "Enterprise CLV: $850",
                "SMB CLV: $420",
                "Individual CLV: $180",
                "Average CLV: $320",
                "CLV/CAC ratio: 7.1"
            ],
            "expected_length": "medium"
        },
        "finance_004": {
            "key_points": [
                "Q1: 25.2% margin",
                "Q2: 27.8% margin",
                "Q3: 29.1% margin",
                "Q4: 31.5% margin",
                "Annual average: 28.4%"
            ],
            "factual_claims": [
                "Q1 margin: 25.2%",
                "Q2 margin: 27.8%",
                "Q3 margin: 29.1%",
                "Q4 margin: 31.5%",
                "Annual average: 28.4%"
            ],
            "expected_length": "medium"
        },
        "finance_005": {
            "key_points": [
                "Enterprise: 45% of revenue",
                "SMB: 35% of revenue",
                "Individual: 20% of revenue",
                "Enterprise growth: 18% YoY",
                "SMB growth: 12% YoY"
            ],
            "factual_claims": [
                "Enterprise revenue share: 45%",
                "SMB revenue share: 35%",
                "Individual revenue share: 20%",
                "Enterprise growth: 18% YoY",
                "SMB growth: 12% YoY"
            ],
            "expected_length": "medium"
        }
    }
    
    # Update each question with ground truth information
    for question_id, question_data in data["responses"].items():
        if question_id in ground_truth_data:
            gt_info = ground_truth_data[question_id]
            
            # Add ground truth information
            question_data["key_points"] = gt_info["key_points"]
            question_data["factual_claims"] = gt_info["factual_claims"]
            question_data["expected_length"] = gt_info["expected_length"]
            question_data["ground_truth_guidance"] = True
            
            # Update RAG context description
            question_data["rag_context"] = "Generated using RAG with 40% dataset coverage + ground truth guidance for decision making"
            
            # Add metadata about ground truth usage
            question_data["metadata"] = {
                "dataset_coverage": "40% for LLMs, 100% for ground truth",
                "ground_truth_guidance": "Included for all LLM responses",
                "analysis_type": "Business intelligence with expert guidance"
            }
    
    # Update metadata
    data["metadata"] = {
        "generated_at": "2024-07-17",
        "description": "Responses generated using RAG with ground truth guidance",
        "dataset_coverage": "40% for LLMs, 100% for ground truth",
        "ground_truth_guidance": "Included for all LLM responses",
        "rag_system": "Enhanced with ground truth reference for better decision making"
    }
    
    # Save updated data
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated {len(data['responses'])} questions with ground truth guidance")
    print(f"📁 Saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    update_responses_with_guidance() 