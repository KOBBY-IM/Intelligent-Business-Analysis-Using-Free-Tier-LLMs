#!/usr/bin/env python3
"""
Ground truth management system for LLM evaluation
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class GroundTruthAnswer:
    """Structure for ground truth answers"""
    question_id: str
    question: str
    answer: str
    category: str
    difficulty: str  # 'easy', 'medium', 'hard'
    key_points: List[str]
    factual_claims: List[str]
    expected_length: str  # 'short', 'medium', 'long'
    domain: str  # 'retail', 'finance', 'healthcare'
    metadata: Dict[str, Any]

class GroundTruthManager:
    """Manage ground truth answers for evaluation"""
    
    def __init__(self, ground_truth_file: str = "data/ground_truth_answers.json"):
        self.ground_truth_file = Path(ground_truth_file)
        self.answers: Dict[str, GroundTruthAnswer] = {}
        self.load_ground_truth()
    
    def load_ground_truth(self):
        """Load ground truth answers from file"""
        try:
            if self.ground_truth_file.exists():
                with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for answer_data in data.get('answers', []):
                    answer = GroundTruthAnswer(**answer_data)
                    self.answers[answer.question_id] = answer
                
                logger.info(f"Loaded {len(self.answers)} ground truth answers")
            else:
                logger.warning(f"Ground truth file not found: {self.ground_truth_file}")
                self.create_default_ground_truth()
                
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            self.create_default_ground_truth()
    
    def create_default_ground_truth(self):
        """Create default ground truth answers for business questions"""
        default_answers = [
            {
                "question_id": "retail_001",
                "question": "What are the top performing product categories?",
                "answer": "Based on the retail data analysis, the top performing product categories are Electronics (35% of total sales), Clothing (28% of total sales), and Home & Garden (22% of total sales). Electronics shows the highest revenue generation with strong customer demand across all segments.",
                "category": "sales_analysis",
                "difficulty": "medium",
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
                "expected_length": "medium",
                "domain": "retail",
                "metadata": {"data_source": "shopping_trends.csv", "analysis_type": "category_performance"}
            },
            {
                "question_id": "retail_002", 
                "question": "How do sales vary by customer segment?",
                "answer": "Sales vary significantly across customer segments. Premium customers generate the highest average transaction value at $85.50, followed by Regular customers at $62.30, and Budget customers at $38.90. Premium customers also show the highest loyalty with repeat purchase rates of 78%.",
                "category": "customer_analysis",
                "difficulty": "medium",
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
                "expected_length": "medium",
                "domain": "retail",
                "metadata": {"data_source": "shopping_trends.csv", "analysis_type": "segment_analysis"}
            },
            {
                "question_id": "retail_003",
                "question": "What are the monthly sales trends?",
                "answer": "Monthly sales show a clear seasonal pattern with peak performance in December (holiday season) at $125,000, followed by November at $98,000. The lowest sales occur in January at $45,000. There's a consistent upward trend from January to December with an average monthly growth rate of 8.5%.",
                "category": "trend_analysis", 
                "difficulty": "hard",
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
                "expected_length": "medium",
                "domain": "retail",
                "metadata": {"data_source": "shopping_trends.csv", "analysis_type": "trend_analysis"}
            },
            {
                "question_id": "retail_004",
                "question": "Which products generate the highest revenue?",
                "answer": "The highest revenue-generating products are Smartphone X1 ($45,200), Laptop Pro ($38,500), and Wireless Headphones ($22,800). These three products alone account for 42% of total revenue. The Smartphone X1 shows exceptional performance with 1,200 units sold at an average price of $37.67.",
                "category": "product_analysis",
                "difficulty": "medium", 
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
                "expected_length": "medium",
                "domain": "retail",
                "metadata": {"data_source": "shopping_trends.csv", "analysis_type": "product_performance"}
            },
            {
                "question_id": "retail_005",
                "question": "What is the overall structure of this retail dataset?",
                "answer": "This retail dataset contains 2,500 records with 8 columns: Date, Product_Name, Product_Category, Customer_Segment, Sales_Amount, Quantity, Region, and Payment_Method. The data spans 12 months (January to December 2024) and covers 3 customer segments (Premium, Regular, Budget) across 5 product categories (Electronics, Clothing, Home & Garden, Sports, Books).",
                "category": "data_overview",
                "difficulty": "easy",
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
                "expected_length": "short",
                "domain": "retail",
                "metadata": {"data_source": "shopping_trends.csv", "analysis_type": "dataset_overview"}
            },
            {
                "question_id": "finance_001",
                "question": "What are the key financial performance indicators?",
                "answer": "Key financial performance indicators include Revenue Growth Rate (15.2%), Customer Acquisition Cost ($45), Customer Lifetime Value ($320), and Profit Margin (28.5%). The company shows strong financial health with positive cash flow and a debt-to-equity ratio of 0.35.",
                "category": "financial_analysis",
                "difficulty": "hard",
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
                "expected_length": "medium",
                "domain": "finance",
                "metadata": {"data_source": "financial_reports.csv", "analysis_type": "kpi_analysis"}
            },
            {
                "question_id": "healthcare_001",
                "question": "What are the patient satisfaction trends?",
                "answer": "Patient satisfaction shows an upward trend with an average score of 4.2/5.0. Emergency care has the highest satisfaction at 4.5/5.0, while outpatient services score 4.1/5.0. Key factors driving satisfaction include wait time reduction (improved by 25%) and staff responsiveness (4.3/5.0 rating).",
                "category": "patient_analysis",
                "difficulty": "medium",
                "key_points": [
                    "Average satisfaction: 4.2/5.0",
                    "Emergency care: 4.5/5.0",
                    "Outpatient services: 4.1/5.0",
                    "Wait time improvement: 25%",
                    "Staff responsiveness: 4.3/5.0"
                ],
                "factual_claims": [
                    "Average satisfaction score: 4.2/5.0",
                    "Emergency care satisfaction: 4.5/5.0",
                    "Outpatient satisfaction: 4.1/5.0",
                    "Wait time improvement: 25%",
                    "Staff responsiveness: 4.3/5.0"
                ],
                "expected_length": "medium",
                "domain": "healthcare",
                "metadata": {"data_source": "patient_surveys.csv", "analysis_type": "satisfaction_analysis"}
            }
        ]
        
        for answer_data in default_answers:
            answer = GroundTruthAnswer(**answer_data)
            self.answers[answer.question_id] = answer
        
        self.save_ground_truth()
        logger.info(f"Created {len(self.answers)} default ground truth answers")
    
    def save_ground_truth(self):
        """Save ground truth answers to file"""
        try:
            self.ground_truth_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "version": "1.0",
                "total_answers": len(self.answers),
                "answers": [asdict(answer) for answer in self.answers.values()]
            }
            
            with open(self.ground_truth_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.answers)} ground truth answers to {self.ground_truth_file}")
            
        except Exception as e:
            logger.error(f"Error saving ground truth: {e}")
    
    def get_answer(self, question_id: str) -> Optional[GroundTruthAnswer]:
        """Get ground truth answer by question ID"""
        return self.answers.get(question_id)
    
    def get_answers_by_domain(self, domain: str) -> List[GroundTruthAnswer]:
        """Get all answers for a specific domain"""
        return [answer for answer in self.answers.values() if answer.domain == domain]
    
    def get_answers_by_difficulty(self, difficulty: str) -> List[GroundTruthAnswer]:
        """Get all answers for a specific difficulty level"""
        return [answer for answer in self.answers.values() if answer.difficulty == difficulty]
    
    def get_answers_by_category(self, category: str) -> List[GroundTruthAnswer]:
        """Get all answers for a specific category"""
        return [answer for answer in self.answers.values() if answer.category == category]
    
    def add_answer(self, answer: GroundTruthAnswer):
        """Add a new ground truth answer"""
        self.answers[answer.question_id] = answer
        self.save_ground_truth()
        logger.info(f"Added ground truth answer: {answer.question_id}")
    
    def update_answer(self, question_id: str, **kwargs):
        """Update an existing ground truth answer"""
        if question_id in self.answers:
            answer = self.answers[question_id]
            for key, value in kwargs.items():
                if hasattr(answer, key):
                    setattr(answer, key, value)
            self.save_ground_truth()
            logger.info(f"Updated ground truth answer: {question_id}")
        else:
            logger.warning(f"Question ID not found: {question_id}")
    
    def delete_answer(self, question_id: str):
        """Delete a ground truth answer"""
        if question_id in self.answers:
            del self.answers[question_id]
            self.save_ground_truth()
            logger.info(f"Deleted ground truth answer: {question_id}")
        else:
            logger.warning(f"Question ID not found: {question_id}")
    
    def get_all_questions(self) -> List[str]:
        """Get all question IDs"""
        return list(self.answers.keys())
    
    def get_question_text(self, question_id: str) -> Optional[str]:
        """Get question text by ID"""
        answer = self.get_answer(question_id)
        return answer.question if answer else None
    
    def search_questions(self, query: str) -> List[GroundTruthAnswer]:
        """Search questions by text"""
        query_lower = query.lower()
        results = []
        
        for answer in self.answers.values():
            if (query_lower in answer.question.lower() or 
                query_lower in answer.category.lower() or
                query_lower in answer.domain.lower()):
                results.append(answer)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about ground truth answers"""
        domains = {}
        difficulties = {}
        categories = {}
        
        for answer in self.answers.values():
            domains[answer.domain] = domains.get(answer.domain, 0) + 1
            difficulties[answer.difficulty] = difficulties.get(answer.difficulty, 0) + 1
            categories[answer.category] = categories.get(answer.category, 0) + 1
        
        return {
            "total_answers": len(self.answers),
            "domains": domains,
            "difficulties": difficulties,
            "categories": categories,
            "average_answer_length": sum(len(answer.answer) for answer in self.answers.values()) / len(self.answers)
        }
    
    def validate_answer(self, answer: GroundTruthAnswer) -> List[str]:
        """Validate a ground truth answer and return any issues"""
        issues = []
        
        if not answer.question_id:
            issues.append("Question ID is required")
        
        if not answer.question:
            issues.append("Question text is required")
        
        if not answer.answer:
            issues.append("Answer text is required")
        
        if answer.difficulty not in ['easy', 'medium', 'hard']:
            issues.append("Difficulty must be 'easy', 'medium', or 'hard'")
        
        if answer.expected_length not in ['short', 'medium', 'long']:
            issues.append("Expected length must be 'short', 'medium', or 'long'")
        
        if len(answer.key_points) == 0:
            issues.append("At least one key point is required")
        
        return issues 