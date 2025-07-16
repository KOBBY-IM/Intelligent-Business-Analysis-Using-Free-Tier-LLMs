#!/usr/bin/env python3
"""
Question Sampler for Blind Evaluation Protocol

Provides methods to randomly sample questions from domain-specific pools
for user evaluation sessions.
"""

import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class QuestionSampler:
    """
    Handles random sampling of questions for blind evaluation sessions.
    """

    def __init__(self, questions_file: str = "data/evaluation_questions.yaml"):
        """
        Initialize the question sampler.

        Args:
            questions_file (str): Path to the YAML file containing question pools.
        """
        self.questions_file = Path(questions_file)
        self.data = self._load_question_data()
        self.question_pools = self._extract_question_pools()

    def _load_question_data(self) -> Dict:
        """Load all data from YAML file."""
        if not self.questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {self.questions_file}")
        
        with open(self.questions_file, 'r') as f:
            return yaml.safe_load(f)

    def _extract_question_pools(self) -> Dict:
        """Extract just the question pools from the loaded data."""
        question_pools = {}
        for key, value in self.data.items():
            if key not in ['dataset_context', 'evaluation_guidelines'] and isinstance(value, dict) and 'questions' in value:
                question_pools[key] = value
        return question_pools

    def get_dataset_context(self, domain: str = None) -> Dict:
        """
        Get dataset context information for a specific domain or all domains.

        Args:
            domain (str, optional): Specific domain to get context for.

        Returns:
            dict: Dataset context information.
        """
        dataset_context = self.data.get('dataset_context', {})
        
        if domain:
            return dataset_context.get(domain, {})
        
        return dataset_context

    def get_evaluation_guidelines(self) -> Dict:
        """
        Get evaluation guidelines.

        Returns:
            dict: Evaluation guidelines.
        """
        return self.data.get('evaluation_guidelines', {})

    def get_available_domains(self) -> List[str]:
        """
        Get list of available question domains.

        Returns:
            list: Available domain names.
        """
        return list(self.question_pools.keys())
    
    def get_all_questions(self) -> List[Dict]:
        """
        Get all questions from all domains with domain information.
        
        Returns:
            list: All questions with domain and question_idx information.
        """
        all_questions = []
        
        for domain, domain_data in self.question_pools.items():
            questions = domain_data.get('questions', [])
            for idx, question in enumerate(questions):
                question_with_domain = question.copy()
                question_with_domain['domain'] = domain
                question_with_domain['question_idx'] = idx
                all_questions.append(question_with_domain)
        
        return all_questions

    def sample_questions(self, domain: str, sample_size: int = 5, session_id: str = None) -> Dict:
        """
        Sample questions from a specific domain.

        Args:
            domain (str): Domain to sample from.
            sample_size (int): Number of questions to sample.
            session_id (str, optional): Session ID for tracking.

        Returns:
            dict: Sample data with questions, metadata, and context.
        """
        if domain not in self.question_pools:
            raise ValueError(f"Domain '{domain}' not found. Available domains: {list(self.question_pools.keys())}")
        
        domain_data = self.question_pools[domain]
        questions = domain_data['questions']
        
        if sample_size > len(questions):
            raise ValueError(f"Sample size {sample_size} exceeds available questions ({len(questions)}) for domain '{domain}'")
        
        # Random sampling without replacement
        sampled_questions = random.sample(questions, sample_size)
        
        # Add sequential numbering for this session
        for i, question in enumerate(sampled_questions):
            question['session_order'] = i + 1
        
        # Build sample data
        sample_data = {
            'domain': domain,
            'sample_size': sample_size,
            'session_id': session_id or str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'questions': sampled_questions,
            'domain_context': domain_data.get('context', {}),
            'evaluation_criteria': domain_data.get('evaluation_criteria', [])
        }
        
        return sample_data

    def create_evaluation_session(self, domains: List[str] = None, sample_size: int = 5) -> Dict:
        """
        Create a complete evaluation session with questions from multiple domains.

        Args:
            domains (list, optional): List of domains to include. If None, includes all available domains.
            sample_size (int): Number of questions to sample per domain.

        Returns:
            dict: Complete session data with questions from all specified domains.
        """
        if domains is None:
            domains = self.get_available_domains()
        
        # Validate domains
        available_domains = self.get_available_domains()
        invalid_domains = [d for d in domains if d not in available_domains]
        if invalid_domains:
            raise ValueError(f"Invalid domains: {invalid_domains}. Available: {available_domains}")
        
        session_id = str(uuid.uuid4())
        session_data = {
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat(),
            'domains': {},
            'total_questions': 0,
            'evaluation_guidelines': self.get_evaluation_guidelines()
        }

        for domain in domains:
            domain_sample = self.sample_questions(domain, sample_size, session_id)
            session_data['domains'][domain] = domain_sample
            session_data['total_questions'] += len(domain_sample['questions'])

        return session_data

    def get_question_by_id(self, question_id: str) -> Optional[Dict]:
        """
        Get a specific question by its ID.

        Args:
            question_id (str): The question ID to look up.

        Returns:
            dict: Question data if found, None otherwise.
        """
        for domain_data in self.question_pools.values():
            for question in domain_data['questions']:
                if question['id'] == question_id:
                    return question
        return None

    def log_session(self, session_data: Dict, log_file: str = "data/evaluation_results/question_sessions.json"):
        """
        Log the session data for reproducibility and analysis.

        Args:
            session_data (dict): Session data to log.
            log_file (str): Path to the log file.
        """
        import json
        
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing logs
        if log_path.exists():
            with open(log_path, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Append new session
        logs.append(session_data)
        
        # Save updated logs
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2) 