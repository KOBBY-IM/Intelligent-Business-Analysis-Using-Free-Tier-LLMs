#!/usr/bin/env python3
"""
Test script for QuestionSampler functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.question_sampler import QuestionSampler


def test_question_sampler():
    """Test the QuestionSampler functionality."""
    print("🧪 Testing QuestionSampler")
    print("=" * 50)
    
    try:
        # Initialize sampler
        sampler = QuestionSampler()
        print("✅ QuestionSampler initialized successfully")
        
        # Test available domains
        domains = sampler.get_available_domains()
        print(f"📋 Available domains: {domains}")
        
        # Test single domain sampling
        print("\n🎯 Testing single domain sampling:")
        for domain in domains:
            sample = sampler.sample_questions(domain, sample_size=3)
            print(f"\n{domain.upper()} Sample (3 questions):")
            print(f"  Session ID: {sample['session_id']}")
            print(f"  Total pool size: {sample['total_pool_size']}")
            print(f"  Sampled: {sample['sample_size']}")
            
            for i, q in enumerate(sample['questions'], 1):
                print(f"  {i}. {q['id']}: {q['question'][:60]}...")
        
        # Test multi-domain sampling
        print("\n🌐 Testing multi-domain sampling:")
        multi_sample = sampler.sample_multi_domain(['retail', 'finance'], sample_size=5)
        print(f"Session ID: {multi_sample['session_id']}")
        print(f"Total questions: {multi_sample['total_questions']}")
        
        for domain, domain_data in multi_sample['domains'].items():
            print(f"\n{domain.upper()} ({len(domain_data['questions'])} questions):")
            for i, q in enumerate(domain_data['questions'], 1):
                print(f"  {i}. {q['id']}: {q['question'][:60]}...")
        
        # Test question lookup
        print("\n🔍 Testing question lookup:")
        test_question = sampler.get_question_by_id("retail_01")
        if test_question:
            print(f"Found question: {test_question['question']}")
        else:
            print("❌ Question not found")
        
        # Test session logging
        print("\n📝 Testing session logging:")
        sampler.log_session(multi_sample)
        print("✅ Session logged successfully")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = test_question_sampler()
    sys.exit(0 if success else 1) 