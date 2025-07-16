#!/usr/bin/env python3
"""
Test Dataset Context Integration

This script tests the dataset context functionality for the blind evaluation system.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.question_sampler import QuestionSampler


def test_dataset_context_loading():
    """Test that dataset context loads correctly."""
    print("🧪 Testing Dataset Context Loading...")
    
    try:
        sampler = QuestionSampler()
        print("✅ QuestionSampler initialized successfully")
        
        # Test context loading
        context = sampler.get_dataset_context()
        print(f"✅ Loaded context for domains: {list(context.keys())}")
        
        # Test guidelines loading
        guidelines = sampler.get_evaluation_guidelines()
        print(f"✅ Loaded evaluation guidelines: {bool(guidelines)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset context: {e}")
        return False


def test_retail_context():
    """Test retail dataset context details."""
    print("\n🛍️ Testing Retail Dataset Context...")
    
    try:
        sampler = QuestionSampler()
        retail_ctx = sampler.get_dataset_context('retail')
        
        print(f"📊 Dataset: {retail_ctx.get('name', 'N/A')}")
        print(f"📝 Description: {retail_ctx.get('description', 'N/A')}")
        print(f"📈 Size: {retail_ctx.get('size', 'N/A')}")
        
        # Test key fields
        if 'key_fields' in retail_ctx:
            fields = retail_ctx['key_fields']
            print(f"🔑 Key field categories: {list(fields.keys())}")
            
            # Show sample fields
            if 'demographic' in fields:
                print(f"   👥 Demographics: {len(fields['demographic'])} fields")
            if 'transaction' in fields:
                print(f"   💰 Transaction: {len(fields['transaction'])} fields")
            if 'behavior' in fields:
                print(f"   📊 Behavior: {len(fields['behavior'])} fields")
        
        # Test evaluation criteria
        if 'evaluation_criteria' in retail_ctx:
            criteria = retail_ctx['evaluation_criteria']
            if 'good_answers_include' in criteria:
                print(f"✅ Good answer criteria: {len(criteria['good_answers_include'])} items")
            if 'poor_answers_include' in criteria:
                print(f"❌ Poor answer criteria: {len(criteria['poor_answers_include'])} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing retail context: {e}")
        return False


def test_finance_context():
    """Test finance dataset context details."""
    print("\n📈 Testing Finance Dataset Context...")
    
    try:
        sampler = QuestionSampler()
        finance_ctx = sampler.get_dataset_context('finance')
        
        print(f"📊 Dataset: {finance_ctx.get('name', 'N/A')}")
        print(f"📝 Description: {finance_ctx.get('description', 'N/A')}")
        print(f"📈 Size: {finance_ctx.get('size', 'N/A')}")
        
        # Test key fields
        if 'key_fields' in finance_ctx:
            fields = finance_ctx['key_fields']
            print(f"🔑 Key field categories: {list(fields.keys())}")
            
            # Show sample fields
            if 'pricing' in fields:
                print(f"   💰 Pricing: {len(fields['pricing'])} fields")
            if 'volume' in fields:
                print(f"   📊 Volume: {len(fields['volume'])} fields")
            if 'calculated_metrics' in fields:
                print(f"   🧮 Calculated: {len(fields['calculated_metrics'])} fields")
        
        # Test evaluation criteria
        if 'evaluation_criteria' in finance_ctx:
            criteria = finance_ctx['evaluation_criteria']
            if 'good_answers_include' in criteria:
                print(f"✅ Good answer criteria: {len(criteria['good_answers_include'])} items")
            if 'poor_answers_include' in criteria:
                print(f"❌ Poor answer criteria: {len(criteria['poor_answers_include'])} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing finance context: {e}")
        return False


def test_evaluation_guidelines():
    """Test evaluation guidelines functionality."""
    print("\n📋 Testing Evaluation Guidelines...")
    
    try:
        sampler = QuestionSampler()
        guidelines = sampler.get_evaluation_guidelines()
        
        if 'rating_scale' in guidelines:
            print(f"📊 Rating scale items: {len(guidelines['rating_scale'])}")
            for rating, description in guidelines['rating_scale'].items():
                rating_num = rating.split('_')[0]
                print(f"   {rating_num}: {description[:50]}...")
        
        if 'what_to_look_for' in guidelines:
            print(f"🔍 Evaluation criteria: {len(guidelines['what_to_look_for'])} items")
            for i, criterion in enumerate(guidelines['what_to_look_for'][:3], 1):
                print(f"   {i}. {criterion}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing guidelines: {e}")
        return False


def test_question_sampling_with_context():
    """Test question sampling with context information."""
    print("\n🎯 Testing Question Sampling with Context...")
    
    try:
        sampler = QuestionSampler()
        
        # Sample questions from retail
        retail_session = sampler.sample_questions('retail', sample_size=2)
        print(f"✅ Sampled {len(retail_session['questions'])} retail questions")
        
        # Check if context info is available for questions
        for question in retail_session['questions']:
            context_domain = question.get('context', '')
            if context_domain:
                context_info = sampler.get_dataset_context(context_domain)
                print(f"   📊 Question {question['id']}: {context_info.get('name', 'Unknown')}")
        
        # Sample questions from finance
        finance_session = sampler.sample_questions('finance', sample_size=2)
        print(f"✅ Sampled {len(finance_session['questions'])} finance questions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing question sampling: {e}")
        return False


def main():
    """Run all dataset context tests."""
    print("🧪 Dataset Context Integration Test")
    print("=" * 50)
    
    tests = [
        test_dataset_context_loading,
        test_retail_context,
        test_finance_context,
        test_evaluation_guidelines,
        test_question_sampling_with_context
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All dataset context tests passed!")
    else:
        print("⚠️ Some tests failed. Check the output above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 