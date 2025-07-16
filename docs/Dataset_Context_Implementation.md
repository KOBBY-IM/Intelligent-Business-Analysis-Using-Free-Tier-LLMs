# Dataset Context Implementation for LLM Evaluation

## Overview

This document describes the implementation of dataset context for the blind LLM evaluation system. The context provides testers with essential information about the datasets they're evaluating, helping them make informed assessments of LLM responses.

## Implementation Summary

### Key Features Added:
1. **Comprehensive Dataset Information**: Detailed context about retail and finance datasets
2. **Evaluation Guidelines**: Clear 5-point rating scale and evaluation criteria
3. **Domain-Specific Criteria**: Tailored expectations for retail vs finance responses
4. **Interactive UI Elements**: Expandable sections and contextual hints
5. **Tester-Friendly Interface**: Clear presentation of what constitutes good answers

### Files Modified:
- `data/evaluation_questions.yaml` - Added dataset context and guidelines
- `src/utils/question_sampler.py` - Enhanced to provide context access
- `src/ui/pages/1_Blind_Evaluation.py` - Added context display functions

## Dataset Context Details

### Retail Dataset (3,901 transactions)
- **Demographics**: Age, Gender, Location
- **Transaction Data**: Products, Categories, Prices, Seasons  
- **Customer Behavior**: Reviews, Loyalty, Payment, Shipping

### Finance Dataset (3,783 trading days)
- **Pricing Data**: Open, High, Low, Close prices
- **Volume Data**: Shares traded
- **Calculated Metrics**: Ranges, Changes, Volatility

## Benefits for Testers

1. **Informed Evaluation**: Clear understanding of underlying data
2. **Consistent Standards**: Uniform evaluation criteria across testers
3. **Domain Awareness**: Different expectations for retail vs finance
4. **Quality Benchmarks**: Specific examples of good vs poor answers
5. **Fair Assessment**: Equal context provided to all evaluators

## Test Results

All functionality tests pass:
- ✅ Dataset context loading (retail + finance domains)
- ✅ Evaluation guidelines (5-point scale + criteria)
- ✅ Question sampling integration
- ✅ UI context display functions 