#!/usr/bin/env python3
"""
Pre-generate LLM Responses for Blind Evaluation
Generates responses from all LLM models with RAG context for consistent testing
Enhanced with robust retry mechanisms and fair LLM opportunity distribution
"""

import json
import sys
import time
import hashlib
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.question_sampler import QuestionSampler
from llm_providers.provider_manager import ProviderManager
from rag.csv_rag_pipeline import CSVRAGPipeline


def setup_rag_pipeline(domain):
    """Set up RAG pipeline for the given domain."""
    print(f"Setting up RAG pipeline for {domain}...")
    
    # Map domain to dataset
    dataset_files = {
        "retail": ["data/shopping_trends.csv"],
        "finance": ["data/Tesla_stock_data.csv"]
    }
    
    csv_files = dataset_files.get(domain, ["data/shopping_trends.csv"])
    
    # Initialize and build RAG pipeline
    rag_pipeline = CSVRAGPipeline()
    rag_pipeline.build_index(csv_files, chunk_size=200)
    
    print(f"✅ RAG pipeline ready for {domain} with {len(csv_files)} dataset(s)")
    return rag_pipeline, csv_files


def generate_response_with_rag(manager, provider_name, model_name, question, rag_context, domain, max_retries=5, base_delay=3):
    """Generate a single response with RAG context with enhanced retry logic."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Progressive delay: 3s, 6s, 12s, 24s, 48s
                delay = base_delay * (2 ** (attempt - 1))
                # Add random jitter to avoid thundering herd
                jitter = random.uniform(0.5, 1.5)
                actual_delay = delay * jitter
                
                print(f"      ⏳ Retry {attempt + 1}/{max_retries} for {model_name} in {actual_delay:.1f}s...")
                time.sleep(actual_delay)
            
            start_time = time.time()
            
            # Create enhanced prompt with RAG context
            enhanced_prompt = f"""
Business Scenario: {question['question']}

Dataset Context: {rag_context}

Additional Context: {question.get('context', '')}

Based on the business scenario above and the relevant data insights from the {domain} dataset, please provide a comprehensive analysis and actionable recommendations. Focus on:
1. Data-driven insights from the provided context
2. Practical business recommendations
3. Specific actions that can be taken based on the data patterns
4. Key performance indicators to monitor

Please ensure your response is grounded in the actual data provided and offers specific, actionable business intelligence.
"""
            
            response = manager.generate_response(
                provider_name=provider_name,
                query=enhanced_prompt,
                model=model_name
            )
            end_time = time.time()
            
            if response and response.success and response.text:
                # Calculate quality metrics
                response_text = response.text
                word_count = len(response_text.split())
                sentence_count = len([s for s in response_text.split('.') if s.strip()])
                
                return {
                    'model': model_name,
                    'provider': provider_name.lower(),
                    'response': response_text,
                    'metadata': {
                        'latency': round(end_time - start_time, 2),
                        'word_count': word_count,
                        'sentence_count': sentence_count,
                        'character_count': len(response_text),
                        'timestamp': datetime.utcnow().isoformat(),
                        'has_code_blocks': '```' in response_text,
                        'has_bullet_points': any(line.strip().startswith(('•', '-', '*', '1.', '2.')) 
                                               for line in response_text.split('\n')),
                        'response_structure_score': min(1.0, (word_count / 100) * 0.5 + 
                                                      (sentence_count / 10) * 0.3 + 
                                                      (0.2 if '```' in response_text else 0)),
                        'uses_rag': True,
                        'domain': domain,
                        'rag_context_length': len(rag_context),
                        'rag_context_preview': rag_context[:200] + "..." if len(rag_context) > 200 else rag_context,
                        'retry_attempts': attempt + 1,
                        'total_retry_time': (time.time() - start_time) if attempt > 0 else 0
                    }
                }
            else:
                # Store error for potential retry
                error_msg = response.error if response and hasattr(response, 'error') else "No response generated"
                last_error = error_msg
                print(f"      ⚠️  Attempt {attempt + 1} failed: {error_msg}")
                if attempt < max_retries - 1:
                    continue  # Try again
                
        except Exception as e:
            last_error = str(e)
            print(f"      ⚠️  Attempt {attempt + 1} exception: {str(e)}")
            if attempt < max_retries - 1:
                continue  # Try again
    
    # All retries failed, return error response
    print(f"   ❌ All {max_retries} attempts failed for {model_name}: {last_error}")
    return {
        'model': model_name,
        'provider': provider_name.lower(),
        'response': f"❌ Error after {max_retries} attempts: {last_error}",
        'metadata': {
            'latency': 0,
            'word_count': 0,
            'sentence_count': 0,
            'character_count': 0,
            'timestamp': datetime.utcnow().isoformat(),
            'has_error': True,
            'error_message': last_error,
            'uses_rag': True,
            'domain': domain,
            'rag_context_length': len(rag_context),
            'rag_context_preview': rag_context[:200] + "..." if len(rag_context) > 200 else rag_context,
            'retry_attempts': max_retries,
            'failed_after_retries': True
        }
    }


def check_question_completion_status(results, total_models):
    """
    Check which questions are fully answered and which still need retries.
    
    Args:
        results: Current results structure
        total_models: Total number of models expected to respond
        
    Returns:
        dict: Question completion status by domain and question index
    """
    completion_status = {}
    
    for domain, questions in results['responses_by_domain'].items():
        completion_status[domain] = {}
        for q_idx, question_data in enumerate(questions):
            successful_responses = len([r for r in question_data['responses'] 
                                      if not r['metadata'].get('has_error')])
            completion_status[domain][q_idx] = {
                'successful': successful_responses,
                'total': len(question_data['responses']),
                'fully_answered': successful_responses == total_models,
                'needs_retry': successful_responses < total_models
            }
    
    return completion_status


def filter_retries_by_question_status(failed_responses, completion_status):
    """
    Filter failed responses to only include those from questions that still need answers.
    
    Args:
        failed_responses: List of failed response attempts
        completion_status: Question completion status from check_question_completion_status
        
    Returns:
        tuple: (filtered_failures, skipped_count, fully_answered_questions)
    """
    filtered_failures = []
    skipped_count = 0
    fully_answered_questions = set()
    
    for failure in failed_responses:
        domain = failure['domain']
        q_idx = failure['question_id']
        
        # Check if this question still needs retries
        question_status = completion_status.get(domain, {}).get(q_idx, {})
        
        if question_status.get('needs_retry', True):
            filtered_failures.append(failure)
        else:
            skipped_count += 1
            fully_answered_questions.add(f"{domain}_Q{q_idx + 1}")
    
    return filtered_failures, skipped_count, fully_answered_questions


def retry_failed_responses(manager, failed_responses, rag_contexts, results, total_models, retry_round=1, max_retry_rounds=3):
    """
    Retry failed responses in batches to give all models fair opportunities.
    Only retries questions that haven't been fully answered by all models.
    
    Args:
        manager: ProviderManager instance
        failed_responses: List of failed response attempts with their question context
        rag_contexts: Dictionary mapping question IDs to their RAG contexts
        results: Current results structure to check completion status
        total_models: Total number of models expected to respond
        retry_round: Current retry round (1-based)
        max_retry_rounds: Maximum number of retry rounds
    """
    if not failed_responses or retry_round > max_retry_rounds:
        return []
    
    print(f"\n🔄 RETRY ROUND {retry_round}/{max_retry_rounds}")
    print(f"📊 Initial failures to consider: {len(failed_responses)}")
    
    # Check which questions are already fully answered
    completion_status = check_question_completion_status(results, total_models)
    
    # Filter out failures from questions that are already fully answered
    filtered_failures, skipped_count, fully_answered = filter_retries_by_question_status(
        failed_responses, completion_status
    )
    
    if skipped_count > 0:
        print(f"✅ Skipping {skipped_count} failures from {len(fully_answered)} fully answered questions:")
        for question_id in sorted(fully_answered):
            print(f"   - {question_id} (all models responded)")
    
    if not filtered_failures:
        print("🎉 All remaining questions are fully answered! No retries needed.")
        return []
    
    print(f"🔄 Retrying {len(filtered_failures)} failures from incomplete questions...")
    
    # Group failures by model to understand patterns
    failure_by_model = defaultdict(int)
    failure_by_question = defaultdict(int)
    
    for failure in filtered_failures:
        failure_by_model[failure['model']] += 1
        question_key = f"{failure['domain']}_Q{failure['question_id'] + 1}"
        failure_by_question[question_key] += 1
    
    print("📈 Failure distribution:")
    print("   By model:")
    for model, count in failure_by_model.items():
        print(f"     {model}: {count} failures")
    print("   By question:")
    for question, count in sorted(failure_by_question.items()):
        print(f"     {question}: {count} models failed")
    
    # Randomize order to give different models first chance
    random.shuffle(filtered_failures)
    
    successful_retries = []
    still_failed = []
    
    for idx, failure_info in enumerate(filtered_failures, 1):
        model_name = failure_info['model']
        provider_name = failure_info['provider']
        question = failure_info['question']
        domain = failure_info['domain']
        question_id = failure_info['question_id']
        
        print(f"   🔄 Retry {idx}/{len(filtered_failures)}: {model_name} for Q{question_id + 1}")
        
        # Get RAG context for this question
        rag_context = rag_contexts.get(f"{domain}_{question_id}", "")
        
        # Retry with increased delays for this round
        response = generate_response_with_rag(
            manager, provider_name, model_name, question, rag_context, domain,
            max_retries=3,  # Fewer retries per attempt in batch mode
            base_delay=5 * retry_round  # Longer base delay for each round
        )
        
        if response['metadata'].get('has_error'):
            still_failed.append(failure_info)
            print(f"      ❌ Still failed")
        else:
            successful_retries.append(response)
            print(f"      ✅ Success! ({response['metadata']['word_count']} words)")
        
        # Brief pause between models to avoid overwhelming APIs
        if idx < len(filtered_failures):
            time.sleep(2)
    
    print(f"🎯 Retry round {retry_round} complete: {len(successful_retries)} recovered, {len(still_failed)} still failing")
    
    # If we have more rounds and still have failures, continue
    if still_failed and retry_round < max_retry_rounds:
        print(f"⏭️  Proceeding to retry round {retry_round + 1}...")
        additional_recoveries = retry_failed_responses(
            manager, still_failed, rag_contexts, results, total_models, retry_round + 1, max_retry_rounds
        )
        successful_retries.extend(additional_recoveries)
    
    return successful_retries


def generate_all_responses():
    """Generate responses for all questions across all domains with enhanced retry logic and LLM health checks."""
    print("🚀 Starting pre-generation of LLM responses with RAG context...")
    print("🔧 Enhanced with progressive retry delays, health checks, and fair opportunity distribution")
    
    # Initialize components
    sampler = QuestionSampler()
    manager = ProviderManager()
    
    # Get available models
    all_models = manager.get_all_models()
    model_provider_pairs = []
    for provider_name, provider_models in all_models.items():
        for model_name in provider_models:
            model_provider_pairs.append((provider_name, model_name))
    
    print(f"📊 Found {len(model_provider_pairs)} model-provider pairs:")
    for provider, model in model_provider_pairs:
        print(f"   - {provider}: {model}")
    
    # Get available domains
    domains = sampler.get_available_domains()
    print(f"🏢 Processing domains: {domains}")
    
    # Results structure
    results = {
        'generation_metadata': {
            'timestamp': datetime.utcnow().isoformat(),
            'total_models': len(model_provider_pairs),
            'domains': domains,
            'models': [f"{p}:{m}" for p, m in model_provider_pairs],
            'enhanced_retry_system': True,
            'llm_health_check': True,
            'llm_health_retry_minutes': 10
        },
        'responses_by_domain': {}
    }
    
    total_questions = 0
    all_failed_responses = []  # Track all failures for batch retry
    all_rag_contexts = {}  # Store RAG contexts for retry
    
    # Health check cache (provider_name: bool)
    health_status = {}
    health_retry_queue = set()
    health_retry_attempts = {}
    MAX_HEALTH_RETRIES = 6  # 1 hour max
    
    # Process each domain
    for domain in domains:
        print(f"\n🔍 Processing {domain.upper()} domain...")
        
        # Set up RAG pipeline for this domain
        rag_pipeline, csv_files = setup_rag_pipeline(domain)
        
        # Get ALL questions for this domain (not just 5 samples)
        domain_pool = sampler.question_pools[domain]
        questions = domain_pool['questions']
        
        print(f"📝 Found {len(questions)} questions for {domain}")
        
        domain_results = []
        
        # Process each question
        for q_idx, question in enumerate(questions, 1):
            print(f"\n   📋 Question {q_idx}/{len(questions)}: {question['question'][:60]}...")
            
            # Generate RAG context for the question with optimized retrieval
            question_id = f"{domain}_{q_idx-1}"
            rag_context = rag_pipeline.generate_context(question['question'], top_k=5, question_id=question_id)
            print(f"   🔍 Generated RAG context: {len(rag_context)} characters")
            
            # Store RAG context for potential retries
            all_rag_contexts[question_id] = rag_context
            
            question_responses = []
            
            # Health check and retry logic for each model
            pending_models = list(model_provider_pairs)
            health_retry_queue.clear()
            health_retry_attempts.clear()
            
            while pending_models:
                next_pending = []
                for m_idx, (provider_name, model_name) in enumerate(pending_models, 1):
                    # Health check (cache per provider)
                    if provider_name not in health_status:
                        print(f"   🩺 Checking health for provider {provider_name}...")
                        is_healthy = manager.health_check(provider_name)
                        health_status[provider_name] = is_healthy
                        if not is_healthy:
                            print(f"      ❌ Provider {provider_name} is UNAVAILABLE. Will retry after 10 minutes.")
                            health_retry_queue.add(provider_name)
                            health_retry_attempts[provider_name] = health_retry_attempts.get(provider_name, 0) + 1
                            next_pending.append((provider_name, model_name))
                            continue
                    elif not health_status[provider_name]:
                        print(f"   ⏳ Skipping {provider_name}:{model_name} (provider still unavailable)")
                        next_pending.append((provider_name, model_name))
                        continue
                    print(f"   🤖 Generating response {m_idx}/{len(model_provider_pairs)}: {model_name}...")
                    response = generate_response_with_rag(
                        manager, provider_name, model_name, question, rag_context, domain,
                        max_retries=3,  # Initial attempt with moderate retries
                        base_delay=2    # Start with shorter delays
                    )
                    question_responses.append(response)
                    if response['metadata'].get('has_error'):
                        all_failed_responses.append({
                            'model': model_name,
                            'provider': provider_name,
                            'question': question,
                            'domain': domain,
                            'question_id': q_idx - 1,
                            'domain_key': domain,
                            'response_index': len(question_responses) - 1
                        })
                        print(f"      ❌ Failed (will retry in batch)")
                    else:
                        print(f"      ✅ Success ({response['metadata']['word_count']} words)")
                    # Small delay between models to be respectful to APIs
                    if m_idx < len(model_provider_pairs):
                        time.sleep(1)
                # If any providers were unavailable, wait 10 minutes and retry only those
                if next_pending:
                    print(f"   ⏳ Waiting 10 minutes before retrying unavailable providers: {list(health_retry_queue)}")
                    time.sleep(600)  # 10 minutes
                    # Re-check health for all unavailable providers
                    for provider_name in list(health_retry_queue):
                        print(f"   🩺 Re-checking health for provider {provider_name}...")
                        is_healthy = manager.health_check(provider_name)
                        health_status[provider_name] = is_healthy
                        if is_healthy:
                            print(f"      ✅ Provider {provider_name} is now AVAILABLE.")
                            health_retry_queue.remove(provider_name)
                        else:
                            print(f"      ❌ Provider {provider_name} still UNAVAILABLE.")
                            health_retry_attempts[provider_name] += 1
                            if health_retry_attempts[provider_name] >= MAX_HEALTH_RETRIES:
                                print(f"      🚨 Provider {provider_name} exceeded max health retries. Skipping for this question.")
                                # Remove from queue and skip for this question
                                health_retry_queue.remove(provider_name)
                                # Remove all models for this provider from next_pending
                                next_pending = [mp for mp in next_pending if mp[0] != provider_name]
                    pending_models = next_pending
                else:
                    break  # All providers healthy or processed
            # Store question and its responses
            question_data = {
                'question': question['question'],
                'context': question.get('context', ''),
                'question_metadata': {
                    'domain': domain,
                    'question_index': q_idx - 1,
                    'rag_context_length': len(rag_context),
                    'dataset_files': csv_files
                },
                'responses': question_responses
            }
            domain_results.append(question_data)
            total_questions += 1
        results['responses_by_domain'][domain] = domain_results
        print(f"✅ Completed {domain}: {len(questions)} questions, {len(model_provider_pairs)} models each")
    # Now perform batch retries for all failed responses
    if all_failed_responses:
        print(f"\n🔄 BATCH RETRY PHASE")
        print(f"📊 Total failures to retry: {len(all_failed_responses)}")
        # Perform batch retries with enhanced logic
        recovered_responses = retry_failed_responses_with_health(
            manager, all_failed_responses, all_rag_contexts, results, len(model_provider_pairs),
            retry_round=1, max_retry_rounds=3
        )
        print(f"🎉 Batch retry recovered {len(recovered_responses)} responses!")
        # Update results with recovered responses
        updated_count = 0
        for recovered in recovered_responses:
            # Find the original failed response and replace it
            found = False
            for domain_name, domain_data in results['responses_by_domain'].items():
                if found:
                    break
                for q_idx, question_data in enumerate(domain_data):
                    if found:
                        break
                    for i, response in enumerate(question_data['responses']):
                        if (response['model'] == recovered['model'] and 
                            response['provider'] == recovered['provider'] and
                            response['metadata'].get('has_error')):
                            question_data['responses'][i] = recovered
                            updated_count += 1
                            print(f"   ✅ Updated {recovered['model']} response for {domain_name} Q{q_idx + 1}")
                            found = True
                            break
        print(f"📝 Successfully updated {updated_count}/{len(recovered_responses)} recovered responses in results")
    # Calculate summary statistics
    total_responses = 0
    successful_responses = 0
    for domain_data in results['responses_by_domain'].values():
        for question_data in domain_data:
            for response in question_data['responses']:
                total_responses += 1
                if not response['metadata'].get('has_error'):
                    successful_responses += 1
    results['generation_metadata'].update({
        'total_questions': total_questions,
        'total_responses': total_responses,
        'successful_responses': successful_responses,
        'success_rate': (successful_responses / total_responses) * 100 if total_responses > 0 else 0,
        'initial_failures': len(all_failed_responses),
        'recovered_responses': len(recovered_responses) if all_failed_responses else 0
    })
    return results

# --- New batch retry with health check logic ---
def retry_failed_responses_with_health(manager, failed_responses, rag_contexts, results, total_models, retry_round=1, max_retry_rounds=3):
    """
    Retry failed responses in batches, but check LLM health before retrying. If a provider is unhealthy, wait 10 minutes and retry only those.
    """
    if not failed_responses or retry_round > max_retry_rounds:
        return []
    print(f"\n🔄 RETRY ROUND {retry_round}/{max_retry_rounds} (with health checks)")
    # Check which questions are already fully answered
    completion_status = check_question_completion_status(results, total_models)
    filtered_failures, skipped_count, fully_answered = filter_retries_by_question_status(
        failed_responses, completion_status
    )
    if skipped_count > 0:
        print(f"✅ Skipping {skipped_count} failures from {len(fully_answered)} fully answered questions:")
        for question_id in sorted(fully_answered):
            print(f"   - {question_id} (all models responded)")
    if not filtered_failures:
        print("🎉 All remaining questions are fully answered! No retries needed.")
        return []
    # Group by provider for health check
    provider_to_failures = {}
    for failure in filtered_failures:
        provider_to_failures.setdefault(failure['provider'], []).append(failure)
    health_status = {}
    health_retry_queue = set()
    health_retry_attempts = {}
    MAX_HEALTH_RETRIES = 6
    successful_retries = []
    still_failed = []
    pending_failures = filtered_failures.copy()
    while pending_failures:
        next_pending = []
        for failure_info in pending_failures:
            provider_name = failure_info['provider']
            model_name = failure_info['model']
            question = failure_info['question']
            domain = failure_info['domain']
            question_id = failure_info['question_id']
            # Health check (cache per provider)
            if provider_name not in health_status:
                print(f"   🩺 Checking health for provider {provider_name}...")
                is_healthy = manager.health_check(provider_name)
                health_status[provider_name] = is_healthy
                if not is_healthy:
                    print(f"      ❌ Provider {provider_name} is UNAVAILABLE. Will retry after 10 minutes.")
                    health_retry_queue.add(provider_name)
                    health_retry_attempts[provider_name] = health_retry_attempts.get(provider_name, 0) + 1
                    next_pending.append(failure_info)
                    continue
            elif not health_status[provider_name]:
                print(f"   ⏳ Skipping {provider_name}:{model_name} (provider still unavailable)")
                next_pending.append(failure_info)
                continue
            print(f"   🔄 Retrying {model_name} for Q{question_id + 1}")
            rag_context = rag_contexts.get(f"{domain}_{question_id}", "")
            response = generate_response_with_rag(
                manager, provider_name, model_name, question, rag_context, domain,
                max_retries=3, base_delay=5 * retry_round
            )
            if response['metadata'].get('has_error'):
                still_failed.append(failure_info)
                print(f"      ❌ Still failed")
            else:
                successful_retries.append(response)
                print(f"      ✅ Success! ({response['metadata']['word_count']} words)")
            time.sleep(2)
        # If any providers were unavailable, wait 10 minutes and retry only those
        if next_pending:
            print(f"   ⏳ Waiting 10 minutes before retrying unavailable providers: {list(health_retry_queue)}")
            time.sleep(600)
            for provider_name in list(health_retry_queue):
                print(f"   🩺 Re-checking health for provider {provider_name}...")
                is_healthy = manager.health_check(provider_name)
                health_status[provider_name] = is_healthy
                if is_healthy:
                    print(f"      ✅ Provider {provider_name} is now AVAILABLE.")
                    health_retry_queue.remove(provider_name)
                else:
                    print(f"      ❌ Provider {provider_name} still UNAVAILABLE.")
                    health_retry_attempts[provider_name] += 1
                    if health_retry_attempts[provider_name] >= MAX_HEALTH_RETRIES:
                        print(f"      🚨 Provider {provider_name} exceeded max health retries. Skipping for this batch.")
                        health_retry_queue.remove(provider_name)
                        # Remove all failures for this provider from next_pending
                        next_pending = [f for f in next_pending if f['provider'] != provider_name]
            pending_failures = next_pending
        else:
            break
    print(f"🎯 Retry round {retry_round} complete: {len(successful_retries)} recovered, {len(still_failed)} still failing")
    # If we have more rounds and still have failures, continue
    if still_failed and retry_round < max_retry_rounds:
        print(f"⏭️  Proceeding to retry round {retry_round + 1}...")
        additional_recoveries = retry_failed_responses_with_health(
            manager, still_failed, rag_contexts, results, total_models, retry_round + 1, max_retry_rounds
        )
        successful_retries.extend(additional_recoveries)
    return successful_retries


def save_results(results):
    """Save pre-generated responses to file."""
    output_file = Path("data/pre_generated_blind_responses.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save a backup with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = Path(f"data/backups/pre_generated_responses_{timestamp}.json")
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(backup_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved:")
    print(f"   Main file: {output_file}")
    print(f"   Backup: {backup_file}")
    
    return output_file


def print_summary(results):
    """Print generation summary with enhanced retry statistics."""
    metadata = results['generation_metadata']
    
    print(f"\n📊 GENERATION SUMMARY")
    print(f"=" * 50)
    print(f"🕒 Generated at: {metadata['timestamp']}")
    print(f"🏢 Domains: {len(metadata['domains'])} ({', '.join(metadata['domains'])})")
    print(f"🤖 Models: {metadata['total_models']}")
    print(f"📝 Total questions: {metadata['total_questions']} (ALL questions for comprehensive pool)")
    print(f"💬 Total responses: {metadata['total_responses']}")
    print(f"✅ Successful: {metadata['successful_responses']}")
    print(f"📈 Success rate: {metadata['success_rate']:.1f}%")
    
    # Enhanced retry statistics
    if metadata.get('enhanced_retry_system'):
        print(f"\n🔄 RETRY SYSTEM PERFORMANCE")
        print(f"-" * 35)
        initial_failures = metadata.get('initial_failures', 0)
        recovered = metadata.get('recovered_responses', 0)
        
        print(f"🚨 Initial failures: {initial_failures}")
        print(f"🔧 Recovered responses: {recovered}")
        if initial_failures > 0:
            recovery_rate = (recovered / initial_failures) * 100
            print(f"🎯 Recovery rate: {recovery_rate:.1f}%")
        print(f"✨ Progressive delays & jitter enabled")
        print(f"🎲 Randomized retry order for fairness")
    
    print(f"\n🔍 DOMAIN BREAKDOWN")
    print(f"-" * 30)
    for domain, questions in results['responses_by_domain'].items():
        print(f"{domain.upper()}: {len(questions)} questions (evaluators will see 5 random)")
        
        # Show summary stats with retry information
        total_successful = 0
        total_responses = 0
        retry_stats = {'no_retry': 0, 'with_retry': 0, 'failed_final': 0}
        failed_questions = []
        
        for i, q in enumerate(questions, 1):
            successful = 0
            total = len(q['responses'])
            
            for response in q['responses']:
                if not response['metadata'].get('has_error'):
                    successful += 1
                    # Check if this was recovered through retries
                    retry_attempts = response['metadata'].get('retry_attempts', 1)
                    if retry_attempts > 1:
                        retry_stats['with_retry'] += 1
                    else:
                        retry_stats['no_retry'] += 1
                else:
                    retry_stats['failed_final'] += 1
            
            total_successful += successful
            total_responses += total
            
            if successful < total:
                failed_questions.append(f"Q{i}: {successful}/{total}")
        
        print(f"  Overall: {total_successful}/{total_responses} responses successful")
        print(f"  🎯 First-try success: {retry_stats['no_retry']}")
        print(f"  🔄 Recovered via retry: {retry_stats['with_retry']}")
        if retry_stats['failed_final'] > 0:
            print(f"  ❌ Final failures: {retry_stats['failed_final']}")
        if failed_questions:
            print(f"  ⚠️  Questions with partial failures: {', '.join(failed_questions[:5])}")
            if len(failed_questions) > 5:
                print(f"     ... and {len(failed_questions) - 5} more")
    
    print(f"\n💡 NOTE: Evaluators will randomly see 5 questions per domain from this complete pool.")
    print(f"🔧 Enhanced retry system ensures maximum LLM participation fairness.")


def print_missing_responses_stats(results):
    print("\n=== Missing Model Responses Stats ===")
    for domain, questions in results['responses_by_domain'].items():
        print(f"\nDomain: {domain}")
        for i, q in enumerate(questions):
            missing = [r['model'] for r in q['responses'] if r['metadata'].get('has_error')]
            if missing:
                print(f"  Q{i+1}: MISSING -> {missing}")
            else:
                print(f"  Q{i+1}: OK (all models answered)")


def main():
    """Main execution function."""
    try:
        print("🎯 Pre-generating LLM responses for consistent blind evaluation...")
        
        # Generate all responses
        results = generate_all_responses()
        
        # Save results
        output_file = save_results(results)
        
        # Print summary
        print_summary(results)
        
        print(f"\n🎉 Pre-generation complete!")
        print(f"📁 Responses saved to: {output_file}")
        print(f"🔄 The blind evaluation will now use these consistent responses.")
        
    except Exception as e:
        print(f"\n❌ Error during pre-generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def retry_kimi_k2_finance_q6_q10():
    import json
    results_path = "data/pre_generated_blind_responses.json"
    with open(results_path, "r") as f:
        results = json.load(f)
    
    finance = results['responses_by_domain']['finance']
    # Questions 6-10 are indices 5-9
    target_indices = range(5, 10)
    target_model = 'moonshotai/kimi-k2:free'
    missing = []
    rag_contexts = {}
    from utils.question_sampler import QuestionSampler
    from llm_providers.provider_manager import ProviderManager
    from rag.csv_rag_pipeline import CSVRAGPipeline
    sampler = QuestionSampler()
    manager = ProviderManager()
    rag_pipeline, _ = setup_rag_pipeline('finance')
    questions = sampler.question_pools['finance']['questions']
    for idx in target_indices:
        q = finance[idx]
        # Check if kimi-k2:free is missing or errored
        found = False
        for r in q['responses']:
            if r['model'] == target_model:
                if r['metadata'].get('has_error'):
                    found = True
                else:
                    found = False
                break
        else:
            found = True  # Not present at all
        if found:
            # Prepare retry info
            rag_context = rag_pipeline.generate_context(questions[idx]['question'], top_k=3)
            rag_contexts[f'finance_{idx}'] = rag_context
            missing.append({
                'model': 'kimi-k2:free',
                'provider': 'openrouter',
                'question': questions[idx],
                'domain': 'finance',
                'question_id': idx,
                'domain_key': 'finance',
                'response_index': None  # Will update below
            })
    if not missing:
        print("No missing kimi-k2:free responses for finance Q6-Q10.")
        return
    print(f"Retrying {len(missing)} missing kimi-k2:free responses for finance Q6-Q10...")
    # Use retry logic
    recovered = retry_failed_responses_with_health(manager, missing, rag_contexts, results, results['generation_metadata']['total_models'], retry_round=1, max_retry_rounds=3)
    # Update results
    updated = 0
    for m in missing:
        idx = m['question_id']
        q = finance[idx]
        for i, r in enumerate(q['responses']):
            if r['model'] == target_model and r['metadata'].get('has_error'):
                # Replace with recovered if available
                for rec in recovered:
                    if rec['model'] == target_model and rec['provider'] == 'openrouter':
                        q['responses'][i] = rec
                        updated += 1
                        break
                break
        else:
            # Not present, append if recovered
            for rec in recovered:
                if rec['model'] == target_model and rec['provider'] == 'openrouter':
                    q['responses'].append(rec)
                    updated += 1
                    break
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Updated {updated} kimi-k2:free responses for finance Q6-Q10.")
    print_missing_responses_stats(results)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--retry-kimi-k2-finance":
        retry_kimi_k2_finance_q6_q10()
    else:
        import json
        results_path = "data/pre_generated_blind_responses.json"
        with open(results_path, "r") as f:
            results = json.load(f)
        print_missing_responses_stats(results)
        exit_code = main()
        sys.exit(exit_code) 