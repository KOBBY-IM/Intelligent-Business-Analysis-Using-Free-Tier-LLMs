import json

removed = {'gemini-2.5-flash', 'moonshotai/kimi-k2:free'}
found = False

with open('data/pre_generated_blind_responses.json') as f:
    d = json.load(f)

for domain, questions in d['responses_by_domain'].items():
    for i, q in enumerate(questions):
        for r in q['responses']:
            if r.get('model', '') in removed:
                print(f"Found {r['model']} in {domain} Q{i+1}")
                found = True

print('Check complete.')
print('Any removed models found?', found)
print('Current models in metadata:', d['generation_metadata']['models']) 