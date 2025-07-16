import json

REMOVE_MODELS = {"gemini-2.5-flash", "moonshotai/kimi-k2:free"}
JSON_PATH = "data/pre_generated_blind_responses.json"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

# Remove models from metadata
data["generation_metadata"]["models"] = [
    m for m in data["generation_metadata"]["models"]
    if not any(rm in m for rm in REMOVE_MODELS)
]
data["generation_metadata"]["total_models"] = len(data["generation_metadata"]["models"])

# Remove responses from each question
total_responses = 0
successful_responses = 0
for domain, questions in data["responses_by_domain"].items():
    for q in questions:
        q["responses"] = [
            r for r in q["responses"] if r["model"] not in REMOVE_MODELS
        ]
        total_responses += len(q["responses"])
        successful_responses += sum(
            not r["metadata"].get("has_error") for r in q["responses"]
        )

data["generation_metadata"]["total_responses"] = total_responses
data["generation_metadata"]["successful_responses"] = successful_responses
data["generation_metadata"]["success_rate"] = (
    (successful_responses / total_responses) * 100 if total_responses else 0
)

with open(JSON_PATH, "w") as f:
    json.dump(data, f, indent=2)

print("Cleaned data file. Only 4 models remain. Metadata updated.") 