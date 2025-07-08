"""
Script to generate a comprehensive Markdown evaluation report for LLM model performance.

- Extracts all model performance data from the latest model_responses.json file
- Converts results into Markdown tables
- Generates an executive summary, per-prompt tables, and appendices (prompt list, scoring rubric)
- Outputs to docs/EvaluationSummary.md
"""

from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.utils.common import load_json_file, load_yaml_file

REPORT_PATH = Path(__file__).parent.parent / "docs" / "EvaluationSummary.md"
LOG_PATH = Path(__file__).parent.parent / "data" / "results"
RULES_PATH = Path(__file__).parent.parent / "config" / "evaluation_rules.yaml"

def load_json_log():
    """
    Load the latest model_responses.json file from the results directory.

    Returns:
        dict: Parsed JSON log data.
    Raises:
        FileNotFoundError: If no log file is found.
    """
    logs = sorted(LOG_PATH.glob("*model_responses.json"), reverse=True)
    if not logs:
        raise FileNotFoundError("No model_responses.json file found in data/results/")
    return load_json_file(logs[0])

def load_rules():
    """
    Load the evaluation scoring rules from the YAML config.

    Returns:
        dict: Parsed YAML rules.
    """
    return load_yaml_file(RULES_PATH)

def truncate(text, n=120):
    """
    Truncate a string to n characters, adding ellipsis if needed.

    Args:
        text (str): The string to truncate.
        n (int): Max length.
    Returns:
        str: Truncated string.
    """
    return (text[:n] + "...") if text and len(text) > n else (text or "")

def make_table(headers, rows):
    """
    Create a Markdown table from headers and rows.

    Args:
        headers (list): List of column headers.
        rows (list): List of row lists.
    Returns:
        str: Markdown table as a string.
    """
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        md += "| " + " | ".join(str(x) for x in row) + " |\n"
    return md

def main():
    """
    Main entry point: generates the evaluation report and writes it to disk.
    """
    log = load_json_log()
    rules = load_rules()
    summary = log.get("summary", {})
    prompts = log["prompts"]
    results = log["results"]

    # Executive summary
    report = f"""# LLM Evaluation Summary\n\n"""
    report += (
        f"**Total Evaluations:** {summary.get('total_evaluations', len(results))}  \n"
    )
    report += f"**Successful Evaluations:** {summary.get('successful_evaluations', sum(1 for r in results if r.get('success')))}  \n"
    report += f"**Success Rate:** {summary.get('success_rate', round(sum(1 for r in results if r.get('success'))/len(results), 2))}  \n"
    report += f"**Average Latency (s):** {summary.get('average_latency', round(mean([r['latency_seconds'] for r in results if r.get('latency_seconds') is not None]), 2))}  \n"
    report += f"**Total Tokens Used:** {summary.get('total_tokens', sum(r.get('tokens_used') or 0 for r in results))}  \n\n"

    # Aggregated model performance
    perf = defaultdict(list)
    for r in results:
        if r.get("success"):
            key = (r["provider"], r["model"])
            perf[key].append(r)
    agg_rows = []
    for (provider, model), recs in perf.items():
        agg_rows.append(
            [
                provider,
                model,
                len(recs),
                round(mean([x["latency_seconds"] for x in recs]), 2),
                (
                    round(
                        mean(
                            [x["tokens_used"] for x in recs if x.get("tokens_used") is not None]
                        ),
                        1,
                    )
                    if any(x.get("tokens_used") for x in recs)
                    else "N/A"
                ),
                round(sum(1 for x in recs if x.get("success")) / len(recs), 2),
            ]
        )
    report += "## Aggregated Model Performance\n\n"
    report += make_table(
        ["Provider", "Model", "Num Success", "Avg Latency (s)", "Avg Tokens", "Success Rate"],
        agg_rows,
    )
    report += "\n\n"

    # Per-prompt tables
    report += "## Per-Prompt Model Results\n\n"
    for i, prompt in enumerate(prompts):
        report += f"### Prompt {i+1}: {prompt['prompt']}\n"
        report += f"*Category:* {prompt.get('category','')}  |  *Context:* {prompt.get('context','')}\n\n"
        rows = []
        for r in results:
            if r["prompt_index"] == i:
                rows.append(
                    [
                        r["provider"],
                        r["model"],
                        "✅" if r.get("success") else "❌",
                        round(r["latency_seconds"], 2) if r.get("latency_seconds") is not None else "N/A",
                        r.get("tokens_used") or "N/A",
                        truncate(r.get("response_text"), 80),
                    ]
                )
        report += make_table(
            ["Provider", "Model", "Success", "Latency (s)", "Tokens", "Response (truncated)"],
            rows,
        )
        report += "\n\n"

    # Appendix A: Prompt list
    report += "---\n\n## Appendix A: Prompt List\n\n"
    for i, prompt in enumerate(prompts):
        report += f"**Prompt {i+1}**\n- Category: {prompt.get('category','')}\n- Prompt: {prompt['prompt']}\n- Context: {prompt.get('context','')}\n\n"

    # Appendix B: Scoring Rubric
    report += "---\n\n## Appendix B: Scoring Rubric\n\n"
    import yaml
    report += "```yaml\n"
    report += yaml.dump(rules, sort_keys=False)
    report += "```\n"

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {REPORT_PATH}")

if __name__ == "__main__":
    main()
