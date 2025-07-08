import argparse
from pathlib import Path

import pandas as pd

from src.utils.common import load_json_file

LOG_PATH = Path(__file__).parent.parent / "data" / "results" / "evaluation_log.jsonl"
EXPORT_PATH = (
    Path(__file__).parent.parent / "data" / "results" / "evaluation_results.csv"
)


def load_logs(log_path):
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return []
    # Use the new utility for JSONL
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(load_json_file(line))
            except Exception as e:
                print(f"Skipping invalid line: {e}")
    return records


def main(provider=None, model=None):
    records = load_logs(LOG_PATH)
    if not records:
        print("No records found.")
        return
    df = pd.DataFrame(records)
    # Only keep relevant columns if they exist
    keep_cols = ["prompt", "model", "provider", "response", "score", "latency"]
    cols = [c for c in keep_cols if c in df.columns]
    df = df[cols]
    # Apply filters
    if provider:
        df = df[df["provider"] == provider]
    if model:
        df = df[df["model"] == model]
    # Export
    df.to_csv(EXPORT_PATH, index=False)
    print(f"Exported {len(df)} records to {EXPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate evaluation logs to CSV.")
    parser.add_argument("--provider", type=str, help="Filter by provider name")
    parser.add_argument("--model", type=str, help="Filter by model name")
    args = parser.parse_args()
    main(provider=args.provider, model=args.model)
