import json
from pathlib import Path
from typing import List

def get_available_industries(data_path: str = "data/blind_responses.json") -> List[str]:
    """Return a list of all available industries in the blind_responses.json file."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"{data_path} not found.")
    with open(path, 'r') as f:
        data = json.load(f)
    return list(data.keys()) 