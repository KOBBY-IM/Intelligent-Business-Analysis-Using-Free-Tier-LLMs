"""
ModelMetadataLoader utility for loading and enriching model metadata from YAML config.

Provides methods to retrieve metadata for models and enrich log entries with tags and attributes.
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml


class ModelMetadataLoader:
    """
    Loader for model metadata from YAML config.

    Provides methods to retrieve metadata for models and enrich log entries.
    """
    def __init__(self, metadata_path: str = None):
        """
        Initialize the loader and load metadata from YAML file.

        Args:
            metadata_path (str, optional): Path to the YAML metadata file. Defaults to config/model_metadata.yaml.
        """
        if metadata_path is None:
            metadata_path = str(
                Path(__file__).parent.parent.parent / "config" / "model_metadata.yaml"
            )
        with open(metadata_path, "r") as f:
            self.metadata_list = yaml.safe_load(f)
        # Index metadata by (provider, name) tuple for fast lookup
        self.meta_by_key = {(m["provider"], m["name"]): m for m in self.metadata_list}

    def get_metadata(self, provider: str, name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific provider/model.

        Args:
            provider (str): Provider name.
            name (str): Model name.
        Returns:
            dict: Metadata dictionary or empty dict if not found.
        """
        return self.meta_by_key.get((provider, name), {})

    def enrich_log(self, log_entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a log entry with model metadata fields (family, base_model, token_limit, speed_category, tags).

        Args:
            log_entry (dict): Log entry with at least 'provider' and 'model'.
        Returns:
            dict: Enriched log entry.
        """
        provider = log_entry.get("provider")
        model = log_entry.get("model")
        meta = self.get_metadata(provider, model)
        enriched = log_entry.copy()
        enriched.update(
            {
                "family": meta.get("family"),
                "base_model": meta.get("base_model"),
                "token_limit": meta.get("token_limit"),
                "speed_category": meta.get("speed_category"),
                "tags": meta.get("tags", []),
            }
        )
        return enriched

    def get_tagged_models(self, tag: str) -> List[Dict[str, Any]]:
        """
        Get all models with a given tag.

        Args:
            tag (str): Tag to filter by.
        Returns:
            list: List of metadata dicts for models with the tag.
        """
        return [m for m in self.metadata_list if tag in m.get("tags", [])]
