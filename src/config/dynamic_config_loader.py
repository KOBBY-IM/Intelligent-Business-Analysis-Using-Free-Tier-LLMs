from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, ValidationError, validator

from src.utils.common import load_yaml_file


class ModelConfig(BaseModel):
    name: str
    max_tokens: int

    @validator("name")
    def name_must_not_be_empty(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Model name must be a non-empty string")
        return v

    @validator("max_tokens")
    def tokens_must_be_positive(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("max_tokens must be a positive integer")
        return v


class LLMConfigLoader:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = str(
                Path(__file__).parent.parent.parent / "config" / "llm_config.yaml"
            )
        self.config_path = config_path
        self._config = self._load_and_validate()

    def _load_and_validate(self) -> Dict[str, List[ModelConfig]]:
        raw = load_yaml_file(self.config_path)
        config = {}
        for provider, models in raw.items():
            if not isinstance(models, list):
                raise ValueError(f"Models for provider {provider} must be a list")
            validated = []
            for entry in models:
                try:
                    validated.append(ModelConfig(**entry))
                except ValidationError as e:
                    raise ValueError(f"Invalid model config for {provider}: {e}")
            config[provider] = validated
        return config

    def get_models(self, provider: str) -> List[ModelConfig]:
        if provider not in self._config:
            raise ValueError(f"Provider '{provider}' not found in config")
        return self._config[provider]

    def get_all(self) -> Dict[str, List[ModelConfig]]:
        return self._config
