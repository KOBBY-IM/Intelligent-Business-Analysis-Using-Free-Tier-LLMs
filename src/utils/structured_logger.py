"""
StructuredLogger utility for thread-safe, structured logging of LLM evaluation results.

This logger writes JSONL logs with prompt, provider, model, response, latency, tokens, and timestamp.
Logs are organized by provider and run, with optional verbosity.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredLogger:
    """
    Thread-safe, structured logger for LLM evaluation results.

    Logs are written as JSON lines, organized by provider and run ID.
    """
    def __init__(
        self, provider: str, run_id: Optional[str] = None, verbose: bool = False
    ):
        """
        Initialize the logger.

        Args:
            provider (str): Provider name (used for log directory).
            run_id (Optional[str]): Unique run identifier (timestamp if None).
            verbose (bool): If True, set log level to DEBUG.
        """
        self.provider = provider
        self.verbose = verbose
        if run_id is None:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.log_dir = (
            Path(__file__).parent.parent.parent / "data" / "results" / provider
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"evaluation_log_{self.run_id}.jsonl"
        self._lock = threading.Lock()
        self._setup_logger()

    def _setup_logger(self):
        """
        Set up the internal Python logger for console output.
        """
        self._logger = logging.getLogger(
            f"StructuredLogger.{self.provider}.{self.run_id}"
        )
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        self._logger.handlers = []
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

    def log(
        self,
        prompt: str,
        provider: str,
        model: str,
        response: str,
        latency: float,
        tokens: int,
        score: Any = None,
        timestamp: str = None,
        extra: Dict[str, Any] = None,
    ):
        """
        Write a structured log entry to the JSONL file.

        Args:
            prompt (str): The prompt/query.
            provider (str): Provider name.
            model (str): Model name.
            response (str): Model response.
            latency (float): Latency in seconds.
            tokens (int): Token count.
            score (Any): Optional score or metrics.
            timestamp (str): Optional timestamp (UTC ISO format).
            extra (dict): Optional extra fields.
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        entry = {
            "prompt": prompt,
            "provider": provider,
            "model": model,
            "response": response,
            "latency": latency,
            "tokens": tokens,
            "score": score,
            "timestamp": timestamp,
        }
        if extra:
            entry.update(extra)
        # Thread-safe write
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        if self.verbose:
            self._logger.debug(f"Logged entry: {entry}")

    def get_log_path(self) -> str:
        """
        Get the path to the current log file.

        Returns:
            str: Path to the log file.
        """
        return str(self.log_path)
