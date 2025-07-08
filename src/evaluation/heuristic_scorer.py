import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class HeuristicScorer:
    def __init__(self, rules_path: Optional[str] = None):
        if rules_path is None:
            rules_path = str(
                Path(__file__).parent.parent.parent / "config" / "evaluation_rules.yaml"
            )
        with open(rules_path, "r") as f:
            self.rules = yaml.safe_load(f)

    def score(
        self,
        response: str,
        context: str,
        model_meta: Dict[str, Any],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        scores = {}
        # Relevance
        scores["relevance"] = self._score_relevance(response, context)
        # Length coherence
        scores["length_coherence"] = self._score_length_coherence(response)
        # Factual accuracy
        if (
            self.rules.get("factual_accuracy", {}).get("enabled", False)
            and ground_truth
        ):
            scores["factual_accuracy"] = self._score_factual_accuracy(
                response, ground_truth
            )
        else:
            scores["factual_accuracy"] = None
        # Attach model metadata
        scores["model"] = model_meta.get("name")
        scores["provider"] = model_meta.get("provider")
        scores["tokens"] = model_meta.get("tokens")
        return scores

    def _score_relevance(self, response: str, context: str) -> float:
        rel_rules = self.rules.get("relevance", {})
        keywords = rel_rules.get("keywords", {}).get(context, [])
        if not keywords:
            return 0.0
        matches = sum(
            1
            for kw in keywords
            if re.search(rf"\\b{re.escape(kw)}\\b", response, re.IGNORECASE)
        )
        score = matches / len(keywords)
        return max(score, rel_rules.get("min_score", 0.0))

    def _score_length_coherence(self, response: str) -> float:
        length_rules = self.rules.get("length_coherence", {})
        min_tokens = length_rules.get("min_tokens", 10)
        max_tokens = length_rules.get("max_tokens", 200)
        penalty_short = length_rules.get("penalty_short", 0.5)
        penalty_long = length_rules.get("penalty_long", 0.5)
        tokens = len(response.split())
        if tokens < min_tokens:
            return penalty_short
        if tokens > max_tokens:
            return penalty_long
        return 1.0

    def _score_factual_accuracy(self, response: str, ground_truth: str) -> float:
        fact_rules = self.rules.get("factual_accuracy", {})
        method = fact_rules.get("method", "substring")
        threshold = fact_rules.get("match_threshold", 0.7)
        if method == "exact":
            return 1.0 if response.strip() == ground_truth.strip() else 0.0
        elif method == "substring":
            # Score by fraction of ground truth tokens present in response
            gt_tokens = set(ground_truth.lower().split())
            resp_tokens = set(response.lower().split())
            match = len(gt_tokens & resp_tokens) / max(1, len(gt_tokens))
            return 1.0 if match >= threshold else match
        elif method == "fuzzy":
            try:
                from difflib import SequenceMatcher

                ratio = SequenceMatcher(None, response, ground_truth).ratio()
                return 1.0 if ratio >= threshold else ratio
            except ImportError:
                return 0.0
        return 0.0
