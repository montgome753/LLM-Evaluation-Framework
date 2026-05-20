"""
Accuracy metric for LLM evaluation.

Supports exact match, normalized match, contains-answer, and fuzzy match
strategies, automatically selecting the best one per sample type.
"""

from __future__ import annotations

import re
import string
from difflib import SequenceMatcher


class AccuracyMetric:
    """
    Multi-strategy accuracy scorer.

    Tries exact match → normalized match → fuzzy match in order,
    returning True if any strategy clears the threshold.
    """

    def __init__(self, fuzzy_threshold: float = 0.85):
        self.fuzzy_threshold = fuzzy_threshold

    def score(self, prediction: str, reference: str) -> bool:
        """Return True if prediction is considered correct vs reference."""
        if not prediction or not reference:
            return False

        pred = self._normalize(prediction)
        ref = self._normalize(reference)

        if pred == ref:
            return True

        # Multiple-choice: check if model picked the right letter
        if self._is_mc_answer(ref):
            return self._mc_match(prediction, ref)

        # Short answers: fuzzy match
        return self._fuzzy_match(pred, ref)

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        # Remove common filler prefixes
        for prefix in ("the answer is", "answer:", "answer is", "therefore"):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text

    def _is_mc_answer(self, ref: str) -> bool:
        return bool(re.fullmatch(r"[a-d]", ref.strip().lower()))

    def _mc_match(self, prediction: str, ref: str) -> bool:
        # Find the first letter A-D mentioned in the prediction
        match = re.search(r"\b([A-Da-d])\b", prediction)
        if match:
            return match.group(1).lower() == ref.lower()
        return False

    def _fuzzy_match(self, pred: str, ref: str) -> bool:
        ratio = SequenceMatcher(None, pred, ref).ratio()
        return ratio >= self.fuzzy_threshold

    def batch_score(
        self, predictions: list[str], references: list[str]
    ) -> tuple[list[bool], float]:
        """Score a batch, returning per-sample booleans and overall accuracy."""
        results = [self.score(p, r) for p, r in zip(predictions, references)]
        acc = sum(results) / len(results) if results else 0.0
        return results, acc
