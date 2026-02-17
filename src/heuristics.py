"""
Weak supervision heuristics for Turkish spam review detection.

Generates noisy labels using structural and linguistic features
when ground-truth annotations are unavailable.
"""

import re
import numpy as np
from typing import List, Dict, Tuple


class SpamHeuristics:
    """Rule-based weak labeling for spam detection."""

    def __init__(self, generic_keywords: List[str]):
        """
        Args:
            generic_keywords: List of common generic review words.
        """
        self.generic_keywords = [kw.lower() for kw in generic_keywords]

    def extract_structural_features(self, text: str) -> Dict[str, int]:
        """
        Extract structural features from raw text for weak labeling.

        Features:
            - is_short: text has fewer than 5 words
            - is_generic: text contains only generic keywords
            - has_excessive_punctuation: excessive use of ! or ?
            - has_all_caps: text is entirely uppercase
            - has_emoji: text contains emoji characters
            - has_repetition: repeated characters (3+ in a row)
            - has_url: text contains a URL
            - word_count: number of words
        """
        if not isinstance(text, str):
            return {k: 0 for k in [
                "is_short", "is_generic", "has_excessive_punctuation",
                "has_all_caps", "has_emoji", "has_repetition",
                "has_url", "word_count"
            ]}

        words = text.split()
        word_count = len(words)
        text_lower = text.lower()

        is_short = 1 if word_count < 5 else 0
        is_generic = 1 if all(
            w.lower() in self.generic_keywords for w in words
        ) and word_count > 0 else 0
        has_excessive_punct = 1 if len(re.findall(r"[!?]", text)) > 3 else 0
        has_all_caps = 1 if text.isupper() and word_count > 1 else 0
        has_emoji = 1 if re.search(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]", text
        ) else 0
        has_repetition = 1 if re.search(r"(.)\1{2,}", text) else 0
        has_url = 1 if re.search(r"http|www\.", text_lower) else 0

        return {
            "is_short": is_short,
            "is_generic": is_generic,
            "has_excessive_punctuation": has_excessive_punct,
            "has_all_caps": has_all_caps,
            "has_emoji": has_emoji,
            "has_repetition": has_repetition,
            "has_url": has_url,
            "word_count": word_count,
        }

    def generate_weak_label(self, features: Dict[str, int]) -> int:
        """
        Generate a weak label from structural features.

        Labeling rules:
            - spam_score >= 2 → spam (1)
            - word_count >= 20 → genuine (0)
            - otherwise → genuine (0)

        Returns:
            0 (genuine) or 1 (spam)
        """
        spam_score = (
            features["is_short"]
            + features["is_generic"]
            + features["has_excessive_punctuation"]
            + features["has_all_caps"]
            + features["has_emoji"]
            + features["has_repetition"]
            + features["has_url"]
        )

        if spam_score >= 1:
            return 1
        if features["word_count"] >= 20:
            return 0
        return 0

    def label_dataset(self, texts: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Label an entire dataset and return labels with statistics.

        Args:
            texts: List of raw text strings.

        Returns:
            Tuple of (labels_array, stats_dict).
        """
        labels = []
        for text in texts:
            feats = self.extract_structural_features(text)
            labels.append(self.generate_weak_label(feats))

        labels_arr = np.array(labels)
        n_total = len(labels_arr)
        n_spam = int(labels_arr.sum())
        n_genuine = n_total - n_spam

        stats = {
            "total_samples": n_total,
            "genuine_count": n_genuine,
            "spam_count": n_spam,
            "genuine_pct": round(n_genuine / n_total * 100, 2),
            "spam_pct": round(n_spam / n_total * 100, 2),
        }

        return labels_arr, stats