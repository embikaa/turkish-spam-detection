"""
Text preprocessing for Turkish spam detection.

Provides two preprocessing pipelines:
  - clean_text_for_bert(): minimal cleaning for BERT embeddings
  - clean_text_for_tfidf(): aggressive cleaning for TF-IDF features
"""

import re
from typing import List

# Turkish stopwords
TURKISH_STOPWORDS = {
    "bir", "bu", "da", "de", "ve", "ile", "için", "ama", "çok",
    "daha", "en", "var", "ne", "ben", "sen", "o", "biz", "siz",
    "onlar", "gibi", "kadar", "sonra", "önce", "her", "bazı",
    "diğer", "olan", "olarak", "üzere", "göre", "karşı", "rağmen"
}


def clean_text_for_bert(text: str) -> str:
    """
    Minimal cleaning for BERT tokenizer input.
    Preserves digits and most punctuation since BERT handles them natively.

    Args:
        text: Raw input text.

    Returns:
        Lightly cleaned text suitable for BERT.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_text_for_tfidf(text: str) -> str:
    """
    Aggressive cleaning for TF-IDF vectorization.
    Removes URLs, punctuation, stopwords, and normalizes whitespace.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text suitable for TF-IDF.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = [w for w in text.split() if w not in TURKISH_STOPWORDS and len(w) > 1]
    return " ".join(words)


def preprocess_batch(texts: List[str], method: str = "bert") -> List[str]:
    """
    Preprocess a batch of texts.

    Args:
        texts: List of raw text strings.
        method: 'bert' or 'tfidf'.

    Returns:
        List of cleaned text strings.
    """
    func = clean_text_for_bert if method == "bert" else clean_text_for_tfidf
    return [func(t) for t in texts]