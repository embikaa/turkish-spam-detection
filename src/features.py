"""
Feature engineering: TF-IDF + BERTurk embeddings.

Provides the FeatureEngineer class that handles both feature types
with proper train/test separation to prevent data leakage.
"""

import numpy as np
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """Extract TF-IDF and BERT features for text classification."""

    def __init__(self, max_features: int = 500,
                 bert_model_name: str = "dbmdz/bert-base-turkish-cased",
                 tfidf_model: Optional[TfidfVectorizer] = None):
        """
        Args:
            max_features: Maximum number of TF-IDF features.
            bert_model_name: HuggingFace model identifier.
            tfidf_model: Pre-fitted TF-IDF model (for inference).
        """
        self.max_features = max_features
        self.bert_model_name = bert_model_name

        if tfidf_model is not None:
            self.tfidf = tfidf_model
            logger.info("Using pre-fitted TF-IDF model")
        else:
            self.tfidf = TfidfVectorizer(max_features=max_features)
            logger.info(f"Created new TF-IDF vectorizer (max_features={max_features})")

        self._bert_model = None
        self._bert_tokenizer = None

    def _load_bert(self):
        """Lazy-load BERT model and tokenizer."""
        if self._bert_model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModel

        logger.info(f"Loading BERT model: {self.bert_model_name}")
        self._bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self._bert_model = AutoModel.from_pretrained(self.bert_model_name)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._bert_model.to(self._device)
        self._bert_model.eval()
        logger.info(f"BERT loaded on {self._device}")

    def fit_transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """Fit TF-IDF on training data and return transformed features."""
        logger.info(f"Fitting and transforming TF-IDF on {len(texts)} samples")
        return self.tfidf.fit_transform(texts).toarray()

    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """Transform texts using pre-fitted TF-IDF model."""
        logger.info(f"Transforming TF-IDF for {len(texts)} samples")
        return self.tfidf.transform(texts).toarray()

    def get_bert_embeddings(self, texts: List[str], batch_size: int = 16,
                            max_length: int = 128) -> np.ndarray:
        """
        Extract BERT [CLS] embeddings for a list of texts.

        Args:
            texts: List of preprocessed text strings.
            batch_size: Batch size for inference.
            max_length: Maximum token sequence length.

        Returns:
            Numpy array of shape (n_samples, hidden_size).
        """
        import torch

        self._load_bert()
        logger.info(f"Extracting BERT embeddings for {len(texts)} samples")

        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="BERT embeddings"):
            batch = texts[i:i + batch_size]
            encoded = self._bert_tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            ).to(self._device)

            try:
                with torch.no_grad():
                    outputs = self._bert_model(**encoded)
                cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.warning("CUDA OOM, falling back to CPU")
                    self._bert_model.to("cpu")
                    self._device = torch.device("cpu")
                    encoded = {k: v.to("cpu") for k, v in encoded.items()}
                    with torch.no_grad():
                        outputs = self._bert_model(**encoded)
                    cls = outputs.last_hidden_state[:, 0, :].numpy()
                    all_embeddings.append(cls)
                else:
                    raise

        return np.vstack(all_embeddings)