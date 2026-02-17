"""
Configuration settings for the Turkish spam detection project.
"""

import os


class Config:
    """Central configuration for model training and evaluation."""

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "veri_seti_200k.csv")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    LOG_FILE = os.path.join(LOGS_DIR, "training.log")

    # BERT settings
    BERT_MODEL_NAME = "dbmdz/bert-base-turkish-cased"
    BERT_MAX_LENGTH = 128
    BERT_BATCH_SIZE = 16

    # TF-IDF settings
    TFIDF_MAX_FEATURES = 500

    # Training settings
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    SMOTE_RATIO = 1.0
    SAMPLE_SIZE = 5000  # Set to None for full dataset

    # Weak supervision keywords (Turkish generic review words)
    GENERIC_KEYWORDS = [
        "güzel", "iyi", "süper", "harika", "mükemmel",
        "kötü", "berbat", "rezalet", "fena", "idare"
    ]

    @classmethod
    def validate(cls):
        """Validate configuration and create required directories."""
        if not os.path.exists(cls.DATA_PATH):
            raise FileNotFoundError(
                f"Dataset not found at {cls.DATA_PATH}. "
                f"Place 'veri_seti_200k.csv' in data/raw/ directory."
            )
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)