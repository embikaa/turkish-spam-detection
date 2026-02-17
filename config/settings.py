import torch
import os
from typing import Optional


class Config:
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "veri_seti_200k.csv")
    
    
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    
    
    MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
    TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf.pkl")
    SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
    
    
    BERT_MODEL_NAME = "dbmdz/bert-base-turkish-cased"
    MAX_LEN = 128
    BATCH_SIZE = 16  
    
    TFIDF_MAX_FEATURES = 500
    
    
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    SMOTE_RATIO = 0.5
    SAMPLE_SIZE: Optional[int] = 5000  
    
    
    CV_FOLDS = 5
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    LOG_LEVEL = "INFO"
    LOG_FILE = os.path.join(LOGS_DIR, "training.log")
    
    GENERIC_KEYWORDS = [
        'güzel', 'iyi', 'harika', 'süper', 'ok', 'tavsiye', 'bayıldım', 
        'aldım', 'beğendim', 'teşekkürler', 'hızlı', 'kargo', 'paketleme',
        'fena', 'değil', 'ürün', 'elime', 'ulaştı', 'kötü', 'berbat'
    ]
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If any configuration value is invalid
            
        Returns:
            True if all validations pass
        """
        errors = []
        
        if cls.TEST_SIZE <= 0 or cls.TEST_SIZE >= 1:
            errors.append(f"TEST_SIZE must be in (0, 1), got {cls.TEST_SIZE}")
        
        if cls.TFIDF_MAX_FEATURES <= 0:
            errors.append(f"TFIDF_MAX_FEATURES must be positive, got {cls.TFIDF_MAX_FEATURES}")
        
        if cls.BATCH_SIZE <= 0:
            errors.append(f"BATCH_SIZE must be positive, got {cls.BATCH_SIZE}")
        
        if cls.SMOTE_RATIO <= 0 or cls.SMOTE_RATIO > 1:
            errors.append(f"SMOTE_RATIO must be in (0, 1], got {cls.SMOTE_RATIO}")
        
        if cls.CV_FOLDS < 2:
            errors.append(f"CV_FOLDS must be >= 2, got {cls.CV_FOLDS}")
        
        if cls.SAMPLE_SIZE is not None and cls.SAMPLE_SIZE <= 0:
            errors.append(f"SAMPLE_SIZE must be positive or None, got {cls.SAMPLE_SIZE}")
        
        if not os.path.exists(os.path.dirname(cls.DATA_PATH)):
            errors.append(f"Data directory does not exist: {os.path.dirname(cls.DATA_PATH)}")
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True