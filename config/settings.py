import torch
import os

class Config:
    # Dosya Yolları
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # CSV dosyasının adı 'veri_seti_200k.csv' olmalı veya burayı değiştirmelisin
    DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "veri_seti_200k.csv")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
    TFIDF_PATH = os.path.join(BASE_DIR, "models", "tfidf.pkl")
    
    # Model Parametreleri
    BERT_MODEL_NAME = "dbmdz/bert-base-turkish-cased"
    MAX_LEN = 128
    BATCH_SIZE = 16  # Bilgisayarın kasarsa bunu 8 yap
    
    TFIDF_MAX_FEATURES = 500
    
    # Eğitim Parametreleri
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    SMOTE_RATIO = 0.5
    
    # Donanım (GPU varsa kullanır, yoksa CPU)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Zayıf Etiketleme için Anahtar Kelimeler
    GENERIC_KEYWORDS = [
        'güzel', 'iyi', 'harika', 'süper', 'ok', 'tavsiye', 'bayıldım', 
        'aldım', 'beğendim', 'teşekkürler', 'hızlı', 'kargo', 'paketleme',
        'fena', 'değil', 'ürün', 'elime', 'ulaştı', 'kötü', 'berbat'
    ]