import joblib
import torch
import numpy as np
import sys
import os

# Python'un src klasörünü görmesi için
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.settings import Config
from src.preprocessing import clean_text
from src.features import FeatureEngineer
from src.heuristics import SpamHeuristics

def load_system():
    print(">> Sistem ve Modeller Yükleniyor (Biraz sürebilir)...")
    
    # Modelleri yükle
    try:
        model = joblib.load(Config.MODEL_SAVE_PATH)
        # TF-IDF modelini train.py içinde kaydetmiştik, onu yüklüyoruz
        tfidf = joblib.load(Config.TFIDF_PATH) 
    except FileNotFoundError:
        print("HATA: Modeller bulunamadı. Önce 'python train.py' çalıştır.")
        sys.exit(1)
        
    # BERT motorunu başlat (FeatureEngineer içinde)
    fe = FeatureEngineer()
    
    # Sezgisel kuralları başlat
    heuristics = SpamHeuristics(Config.GENERIC_KEYWORDS)
    
    print(">> Sistem Hazır!\n")
    return model, tfidf, fe, heuristics

def predict_comment(text, model, tfidf, fe, heuristics):
    # 1. Temizlik
    cleaned = clean_text(text)
    
    # 2. Öznitelik Çıkarımı
    # A. Yapısal (Analiz için ekrana basacağız ama modele girmeyecek, çünkü model BERT+TFIDF ile eğitildi)
    struct_feats = heuristics.extract_structural_features(text)
    weak_label = heuristics.generate_weak_label(struct_feats)
    
    # B. Model Girdisi (BERT + TF-IDF)
    vec_tfidf = tfidf.transform([cleaned]).toarray()
    vec_bert = fe.get_bert_embeddings([cleaned]) # Tek satır için embedding
    
    X_input = np.hstack([vec_tfidf, vec_bert])
    
    # 3. Tahmin
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1] # Spam olma ihtimali
    
    return prediction, probability, weak_label

if __name__ == "__main__":
    # Sistemi bir kez yükle
    rf_model, tfidf_model, fe_engine, heuristic_engine = load_system()
    
    print("Çıkış için 'q' yazıp Enter'a basın.")
    print("-" * 50)
    
    while True:
        user_input = input("\nYorumu Girin: ")
        
        if user_input.lower() == 'q':
            break
            
        if len(user_input.strip()) < 2:
            continue
            
        pred, prob, weak_lbl = predict_comment(
            user_input, rf_model, tfidf_model, fe_engine, heuristic_engine
        )
        
        status = "SPAM 🔴" if pred == 1 else "GERÇEK (GENUINE) 🟢"
        
        print(f"\nSonuç: {status}")
        print(f"Spam İhtimali: %{prob*100:.2f}")
        print(f"Kural Tabanlı Tahmin (Referans): {'Spam' if weak_lbl==1 else 'Gerçek'}")
        print("-" * 50)