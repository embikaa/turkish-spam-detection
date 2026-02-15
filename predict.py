import joblib
import torch
import numpy as np
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.settings import Config
from src.preprocessing import clean_text
from src.features import FeatureEngineer
from src.heuristics import SpamHeuristics

def load_system():
    print(">> Sistem ve Modeller Yükleniyor (Biraz sürebilir)...")
    
    
    try:
        model = joblib.load(Config.MODEL_SAVE_PATH)
        
        tfidf = joblib.load(Config.TFIDF_PATH) 
    except FileNotFoundError:
        print("HATA: Modeller bulunamadı. Önce 'python train.py' çalıştır.")
        sys.exit(1)
        
    
    fe = FeatureEngineer()
   
    heuristics = SpamHeuristics(Config.GENERIC_KEYWORDS)
    
    print(">> Sistem Hazır!\n")
    return model, tfidf, fe, heuristics

def predict_comment(text, model, tfidf, fe, heuristics):
  
    cleaned = clean_text(text)
    
   
    struct_feats = heuristics.extract_structural_features(text)
    weak_label = heuristics.generate_weak_label(struct_feats)
    
    vec_tfidf = tfidf.transform([cleaned]).toarray()
    vec_bert = fe.get_bert_embeddings([cleaned])
    
    X_input = np.hstack([vec_tfidf, vec_bert])

    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1] 
    
    return prediction, probability, weak_label

if __name__ == "__main__":
   
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
