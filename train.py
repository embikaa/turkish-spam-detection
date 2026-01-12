import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import sys
import os

# Python'un src klasörünü görmesi için
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.settings import Config
from src.preprocessing import clean_text
from src.features import FeatureEngineer
from src.heuristics import SpamHeuristics

def main():
    print("--- SİSTEM BAŞLATILIYOR ---")
    
    # 1. VERİ OKUMA
    if not os.path.exists(Config.DATA_PATH):
        print(f"HATA: Dosya bulunamadı: {Config.DATA_PATH}")
        print("Lütfen veri_seti_200k.csv dosyasını 'data/raw' klasörüne koy.")
        return

    df = pd.read_csv(Config.DATA_PATH, low_memory=False, usecols=['comment'])
    # Hızlı test için 5000 satır alıyoruz. Savunma için bunu 50.000 yapabilirsin.
    df = df.dropna(subset=['comment']).sample(n=5000, random_state=Config.RANDOM_STATE)
    
    # 2. ETİKETLEME (WEAK SUPERVISION)
    print(">> Otomatik etiketleme yapılıyor...")
    df['clean_text'] = df['comment'].apply(clean_text)
    
    heuristics = SpamHeuristics(Config.GENERIC_KEYWORDS)
    df['struct_feats'] = df['comment'].apply(heuristics.extract_structural_features)
    df['weak_label'] = df['struct_feats'].apply(heuristics.generate_weak_label)
    
    y = df['weak_label'].values
    print(f"Etiket Dağılımı -> Normal: {np.sum(y==0)}, Spam: {np.sum(y==1)}")

    # 3. ÖZNİTELİK ÇIKARMA
    fe = FeatureEngineer()
    X_tfidf = fe.fit_transform_tfidf(df['clean_text'].tolist())
    X_bert = fe.get_bert_embeddings(df['clean_text'].tolist())
    X_final = np.hstack([X_tfidf, X_bert])
    
    # 4. EĞİTİM
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=Config.TEST_SIZE, stratify=y, random_state=Config.RANDOM_STATE)
    
    smote = SMOTE(sampling_strategy=Config.SMOTE_RATIO, random_state=Config.RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(">> Model eğitiliyor...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=Config.RANDOM_STATE)
    rf.fit(X_train_res, y_train_res)
    
    # 5. SONUÇ
    print(">> Sonuçlar:")
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Kayıt
    if not os.path.exists("models"): os.makedirs("models")
    joblib.dump(rf, Config.MODEL_SAVE_PATH)
    joblib.dump(fe.tfidf, Config.TFIDF_PATH)
    print("Başarıyla tamamlandı.")

if __name__ == "__main__":
    main()