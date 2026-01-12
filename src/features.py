import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from config.settings import Config

class FeatureEngineer:
    def __init__(self):
        print(f"BERT Modeli Yükleniyor: {Config.BERT_MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        self.bert_model = AutoModel.from_pretrained(Config.BERT_MODEL_NAME).to(Config.DEVICE)
        self.bert_model.eval()
        self.tfidf = TfidfVectorizer(max_features=Config.TFIDF_MAX_FEATURES)
        
    def get_bert_embeddings(self, text_list):
        all_embeddings = []
        total = len(text_list)
        print(f"BERT Embedding çıkarılıyor ({total} örnek)...")
        
        for i in range(0, total, Config.BATCH_SIZE):
            batch_texts = text_list[i:i+Config.BATCH_SIZE]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=Config.MAX_LEN)
            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(embeddings)
            
            if i % 100 == 0:
                print(f"İşlenen: {i}/{total}")
            
        return np.vstack(all_embeddings)

    def fit_transform_tfidf(self, clean_text_list):
        print("TF-IDF dönüşümü yapılıyor...")
        return self.tfidf.fit_transform(clean_text_list).toarray()