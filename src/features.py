import numpy as np
import numpy.typing as npt
import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from config.settings import Config
from src.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    
    def __init__(self, tfidf_model: Optional[TfidfVectorizer] = None):
    
        logger.info(f"Loading BERT model: {Config.BERT_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        self.bert_model = AutoModel.from_pretrained(Config.BERT_MODEL_NAME).to(Config.DEVICE)
        self.bert_model.eval()
        
        if tfidf_model is not None:
            self.tfidf = tfidf_model
            logger.info("Loaded pre-fitted TF-IDF model")
        else:
            self.tfidf = TfidfVectorizer(max_features=Config.TFIDF_MAX_FEATURES)
            logger.info(f"Created new TF-IDF vectorizer (max_features={Config.TFIDF_MAX_FEATURES})")
    
    def get_bert_embeddings(self, text_list: List[str]) -> npt.NDArray[np.float32]:
        
        all_embeddings = []
        total = len(text_list)
        logger.info(f"Extracting BERT embeddings for {total} samples")
        
        try:
            for i in tqdm(range(0, total, Config.BATCH_SIZE), desc="BERT embeddings"):
                batch_texts = text_list[i:i+Config.BATCH_SIZE]
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=Config.MAX_LEN
                )
                inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_embeddings.append(embeddings)
            
            return np.vstack(all_embeddings)
        
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU out of memory. Switching to CPU...")
            self.bert_model = self.bert_model.cpu()
            Config.DEVICE = torch.device("cpu")
            return self.get_bert_embeddings(text_list)
    
    def fit_transform_tfidf(self, text_list: List[str]) -> npt.NDArray[np.float64]:
        
        logger.info(f"Fitting and transforming TF-IDF on {len(text_list)} samples")
        return self.tfidf.fit_transform(text_list).toarray()
    
    def transform_tfidf(self, text_list: List[str]) -> npt.NDArray[np.float64]:
        
        if not hasattr(self.tfidf, 'vocabulary_'):
            raise ValueError("TF-IDF vectorizer has not been fitted. Call fit_transform_tfidf first.")
        
        logger.info(f"Transforming TF-IDF for {len(text_list)} samples")
        return self.tfidf.transform(text_list).toarray()