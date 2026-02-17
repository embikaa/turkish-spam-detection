import numpy as np
import numpy.typing as npt
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from config.settings import Config
from src.preprocessing import clean_text, clean_text_for_bert, clean_text_for_tfidf
from src.features import FeatureEngineer
from src.heuristics import SpamHeuristics
from src.evaluation import evaluate_model
from src.utils import ensure_dir, save_json
from src.logger import setup_logger

logger = setup_logger(__name__)


class SpamDetectionPipeline:
    def __init__(self, config: Optional[Any] = None):
        
        self.config = config or Config
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.heuristics = SpamHeuristics(self.config.GENERIC_KEYWORDS)
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.metadata: Dict[str, Any] = {}
    
    def _generate_weak_labels(self, texts: List[str]) -> npt.NDArray:
        
        logger.info("Generating weak labels using heuristics...")
        labels = []
        for text in texts:
            struct_feats = self.heuristics.extract_structural_features(text)
            label = self.heuristics.generate_weak_label(struct_feats)
            labels.append(label)
        
        labels = np.array(labels)
        n_genuine = np.sum(labels == 0)
        n_spam = np.sum(labels == 1)
        logger.info(f"Label distribution - Genuine: {n_genuine}, Spam: {n_spam}")
        
        return labels
    
    def fit(
        self,
        texts: List[str],
        labels: Optional[npt.NDArray] = None,
        use_cross_validation: bool = False
    ) -> 'SpamDetectionPipeline':
        
        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 60)
        
        if labels is None:
            labels = self._generate_weak_labels(texts)
        
        logger.info(f"Splitting data: {len(texts)} samples, test_size={self.config.TEST_SIZE}")
        texts_array = np.array(texts)
        
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts_array,
            labels,
            test_size=self.config.TEST_SIZE,
            stratify=labels,
            random_state=self.config.RANDOM_STATE
        )
        
        logger.info(f"Train set: {len(X_train_texts)} samples")
        logger.info(f"Test set: {len(X_test_texts)} samples")
        
        logger.info("Preprocessing texts...")
        X_train_tfidf_texts = [clean_text_for_tfidf(t) for t in X_train_texts]
        X_train_bert_texts = [clean_text_for_bert(t) for t in X_train_texts]
        X_test_tfidf_texts = [clean_text_for_tfidf(t) for t in X_test_texts]
        X_test_bert_texts = [clean_text_for_bert(t) for t in X_test_texts]
        
        self.feature_engineer = FeatureEngineer()
        
        logger.info("Extracting features from TRAINING set...")
        X_train_tfidf = self.feature_engineer.fit_transform_tfidf(X_train_tfidf_texts)
        X_train_bert = self.feature_engineer.get_bert_embeddings(X_train_bert_texts)
        X_train = np.hstack([X_train_tfidf, X_train_bert])
        
        logger.info("Extracting features from TEST set...")
        X_test_tfidf = self.feature_engineer.transform_tfidf(X_test_tfidf_texts)
        X_test_bert = self.feature_engineer.get_bert_embeddings(X_test_bert_texts)
        X_test = np.hstack([X_test_tfidf, X_test_bert])
        
        logger.info(f"Feature shape: {X_train.shape}")
        
        logger.info("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Applying SMOTE (ratio={self.config.SMOTE_RATIO})...")
        smote = SMOTE(
            sampling_strategy=self.config.SMOTE_RATIO,
            random_state=self.config.RANDOM_STATE
        )
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        logger.info(f"After SMOTE: {len(X_train_resampled)} samples")
        
        logger.info("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1
        )
        self.model.fit(X_train_resampled, y_train_resampled)
        
        logger.info("Evaluating on test set...")
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        
        self.metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'bert_model': self.config.BERT_MODEL_NAME,
                'tfidf_max_features': self.config.TFIDF_MAX_FEATURES,
                'random_state': self.config.RANDOM_STATE,
                'test_size': self.config.TEST_SIZE,
                'smote_ratio': self.config.SMOTE_RATIO
            },
            'dataset': {
                'total_samples': len(texts),
                'train_samples': len(X_train_texts),
                'test_samples': len(X_test_texts)
            },
            'metrics': metrics
        }
        
        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info("=" * 60)
        
        return self
    
    def predict(self, texts: List[str]) -> npt.NDArray:
      
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        tfidf_texts = [clean_text_for_tfidf(t) for t in texts]
        bert_texts = [clean_text_for_bert(t) for t in texts]
        
        X_tfidf = self.feature_engineer.transform_tfidf(tfidf_texts)
        X_bert = self.feature_engineer.get_bert_embeddings(bert_texts)
        X = np.hstack([X_tfidf, X_bert])
        
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, texts: List[str]) -> npt.NDArray:
       
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        tfidf_texts = [clean_text_for_tfidf(t) for t in texts]
        bert_texts = [clean_text_for_bert(t) for t in texts]
        
        X_tfidf = self.feature_engineer.transform_tfidf(tfidf_texts)
        X_bert = self.feature_engineer.get_bert_embeddings(bert_texts)
        X = np.hstack([X_tfidf, X_bert])
        
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)
    
    def save(self, version: Optional[str] = None) -> str:
        
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_dir = ensure_dir(f"{self.config.MODELS_DIR}/{version}")
        
        joblib.dump(self.model, f"{model_dir}/model.pkl")
        joblib.dump(self.feature_engineer.tfidf, f"{model_dir}/tfidf.pkl")
        joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")
        
        save_json(self.metadata, f"{model_dir}/metadata.json")
        
        latest_link = f"{self.config.MODELS_DIR}/latest"
        import os
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(version, latest_link)
        
        logger.info(f"Pipeline saved to {model_dir}")
        return str(model_dir)
    
    @classmethod
    def load(cls, path: str) -> 'SpamDetectionPipeline':
        
        logger.info(f"Loading pipeline from {path}")
        
        model = joblib.load(f"{path}/model.pkl")
        tfidf = joblib.load(f"{path}/tfidf.pkl")
        scaler = joblib.load(f"{path}/scaler.pkl")
        
        from src.utils import load_json
        metadata = load_json(f"{path}/metadata.json")
        
        pipeline = cls()
        pipeline.model = model
        pipeline.scaler = scaler
        pipeline.feature_engineer = FeatureEngineer(tfidf_model=tfidf)
        pipeline.metadata = metadata
        
        logger.info("Pipeline loaded successfully")
        return pipeline
