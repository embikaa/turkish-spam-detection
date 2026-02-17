"""
Multi-model training script for Turkish spam detection.

Trains 10 different ML models using shared TF-IDF + BERTurk features
and saves all results for dashboard comparison.
"""

import numpy as np
import pandas as pd
import json
import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import joblib

from config.settings import Config
from src.preprocessing import clean_text_for_bert, clean_text_for_tfidf
from src.features import FeatureEngineer
from src.heuristics import SpamHeuristics
from src.utils import set_seed, ensure_dir, save_json
from src.logger import setup_logger

logger = setup_logger(__name__, log_file=Config.LOG_FILE)

# Try importing optional boosting libraries
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed, skipping")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed, skipping")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.warning("CatBoost not installed, skipping")


def get_models():
    """Return dictionary of all models to train."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=Config.RANDOM_STATE, n_jobs=-1
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1
        ),
        "SVM": SVC(
            kernel='rbf', probability=True, random_state=Config.RANDOM_STATE
        ),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=300,
            random_state=Config.RANDOM_STATE, early_stopping=True
        ),
        "CART": DecisionTreeClassifier(
            random_state=Config.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=Config.RANDOM_STATE, n_jobs=-1
        ),
        "GBM": GradientBoostingClassifier(
            n_estimators=100, random_state=Config.RANDOM_STATE
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=Config.RANDOM_STATE,
            use_label_encoder=False, eval_metric='logloss', n_jobs=-1
        )

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=100, random_state=Config.RANDOM_STATE,
            verbose=-1, n_jobs=-1
        )

    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=100, random_seed=Config.RANDOM_STATE,
            verbose=0
        )

    return models


def evaluate(y_true, y_pred, y_proba=None):
    """Compute all evaluation metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

    unique, counts = np.unique(y_true, return_counts=True)
    metrics['class_distribution'] = {
        ['Genuine', 'Spam'][int(i)]: int(c) for i, c in zip(unique, counts)
    }

    report = classification_report(y_true, y_pred, target_names=['Genuine', 'Spam'], output_dict=True)
    metrics['classification_report'] = report

    if y_proba is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))

    return metrics


def main():
    logger.info("=" * 70)
    logger.info("MULTI-MODEL TRAINING PIPELINE")
    logger.info("=" * 70)

    # ── 1. Setup ──────────────────────────────────────────────────────────
    Config.validate()
    set_seed(Config.RANDOM_STATE)

    # ── 2. Load data ──────────────────────────────────────────────────────
    logger.info(f"Loading data from {Config.DATA_PATH}")
    df = pd.read_csv(Config.DATA_PATH, low_memory=False, usecols=['comment'])
    df = df.dropna(subset=['comment'])

    if Config.SAMPLE_SIZE is not None:
        df = df.sample(n=min(Config.SAMPLE_SIZE, len(df)),
                       random_state=Config.RANDOM_STATE)
    logger.info(f"Using {len(df)} samples")

    texts = df['comment'].tolist()

    # ── 3. Weak labels ───────────────────────────────────────────────────
    logger.info("Generating weak labels...")
    heuristics = SpamHeuristics(Config.GENERIC_KEYWORDS)
    labels = np.array([
        heuristics.generate_weak_label(
            heuristics.extract_structural_features(t)
        ) for t in texts
    ])
    logger.info(f"Labels → Genuine: {(labels==0).sum()}, Spam: {(labels==1).sum()}")

    # ── 4. Train / Test split ─────────────────────────────────────────────
    texts_arr = np.array(texts)
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts_arr, labels,
        test_size=Config.TEST_SIZE,
        stratify=labels,
        random_state=Config.RANDOM_STATE
    )
    logger.info(f"Train: {len(X_train_txt)}  |  Test: {len(X_test_txt)}")

    # ── 5. Feature extraction (shared) ────────────────────────────────────
    logger.info("Preprocessing texts...")
    train_tfidf = [clean_text_for_tfidf(t) for t in X_train_txt]
    train_bert  = [clean_text_for_bert(t)  for t in X_train_txt]
    test_tfidf  = [clean_text_for_tfidf(t) for t in X_test_txt]
    test_bert   = [clean_text_for_bert(t)  for t in X_test_txt]

    fe = FeatureEngineer()

    logger.info("Extracting TRAINING features...")
    X_train_tfidf = fe.fit_transform_tfidf(train_tfidf)
    X_train_bert  = fe.get_bert_embeddings(train_bert)
    X_train = np.hstack([X_train_tfidf, X_train_bert])

    logger.info("Extracting TEST features...")
    X_test_tfidf = fe.transform_tfidf(test_tfidf)
    X_test_bert  = fe.get_bert_embeddings(test_bert)
    X_test = np.hstack([X_test_tfidf, X_test_bert])

    logger.info(f"Feature shape: {X_train.shape}")

    # ── 6. Scale ──────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── 7. SMOTE ──────────────────────────────────────────────────────────
    logger.info(f"Applying SMOTE (ratio={Config.SMOTE_RATIO})...")
    smote = SMOTE(sampling_strategy=Config.SMOTE_RATIO,
                  random_state=Config.RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    logger.info(f"After SMOTE: {len(X_train_res)} samples")

    # ── 8. Train every model ──────────────────────────────────────────────
    models = get_models()
    all_results = {}
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(f"{Config.MODELS_DIR}/{version}")

    logger.info(f"\nTraining {len(models)} models...")
    logger.info("-" * 70)

    for name, model in models.items():
        logger.info(f"\n▶ Training: {name}")
        t0 = time.time()

        try:
            model.fit(X_train_res, y_train_res)
            elapsed = time.time() - t0

            y_pred = model.predict(X_test_scaled)

            y_proba = None
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_scaled)[:, 1]

            metrics = evaluate(y_test, y_pred, y_proba)
            metrics['training_time'] = round(elapsed, 2)

            all_results[name] = metrics

            # Save individual model
            safe_name = name.lower().replace(' ', '_')
            joblib.dump(model, f"{run_dir}/{safe_name}_model.pkl")

            logger.info(
                f"  ✅ {name}: Acc={metrics['accuracy']:.4f}  "
                f"F1={metrics['f1_score']:.4f}  "
                f"Prec={metrics['precision']:.4f}  "
                f"Rec={metrics['recall']:.4f}  "
                f"({elapsed:.1f}s)"
            )

        except Exception as e:
            logger.error(f"  ❌ {name} failed: {e}")
            all_results[name] = {'error': str(e)}

    # ── 9. Save shared artifacts ──────────────────────────────────────────
    joblib.dump(fe.tfidf, f"{run_dir}/tfidf.pkl")
    joblib.dump(scaler, f"{run_dir}/scaler.pkl")

    meta = {
        'timestamp': datetime.now().isoformat(),
        'version': version,
        'config': {
            'bert_model': Config.BERT_MODEL_NAME,
            'tfidf_max_features': Config.TFIDF_MAX_FEATURES,
            'random_state': Config.RANDOM_STATE,
            'test_size': Config.TEST_SIZE,
            'smote_ratio': Config.SMOTE_RATIO,
            'sample_size': Config.SAMPLE_SIZE,
        },
        'dataset': {
            'total': len(texts),
            'train': len(X_train_txt),
            'test': len(X_test_txt),
        },
        'models': all_results,
    }
    save_json(meta, f"{run_dir}/metadata.json")

    # Also save to a well-known location for the dashboard
    save_json(meta, f"{Config.MODELS_DIR}/multi_model_results.json")

    # Update latest symlink
    latest = f"{Config.MODELS_DIR}/latest"
    if os.path.islink(latest):
        os.remove(latest)
    elif os.path.exists(latest):
        os.remove(latest)
    os.symlink(version, latest)

    # ── 10. Summary ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'AUC':>10} {'Time':>8}")
    logger.info("-" * 85)

    for name, m in sorted(all_results.items(),
                          key=lambda x: x[1].get('f1_score', 0),
                          reverse=True):
        if 'error' in m:
            logger.info(f"{name:<25} {'ERROR':>10}")
            continue
        auc_str = f"{m.get('roc_auc', 0):.4f}" if 'roc_auc' in m else "N/A"
        logger.info(
            f"{name:<25} {m['accuracy']:>10.4f} {m['f1_score']:>10.4f} "
            f"{m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{auc_str:>10} {m['training_time']:>7.1f}s"
        )

    logger.info("=" * 70)
    logger.info(f"Results saved to: {run_dir}")
    logger.info(f"Dashboard data: {Config.MODELS_DIR}/multi_model_results.json")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
