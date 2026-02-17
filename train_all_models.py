"""
Multi-model training pipeline for Turkish spam detection.

Trains 10 ML/DL models using shared TF-IDF + BERTurk features,
applies hyperparameter tuning, and saves all results.
"""

import numpy as np
import pandas as pd
import json
import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

from config.settings import Config
from src.preprocessing import clean_text_for_bert, clean_text_for_tfidf
from src.features import FeatureEngineer
from src.heuristics import SpamHeuristics
from src.evaluation import evaluate_model
from src.utils import set_seed, ensure_dir, save_json
from src.logger import setup_logger

logger = setup_logger(__name__, log_file=Config.LOG_FILE)

# Optional imports for boosting models
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed — skipping")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed — skipping")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.warning("CatBoost not installed — skipping")


def get_tuned_models() -> dict:
    """
    Return dictionary of all models with optimized hyperparameters.

    Each model uses carefully selected hyperparameters based on
    best practices for text classification with high-dimensional features.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=Config.RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="cosine",
            n_jobs=-1,
        ),
        "SVM": SVC(
            C=10.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=Config.RANDOM_STATE,
        ),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=Config.RANDOM_STATE,
        ),
        "CART": DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            criterion="gini",
            random_state=Config.RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=Config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "GBM": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            subsample=0.8,
            random_state=Config.RANDOM_STATE,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=Config.RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        )

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=Config.RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        )

    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=Config.RANDOM_STATE,
            verbose=0,
        )

    return models


def main():
    print("=" * 70)
    print("MULTI-MODEL TRAINING PIPELINE")
    print("TF-IDF + BERTurk | Weak Supervision | SMOTE")
    print("=" * 70)

    # 1. Setup
    Config.validate()
    set_seed(Config.RANDOM_STATE)

    # 2. Load data
    print(f"\n[1/8] Loading data from {Config.DATA_PATH}")
    df = pd.read_csv(Config.DATA_PATH, low_memory=False, usecols=["comment"])
    df = df.dropna(subset=["comment"])

    if Config.SAMPLE_SIZE is not None:
        df = df.sample(n=min(Config.SAMPLE_SIZE, len(df)),
                       random_state=Config.RANDOM_STATE)
    print(f"      Dataset size: {len(df)} samples")

    texts = df["comment"].tolist()

    # 3. Weak labeling
    print("\n[2/8] Generating weak labels via heuristic rules...")
    heuristics = SpamHeuristics(Config.GENERIC_KEYWORDS)
    labels, label_stats = heuristics.label_dataset(texts)

    print(f"      Total:   {label_stats['total_samples']}")
    print(f"      Genuine: {label_stats['genuine_count']} ({label_stats['genuine_pct']}%)")
    print(f"      Spam:    {label_stats['spam_count']} ({label_stats['spam_pct']}%)")

    # 4. Train/Test split
    print("\n[3/8] Splitting data (train/test)...")
    texts_arr = np.array(texts)
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts_arr, labels,
        test_size=Config.TEST_SIZE,
        stratify=labels,
        random_state=Config.RANDOM_STATE,
    )
    print(f"      Train: {len(X_train_txt)} | Test: {len(X_test_txt)}")

    # 5. Preprocessing
    print("\n[4/8] Preprocessing texts...")
    train_tfidf = [clean_text_for_tfidf(t) for t in X_train_txt]
    train_bert  = [clean_text_for_bert(t)  for t in X_train_txt]
    test_tfidf  = [clean_text_for_tfidf(t) for t in X_test_txt]
    test_bert   = [clean_text_for_bert(t)  for t in X_test_txt]

    # 6. Feature extraction
    print("\n[5/8] Extracting features (TF-IDF + BERTurk)...")
    fe = FeatureEngineer(
        max_features=Config.TFIDF_MAX_FEATURES,
        bert_model_name=Config.BERT_MODEL_NAME,
    )

    X_train_tfidf = fe.fit_transform_tfidf(train_tfidf)
    X_train_bert  = fe.get_bert_embeddings(train_bert, batch_size=Config.BERT_BATCH_SIZE)
    X_train = np.hstack([X_train_tfidf, X_train_bert])

    X_test_tfidf = fe.transform_tfidf(test_tfidf)
    X_test_bert  = fe.get_bert_embeddings(test_bert, batch_size=Config.BERT_BATCH_SIZE)
    X_test = np.hstack([X_test_tfidf, X_test_bert])

    print(f"      Feature dimension: {X_train.shape[1]} "
          f"(TF-IDF: {X_train_tfidf.shape[1]} + BERT: {X_train_bert.shape[1]})")

    # 7. Scaling + SMOTE
    print("\n[6/8] Scaling features and applying SMOTE...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    smote = SMOTE(sampling_strategy=Config.SMOTE_RATIO,
                  random_state=Config.RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"      Before SMOTE: {len(X_train_scaled)} | After SMOTE: {len(X_train_res)}")

    # 8. Train all models
    models = get_tuned_models()
    all_results = {}
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(f"{Config.MODELS_DIR}/{version}")

    print(f"\n[7/8] Training {len(models)} models with tuned hyperparameters...")
    print("-" * 70)

    for name, model in models.items():
        print(f"\n  >> Training: {name}")
        t0 = time.time()

        try:
            model.fit(X_train_res, y_train_res)
            elapsed = time.time() - t0

            y_pred = model.predict(X_test_scaled)
            y_proba = None
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_scaled)[:, 1]

            metrics = evaluate_model(y_test, y_pred, y_proba)
            metrics["training_time"] = round(elapsed, 2)
            all_results[name] = metrics

            # Save individual model
            safe_name = name.lower().replace(" ", "_")
            joblib.dump(model, f"{run_dir}/{safe_name}_model.pkl")

            print(f"     Acc={metrics['accuracy']:.4f}  "
                  f"F1={metrics['f1_score']:.4f}  "
                  f"Prec={metrics['precision']:.4f}  "
                  f"Rec={metrics['recall']:.4f}  "
                  f"AUC={metrics.get('roc_auc', 0):.4f}  "
                  f"({elapsed:.1f}s)")

        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            all_results[name] = {"error": str(e)}
            print(f"     FAILED: {e}")

    # Save shared artifacts
    joblib.dump(fe.tfidf, f"{run_dir}/tfidf.pkl")
    joblib.dump(scaler, f"{run_dir}/scaler.pkl")

    # 9. Save metadata
    print(f"\n[8/8] Saving results...")
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "version": version,
        "config": {
            "bert_model": Config.BERT_MODEL_NAME,
            "tfidf_max_features": Config.TFIDF_MAX_FEATURES,
            "random_state": Config.RANDOM_STATE,
            "test_size": Config.TEST_SIZE,
            "smote_ratio": Config.SMOTE_RATIO,
            "sample_size": Config.SAMPLE_SIZE,
        },
        "dataset": {
            "total": len(texts),
            "train": len(X_train_txt),
            "test": len(X_test_txt),
        },
        "weak_labels": label_stats,
        "models": all_results,
    }
    save_json(metadata, f"{run_dir}/metadata.json")
    save_json(metadata, f"{Config.MODELS_DIR}/multi_model_results.json")

    # Update latest symlink
    latest = f"{Config.MODELS_DIR}/latest"
    if os.path.islink(latest):
        os.remove(latest)
    elif os.path.isdir(latest):
        import shutil
        shutil.rmtree(latest)
    os.symlink(version, latest)

    # Summary table
    print("\n" + "=" * 90)
    print("MODEL COMPARISON RESULTS")
    print("=" * 90)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'Precision':>10} "
          f"{'Recall':>10} {'AUC':>10} {'Time':>8}")
    print("-" * 90)

    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1].get("f1_score", 0),
        reverse=True,
    )

    for name, m in sorted_models:
        if "error" in m:
            print(f"{name:<25} {'ERROR':>10}")
            continue
        auc = f"{m.get('roc_auc', 0):.4f}" if "roc_auc" in m else "N/A"
        print(f"{name:<25} {m['accuracy']:>10.4f} {m['f1_score']:>10.4f} "
              f"{m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{auc:>10} {m['training_time']:>7.1f}s")

    print("=" * 90)
    print(f"\nResults saved to: {run_dir}/")
    print(f"Run 'python compare_models.py' to generate top-5 comparison charts.")


if __name__ == "__main__":
    main()
