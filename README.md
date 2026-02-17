# Turkish Spam Detection — Hybrid ML System

Multi-model machine learning system for detecting spam reviews in Turkish e-commerce platforms, combining **BERTurk** contextual embeddings with **TF-IDF** statistical features under a **weak supervision** labeling framework.

## Architecture

```
Raw Text (Turkish reviews)
    │
    ├── Preprocessing
    │   ├── clean_text_for_bert()   → minimal cleaning
    │   └── clean_text_for_tfidf() → aggressive cleaning + stopword removal
    │
    ├── Feature Engineering
    │   ├── TF-IDF (500 features)
    │   └── BERTurk [CLS] embeddings (768 features)
    │   → Combined: 1268-dimensional feature vector
    │
    ├── Weak Supervision (Heuristic Labeling)
    │   └── Rule-based labels using structural features
    │
    ├── Class Balancing (SMOTE)
    │
    └── Model Training (10 classifiers)
        ├── Logistic Regression
        ├── K-Nearest Neighbors
        ├── Support Vector Machine
        ├── Artificial Neural Network (MLP)
        ├── Decision Tree (CART)
        ├── Random Forest
        ├── Gradient Boosting (GBM)
        ├── XGBoost
        ├── LightGBM
        └── CatBoost
```

## Project Structure

```
turkish-spam-detection/
├── config/
│   └── settings.py           # Central configuration
├── src/
│   ├── preprocessing.py      # Text cleaning (BERT & TF-IDF pipelines)
│   ├── features.py           # Feature extraction (TF-IDF + BERTurk)
│   ├── heuristics.py         # Weak supervision labeling rules
│   ├── evaluation.py         # Metrics computation
│   ├── logger.py             # Centralized logging
│   └── utils.py              # Seed, I/O, directory utilities
├── data/
│   └── raw/                  # Place dataset here
├── models/                   # Trained model artifacts
├── results/                  # Comparison charts & tables
├── train_all_models.py       # Train all 10 models
├── compare_models.py         # Top-5 comparison & visualization
├── requirements.txt          # Dependencies
└── README.md
```

## Quick Start

### 1. Setup

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Prepare Data

Place `veri_seti_200k.csv` in `data/raw/` directory. The dataset must have a `comment` column.

### 3. Train All Models

```bash
python train_all_models.py
```

This will:
- Load and preprocess the dataset
- Generate weak labels using heuristic rules
- Extract TF-IDF + BERTurk features
- Apply SMOTE for class balancing
- Train 10 models with optimized hyperparameters
- Save results to `models/` directory

### 4. Compare & Visualize

```bash
python compare_models.py
```

Generates in `results/`:
- `comparison_table.txt` — Top-5 model comparison table
- `comparison_chart.png` — Grouped bar chart of all metrics
- `weak_label_distribution.png` — Weak labeling stats (count + percentage)
- `confusion_matrices.png` — Side-by-side confusion matrices
- `all_models_ranking.png` — Full ranking of all models by F1 score

## Models & Hyperparameters

| Model | Key Hyperparameters |
|-------|-------------------|
| Logistic Regression | C=1.0, penalty=l2, solver=lbfgs |
| KNN | k=7, weights=distance, metric=cosine |
| SVM | C=10.0, kernel=rbf, gamma=scale |
| ANN (MLP) | layers=(256,128,64), adam, early_stopping |
| CART | max_depth=20, min_samples_split=10 |
| Random Forest | n_estimators=200, max_depth=30 |
| GBM | n_estimators=200, lr=0.1, max_depth=5 |
| XGBoost | n_estimators=200, lr=0.1, max_depth=6 |
| LightGBM | n_estimators=200, lr=0.1, num_leaves=31 |
| CatBoost | iterations=200, lr=0.1, depth=6 |

## Methodology

### Weak Supervision

Since the dataset lacks ground-truth labels, heuristic rules generate noisy labels based on structural features:

- **Short reviews** (< 5 words) → likely spam
- **Generic keywords only** → likely spam
- **Excessive punctuation** (!!!???) → likely spam
- **All caps text** → likely spam
- **Emoji-heavy** → likely spam
- **Character repetition** (aaaa, güzellll) → likely spam
- **Contains URL** → likely spam
- **Long reviews** (≥ 20 words) → likely genuine

A spam score ≥ 2 classifies as spam.

### Data Leakage Prevention

- Train/test split performed **before** feature extraction
- TF-IDF fitted only on training data, then transformed on test data
- SMOTE applied only to training data

### Feature Engineering

- **TF-IDF**: 500 most informative n-grams from aggressively cleaned text
- **BERTurk**: 768-dim [CLS] embeddings from minimally cleaned text
- **Combined**: 1268-dimensional feature vectors

## Requirements

- Python ≥ 3.10
- PyTorch
- Transformers (HuggingFace)
- scikit-learn
- XGBoost, LightGBM (optional: CatBoost)

See `requirements.txt` for full list.

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | BERTurk (dbmdz/bert-base-turkish-cased) |
| NLP | TF-IDF, Turkish stopword removal |
| ML Framework | scikit-learn, XGBoost, LightGBM |
| Labeling | Weak supervision (heuristic rules) |
| Balancing | SMOTE (imbalanced-learn) |
| Visualization | Matplotlib |
