# Hybrid Spam Review Detection in Turkish E-Commerce Platforms

**Authors:** Muhammet Burak KILIÇ, İsmail Hakkı Birgez  
**Domain:** NLP, Machine Learning, Weak Supervision  
**Status:** Production-Ready (v2.0)

---

## Project Overview

This project detects deceptive (spam) reviews in Turkish e-commerce platforms using a **Hybrid Feature Fusion** approach. The system combines:

1. **Contextual Embeddings** from BERTurk (Transformer-based)
2. **Lexical Features** using TF-IDF vectors
3. **Weak Supervision** through rule-based heuristics for unlabeled data

### Key Improvements (v2.0)

✅ **Fixed critical data leakage** - Train-test split now occurs BEFORE feature extraction  
✅ **Unified pipeline architecture** - Consistent train/inference workflow  
✅ **Production-ready code** - Comprehensive error handling, logging, and type hints  
✅ **Model versioning** - Automatic timestamped model saves with metadata  
✅ **API deployment** - FastAPI REST endpoint for production use  

---

## Technology Stack

- **Language:** Python 3.8+
- **Core ML:** PyTorch, Transformers (HuggingFace), Scikit-Learn
- **Model:** Random Forest (trained on BERT + TF-IDF hybrid features)
- **Data Handling:** Pandas, NumPy, Imbalanced-Learn (SMOTE)
- **Deployment:** FastAPI, Uvicorn

---

## Project Structure

```text
turkish-spam-detection/
├── config/                 # Configuration and settings
│   └── settings.py         # Centralized configuration with validation
├── data/                   # Data storage (not in repository)
│   └── raw/                # Place veri_seti_200k.csv here
├── models/                 # Versioned trained models
│   └── YYYYMMDD_HHMMSS/    # Timestamped model directories
├── logs/                   # Training and application logs
├── src/                    # Source code modules
│   ├── features.py         # BERT and TF-IDF feature extraction
│   ├── heuristics.py       # Weak labeling logic
│   ├── preprocessing.py    # Text cleaning (BERT vs TF-IDF optimized)
│   ├── pipeline.py         # Unified ML pipeline
│   ├── evaluation.py       # Metrics and visualization
│   ├── logger.py           # Logging utilities
│   └── utils.py            # Helper functions
├── train.py                # Training script
├── predict.py              # Interactive inference demo
├── api.py                  # FastAPI deployment
└── requirements.txt        # Python dependencies
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/embikaa/turkish-spam-detection.git
cd turkish-spam-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Data

Place your dataset `veri_seti_200k.csv` in the `data/raw/` directory.

```bash
mkdir -p data/raw
```

---

## Usage

### Training the Model

```bash
python train.py
```

**What happens:**
- Loads and validates data
- Generates weak labels using heuristics
- Splits data (80/20) **BEFORE** feature extraction
- Extracts BERT + TF-IDF features
- Applies SMOTE for class balancing
- Trains Random Forest classifier
- Evaluates on test set
- Saves versioned model with metadata

**Output:**
```
models/20260215_103045/
├── model.pkl           # Trained Random Forest
├── tfidf.pkl           # Fitted TF-IDF vectorizer
├── scaler.pkl          # Fitted StandardScaler
└── metadata.json       # Metrics, config, timestamps
```

### Interactive Testing

```bash
python predict.py
```

**Example:**
```
Yorumu Girin: Harika ürün, çok memnun kaldım!
🟢 GENUINE
Spam İhtimali: %15.32
Güven Seviyesi: Yüksek
```

### API Deployment

```bash
# Install API dependencies
pip install -r requirements-api.txt

# Run API server
python api.py
```

**API Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /model-info` - Model metrics and version
- `POST /predict` - Spam prediction

**Example API Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Harika ürün!"}'
```

**Response:**
```json
{
  "is_spam": false,
  "spam_probability": 0.1532,
  "confidence": "Yüksek",
  "model_version": "2026-02-15T10:30:45"
}
```

---

## Methodology

### Weak Supervision

Since the dataset is unlabeled, heuristic rules generate "silver labels":

- **Brevity Penalty:** Very short reviews → Spam
- **Generic Density:** High filler word ratio → Spam  
- **Structural Complexity:** Long reviews with digits → Genuine
- **Capitalization:** ALL CAPS reviews → Spam

### Hybrid Feature Fusion

The classifier receives concatenated features:

```
V_final = [V_TF-IDF ⊕ V_BERT]
```

This captures both:
- **Lexical patterns** (TF-IDF): Keyword-based signals
- **Semantic meaning** (BERT): Deep contextual understanding

### Data Leakage Prevention

**CRITICAL FIX:** Train-test split occurs **BEFORE** feature extraction:

```python
# 1. Split data first
X_train, X_test, y_train, y_test = train_test_split(texts, labels)

# 2. Fit TF-IDF only on training data
tfidf.fit(X_train)

# 3. Transform both sets
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)  # No leakage!
```

---

## Configuration

Edit `config/settings.py` to customize:

```python
class Config:
    # Model parameters
    BERT_MODEL_NAME = "dbmdz/bert-base-turkish-cased"
    TFIDF_MAX_FEATURES = 500
    
    # Training parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    SMOTE_RATIO = 0.5
    SAMPLE_SIZE = 5000  # None for full dataset
    
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## Results

> [!IMPORTANT]
> **Metrics updated after fixing data leakage (v2.0)**

Performance on test set (5000 samples):

- **Accuracy:** ~85-88% (down from inflated 90% due to leakage fix)
- **F1-Score (Spam):** ~0.78-0.82
- **Precision:** ~0.80-0.85
- **Recall:** ~0.75-0.80

**Note:** Results vary with random seed and sample size. Use cross-validation for robust estimates.

---

## Troubleshooting

### GPU Out of Memory

Reduce batch size in `config/settings.py`:
```python
BATCH_SIZE = 8  # Default is 16
```

### Model Not Found

Ensure you've trained a model first:
```bash
python train.py
```

### Data File Missing

Place `veri_seti_200k.csv` in `data/raw/` directory.

---

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Type checking
mypy src/

# Linting
flake8 src/
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{kilic2026turkish,
  author = {Kılıç, Muhammet Burak and Birgez, İsmail Hakkı},
  title = {Hybrid Spam Review Detection in Turkish E-Commerce Platforms},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/turkish-spam-detection}
}
```

---

## License

This project is developed for academic research purposes.

---

## Changelog

### v2.0 (2026-02-15)
- 🔴 **CRITICAL:** Fixed data leakage in feature extraction
- ✨ Added unified `SpamDetectionPipeline` class
- ✨ Implemented model versioning with metadata
- ✨ Added comprehensive logging and error handling
- ✨ Created FastAPI deployment
- 📝 Added type hints and docstrings throughout
- 🐛 Fixed preprocessing inconsistency (digit removal)
- 🐛 Fixed train-inference feature mismatch

### v1.0 (Initial Release)
- Basic hybrid model implementation
- Weak supervision with heuristics
- BERT + TF-IDF feature fusion
