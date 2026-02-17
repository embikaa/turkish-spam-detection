"""
Web Dashboard API for Turkish spam detection — multi-model version.

Serves dashboard and provides endpoints for prediction & metrics.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.settings import Config
from src.pipeline import SpamDetectionPipeline
from src.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(title="Spam Detection Dashboard", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

pipeline: Optional[SpamDetectionPipeline] = None
multi_results: Optional[dict] = None


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=2)


class ReviewResponse(BaseModel):
    is_spam: bool
    spam_probability: float = Field(ge=0, le=1)
    confidence: str
    model_version: Optional[str] = None


@app.on_event("startup")
async def startup():
    global pipeline, multi_results

    # Load pipeline for predictions
    model_path = f"{Config.MODELS_DIR}/latest"
    if os.path.exists(model_path):
        try:
            pipeline = SpamDetectionPipeline.load(model_path)
            logger.info("Pipeline loaded for predictions")
        except Exception as e:
            logger.error(f"Pipeline load failed: {e}")

    # Load multi-model results
    results_path = f"{Config.MODELS_DIR}/multi_model_results.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            multi_results = json.load(f)
        logger.info(f"Loaded results for {len(multi_results.get('models', {}))} models")
    else:
        # Fall back to single-model metadata
        meta_path = f"{model_path}/metadata.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            multi_results = {
                'version': meta.get('timestamp', 'unknown'),
                'config': meta.get('config', {}),
                'dataset': meta.get('dataset', {}),
                'models': {'Random Forest': meta.get('metrics', {})}
            }
            logger.info("Loaded single-model fallback results")


@app.get("/")
async def dashboard():
    return FileResponse("templates/index.html")


@app.get("/model-info")
async def model_info():
    if multi_results is None:
        raise HTTPException(503, "No results loaded")
    return multi_results


@app.get("/health")
async def health():
    return {"status": "healthy", "pipeline": pipeline is not None,
            "results": multi_results is not None}


@app.post("/predict", response_model=ReviewResponse)
async def predict(req: ReviewRequest):
    if pipeline is None:
        raise HTTPException(503, "Model not loaded")
    try:
        pred = pipeline.predict([req.text])[0]
        proba = pipeline.predict_proba([req.text])[0]
        spam_p = float(proba[1])
        conf_score = abs(spam_p - 0.5) * 2
        confidence = "Yüksek" if conf_score > .6 else ("Orta" if conf_score > .3 else "Düşük")
        return ReviewResponse(is_spam=bool(pred), spam_probability=spam_p,
                              confidence=confidence,
                              model_version=pipeline.metadata.get('timestamp'))
    except Exception as e:
        logger.error(f"Predict error: {e}")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 Spam Detection Dashboard — http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
