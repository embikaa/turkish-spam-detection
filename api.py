from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.settings import Config
from src.pipeline import SpamDetectionPipeline
from src.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Turkish Spam Detection API",
    description="Hybrid ML system for detecting spam reviews in Turkish e-commerce platforms",
    version="1.0.0"
)

pipeline: Optional[SpamDetectionPipeline] = None


class ReviewRequest(BaseModel):
    
    text: str = Field(..., min_length=2, description="Review text to analyze")


class ReviewResponse(BaseModel):
    
    is_spam: bool = Field(..., description="Whether the review is classified as spam")
    spam_probability: float = Field(..., ge=0, le=1, description="Probability of being spam (0-1)")
    confidence: str = Field(..., description="Confidence level: Yüksek, Orta, or Düşük")
    model_version: Optional[str] = Field(None, description="Model version timestamp")


@app.on_event("startup")
async def load_model():
    
    global pipeline
    
    model_path = f"{Config.MODELS_DIR}/latest"
    
    if not os.path.exists(model_path):
        logger.error("No trained model found! Please train a model first.")
        raise RuntimeError("Model not found")
    
    try:
        pipeline = SpamDetectionPipeline.load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/")
async def root():
   
    return {
        "message": "Turkish Spam Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model-info"
        }
    }


@app.get("/health")
async def health_check():
  
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.get("/model-info")
async def model_info():
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "version": pipeline.metadata.get('timestamp', 'unknown'),
        "metrics": pipeline.metadata.get('metrics', {}),
        "config": pipeline.metadata.get('config', {})
    }


@app.post("/predict", response_model=ReviewResponse)
async def predict_review(request: ReviewRequest):

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.text.strip()) < 2:
        raise HTTPException(status_code=400, detail="Text too short (minimum 2 characters)")
    
    try:
        prediction = pipeline.predict([request.text])[0]
        probabilities = pipeline.predict_proba([request.text])[0]
        spam_probability = float(probabilities[1])
        
        confidence_score = abs(spam_probability - 0.5) * 2
        if confidence_score > 0.6:
            confidence = "Yüksek"
        elif confidence_score > 0.3:
            confidence = "Orta"
        else:
            confidence = "Düşük"
        
        return ReviewResponse(
            is_spam=bool(prediction),
            spam_probability=spam_probability,
            confidence=confidence,
            model_version=pipeline.metadata.get('timestamp')
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
