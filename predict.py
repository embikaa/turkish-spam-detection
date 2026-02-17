import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.settings import Config
from src.pipeline import SpamDetectionPipeline
from src.logger import setup_logger

logger = setup_logger(__name__)


def load_pipeline():
    logger.info("Loading trained model...")
    
    model_path = f"{Config.MODELS_DIR}/latest"
    
    if not os.path.exists(model_path):
        logger.error("No trained model found!")
        logger.error("Please run 'python train.py' first to train a model.")
        sys.exit(1)
    
    try:
        pipeline = SpamDetectionPipeline.load(model_path)
        logger.info("Model loaded successfully!")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)


def predict_single(pipeline: SpamDetectionPipeline, text: str) -> dict:
    prediction = pipeline.predict([text])[0]
    probabilities = pipeline.predict_proba([text])[0]
    spam_probability = probabilities[1]
    
    confidence_score = abs(spam_probability - 0.5) * 2  # 0 to 1
    if confidence_score > 0.6:
        confidence = "Yüksek"
    elif confidence_score > 0.3:
        confidence = "Orta"
    else:
        confidence = "Düşük"
    
    return {
        'is_spam': bool(prediction),
        'spam_probability': float(spam_probability),
        'confidence': confidence,
        'confidence_score': float(confidence_score)
    }


def main():
    
    pipeline = load_pipeline()
    
    if 'timestamp' in pipeline.metadata:
        print(f"Model Version: {pipeline.metadata['timestamp']}")
    if 'metrics' in pipeline.metadata:
        metrics = pipeline.metadata['metrics']
        print(f"Model Accuracy: {metrics['accuracy']:.2%}")
        print(f"Model F1 Score: {metrics['f1_score']:.2%}")
    print()
    
    print("Exit for 'q' write and press enter.")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nEnter a comment: ")
            
            if user_input.lower() == 'q':
                print("\nProgram is finishing")
                break
            
            if len(user_input.strip()) < 2:
                print("Please write a long comment")
                continue
            
            result = predict_single(pipeline, user_input)
            
            if result['is_spam']:
                status = "🔴 SPAM"
                color_code = "\033[91m"  
            else:
                status = "🟢 GENUINE"
                color_code = "\033[92m" 
            
            reset_code = "\033[0m"
            
            print(f"\n{color_code}Sonuç: {status}{reset_code}")
            print(f"Spam Rate: %{result['spam_probability']*100:.2f}")
            print(f"Trust Level: {result['confidence']}")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nProgram is finishing")
            break
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()