import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.settings import Config
from src.pipeline import SpamDetectionPipeline
from src.utils import set_seed, ensure_dir
from src.logger import setup_logger
from src.evaluation import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    analyze_feature_importance
)

logger = setup_logger(__name__, log_file=Config.LOG_FILE)


def main():
    logger.info("=" * 80)
    logger.info("TURKISH SPAM DETECTION - TRAINING PIPELINE")
    logger.info("=" * 80)
    
    try:
        logger.info("Validating configuration...")
        Config.validate()
        
        logger.info(f"Setting random seed: {Config.RANDOM_STATE}")
        set_seed(Config.RANDOM_STATE)
        
        logger.info(f"Loading data from: {Config.DATA_PATH}")
        if not os.path.exists(Config.DATA_PATH):
            raise FileNotFoundError(
                f"Data file not found: {Config.DATA_PATH}\n"
                f"Please place veri_seti_200k.csv in data/raw/ directory."
            )
        
        df = pd.read_csv(Config.DATA_PATH, low_memory=False, usecols=['comment'])
        logger.info(f"Loaded {len(df)} total samples")
        
        df = df.dropna(subset=['comment'])
        logger.info(f"After removing NaN: {len(df)} samples")
        
        if Config.SAMPLE_SIZE is not None:
            sample_size = min(Config.SAMPLE_SIZE, len(df))
            df = df.sample(n=sample_size, random_state=Config.RANDOM_STATE)
            logger.info(f"Sampled {sample_size} samples for training")
        
        texts = df['comment'].tolist()
        
        logger.info("Initializing pipeline...")
        pipeline = SpamDetectionPipeline(config=Config)
        
        logger.info("Training pipeline (this may take several minutes)...")
        pipeline.fit(texts)
        
        logger.info("Saving model...")
        model_path = pipeline.save()
        logger.info(f"Model saved to: {model_path}")
        
        logger.info("Generating evaluation plots...")
        ensure_dir(f"{model_path}/plots")
        
      
        logger.info("Evaluation plots will be generated in future iterations")
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Model version: {os.path.basename(model_path)}")
        logger.info(f"Accuracy: {pipeline.metadata['metrics']['accuracy']:.4f}")
        logger.info(f"F1 Score: {pipeline.metadata['metrics']['f1_score']:.4f}")
        logger.info("=" * 80)
        
    except FileNotFoundError as e:
        logger.error(f"File Error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()