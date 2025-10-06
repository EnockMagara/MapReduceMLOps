#!/usr/bin/env python3
"""
PyTorch prediction script for Movie Review Sentiment Analysis.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.settings import Config
from utils.logger import setup_logger
from models.pytorch_trainer import PyTorchTrainer
import click


@click.command()
@click.option("--model-path", required=True, help="Path to trained PyTorch model")
@click.option("--model-type", required=True, help="Model type: lstm, transformer, bert")
@click.option("--text", help="Text to predict")
@click.option("--input-file", help="Input file with texts to predict")
@click.option("--output-file", help="Output file for predictions")
def main(model_path: str, model_type: str, text: str = None, input_file: str = None, output_file: str = None):
    """Main PyTorch prediction function."""
    
    logger = setup_logger()
    logger.info("Starting PyTorch prediction")
    
    try:
        # Load configuration
        config = Config()
        
        # Initialize trainer
        pytorch_trainer = PyTorchTrainer(config)
        
        # Load model
        pytorch_trainer.load_model(model_path, model_type)
        
        if text:
            # Single text prediction
            predictions, probabilities = pytorch_trainer.predict([text])
            
            logger.info(f"Text: {text}")
            logger.info(f"Prediction: {predictions[0]}")
            logger.info(f"Probability: {probabilities[0]}")
            
        elif input_file:
            # Batch prediction
            import pandas as pd
            
            df = pd.read_csv(input_file)
            texts = df['text'].tolist()
            
            predictions, probabilities = pytorch_trainer.predict(texts)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'text': texts,
                'prediction': predictions,
                'probability_negative': [prob[0] for prob in probabilities],
                'probability_positive': [prob[1] for prob in probabilities]
            })
            
            if output_file:
                results_df.to_csv(output_file, index=False)
                logger.info(f"Predictions saved to {output_file}")
            else:
                print(results_df.to_string())
        
    except Exception as e:
        logger.error(f"PyTorch prediction failed: {str(e)}")
        raise
    
    finally:
        logger.info("PyTorch prediction script completed")


if __name__ == "__main__":
    main()
