#!/usr/bin/env python3
"""
PyTorch training script for Movie Review Sentiment Analysis.

This script implements the PyTorch neural network training pipeline:
1. Data loading and preprocessing
2. Model creation (LSTM, Transformer, BERT)
3. Training with PyTorch
4. Model evaluation and logging
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.settings import Config
from utils.logger import setup_logger
from utils.mlflow_utils import setup_mlflow
from data.pytorch_loader import PyTorchDataLoader
from models.pytorch_trainer import PyTorchTrainer
import click


@click.command()
@click.option("--config", default="config/config.yaml", help="Configuration file path")
@click.option("--model-type", default="lstm", help="Model type: lstm, transformer, bert")
@click.option("--log-level", default="INFO", help="Logging level")
def main(config: str, model_type: str, log_level: str):
    """Main PyTorch training function."""
    
    # Setup logging
    logger = setup_logger(level=log_level)
    logger.info(f"Starting PyTorch training with {model_type} model")
    
    try:
        # Load configuration
        config_obj = Config(config)
        logger.info(f"Configuration loaded from {config}")
        
        # Setup MLflow
        setup_mlflow(config_obj.mlops.__dict__)
        
        # Initialize components
        data_loader = PyTorchDataLoader()
        pytorch_trainer = PyTorchTrainer(config_obj)
        
        # Load training data
        logger.info("Loading training data")
        train_file = Path(config_obj.data.raw_path) / config_obj.data.train_file
        train_texts, train_labels = data_loader.load_train_data(str(train_file))
        
        # Load test data
        logger.info("Loading test data")
        test_file = Path(config_obj.data.raw_path) / config_obj.data.test_file
        test_texts, test_labels = data_loader.load_test_data(str(test_file))
        
        # Get data summary
        train_summary = data_loader.get_data_summary(train_texts, train_labels)
        test_summary = data_loader.get_data_summary(test_texts, test_labels)
        
        logger.info(f"Training data summary: {train_summary}")
        logger.info(f"Test data summary: {test_summary}")
        
        # Prepare data loaders
        logger.info("Preparing data loaders")
        train_loader, val_loader = pytorch_trainer.prepare_data(
            train_texts, train_labels, model_type
        )
        
        # Create vocabulary for LSTM/Transformer models
        vocab_size = None
        if model_type in ["lstm", "transformer"]:
            vocab = data_loader.create_vocabulary(train_texts)
            vocab_size = len(vocab)
            logger.info(f"Created vocabulary with {vocab_size} words")
        
        # Create model
        logger.info(f"Creating {model_type} model")
        model = pytorch_trainer.create_model(model_type, vocab_size)
        
        # Train model
        logger.info("Starting model training")
        training_results = pytorch_trainer.train(train_loader, val_loader, model_type)
        
        # Log final results
        logger.info("PyTorch training completed successfully!")
        logger.info(f"Best validation metrics: {training_results['best_metrics']}")
        
        # Save model
        model_path = f"models/pytorch_{model_type}_model.pth"
        pytorch_trainer.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Test on test set
        logger.info("Evaluating on test set")
        test_predictions, test_probabilities = pytorch_trainer.predict(test_texts)
        
        # Calculate test metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        
        test_accuracy = accuracy_score(test_labels, test_predictions)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            test_labels, test_predictions, average='weighted'
        )
        
        try:
            test_auc = roc_auc_score(test_labels, [prob[1] for prob in test_probabilities])
        except ValueError:
            test_auc = 0.0
        
        test_metrics = {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc
        }
        
        logger.info(f"Test metrics: {test_metrics}")
        
        # Log test metrics to MLflow
        import mlflow
        with mlflow.start_run():
            for metric, value in test_metrics.items():
                mlflow.log_metric(metric, value)
        
    except Exception as e:
        logger.error(f"PyTorch training failed: {str(e)}")
        raise
    
    finally:
        logger.info("PyTorch training script completed")


if __name__ == "__main__":
    main()
