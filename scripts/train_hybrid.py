#!/usr/bin/env python3
"""
Hybrid training script for Movie Review Sentiment Analysis.

This script combines Spark MapReduce processing with PyTorch neural networks:
1. Use Spark for distributed data processing and feature extraction
2. Use PyTorch for advanced neural network training
3. Compare different approaches and models
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.settings import Config
from utils.logger import setup_logger
from utils.spark_utils import create_spark_session, stop_spark_session
from utils.mlflow_utils import setup_mlflow
from data.loader import DataLoader
from data.pytorch_loader import PyTorchDataLoader
from processing.mapreduce import MapReduceProcessor
from models.trainer import ModelTrainer
from models.pytorch_trainer import PyTorchTrainer
import click


@click.command()
@click.option("--config", default="config/config.yaml", help="Configuration file path")
@click.option("--spark-algorithm", default="naive_bayes", help="Spark MLlib algorithm")
@click.option("--pytorch-model", default="lstm", help="PyTorch model type")
@click.option("--compare", is_flag=True, help="Compare Spark and PyTorch results")
@click.option("--log-level", default="INFO", help="Logging level")
def main(config: str, spark_algorithm: str, pytorch_model: str, compare: bool, log_level: str):
    """Main hybrid training function."""
    
    # Setup logging
    logger = setup_logger(level=log_level)
    logger.info("Starting hybrid training pipeline")
    
    try:
        # Load configuration
        config_obj = Config(config)
        logger.info(f"Configuration loaded from {config}")
        
        # Setup MLflow
        setup_mlflow(config_obj.mlops.__dict__)
        
        # Create Spark session
        spark = create_spark_session(config_obj.get_spark_config())
        
        # Initialize components
        spark_data_loader = DataLoader(spark)
        pytorch_data_loader = PyTorchDataLoader()
        mapreduce_processor = MapReduceProcessor(spark, config_obj)
        spark_trainer = ModelTrainer(spark, config_obj)
        pytorch_trainer = PyTorchTrainer(config_obj)
        
        # Load data
        logger.info("Loading data for both Spark and PyTorch")
        train_file = Path(config_obj.data.raw_path) / config_obj.data.train_file
        test_file = Path(config_obj.data.raw_path) / config_obj.data.test_file
        
        # Spark data loading
        spark_train_df = spark_data_loader.load_train_data(str(train_file))
        spark_test_df = spark_data_loader.load_test_data(str(test_file))
        
        # PyTorch data loading
        pytorch_train_texts, pytorch_train_labels = pytorch_data_loader.load_train_data(str(train_file))
        pytorch_test_texts, pytorch_test_labels = pytorch_data_loader.load_test_data(str(test_file))
        
        # Get data summaries
        spark_train_summary = spark_data_loader.get_data_summary(spark_train_df)
        pytorch_train_summary = pytorch_data_loader.get_data_summary(pytorch_train_texts, pytorch_train_labels)
        
        logger.info(f"Spark training data: {spark_train_summary}")
        logger.info(f"PyTorch training data: {pytorch_train_summary}")
        
        # Train Spark model
        logger.info("=" * 50)
        logger.info("TRAINING SPARK MODEL")
        logger.info("=" * 50)
        
        # MapReduce processing
        words_df = mapreduce_processor.map_phase(spark_train_df)
        word_counts = mapreduce_processor.reduce_phase(words_df)
        vocabulary = mapreduce_processor.build_vocabulary(word_counts)
        
        # Feature extraction
        spark_train_features = mapreduce_processor.extract_features(spark_train_df, vocabulary)
        spark_test_features = mapreduce_processor.extract_features(spark_test_df, vocabulary)
        
        # Prepare training data
        spark_train_prepared = spark_trainer.prepare_training_data(spark_train_features)
        spark_test_prepared = spark_trainer.prepare_training_data(spark_test_features)
        
        # Train Spark model
        spark_model = spark_trainer.train_model(spark_train_prepared, spark_algorithm)
        
        # Evaluate Spark model
        spark_metrics = spark_trainer.evaluate_model(spark_model, spark_test_prepared)
        logger.info(f"Spark {spark_algorithm} metrics: {spark_metrics}")
        
        # Train PyTorch model
        logger.info("=" * 50)
        logger.info("TRAINING PYTORCH MODEL")
        logger.info("=" * 50)
        
        # Prepare PyTorch data
        pytorch_train_loader, pytorch_val_loader = pytorch_trainer.prepare_data(
            pytorch_train_texts, pytorch_train_labels, pytorch_model
        )
        
        # Create vocabulary for LSTM/Transformer
        vocab_size = None
        if pytorch_model in ["lstm", "transformer"]:
            vocab = pytorch_data_loader.create_vocabulary(pytorch_train_texts)
            vocab_size = len(vocab)
        
        # Create and train PyTorch model
        pytorch_model = pytorch_trainer.create_model(pytorch_model, vocab_size)
        pytorch_results = pytorch_trainer.train(pytorch_train_loader, pytorch_val_loader, pytorch_model)
        
        # Evaluate PyTorch model on test set
        pytorch_test_predictions, pytorch_test_probabilities = pytorch_trainer.predict(pytorch_test_texts)
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        
        pytorch_test_accuracy = accuracy_score(pytorch_test_labels, pytorch_test_predictions)
        pytorch_test_precision, pytorch_test_recall, pytorch_test_f1, _ = precision_recall_fscore_support(
            pytorch_test_labels, pytorch_test_predictions, average='weighted'
        )
        
        try:
            pytorch_test_auc = roc_auc_score(pytorch_test_labels, [prob[1] for prob in pytorch_test_probabilities])
        except ValueError:
            pytorch_test_auc = 0.0
        
        pytorch_test_metrics = {
            'test_accuracy': pytorch_test_accuracy,
            'test_precision': pytorch_test_precision,
            'test_recall': pytorch_test_recall,
            'test_f1': pytorch_test_f1,
            'test_auc': pytorch_test_auc
        }
        
        logger.info(f"PyTorch {pytorch_model} test metrics: {pytorch_test_metrics}")
        
        # Compare results if requested
        if compare:
            logger.info("=" * 50)
            logger.info("COMPARISON RESULTS")
            logger.info("=" * 50)
            
            logger.info(f"Spark {spark_algorithm} vs PyTorch {pytorch_model}:")
            logger.info(f"Accuracy: {spark_metrics.get('accuracy', 0):.4f} vs {pytorch_test_metrics['test_accuracy']:.4f}")
            logger.info(f"Precision: {spark_metrics.get('precision', 0):.4f} vs {pytorch_test_metrics['test_precision']:.4f}")
            logger.info(f"Recall: {spark_metrics.get('recall', 0):.4f} vs {pytorch_test_metrics['test_recall']:.4f}")
            logger.info(f"F1: {spark_metrics.get('f1', 0):.4f} vs {pytorch_test_metrics['test_f1']:.4f}")
            logger.info(f"AUC: {spark_metrics.get('auc', 0):.4f} vs {pytorch_test_metrics['test_auc']:.4f}")
        
        # Save models
        spark_model_path = f"models/spark_{spark_algorithm}_model"
        spark_model.write().overwrite().save(spark_model_path)
        logger.info(f"Spark model saved to {spark_model_path}")
        
        pytorch_model_path = f"models/pytorch_{pytorch_model}_model.pth"
        pytorch_trainer.save_model(pytorch_model_path)
        logger.info(f"PyTorch model saved to {pytorch_model_path}")
        
        logger.info("Hybrid training completed successfully!")
        
    except Exception as e:
        logger.error(f"Hybrid training failed: {str(e)}")
        raise
    
    finally:
        # Stop Spark session
        if 'spark' in locals():
            stop_spark_session(spark)
        logger.info("Hybrid training script completed")


if __name__ == "__main__":
    main()
