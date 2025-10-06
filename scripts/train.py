#!/usr/bin/env python3
"""
Training script for Movie Review Sentiment Analysis.

This script implements the complete MapReduce + ML pipeline:
1. Data loading and preprocessing
2. MapReduce for feature extraction
3. Model training with Spark MLlib
4. Model evaluation and logging
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
from processing.mapreduce import MapReduceProcessor
from models.trainer import ModelTrainer
import click


@click.command()
@click.option("--config", default="config/config.yaml", help="Configuration file path")
@click.option("--algorithm", default="naive_bayes", help="Model algorithm to use")
@click.option("--cross-validate", is_flag=True, help="Perform cross-validation")
@click.option("--log-level", default="INFO", help="Logging level")
def main(config: str, algorithm: str, cross_validate: bool, log_level: str):
    """Main training function."""
    
    # Setup logging
    logger = setup_logger(level=log_level)
    logger.info("Starting Movie Review Sentiment Analysis training")
    
    try:
        # Load configuration
        config_obj = Config(config)
        logger.info(f"Configuration loaded from {config}")
        
        # Setup MLflow
        setup_mlflow(config_obj.mlops.__dict__)
        
        # Create Spark session
        spark = create_spark_session(config_obj.get_spark_config())
        
        # Initialize components
        data_loader = DataLoader(spark)
        mapreduce_processor = MapReduceProcessor(spark, config_obj)
        model_trainer = ModelTrainer(spark, config_obj)
        
        # Load training data
        logger.info("Loading training data")
        train_file = Path(config_obj.data.raw_path) / config_obj.data.train_file
        train_df = data_loader.load_train_data(str(train_file))
        
        # Load test data
        logger.info("Loading test data")
        test_file = Path(config_obj.data.raw_path) / config_obj.data.test_file
        test_df = data_loader.load_test_data(str(test_file))
        
        # Get data summary
        train_summary = data_loader.get_data_summary(train_df)
        test_summary = data_loader.get_data_summary(test_df)
        
        logger.info(f"Training data summary: {train_summary}")
        logger.info(f"Test data summary: {test_summary}")
        
        # MapReduce processing
        logger.info("Starting MapReduce processing")
        
        # Map phase: Tokenize and create word pairs
        words_df = mapreduce_processor.map_phase(train_df)
        
        # Reduce phase: Aggregate word counts
        word_counts = mapreduce_processor.reduce_phase(words_df)
        
        # Build vocabulary
        vocabulary = mapreduce_processor.build_vocabulary(word_counts)
        
        # Get word statistics
        word_stats = mapreduce_processor.get_word_statistics(word_counts)
        logger.info(f"Word statistics: {word_stats}")
        
        # Feature extraction
        logger.info("Extracting features")
        train_features = mapreduce_processor.extract_features(train_df, vocabulary)
        test_features = mapreduce_processor.extract_features(test_df, vocabulary)
        
        # Prepare training data
        train_prepared = model_trainer.prepare_training_data(train_features)
        test_prepared = model_trainer.prepare_training_data(test_features)
        
        # Train model
        if cross_validate:
            logger.info("Training model with cross-validation")
            model = model_trainer.cross_validate(train_prepared, algorithm)
        else:
            logger.info("Training model")
            model = model_trainer.train_model(train_prepared, algorithm)
        
        # Evaluate model
        logger.info("Evaluating model")
        metrics = model_trainer.evaluate_model(model, test_prepared)
        
        # Log final results
        logger.info("Training completed successfully!")
        logger.info(f"Final metrics: {metrics}")
        
        # Save model
        model_path = f"models/{algorithm}_model"
        model.write().overwrite().save(model_path)
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    finally:
        # Stop Spark session
        if 'spark' in locals():
            stop_spark_session(spark)
        logger.info("Training script completed")


if __name__ == "__main__":
    main()
