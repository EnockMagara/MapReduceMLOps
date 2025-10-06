#!/usr/bin/env python3
"""
Evaluation script for Movie Review Sentiment Analysis.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.settings import Config
from utils.logger import setup_logger
from utils.spark_utils import create_spark_session, stop_spark_session
from data.loader import DataLoader
from models.trainer import ModelTrainer
import click


@click.command()
@click.option("--model-path", required=True, help="Path to trained model")
@click.option("--test-file", help="Test file path")
def main(model_path: str, test_file: str = None):
    """Main evaluation function."""
    
    logger = setup_logger()
    logger.info("Starting model evaluation")
    
    try:
        # Load configuration
        config = Config()
        
        # Create Spark session
        spark = create_spark_session(config.get_spark_config())
        
        # Load test data
        if test_file is None:
            test_file = str(Path(config.data.raw_path) / config.data.test_file)
        
        data_loader = DataLoader(spark)
        test_df = data_loader.load_test_data(test_file)
        
        # Load model
        from pyspark.ml import PipelineModel
        model = PipelineModel.load(model_path)
        
        # Evaluate model
        model_trainer = ModelTrainer(spark, config)
        metrics = model_trainer.evaluate_model(model, test_df)
        
        logger.info("Evaluation completed")
        logger.info(f"Metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
    
    finally:
        if 'spark' in locals():
            stop_spark_session(spark)


if __name__ == "__main__":
    main()
