#!/usr/bin/env python3
"""
Prediction script for Movie Review Sentiment Analysis.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pyspark.ml import PipelineModel
from config.settings import Config
from utils.logger import setup_logger
from utils.spark_utils import create_spark_session, stop_spark_session
import click


@click.command()
@click.option("--model-path", required=True, help="Path to trained model")
@click.option("--text", help="Text to predict")
@click.option("--input-file", help="Input file with texts to predict")
@click.option("--output-file", help="Output file for predictions")
def main(model_path: str, text: str = None, input_file: str = None, output_file: str = None):
    """Main prediction function."""
    
    logger = setup_logger()
    logger.info("Starting prediction")
    
    try:
        # Load configuration
        config = Config()
        
        # Create Spark session
        spark = create_spark_session(config.get_spark_config())
        
        # Load model
        model = PipelineModel.load(model_path)
        
        if text:
            # Single text prediction
            from pyspark.sql import Row
            df = spark.createDataFrame([Row(text=text)])
            prediction = model.transform(df)
            result = prediction.select("text", "prediction", "probability").collect()[0]
            
            logger.info(f"Text: {result['text']}")
            logger.info(f"Prediction: {result['prediction']}")
            logger.info(f"Probability: {result['probability']}")
            
        elif input_file:
            # Batch prediction
            df = spark.read.option("header", True).csv(input_file)
            predictions = model.transform(df)
            
            if output_file:
                predictions.select("text", "prediction", "probability").write.mode("overwrite").csv(output_file)
                logger.info(f"Predictions saved to {output_file}")
            else:
                predictions.select("text", "prediction", "probability").show()
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise
    
    finally:
        if 'spark' in locals():
            stop_spark_session(spark)


if __name__ == "__main__":
    main()
