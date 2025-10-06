"""Spark utilities for the Movie Review Sentiment Analysis project."""

from pyspark.sql import SparkSession
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def create_spark_session(config: Dict[str, Any]) -> SparkSession:
    """
    Create Spark session with configuration.
    
    Args:
        config: Spark configuration dictionary
        
    Returns:
        SparkSession instance
    """
    logger.info("Creating Spark session")
    
    # Create Spark session builder
    builder = SparkSession.builder.appName(config["spark.app.name"])
    
    # Set configuration
    for key, value in config.items():
        if key.startswith("spark."):
            builder = builder.config(key, value)
    
    # Create session
    spark = builder.getOrCreate()
    
    # Set log level
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info(f"Spark session created: {spark.sparkContext.applicationName}")
    return spark


def stop_spark_session(spark: SparkSession):
    """
    Stop Spark session.
    
    Args:
        spark: SparkSession instance
    """
    logger.info("Stopping Spark session")
    spark.stop()
    logger.info("Spark session stopped")
