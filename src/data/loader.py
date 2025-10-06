"""Data loading utilities for movie review sentiment analysis."""

# Import pandas for possible local data manipulation (not used in this file, but may be used elsewhere)
import pandas as pd

# Import SparkSession and DataFrame for Spark operations
from pyspark.sql import SparkSession, DataFrame

# Import Spark SQL functions for column operations and text cleaning
from pyspark.sql.functions import col, regexp_replace, lower, trim

# Import Spark SQL types for explicit type casting
from pyspark.sql.types import StringType, IntegerType

# Import typing for type hints (Tuple, Optional not used here, but may be for future extensibility)
from typing import Tuple, Optional

# Import logging to enable logging of data loading and processing steps
import logging

# Set up a logger for this module to track events and errors
logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader class for handling movie review datasets."""

    def __init__(self, spark: SparkSession):
        """
        Initialize data loader with Spark session.

        Args:
            spark (SparkSession): The Spark session to use for data operations.
        """
        self.spark = spark  # Store the Spark session for use in all data loading methods

    def load_csv(self, file_path: str, header: bool = True) -> DataFrame:
        """
        Load a CSV file into a Spark DataFrame.

        Args:
            file_path (str): Path to the CSV file.
            header (bool): Whether the CSV file has a header row.

        Returns:
            DataFrame: Spark DataFrame containing the loaded data.
        """
        try:
            # Read the CSV file using Spark, with or without header as specified
            df = self.spark.read.option("header", header).csv(file_path)
            # Log the successful load and the number of rows loaded
            logger.info(f"Successfully loaded {file_path} with {df.count()} rows")
            return df
        except Exception as e:
            # Log any errors encountered during loading and re-raise the exception
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    def load_train_data(self, file_path: str) -> DataFrame:
        """
        Load and preprocess training data from a CSV file.

        Args:
            file_path (str): Path to the training data CSV.

        Returns:
            DataFrame: Cleaned and preprocessed Spark DataFrame for training.
        """
        # Load the raw CSV data
        df = self.load_csv(file_path)

        # Select and cast the relevant columns to ensure correct types
        df = df.select(
            col("text").cast(StringType()).alias("text"),   # Ensure 'text' is string
            col("label").cast(IntegerType()).alias("label") # Ensure 'label' is integer
        )

        # Clean the text data step by step:
        # 1. Remove leading/trailing whitespace
        df = df.withColumn("text", trim(col("text")))
        # 2. Replace HTML line breaks with spaces
        df = df.withColumn("text", regexp_replace(col("text"), r'<br\s*/?>', ' '))
        # 3. Remove all non-word and non-space characters (punctuation, etc.)
        df = df.withColumn("text", regexp_replace(col("text"), r'[^\w\s]', ' '))
        # 4. Replace multiple spaces with a single space
        df = df.withColumn("text", regexp_replace(col("text"), r'\s+', ' '))

        # Filter out rows where text is null or empty after cleaning
        df = df.filter(col("text").isNotNull() & (col("text") != ""))

        # Log the number of rows after processing
        logger.info(f"Processed training data: {df.count()} rows")
        return df

    def load_test_data(self, file_path: str) -> DataFrame:
        """
        Load and preprocess test data from a CSV file.

        Args:
            file_path (str): Path to the test data CSV.

        Returns:
            DataFrame: Cleaned and preprocessed Spark DataFrame for testing.
        """
        # Load the raw CSV data
        df = self.load_csv(file_path)

        # Select and cast the relevant columns to ensure correct types
        df = df.select(
            col("text").cast(StringType()).alias("text"),   # Ensure 'text' is string
            col("label").cast(IntegerType()).alias("label") # Ensure 'label' is integer
        )

        # Clean the text data using the same steps as for training data
        df = df.withColumn("text", trim(col("text")))
        df = df.withColumn("text", regexp_replace(col("text"), r'<br\s*/?>', ' '))
        df = df.withColumn("text", regexp_replace(col("text"), r'[^\w\s]', ' '))
        df = df.withColumn("text", regexp_replace(col("text"), r'\s+', ' '))

        # Filter out rows where text is null or empty after cleaning
        df = df.filter(col("text").isNotNull() & (col("text") != ""))

        # Log the number of rows after processing
        logger.info(f"Processed test data: {df.count()} rows")
        return df

    def load_unsupervised_data(self, file_path: str) -> DataFrame:
        """
        Load and preprocess unsupervised data for additional training.

        Args:
            file_path (str): Path to the unsupervised data CSV.

        Returns:
            DataFrame: Cleaned and preprocessed Spark DataFrame for unsupervised learning.
        """
        # Load the raw CSV data
        df = self.load_csv(file_path)

        # Select and cast the relevant columns to ensure correct types
        df = df.select(
            col("text").cast(StringType()).alias("text"),   # Ensure 'text' is string
            col("label").cast(IntegerType()).alias("label") # Ensure 'label' is integer
        )

        # Clean the text data using the same steps as for other datasets
        df = df.withColumn("text", trim(col("text")))
        df = df.withColumn("text", regexp_replace(col("text"), r'<br\s*/?>', ' '))
        df = df.withColumn("text", regexp_replace(col("text"), r'[^\w\s]', ' '))
        df = df.withColumn("text", regexp_replace(col("text"), r'\s+', ' '))

        # Filter out rows where text is null or empty after cleaning
        df = df.filter(col("text").isNotNull() & (col("text") != ""))

        # Log the number of rows after processing
        logger.info(f"Processed unsupervised data: {df.count()} rows")
        return df

    def get_data_summary(self, df: DataFrame) -> dict:
        """
        Get summary statistics of the dataset.

        Args:
            df (DataFrame): The Spark DataFrame to summarize.

        Returns:
            dict: Dictionary containing total row count and label distribution.
        """
        # Count the total number of rows in the DataFrame
        total_rows = df.count()

        # Group by 'label' and count occurrences for each label value
        label_counts = df.groupBy("label").count().collect()

        # Build a summary dictionary with total rows and label distribution
        summary = {
            "total_rows": total_rows,
            # Create a dictionary mapping label values to their counts
            "label_distribution": {row["label"]: row["count"] for row in label_counts}
        }

        # Return the summary statistics
        return summary
