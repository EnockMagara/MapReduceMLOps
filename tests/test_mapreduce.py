"""Tests for MapReduce processor module."""

import pytest
from pyspark.sql import SparkSession
from src.processing.mapreduce import MapReduceProcessor
from src.config.settings import Config


@pytest.fixture
def spark():
    """Create Spark session for testing."""
    spark = SparkSession.builder.appName("test").master("local[*]").getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def config():
    """Create test configuration."""
    return Config("config/config.yaml")


@pytest.fixture
def mapreduce_processor(spark, config):
    """Create MapReduce processor instance."""
    return MapReduceProcessor(spark, config)


def test_mapreduce_processor_initialization(mapreduce_processor):
    """Test MapReduce processor initialization."""
    assert mapreduce_processor is not None
    assert mapreduce_processor.spark is not None
    assert mapreduce_processor.config is not None


def test_map_phase(mapreduce_processor, spark):
    """Test map phase functionality."""
    # Create test DataFrame
    from pyspark.sql import Row
    test_data = [Row(text="hello world", label=1), Row(text="good movie", label=0)]
    df = spark.createDataFrame(test_data)
    
    # Run map phase
    result = mapreduce_processor.map_phase(df)
    
    # Assertions
    assert result.count() > 0
    assert "word" in result.columns
    assert "count" in result.columns
