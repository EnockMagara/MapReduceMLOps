"""Tests for data loader module."""

import pytest
from pyspark.sql import SparkSession
from src.data.loader import DataLoader


@pytest.fixture
def spark():
    """Create Spark session for testing."""
    spark = SparkSession.builder.appName("test").master("local[*]").getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def data_loader(spark):
    """Create data loader instance."""
    return DataLoader(spark)


def test_data_loader_initialization(data_loader):
    """Test data loader initialization."""
    assert data_loader is not None
    assert data_loader.spark is not None


def test_load_csv(data_loader, tmp_path):
    """Test CSV loading functionality."""
    # Create test CSV file
    test_file = tmp_path / "test.csv"
    test_file.write_text("text,label\nhello world,1\ngood movie,0")
    
    # Load CSV
    df = data_loader.load_csv(str(test_file))
    
    # Assertions
    assert df.count() == 2
    assert "text" in df.columns
    assert "label" in df.columns
