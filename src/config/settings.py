"""Configuration management for the Movie Review Sentiment Analysis project."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data configuration settings."""
    raw_path: str
    processed_path: str
    train_file: str
    test_file: str
    unsupervised_file: str


@dataclass
class SparkConfig:
    """Spark configuration settings."""
    app_name: str
    master: str
    driver_memory: str
    executor_memory: str
    max_result_size: str


@dataclass
class TextProcessingConfig:
    """Text processing configuration settings."""
    min_word_length: int
    max_word_length: int
    remove_stopwords: bool
    language: str
    max_vocab_size: int
    min_document_frequency: int
    max_document_frequency: float


@dataclass
class FeaturesConfig:
    """Feature engineering configuration settings."""
    use_tfidf: bool
    use_bow: bool
    ngram_range: list
    max_features: int


@dataclass
class ModelConfig:
    """Model configuration settings."""
    algorithm: str
    test_size: float
    random_state: int
    cv_folds: int


@dataclass
class PyTorchConfig:
    """PyTorch configuration settings."""
    device: str
    batch_size: int
    learning_rate: float
    num_epochs: int
    early_stopping_patience: int
    gradient_clip_norm: float


@dataclass
class LSTMConfig:
    """LSTM model configuration settings."""
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool


@dataclass
class TransformerConfig:
    """Transformer model configuration settings."""
    d_model: int
    nhead: int
    num_layers: int
    dropout: float
    max_length: int


@dataclass
class BERTConfig:
    """BERT model configuration settings."""
    model_name: str
    max_length: int
    freeze_bert: bool
    dropout: float


@dataclass
class NeuralModelsConfig:
    """Neural network models configuration settings."""
    lstm: LSTMConfig
    transformer: TransformerConfig
    bert: BERTConfig


@dataclass
class EvaluationConfig:
    """Evaluation configuration settings."""
    metrics: list


@dataclass
class MLOpsConfig:
    """MLOps configuration settings."""
    experiment_name: str
    tracking_uri: str
    log_artifacts: bool
    log_models: bool


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str
    format: str
    file: str


class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration from YAML file."""
        self.config_path = Path(config_path)
        self._load_config()
        self._setup_directories()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        # Create configuration objects
        self.data = DataConfig(**config_data['data'])
        self.spark = SparkConfig(**config_data['spark'])
        self.text_processing = TextProcessingConfig(**config_data['text_processing'])
        self.features = FeaturesConfig(**config_data['features'])
        self.model = ModelConfig(**config_data['model'])
        self.pytorch = PyTorchConfig(**config_data['pytorch'])
        self.neural_models = NeuralModelsConfig(
            lstm=LSTMConfig(**config_data['neural_models']['lstm']),
            transformer=TransformerConfig(**config_data['neural_models']['transformer']),
            bert=BERTConfig(**config_data['neural_models']['bert'])
        )
        self.evaluation = EvaluationConfig(**config_data['evaluation'])
        self.mlops = MLOpsConfig(**config_data['mlops'])
        self.logging = LoggingConfig(**config_data['logging'])
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.data.raw_path,
            self.data.processed_path,
            "logs",
            "models",
            "artifacts",
            "mlruns"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_spark_config(self) -> Dict[str, str]:
        """Get Spark configuration as dictionary."""
        return {
            "spark.app.name": self.spark.app_name,
            "spark.master": self.spark.master,
            "spark.driver.memory": self.spark.driver_memory,
            "spark.executor.memory": self.spark.executor_memory,
            "spark.driver.maxResultSize": self.spark.max_result_size,
        }
