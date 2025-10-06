"""MLflow utilities for the Movie Review Sentiment Analysis project."""

import mlflow
import mlflow.spark
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def setup_mlflow(config: Dict[str, Any]):
    """
    Set up MLflow tracking.
    
    Args:
        config: MLOps configuration dictionary
    """
    logger.info("Setting up MLflow tracking")
    
    # Set tracking URI
    mlflow.set_tracking_uri(config["tracking_uri"])
    
    # Set experiment
    experiment_name = config["experiment_name"]
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name}")
        else:
            logger.info(f"Using existing experiment: {experiment_name}")
    except Exception as e:
        logger.warning(f"Could not set up experiment: {e}")
    
    mlflow.set_experiment(experiment_name)


def log_model_metrics(metrics: Dict[str, float], model_name: str = "model"):
    """
    Log model metrics to MLflow.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Model name for logging
    """
    logger.info(f"Logging metrics for {model_name}")
    
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value)
        logger.info(f"Logged {metric_name}: {value}")


def log_model_parameters(params: Dict[str, Any], model_name: str = "model"):
    """
    Log model parameters to MLflow.
    
    Args:
        params: Dictionary of parameters
        model_name: Model name for logging
    """
    logger.info(f"Logging parameters for {model_name}")
    
    for param_name, value in params.items():
        mlflow.log_param(param_name, value)
        logger.info(f"Logged {param_name}: {value}")


def log_artifacts(artifacts: Dict[str, str], model_name: str = "model"):
    """
    Log artifacts to MLflow.
    
    Args:
        artifacts: Dictionary of artifact paths and names
        model_name: Model name for logging
    """
    logger.info(f"Logging artifacts for {model_name}")
    
    for artifact_name, artifact_path in artifacts.items():
        mlflow.log_artifact(artifact_path, artifact_name)
        logger.info(f"Logged artifact {artifact_name}: {artifact_path}")
