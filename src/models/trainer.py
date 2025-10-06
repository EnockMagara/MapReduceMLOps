"""Model training utilities for movie review sentiment analysis."""

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.classification import NaiveBayes, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from typing import Dict, Any, Tuple
import logging
import mlflow
import mlflow.spark

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model trainer class for sentiment analysis."""
    
    def __init__(self, spark: SparkSession, config):
        """Initialize model trainer."""
        self.spark = spark
        self.config = config
        self.model = None
        self.evaluator = None
    
    def prepare_training_data(self, df: DataFrame) -> DataFrame:
        """
        Prepare training data for model training.
        
        Args:
            df: DataFrame with features and labels
            
        Returns:
            Prepared DataFrame for training
        """
        logger.info("Preparing training data")
        
        # Assemble features
        feature_cols = []
        if self.config.features.use_bow:
            feature_cols.append("bow_features")
        if self.config.features.use_tfidf:
            feature_cols.append("tfidf_features")
        
        if not feature_cols:
            raise ValueError("No features configured for training")
        
        # Create vector assembler
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features"
        )
        
        # Transform data
        df_prepared = assembler.transform(df)
        
        # Select only necessary columns
        df_prepared = df_prepared.select("features", "label")
        
        logger.info(f"Training data prepared with {df_prepared.count()} samples")
        return df_prepared
    
    def create_model(self, algorithm: str = None) -> Any:
        """
        Create model based on algorithm.
        
        Args:
            algorithm: Model algorithm to use
            
        Returns:
            Model instance
        """
        if algorithm is None:
            algorithm = self.config.model.algorithm
        
        logger.info(f"Creating {algorithm} model")
        
        if algorithm == "naive_bayes":
            model = NaiveBayes(
                featuresCol="features",
                labelCol="label",
                smoothing=1.0
            )
        elif algorithm == "logistic_regression":
            model = LogisticRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=100,
                regParam=0.01
            )
        elif algorithm == "random_forest":
            model = RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                numTrees=100,
                maxDepth=10
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        self.model = model
        return model
    
    def train_model(self, train_df: DataFrame, algorithm: str = None) -> Any:
        """
        Train the model.
        
        Args:
            train_df: Training DataFrame
            algorithm: Model algorithm to use
            
        Returns:
            Trained model
        """
        logger.info("Starting model training")
        
        # Create model
        model = self.create_model(algorithm)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{algorithm}_training"):
            # Log parameters
            mlflow.log_param("algorithm", algorithm)
            mlflow.log_param("train_samples", train_df.count())
            mlflow.log_param("features_used", self._get_feature_info())
            
            # Train model
            trained_model = model.fit(train_df)
            
            # Log model
            if self.config.mlops.log_models:
                mlflow.spark.log_model(trained_model, "model")
            
            logger.info("Model training completed")
            return trained_model
    
    def cross_validate(self, train_df: DataFrame, algorithm: str = None) -> Any:
        """
        Perform cross-validation for hyperparameter tuning.
        
        Args:
            train_df: Training DataFrame
            algorithm: Model algorithm to use
            
        Returns:
            Best model from cross-validation
        """
        logger.info("Starting cross-validation")
        
        # Create model
        model = self.create_model(algorithm)
        
        # Create evaluator
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        # Create parameter grid
        param_grid = self._create_param_grid(algorithm)
        
        # Create cross-validator
        cv = CrossValidator(
            estimator=model,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=self.config.model.cv_folds,
            seed=self.config.model.random_state
        )
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{algorithm}_cv"):
            # Log parameters
            mlflow.log_param("algorithm", algorithm)
            mlflow.log_param("cv_folds", self.config.model.cv_folds)
            mlflow.log_param("train_samples", train_df.count())
            
            # Fit cross-validator
            cv_model = cv.fit(train_df)
            
            # Get best model
            best_model = cv_model.bestModel
            
            # Log best parameters
            best_params = best_model.extractParamMap()
            for param, value in best_params.items():
                mlflow.log_param(f"best_{param.name}", value)
            
            # Log model
            if self.config.mlops.log_models:
                mlflow.spark.log_model(best_model, "model")
            
            logger.info("Cross-validation completed")
            return best_model
    
    def evaluate_model(self, model: Any, test_df: DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            test_df: Test DataFrame
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Create evaluators
        multiclass_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        
        # Calculate metrics
        metrics = {}
        
        for metric in self.config.evaluation.metrics:
            if metric == "accuracy":
                metrics[metric] = multiclass_evaluator.evaluate(predictions)
            elif metric == "precision":
                evaluator = MulticlassClassificationEvaluator(
                    labelCol="label",
                    predictionCol="prediction",
                    metricName="weightedPrecision"
                )
                metrics[metric] = evaluator.evaluate(predictions)
            elif metric == "recall":
                evaluator = MulticlassClassificationEvaluator(
                    labelCol="label",
                    predictionCol="prediction",
                    metricName="weightedRecall"
                )
                metrics[metric] = evaluator.evaluate(predictions)
            elif metric == "f1":
                evaluator = MulticlassClassificationEvaluator(
                    labelCol="label",
                    predictionCol="prediction",
                    metricName="f1"
                )
                metrics[metric] = evaluator.evaluate(predictions)
            elif metric == "auc":
                metrics[metric] = binary_evaluator.evaluate(predictions)
        
        # Log metrics to MLflow
        with mlflow.start_run():
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
        
        logger.info(f"Model evaluation completed. Metrics: {metrics}")
        return metrics
    
    def _create_param_grid(self, algorithm: str) -> list:
        """Create parameter grid for cross-validation."""
        if algorithm == "naive_bayes":
            return ParamGridBuilder() \
                .addGrid(self.model.smoothing, [0.5, 1.0, 2.0]) \
                .build()
        elif algorithm == "logistic_regression":
            return ParamGridBuilder() \
                .addGrid(self.model.regParam, [0.01, 0.1, 1.0]) \
                .addGrid(self.model.maxIter, [50, 100, 200]) \
                .build()
        elif algorithm == "random_forest":
            return ParamGridBuilder() \
                .addGrid(self.model.numTrees, [50, 100, 200]) \
                .addGrid(self.model.maxDepth, [5, 10, 15]) \
                .build()
        else:
            return ParamGridBuilder().build()
    
    def _get_feature_info(self) -> str:
        """Get feature information string."""
        features = []
        if self.config.features.use_bow:
            features.append("bow")
        if self.config.features.use_tfidf:
            features.append("tfidf")
        return "_".join(features)
