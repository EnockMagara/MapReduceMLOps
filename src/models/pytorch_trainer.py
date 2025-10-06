"""PyTorch trainer for neural network models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import os

from .pytorch_models import ModelFactory, SentimentDataset

logger = logging.getLogger(__name__)


class PyTorchTrainer:
    """PyTorch trainer for sentiment analysis models."""
    
    def __init__(self, config):
        """Initialize PyTorch trainer."""
        self.config = config
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
    def _get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.config.pytorch.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Using CUDA device")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using MPS device")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device = torch.device(self.config.pytorch.device)
            logger.info(f"Using specified device: {device}")
        
        return device
    
    def prepare_data(self, texts, labels, model_type: str) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders for training and validation.
        
        Args:
            texts: List of text samples
            labels: List of labels
            model_type: Type of model being trained
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info("Preparing data loaders")
        
        # Get tokenizer
        self.tokenizer = ModelFactory.get_tokenizer(model_type, self._get_model_config(model_type))
        
        # Create dataset
        dataset = SentimentDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self._get_max_length(model_type)
        )
        
        # Split dataset
        val_size = int(len(dataset) * self.config.model.test_size)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.model.random_state)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.pytorch.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.pytorch.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
        return train_loader, val_loader
    
    def create_model(self, model_type: str, vocab_size: Optional[int] = None) -> nn.Module:
        """
        Create and initialize model.
        
        Args:
            model_type: Type of model to create
            vocab_size: Vocabulary size (for LSTM/Transformer)
            
        Returns:
            PyTorch model
        """
        logger.info(f"Creating {model_type} model")
        
        model_config = self._get_model_config(model_type)
        self.model = ModelFactory.create_model(model_type, model_config, vocab_size)
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.pytorch.learning_rate
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
        
        logger.info(f"Model created and moved to {self.device}")
        return self.model
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.model.__class__.__name__ == "BERTClassifier":
                outputs = self.model(input_ids, attention_mask)
            else:
                outputs = self.model(input_ids)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.pytorch.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.pytorch.gradient_clip_norm
                )
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if self.model.__class__.__name__ == "BERTClassifier":
                    outputs = self.model(input_ids, attention_mask)
                else:
                    outputs = self.model(input_ids)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Positive class probability
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
        except ValueError:
            auc = 0.0  # Handle case where only one class is present
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'val_auc': auc
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, model_type: str) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            model_type: Type of model being trained
            
        Returns:
            Training history and best metrics
        """
        logger.info("Starting PyTorch model training")
        
        best_val_loss = float('inf')
        best_metrics = {}
        patience_counter = 0
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': []
        }
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"pytorch_{model_type}_training"):
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("batch_size", self.config.pytorch.batch_size)
            mlflow.log_param("learning_rate", self.config.pytorch.learning_rate)
            mlflow.log_param("num_epochs", self.config.pytorch.num_epochs)
            mlflow.log_param("device", str(self.device))
            
            for epoch in range(self.config.pytorch.num_epochs):
                logger.info(f"Epoch {epoch + 1}/{self.config.pytorch.num_epochs}")
                
                # Train
                train_metrics = self.train_epoch(train_loader)
                
                # Validate
                val_metrics = self.validate_epoch(val_loader)
                
                # Update scheduler
                self.scheduler.step(val_metrics['val_loss'])
                
                # Log metrics
                for key, value in {**train_metrics, **val_metrics}.items():
                    mlflow.log_metric(key, value, step=epoch)
                    history[key].append(value)
                
                # Check for improvement
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    best_metrics = val_metrics.copy()
                    patience_counter = 0
                    
                    # Save best model
                    if self.config.mlops.log_models:
                        mlflow.pytorch.log_model(self.model, "model")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.config.pytorch.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                          f"Val Loss: {val_metrics['val_loss']:.4f}, "
                          f"Val Accuracy: {val_metrics['val_accuracy']:.4f}")
        
        logger.info("Training completed")
        logger.info(f"Best validation metrics: {best_metrics}")
        
        return {
            'history': history,
            'best_metrics': best_metrics
        }
    
    def save_model(self, model_path: str):
        """Save the trained model."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Get model type
        model_type = self.model.__class__.__name__.lower().replace('classifier', '')
        
        # Get model config
        model_config = self._get_model_config(model_type)
        
        # Add vocab_size for LSTM/Transformer models
        if model_type in ['lstm', 'transformer']:
            if hasattr(self.model, 'vocab_size'):
                model_config['vocab_size'] = self.model.vocab_size
            else:
                # Try to get vocab size from embedding layer
                if hasattr(self.model, 'embedding'):
                    model_config['vocab_size'] = self.model.embedding.num_embeddings
                else:
                    # Fallback: use the vocab size from the tokenizer
                    if hasattr(self.tokenizer, 'vocab_size'):
                        model_config['vocab_size'] = self.tokenizer.vocab_size
                    else:
                        # Default vocab size
                        model_config['vocab_size'] = 50000
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': model_config,
            'model_type': model_type,
            'tokenizer_name': self.tokenizer.name_or_path if hasattr(self.tokenizer, 'name_or_path') else 'custom'
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str, model_type: str):
        """Load a trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model type from checkpoint if available
        if 'model_type' in checkpoint:
            model_type = checkpoint['model_type']
        
        # Get model config
        model_config = checkpoint['model_config']
        
        # Create model
        if model_type in ['lstm', 'transformer']:
            vocab_size = model_config.get('vocab_size')
            if vocab_size is None:
                # Fallback vocab size if not found
                vocab_size = 50000
                logger.warning(f"vocab_size not found in model config, using default: {vocab_size}")
        else:
            vocab_size = None
            
        self.model = ModelFactory.create_model(model_type, model_config, vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, texts: list) -> Tuple[list, list]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of texts to predict
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self._get_max_length(self.model.__class__.__name__.lower().replace('classifier', '')),
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                if self.model.__class__.__name__ == "BERTClassifier":
                    outputs = self.model(input_ids, attention_mask)
                else:
                    outputs = self.model(input_ids)
                
                # Get prediction and probability
                prob = torch.softmax(outputs, dim=1)
                pred = torch.argmax(outputs, dim=1)
                
                predictions.append(pred.cpu().item())
                probabilities.append(prob.cpu().numpy()[0])
        
        return predictions, probabilities
    
    def _get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get model configuration based on type."""
        if model_type == "lstm":
            return self.config.neural_models.lstm.__dict__
        elif model_type == "transformer":
            return self.config.neural_models.transformer.__dict__
        elif model_type == "bert":
            return self.config.neural_models.bert.__dict__
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _get_max_length(self, model_type: str) -> int:
        """Get maximum sequence length for model type."""
        if model_type == "bert":
            return self.config.neural_models.bert.max_length
        elif model_type == "transformer":
            return self.config.neural_models.transformer.max_length
        else:
            return 256  # Default for LSTM
