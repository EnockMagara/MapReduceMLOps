"""PyTorch data loading utilities for movie review sentiment analysis."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PyTorchDataLoader:
    """Data loader for PyTorch models."""
    
    def __init__(self):
        """Initialize PyTorch data loader."""
        pass
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV file into pandas DataFrame.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            pandas DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {file_path} with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def load_train_data(self, file_path: str) -> Tuple[List[str], List[int]]:
        """
        Load training data.
        
        Args:
            file_path: Path to training CSV file
            
        Returns:
            Tuple of (texts, labels)
        """
        df = self.load_csv(file_path)
        
        # Clean and prepare the data
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
        
        # Basic text cleaning
        texts = [self._clean_text(text) for text in texts]
        
        # Remove empty texts
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        texts = [texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        logger.info(f"Processed training data: {len(texts)} samples")
        return texts, labels
    
    def load_test_data(self, file_path: str) -> Tuple[List[str], List[int]]:
        """
        Load test data.
        
        Args:
            file_path: Path to test CSV file
            
        Returns:
            Tuple of (texts, labels)
        """
        df = self.load_csv(file_path)
        
        # Clean and prepare the data
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
        
        # Basic text cleaning
        texts = [self._clean_text(text) for text in texts]
        
        # Remove empty texts
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        texts = [texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        logger.info(f"Processed test data: {len(texts)} samples")
        return texts, labels
    
    def load_unsupervised_data(self, file_path: str) -> Tuple[List[str], List[int]]:
        """
        Load unsupervised data.
        
        Args:
            file_path: Path to unsupervised CSV file
            
        Returns:
            Tuple of (texts, labels)
        """
        df = self.load_csv(file_path)
        
        # Clean and prepare the data
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
        
        # Basic text cleaning
        texts = [self._clean_text(text) for text in texts]
        
        # Remove empty texts
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        texts = [texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        logger.info(f"Processed unsupervised data: {len(texts)} samples")
        return texts, labels
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text data.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove HTML tags
        import re
        text = re.sub(r'<br\s*/?>', ' ', text)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def get_data_summary(self, texts: List[str], labels: List[int]) -> dict:
        """
        Get summary statistics of the dataset.
        
        Args:
            texts: List of texts
            labels: List of labels
            
        Returns:
            Dictionary with summary statistics
        """
        total_samples = len(texts)
        label_counts = {}
        
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Calculate text length statistics
        text_lengths = [len(text.split()) for text in texts]
        
        summary = {
            "total_samples": total_samples,
            "label_distribution": label_counts,
            "avg_text_length": np.mean(text_lengths),
            "max_text_length": np.max(text_lengths),
            "min_text_length": np.min(text_lengths)
        }
        
        return summary
    
    def create_vocabulary(self, texts: List[str], min_freq: int = 2) -> dict:
        """
        Create vocabulary from texts.
        
        Args:
            texts: List of texts
            min_freq: Minimum frequency for word inclusion
            
        Returns:
            Dictionary with word to index mapping
        """
        from collections import Counter
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Filter by minimum frequency
        filtered_words = {word: count for word, count in word_counts.items() 
                         if count >= min_freq}
        
        # Create vocabulary
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }
        
        for word in sorted(filtered_words.keys()):
            vocab[word] = len(vocab)
        
        logger.info(f"Created vocabulary with {len(vocab)} words")
        return vocab
