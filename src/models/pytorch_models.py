"""PyTorch neural network models for movie review sentiment analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LSTMClassifier(nn.Module):
    """LSTM-based sentiment classifier."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_layers: int, dropout: float, bidirectional: bool, num_classes: int = 2):
        super(LSTMClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classification layers
        self.dropout_layer = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Classification
        output = self.dropout_layer(hidden)
        output = F.relu(self.fc1(output))
        output = self.dropout_layer(output)
        output = self.fc2(output)
        
        return output


class TransformerClassifier(nn.Module):
    """Transformer-based sentiment classifier."""
    
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, 
                 dropout: float, max_length: int, num_classes: int = 2):
        super(TransformerClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_length = max_length
        self.num_classes = num_classes
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(max_length, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        seq_length = x.size(1)
        
        # Embedding
        embedded = self.embedding(x) * (self.d_model ** 0.5)  # Scale embeddings
        
        # Add positional encoding
        embedded += self.pos_encoding[:seq_length, :].unsqueeze(0)
        
        # Create padding mask
        padding_mask = (x == 0)
        
        # Transformer forward pass
        transformer_out = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # Global average pooling
        mask = (~padding_mask).float().unsqueeze(-1)
        pooled = (transformer_out * mask).sum(dim=1) / mask.sum(dim=1)
        
        # Classification
        output = self.dropout_layer(pooled)
        output = self.fc(output)
        
        return output


class BERTClassifier(nn.Module):
    """BERT-based sentiment classifier."""
    
    def __init__(self, model_name: str, num_classes: int = 2, dropout: float = 0.1, freeze_bert: bool = False):
        super(BERTClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout = dropout
        self.freeze_bert = freeze_bert
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Get BERT hidden size
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Classification head
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert_hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Classification
        output = self.dropout_layer(pooled_output)
        output = self.classifier(output)
        
        return output


class SentimentDataset(torch.utils.data.Dataset):
    """PyTorch dataset for sentiment analysis."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ModelFactory:
    """Factory class for creating PyTorch models."""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any], vocab_size: Optional[int] = None) -> nn.Module:
        """
        Create a PyTorch model based on configuration.
        
        Args:
            model_type: Type of model to create
            config: Model configuration
            vocab_size: Vocabulary size (required for LSTM and Transformer)
            
        Returns:
            PyTorch model instance
        """
        logger.info(f"Creating {model_type} model")
        
        if model_type == "lstm":
            if vocab_size is None:
                raise ValueError("vocab_size is required for LSTM model")
            
            return LSTMClassifier(
                vocab_size=vocab_size,
                embedding_dim=config.get('embedding_dim', 128),
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                bidirectional=config['bidirectional'],
                num_classes=2
            )
        
        elif model_type == "transformer":
            if vocab_size is None:
                raise ValueError("vocab_size is required for Transformer model")
            
            return TransformerClassifier(
                vocab_size=vocab_size,
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                max_length=config['max_length'],
                num_classes=2
            )
        
        elif model_type == "bert":
            return BERTClassifier(
                model_name=config['model_name'],
                num_classes=2,
                dropout=config['dropout'],
                freeze_bert=config['freeze_bert']
            )
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def get_tokenizer(model_type: str, config: Dict[str, Any]):
        """
        Get appropriate tokenizer for the model.
        
        Args:
            model_type: Type of model
            config: Model configuration
            
        Returns:
            Tokenizer instance
        """
        if model_type == "bert":
            return BertTokenizer.from_pretrained(config['model_name'])
        else:
            # For LSTM and Transformer, we'll use a simple tokenizer
            # In practice, you might want to use a more sophisticated tokenizer
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained("bert-base-uncased")
