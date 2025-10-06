#!/usr/bin/env python3
"""
Flask web application for Movie Review Sentiment Analysis.

"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from flask import Flask, render_template, request, jsonify
import torch
from config.settings import Config
from models.pytorch_trainer import PyTorchTrainer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model
config = None
pytorch_trainer = None
model_loaded = False

def load_model():
    """Load the trained PyTorch model."""
    global config, pytorch_trainer, model_loaded
    
    try:
        # Load configuration
        config = Config()
        
        # Initialize trainer
        pytorch_trainer = PyTorchTrainer(config)
        
        # Load the trained LSTM model
        model_path = "models/pytorch_lstm_model.pth"
        pytorch_trainer.load_model(model_path, "lstm")
        
        # Load tokenizer
        from models.pytorch_models import ModelFactory
        model_config = pytorch_trainer._get_model_config("lstm")
        pytorch_trainer.tokenizer = ModelFactory.get_tokenizer("lstm", model_config)
        
        model_loaded = True
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model_loaded = False

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for given text."""
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please train a model first.',
            'success': False
        })
    
    try:
        # Get text from request
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'error': 'Please provide some text to analyze.',
                'success': False
            })
        
        # Make prediction
        predictions, probabilities = pytorch_trainer.predict([text])
        
        # Get prediction result
        prediction = predictions[0]
        prob = probabilities[0]
        
        # Convert to human-readable format
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = float(max(prob)) * 100
        
        # Get detailed probabilities (convert to Python floats)
        negative_prob = float(prob[0]) * 100
        positive_prob = float(prob[1]) * 100
        
        return jsonify({
            'success': True,
            'text': text,
            'sentiment': sentiment,
            'confidence': round(confidence, 2),
            'probabilities': {
                'negative': round(negative_prob, 2),
                'positive': round(positive_prob, 2)
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        })

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5005)
