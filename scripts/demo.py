#!/usr/bin/env python3
"""
Demo script for Movie Review Sentiment Analysis.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.settings import Config
from models.pytorch_trainer import PyTorchTrainer
import click


@click.command()
@click.option("--model-path", default="models/pytorch_lstm_model.pth", help="Path to trained model")
@click.option("--model-type", default="lstm", help="Model type: lstm, transformer, bert")
@click.option("--interactive", is_flag=True, help="Run in interactive mode")
def main(model_path: str, model_type: str, interactive: bool):
    """Demo script for sentiment analysis."""
    
    print("🎬 Movie Review Sentiment Analysis Demo")
    print("=" * 50)
    
    try:
        # Load configuration
        config = Config()
        
        # Initialize trainer
        pytorch_trainer = PyTorchTrainer(config)
        
        # Load model
        print(f"📥 Loading {model_type} model from {model_path}...")
        pytorch_trainer.load_model(model_path, model_type)
        
        # Load tokenizer
        from models.pytorch_models import ModelFactory
        model_config = pytorch_trainer._get_model_config(model_type)
        pytorch_trainer.tokenizer = ModelFactory.get_tokenizer(model_type, model_config)
        
        print("✅ Model loaded successfully!")
        
        if interactive:
            print("\n🔄 Interactive mode - Enter 'quit' to exit")
            print("-" * 50)
            
            while True:
                text = input("\n📝 Enter your movie review: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not text:
                    print("❌ Please enter some text!")
                    continue
                
                # Make prediction
                predictions, probabilities = pytorch_trainer.predict([text])
                
                # Get results
                prediction = predictions[0]
                prob = probabilities[0]
                
                sentiment = "😊 Positive" if prediction == 1 else "😞 Negative"
                confidence = max(prob) * 100
                negative_prob = prob[0] * 100
                positive_prob = prob[1] * 100
                
                print(f"\n📊 Results:")
                print(f"   Sentiment: {sentiment}")
                print(f"   Confidence: {confidence:.1f}%")
                print(f"   Negative: {negative_prob:.1f}%")
                print(f"   Positive: {positive_prob:.1f}%")
        
        else:
            # Demo with sample texts
            sample_texts = [
                "This movie is absolutely amazing! I loved every minute of it.",
                "Terrible movie, waste of time. Don't watch it.",
                "The acting was okay but the plot was confusing.",
                "One of the best films I've ever seen. Highly recommended!",
                "Boring and predictable. Nothing new here."
            ]
            
            print(f"\n🧪 Testing with sample reviews:")
            print("-" * 50)
            
            for i, text in enumerate(sample_texts, 1):
                print(f"\n{i}. \"{text}\"")
                
                # Make prediction
                predictions, probabilities = pytorch_trainer.predict([text])
                
                # Get results
                prediction = predictions[0]
                prob = probabilities[0]
                
                sentiment = "😊 Positive" if prediction == 1 else "😞 Negative"
                confidence = max(prob) * 100
                negative_prob = prob[0] * 100
                positive_prob = prob[1] * 100
                
                print(f"   → {sentiment} ({confidence:.1f}% confidence)")
                print(f"   → Negative: {negative_prob:.1f}%, Positive: {positive_prob:.1f}%")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Make sure you have trained a model first:")
        print("   make train-lstm")


if __name__ == "__main__":
    main()
