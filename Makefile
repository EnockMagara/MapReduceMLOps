# Makefile for Movie Review Sentiment Analysis

.PHONY: help install setup train predict evaluate test clean docker-build docker-run

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install dependencies"
	@echo "  setup        - Setup project directories"
	@echo "  train        - Train the model"
	@echo "  predict      - Make predictions"
	@echo "  evaluate     - Evaluate model"
	@echo "  test         - Run tests"
	@echo "  clean        - Clean up generated files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"

# Install dependencies
install:
	python3 -m pip install -r requirements.txt

# Setup project directories
setup:
	mkdir -p data/raw data/processed logs models artifacts mlruns
	@echo "Project directories created"

# Train model
train:
	python3 scripts/train.py

# Train with cross-validation
train-cv:
	python3 scripts/train.py --cross-validate

# Train PyTorch model
train-pytorch:
	python3 scripts/train_pytorch.py

# Train PyTorch LSTM
train-lstm:
	python3 scripts/train_pytorch.py --model-type lstm

# Train PyTorch Transformer
train-transformer:
	python3 scripts/train_pytorch.py --model-type transformer

# Train PyTorch BERT
train-bert:
	python3 scripts/train_pytorch.py --model-type bert

# Train hybrid (Spark + PyTorch)
train-hybrid:
	python3 scripts/train_hybrid.py --compare

# Make predictions
predict:
	python3 scripts/predict.py --model-path models/naive_bayes_model --text "This movie is great!"

# Make PyTorch predictions
predict-pytorch:
	python3 scripts/predict_pytorch.py --model-path models/pytorch_lstm_model.pth --model-type lstm --text "This movie is great!"

# Evaluate model
evaluate:
	python3 scripts/evaluate.py --model-path models/naive_bayes_model

# Run tests
test:
	python3 -m pytest tests/ -v

# Run tests with coverage
test-coverage:
	python3 -m pytest tests/ --cov=src --cov-report=html

# Run web application
run-app:
	python3 app.py

# Run demo (command-line interface)
demo:
	python3 scripts/demo.py

# Run interactive demo
demo-interactive:
	python3 scripts/demo.py --interactive

# Clean up generated files
clean:
	rm -rf logs/* models/* artifacts/* mlruns/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build Docker image
docker-build:
	docker build -t movie-sentiment-analysis .

# Run Docker container
docker-run:
	docker-compose up

# Format code
format:
	black src/ scripts/ tests/
	flake8 src/ scripts/ tests/

# Type checking
type-check:
	mypy src/ scripts/
