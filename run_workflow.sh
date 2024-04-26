#!/bin/bash
# This script automates the processing, training, and testing of the domino recognition model.

echo "Processing images..."
python -m scripts.process_images data/raw data/processed
python -m scripts.process_images data/validation data/processed/validation

echo "Training model..."
python -m model.train

echo "Testing model..."
python -m tests.test_model

echo "Validating..."
python -m tests.test_validation
