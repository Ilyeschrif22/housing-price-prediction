#!/bin/bash
# Quick local run script
# Usage: bash deploy.sh

set -e

echo "Real Estate Price Prediction API - Local Setup"
echo "================================================"
echo ""

python3 --version

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Training the model..."
python3 train_model.py

echo ""
echo "Starting local server (Ctrl+C to stop)..."
echo "Test at: http://localhost:5000/health"
echo ""

python3 app.py

