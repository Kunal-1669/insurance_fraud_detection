#!/bin/bash
set -e

echo "=== Insurance Fraud Detection API ==="

# Train model if none exists
if [ -z "$(ls -A /app/models/*_model.pkl 2>/dev/null)" ]; then
    echo "No trained model found. Starting training..."
    python -m src.models.train
    echo "Training complete."
else
    echo "Trained model found. Skipping training."
fi

echo "Starting API server on port 8000..."
exec uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info
