#!/bin/bash

# AI Voice Detection API - Run Script

echo "=============================================="
echo "🎙️  AI Voice Detection API"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt --quiet

# Create models cache directory
mkdir -p models_cache

# Run the server
echo ""
echo "=============================================="
echo "🚀 Starting API Server..."
echo "=============================================="
echo "📍 URL: http://localhost:8000"
echo "📚 Docs: http://localhost:8000/docs"
echo "=============================================="
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
