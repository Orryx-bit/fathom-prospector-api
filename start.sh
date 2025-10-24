#!/bin/bash
set -e

echo "🚀 Starting Fathom API with dependency verification..."

echo "Checking Python version..."
python --version

# CRITICAL: Set PYTHONPATH to include current directory for scoring_system module
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "✅ PYTHONPATH set to: $PYTHONPATH"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Verify critical packages
echo "🔍 Verifying installations..."
python -c "import pandas; print(f'✅ pandas {pandas.__version__} installed')"
python -c "import scrapingbee; print(f'✅ ScrapingBee installed')"
echo "🐝 ScrapingBee will handle JavaScript rendering"

echo "🔍 Verifying aiohttp installation..."
python -c "import aiohttp; print(f'✅ aiohttp {aiohttp.__version__} installed - concurrent scraping enabled')"

echo "🔍 Verifying google.generativeai installation..."
python -c "import google.generativeai; print('✅ google.generativeai installed')"

echo "🔍 Verifying requests installation..."
python -c "import requests; print(f'✅ requests installed')"

echo "🔍 Verifying scoring_system module..."
python -c "import scoring_system; print('✅ scoring_system module found!')"

# Start the server with uvicorn (required by Railway)
echo "🎯 Starting API server on port $PORT..."
cd /app || cd "$(pwd)"
exec python -m uvicorn api_server:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 300
