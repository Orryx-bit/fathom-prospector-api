#!/bin/bash terminal
set -e

echo "🚀 Starting Fathom API with dependency verification..."

echo "Checking Python version..."
python --version

# CRITICAL: Set PYTHONPATH to include current directory for scoring_system module
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "✅ PYTHONPATH set to: $PYTHONPATH"

# Verify critical packages (quick checks - no reinstall)
echo "🔍 Verifying installations..."
python -c "import pandas; print(f'✅ pandas {pandas.__version__} installed')"
python -c "import aiohttp; print(f'✅ aiohttp {aiohttp.__version__} installed - concurrent scraping enabled')"
python -c "import scrapingbee; print(f'✅ ScrapingBee installed')"
python -c "import google.generativeai; print('✅ google.generativeai installed')"
python -c "import requests; print(f'✅ requests installed')"
python -c "import scoring_system; print('✅ scoring_system module found!')"

# Start the server with uvicorn (required by Railway)
echo "🎯 Starting API server on port $PORT..."
cd /app || cd "$(pwd)"
exec python -m uvicorn api_server:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 300
