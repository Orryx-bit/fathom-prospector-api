#!/bin/bash
set -e

echo "🚀 Starting Fathom API with dependency verification..."

# CRITICAL: Explicitly define and export the path for Playwright browsers.
# This ensures both the 'install' command and the running Python app use the same directory.
export PLAYWRIGHT_BROWSERS_PATH="/opt/playwright"
echo "✅ Browser cache path set to: $PLAYWRIGHT_BROWSERS_PATH"

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

# Install Playwright browsers to the defined path
echo "🎭 Installing Playwright browsers..."
python -m playwright install --with-deps chromium

# Verify critical packages
echo "🔍 Verifying critical packages..."
python -c "import pandas; print(f'✅ pandas {pandas.__version__} installed')"
python -c "import playwright; from playwright.sync_api import sync_playwright; print('✅ Playwright installed')"

# Start the server with uvicorn (required by Railway)
echo "🎯 Starting API server on port $PORT..."
cd /app || cd "$(pwd)"
exec python -m uvicorn api_server:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 300
