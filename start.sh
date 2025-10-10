#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v python >/dev/null 2>&1; then
    echo "❌ Python is not installed or not on PATH" >&2
    exit 1
fi

if ! command -v pip >/dev/null 2>&1; then
    echo "❌ pip is not installed or not on PATH" >&2
    exit 1
fi

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

# Upgrade pip (can be skipped by exporting SKIP_PIP_UPGRADE=1)
if [[ "${SKIP_PIP_UPGRADE:-0}" != "1" ]]; then
    echo "📦 Upgrading pip..."
    python -m pip install --upgrade pip
else
    echo "⏭️ Skipping pip upgrade because SKIP_PIP_UPGRADE=${SKIP_PIP_UPGRADE:-0}"
fi

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Verify critical packages
echo "🔍 Verifying pandas installation..."
python -c "import pandas; print(f'✅ pandas {pandas.__version__} installed')"

echo "🔍 Verifying google.generativeai installation..."
python -c "import google.generativeai; print('✅ google.generativeai installed')"

echo "🔍 Verifying requests installation..."
python -c "import requests; print(f'✅ requests installed')"

echo "🔍 Verifying scoring_system module..."
python -c "import scoring_system; print('✅ scoring_system module found!')"

# Start the server with uvicorn (required by Railway)
PORT="${PORT:-8000}"
echo "🎯 Starting API server on port $PORT..."
cd /app || cd "$(pwd)"
exec python -m uvicorn api_server:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 300
