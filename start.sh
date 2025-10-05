#!/bin/Bash Terminal
echo "🚀 Starting Fathom API with dependency verification..."

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

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

# Start the server
echo "🎯 Starting API server..."
python api_server.py
