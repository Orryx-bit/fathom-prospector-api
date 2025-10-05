#!/bin/Bash Terminal
set -e

echo "Checking Python version..."
python --version

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Verifying pandas installation..."
python -c "import pandas; print(f'Pandas {pandas.__version__} installed successfully')"

echo "Starting server..."
uvicorn api_server:app --host 0.0.0.0 --port $PORT
