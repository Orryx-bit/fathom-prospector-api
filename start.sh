#!/bin/Bash Terminal
set -e

echo "Checking Python version..."
python --version

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Verifying pandas installation..."
python -c "import pandas; print(f'Pandas {pandas.__version__} installed successfully')"

echo "Starting server on port $PORT..."
exec uvicorn api_server:app --host 0.0.0.0 --port $PORT
