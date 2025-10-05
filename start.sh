#!/bin/Bash Terminal
set -e

echo "Checking Python version..."
python --version

# Set PYTHONPATH to include current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "PYTHONPATH set to: $PYTHONPATH"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Verifying installations..."
python -c "import pandas; print(f'Pandas {pandas.__version__} installed successfully')"
python -c "import scoring_system; print('scoring_system module found!')"

echo "Starting server on port $PORT..."
cd /app || cd "$(pwd)"
exec python -m uvicorn api_server:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 300
