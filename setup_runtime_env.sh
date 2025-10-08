#!/bin/bash
# setup script for project asha runtime environment (python 3.11 + mps)
# purpose: main environment for running the application
# usage: ./setup_runtime_env.sh (run once, then use for all runtime tasks)

set -e

echo "setting up project asha runtime environment..."

# check for python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo "python 3.11 not found. install it first:"
    echo "   brew install python@3.11"
    exit 1
fi

# create python 3.11 virtual environment
python3.11 -m venv asha_env

# activate environment
source asha_env/bin/activate

# upgrade pip
pip install --upgrade pip

echo ""
echo "installing pytorch with mps support for apple silicon..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "installing core dependencies..."
pip install -r requirements.txt

echo ""
echo "verifying mps support..."
python -c "import torch; print('mps available:', torch.backends.mps.is_available())"

echo ""
echo "runtime environment ready"
echo ""
echo "to activate:"
echo "  source asha_env/bin/activate"
echo ""
echo "then run:"
echo "  python src/mano_v1.py        # full ik + lbs"
echo "  python src/mediapipe_v0.py   # mediapipe baseline"
