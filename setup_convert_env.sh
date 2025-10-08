#!/bin/bash
# setup script for mano conversion environment (python 3.10 + chumpy)
# purpose: convert legacy mano .pkl files with chumpy objects to pure numpy
# usage: ./setup_convert_env.sh (run once, then use to convert mano models)

set -e

echo "setting up mano conversion environment..."

# create python 3.10 virtual environment
python3.10 -m venv mano_convert_env

# activate environment
source mano_convert_env/bin/activate

# install dependencies from requirements file
echo "installing conversion dependencies..."
pip install --upgrade pip
pip install -r requirements_convert.txt

echo "conversion environment ready"
echo ""
echo "to activate:"
echo "  source mano_convert_env/bin/activate"
echo ""
echo "then convert mano models:"
echo "  python src/convert_mano_to_numpy.py"
