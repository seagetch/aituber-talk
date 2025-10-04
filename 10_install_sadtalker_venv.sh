#!/bin/bash
set -euo pipefail

# This script should be run from the project root.

# Clone SadTalker repository if it doesn't exist
if [ ! -d "SadTalker" ]; then
    echo "--- 'SadTalker' directory not found. Cloning repository... ---"
    git clone https://github.com/OpenTalker/SadTalker.git
else
    echo "--- 'SadTalker' directory already exists. Skipping clone. ---"
fi

# Create a virtual environment in the project root if it doesn't exist
if [ ! -d "sadtalker_venv" ]; then
    echo "--- 'sadtalker_venv' directory not found. Creating virtual environment... ---"
    python3 -m venv sadtalker_venv
    if [ $? -ne 0 ]; then
        echo "--- ERROR: Failed to create virtual environment. ---"
        echo "--- Please ensure 'python3' and the 'venv' module are installed correctly. ---"
        echo "--- On Debian/Ubuntu, you might need to run: sudo apt install python3-venv ---"
        exit 1
    fi
    echo "--- Virtual environment created. ---"
else
    echo "--- 'sadtalker_venv' directory already exists. Skipping creation. ---"
fi

echo "--- Activating virtual environment... ---"
# Activate the virtual environment
source sadtalker_venv/bin/activate
if [ $? -ne 0 ]; then
    echo "--- ERROR: Failed to activate virtual environment. ---"
    echo "--- File 'sadtalker_venv/bin/activate' not found. ---"
    exit 1
fi
echo "--- Virtual environment activated. ---"


# Upgrade pip
echo "--- Upgrading pip... ---"
pip install --upgrade pip

# Install dependencies
echo "--- Installing dependencies... ---"
pip3 install -U --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu129
pip install -r requirements-sadtalker.txt
pip install scipy
pip install mat73 h5py --upgrade
pip install dlib
pip install opencv-python
pip install python-osc
pip install pygame

echo "------------------------------------------------"
echo "The installation is complete."
echo "The 'sadtalker_venv' virtual environment is ready to use in the project root."
echo "Please install ffmpeg using your system's package manager if you haven't already."
echo "For example:"
echo "sudo apt update && sudo apt install ffmpeg"
echo "sudo pacman -S ffmpeg"
echo "------------------------------------------------"
