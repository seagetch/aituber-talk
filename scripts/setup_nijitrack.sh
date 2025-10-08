#!/bin/bash
# This script installs the nijitrack dependency.

set -e

# --- Helper functions ---

# Function to print messages
info() {
    echo "[INFO] $1"
}

# --- Main script ---

# Check if nijitrack directory exists
if [ -d "nijitrack" ]; then
    info "nijitrack directory already exists. Skipping clone."
else
    info "Cloning nijitrack repository..."
    git clone https://github.com/nijigenerate/nijitrack.git
fi

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo "[ERROR] Virtual environment .venv not found. Please run setup_env.sh first."
    exit 1
fi

# Activate virtual environment and install
info "Activating virtual environment..."
# shellcheck source=./.venv/bin/activate
source ./.venv/bin/activate

info "Installing nijitrack..."
pip install ./nijitrack

info "nijitrack installation complete."
