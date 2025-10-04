# This script sets up the Python virtual environment for the web UI.

# 1. Create the virtual environment
python -m venv webui_venv

# 2. Upgrade pip
./webui_venv/Scripts/python.exe -m pip install --upgrade pip

# 3. Install dependencies
./webui_venv/Scripts/python.exe -m pip install gradio requests

Write-Host "Web UI venv setup complete. Please activate the environment using '.\webui_venv\Scripts\activate'."
