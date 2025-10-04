# This script sets up the Python virtual environment for SadTalker on Windows.

# 1. Clone the SadTalker repository if it doesn't exist
if (-not (Test-Path -Path "SadTalker")) {
    git clone https://github.com/OpenTalker/SadTalker.git
}

# 2. Create the virtual environment
python -m venv sadtalker_venv

# 3. Upgrade pip
./sadtalker_venv/Scripts/python.exe -m pip install --upgrade pip

# 4. Install PyTorch with CUDA support (adjust the CUDA version if needed)
# For example, use 'cu117' for CUDA 11.7, or 'cpu' for CPU-only version.
./sadtalker_venv/Scripts/python.exe -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# 5. Install other dependencies
./sadtalker_venv/Scripts/python.exe -m pip install -r requirements-sadtalker.txt
./sadtalker_venv/Scripts/python.exe -m pip install scipy mat73 h5py dlib opencv-python python-osc pygame

Write-Host "SadTalker venv setup complete. Please activate the environment using '.\sadtalker_venv\Scripts\activate' and then run the model download script."