#!/bin/bash
git clone https://github.com/OpenTalker/SadTalker.git

cd SadTalker

set -euo pipefail
# --- Conda を有効化 ---
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n sadtalker python=3.11

conda activate sadtalker

#pip install torch==2.3.1+cpu torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
#pip install torch torchvision
conda install -y ffmpeg
pip3 install -U --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

pip install -r ../requirements-sadtalker.txt
conda install -y scipy
pip install mat73 h5py --upgrade
#pip install dlib==19.24.0 
pip install dlib
conda install -y opencv
pip install python-osc
conda install -y pygame

### Coqui TTS is optional for gradio demo.
### pip install TTS
