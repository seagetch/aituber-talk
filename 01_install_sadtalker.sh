#!/bin/bash
git clone https://github.com/OpenTalker/SadTalker.git

cd SadTalker

set -euo pipefail
# --- Conda を有効化 ---
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n sadtalker python=3.8

conda activate sadtalker

pip install torch==2.3.1+cpu torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

conda install ffmpeg

pip install -r requirements.txt
pip install mat73 h5py --upgrade
pip install dlib==19.24.0 
conda install opencv

### Coqui TTS is optional for gradio demo.
### pip install TTS
