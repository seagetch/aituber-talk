#!/bin/bash
set -euo pipefail
# --- Conda を有効化 ---
source "$(conda info --base)/etc/profile.d/conda.sh"

conda env create -f environment-webui.yml

# 作成した env をアクティベート
conda activate aituber-talker-webui