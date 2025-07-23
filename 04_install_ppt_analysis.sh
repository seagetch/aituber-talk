#!/bin/bash
set -euo pipefail
# --- Conda を有効化 ---
source "$(conda info --base)/etc/profile.d/conda.sh"

conda env create -f environment-ppt_analysis.yaml

# 作成した env をアクティベート
conda activate slide-analysis-agent