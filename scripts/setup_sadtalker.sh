#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/OpenTalker/SadTalker.git}"
DESTINATION="${1:-SadTalker}"
FORCE="${FORCE:-false}"

log() {
  echo "[sadtalker] $1"
}

ensure_dir() {
  mkdir -p "$1"
}

download_if_missing() {
  local url="$1"
  local target="$2"
  if [[ -f "$target" ]]; then
    log "Already present: $target"
    return
  fi
  log "Downloading $(basename "$target")"
  curl -L --retry 3 --retry-delay 5 -A "Mozilla/5.0" "$url" -o "$target"
}

if [[ -d "$DESTINATION" ]]; then
  if [[ "$FORCE" == "true" ]]; then
    log "Removing existing directory $DESTINATION"
    rm -rf "$DESTINATION"
  else
    log "Existing SadTalker directory detected. Updating."
    git -C "$DESTINATION" fetch --tags
    git -C "$DESTINATION" pull --ff-only
    git -C "$DESTINATION" submodule update --init --recursive
  fi
fi

if [[ ! -d "$DESTINATION" ]]; then
  log "Cloning SadTalker from $REPO_URL"
  git clone "$REPO_URL" "$DESTINATION"
  git -C "$DESTINATION" submodule update --init --recursive
fi

CHECKPOINTS="$DESTINATION/checkpoints"
GFPGAN_WEIGHTS="$DESTINATION/gfpgan/weights"
ensure_dir "$CHECKPOINTS"
ensure_dir "$GFPGAN_WEIGHTS"

while IFS=, read -r url path; do
  [[ -z "$url" ]] && continue
  download_if_missing "$url" "$path"
done <<EOF
https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar,$CHECKPOINTS/mapping_00109-model.pth.tar
https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar,$CHECKPOINTS/mapping_00229-model.pth.tar
https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors,$CHECKPOINTS/SadTalker_V0.0.2_256.safetensors
https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors,$CHECKPOINTS/SadTalker_V0.0.2_512.safetensors
https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth,$GFPGAN_WEIGHTS/alignment_WFLW_4HG.pth
https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth,$GFPGAN_WEIGHTS/detection_Resnet50_Final.pth
https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth,$GFPGAN_WEIGHTS/GFPGANv1.4.pth
https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth,$GFPGAN_WEIGHTS/parsing_parsenet.pth
EOF

SHAPE_COMPRESSED="$CHECKPOINTS/shape_predictor_68_face_landmarks.dat.bz2"
SHAPE_OUTPUT="$CHECKPOINTS/shape_predictor_68_face_landmarks.dat"
if [[ ! -f "$SHAPE_OUTPUT" ]]; then
  download_if_missing "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" "$SHAPE_COMPRESSED"
  log "Decompressing shape_predictor_68_face_landmarks.dat.bz2"
  PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
  if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python"
  fi
  "$PYTHON" "$(pwd)/decompress.py" "$SHAPE_COMPRESSED"
fi

log "SadTalker setup completed."
log "Remember to prepare the Python virtual environment via scripts/setup_env.sh or setup_env.ps1 before running the controller." 
