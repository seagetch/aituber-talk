#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="${1:-.venv}"
FORCE="${FORCE:-false}"

log() {
  echo "[setup] $1"
}
TORCH_CHANNEL="${TORCH_CHANNEL:-cu121}"
CPU_TORCH="${CPU_TORCH:-false}"

install_pytorch() {
  local cpu_flag="${CPU_TORCH,,}"
  if [[ "$cpu_flag" == "true" || "$cpu_flag" == "1" ]]; then
    log "Installing PyTorch (CPU build)"
    if ! "$PYBIN" -m pip install --upgrade torch torchvision torchaudio; then
      echo "Failed to install CPU PyTorch packages" >&2
      exit 1
    fi
    return
  fi

  if [[ -z "${TORCH_CHANNEL}" ]]; then
    log "Torch channel not set; installing CPU build"
    if ! "$PYBIN" -m pip install --upgrade torch torchvision torchaudio; then
      echo "Failed to install CPU PyTorch packages" >&2
      exit 1
    fi
    return
  fi

  log "Installing PyTorch (nightly channel: ${TORCH_CHANNEL})"
  local nightly_url="https://download.pytorch.org/whl/nightly/${TORCH_CHANNEL}"
  if ! "$PYBIN" -m pip install --upgrade --pre torch torchvision torchaudio --index-url "$nightly_url" --extra-index-url https://pypi.org/simple; then
    log "CUDA nightly installation failed; falling back to CPU build"
    if ! "$PYBIN" -m pip install --upgrade torch torchvision torchaudio; then
      echo "Failed to install PyTorch packages" >&2
      exit 1
    fi
  fi
}


if [[ -d "$ENV_PATH" ]]; then
  if [[ "$FORCE" == "true" ]]; then
    log "Removing existing environment at $ENV_PATH"
    rm -rf "$ENV_PATH"
  else
    log "Environment already exists at $ENV_PATH; reuse it. Set FORCE=true to recreate."
  fi
fi

if [[ ! -d "$ENV_PATH" ]]; then
  log "Creating virtual environment"
  python -m venv "$ENV_PATH"
fi

PYBIN="$ENV_PATH/bin/python"
if [[ ! -x "$PYBIN" ]]; then
  echo "Python executable not found at $PYBIN" >&2
  exit 1
fi

log "Upgrading pip"
"$PYBIN" -m pip install --upgrade pip

if [[ ! -f requirements.txt ]]; then
  echo "requirements.txt not found in repo root" >&2
  exit 1
fi

log "Installing requirements"
"$PYBIN" -m pip install -r requirements.txt
install_pytorch


log "Setup complete. Activate with 'source $ENV_PATH/bin/activate'."
