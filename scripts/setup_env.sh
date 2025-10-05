#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="${1:-.venv}"
FORCE="${FORCE:-false}"

log() {
  echo "[setup] $1"
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

log "Setup complete. Activate with 'source $ENV_PATH/bin/activate'."
