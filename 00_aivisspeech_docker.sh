#!/bin/bash
set -x
if [ ! -n "$DEVICE" ]; then
  DEVICE=cpu
fi

if [ "$DEVICE" != "cpu" ]; then
  OPT="--gpus all"
else
  OPT=""
fi
docker pull ghcr.io/aivis-project/aivisspeech-engine:$DEVICE-latest
docker run --rm $OPT -p '10101:10101' \
  -v ~/.local/share/AivisSpeech-Engine:/home/user/.local/share/AivisSpeech-Engine-Dev \
  ghcr.io/aivis-project/aivisspeech-engine:$DEVICE-latest
