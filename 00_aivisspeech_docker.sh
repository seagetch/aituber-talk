#!/bin/bash
docker pull ghcr.io/aivis-project/aivisspeech-engine:cpu-latest
docker run --rm -p '10101:10101' \
  -v ~/.local/share/AivisSpeech-Engine:/home/user/.local/share/AivisSpeech-Engine-Dev \
  ghcr.io/aivis-project/aivisspeech-engine:cpu-latest
