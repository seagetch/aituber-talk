#!/bin/sh
STYLE_ID=888753760 && echo -n "こんにちは、音声合成の世界へようこそ！" > text.txt && \
curl -s -X POST "127.0.0.1:10101/audio_query?speaker=$STYLE_ID" --get --data-urlencode text@text.txt > query.json && \
curl -s -H "Content-Type: application/json" -X POST -d @query.json "127.0.0.1:10101/synthesis?speaker=$STYLE_ID" > audio.wav && rm text.txt query.json
