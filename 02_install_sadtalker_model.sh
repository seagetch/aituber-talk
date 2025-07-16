#!/usr/bin/env bash
cd SadTalker
bash scripts/download_models.sh light
mkdir -p SadTalker/checkpoints
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O SadTalker/checkpoints/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 SadTalker/checkpoints/shape_predictor_68_face_landmarks.dat.bz2
