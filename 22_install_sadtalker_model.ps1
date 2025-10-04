# This script downloads and sets up the models for SadTalker on Windows.

# Create the checkpoints directory
$checkpointsDir = "SadTalker/checkpoints"
if (-not (Test-Path -Path $checkpointsDir)) {
    New-Item -ItemType Directory -Path $checkpointsDir
}

# Download main SadTalker models
Write-Host "Downloading main SadTalker models..."
$sadTalkerModels = @(
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar",
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar",
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors",
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors"
)
foreach ($modelUrl in $sadTalkerModels) {
    $fileName = [System.IO.Path]::GetFileName($modelUrl)
    $outputPath = Join-Path $checkpointsDir $fileName
    curl.exe -L $modelUrl -o $outputPath
}

# Create gfpgan weights directory
$gfpganDir = "SadTalker/gfpgan/weights"
if (-not (Test-Path -Path $gfpganDir)) {
    New-Item -ItemType Directory -Path $gfpganDir
}

# Download GFPGAN models
Write-Host "Downloading GFPGAN models..."
$gfpganModels = @(
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth",
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "https://huggingface.co/gmk123/GFPGAN/resolve/main/GFPGANv1.4.pth",
    "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"
)
foreach ($modelUrl in $gfpganModels) {
    $fileName = [System.IO.Path]::GetFileName($modelUrl)
    $outputPath = Join-Path $gfpganDir $fileName
    curl.exe -L $modelUrl -o $outputPath
}

# Download dlib model
Write-Host "Downloading dlib model..."
$dlibBz2Path = Join-Path $checkpointsDir "shape_predictor_68_face_landmarks.dat.bz2"
curl.exe -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o $dlibBz2Path

# Decompress the dlib model using decompress.py
Write-Host "Decompressing dlib model..."
$pythonExecutable = "./sadtalker_venv/Scripts/python.exe"
$decompressScriptPath = "./decompress.py"
& $pythonExecutable $decompressScriptPath $dlibBz2Path

Write-Host "Model download and setup complete."