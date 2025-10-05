param(
    [string]$RepoUrl = "https://github.com/OpenTalker/SadTalker.git",
    [string]$Destination = "SadTalker",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Write-Info($message) {
    Write-Host "[sadtalker] $message"
}

function Ensure-Directory($path) {
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
}

function Download-IfMissing($url, $targetPath) {
    if (Test-Path $targetPath) {
        Write-Info "Already present: $targetPath"
        return
    }
    Write-Info "Downloading $(Split-Path $targetPath -Leaf)"
    Invoke-WebRequest -Uri $url -OutFile $targetPath -UseBasicParsing -Headers @{ 'User-Agent' = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' }
}

if (Test-Path $Destination) {
    if ($Force) {
        Write-Info "Removing existing directory $Destination"
        Remove-Item -Recurse -Force $Destination
    } else {
        Write-Info "Existing SadTalker directory detected. Pulling latest changes."
        Push-Location $Destination
        try {
            git fetch --tags
            git pull --ff-only
            git submodule update --init --recursive
        } finally {
            Pop-Location
        }
    }
}

if (-not (Test-Path $Destination)) {
    Write-Info "Cloning SadTalker from $RepoUrl"
    git clone $RepoUrl $Destination
    Push-Location $Destination
    try {
        git submodule update --init --recursive
    } finally {
        Pop-Location
    }
}

$checkpoints = Join-Path $Destination "checkpoints"
Ensure-Directory $checkpoints
$gfpganWeights = Join-Path $Destination "gfpgan/weights"
Ensure-Directory $gfpganWeights

$downloads = @(
    @{ Url = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar"; Path = Join-Path $checkpoints "mapping_00109-model.pth.tar" },
    @{ Url = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar"; Path = Join-Path $checkpoints "mapping_00229-model.pth.tar" },
    @{ Url = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors"; Path = Join-Path $checkpoints "SadTalker_V0.0.2_256.safetensors" },
    @{ Url = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors"; Path = Join-Path $checkpoints "SadTalker_V0.0.2_512.safetensors" },
    @{ Url = "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth"; Path = Join-Path $gfpganWeights "alignment_WFLW_4HG.pth" },
    @{ Url = "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"; Path = Join-Path $gfpganWeights "detection_Resnet50_Final.pth" },
    @{ Url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"; Path = Join-Path $gfpganWeights "GFPGANv1.4.pth" },
    @{ Url = "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"; Path = Join-Path $gfpganWeights "parsing_parsenet.pth" }
)

foreach ($item in $downloads) {
    Download-IfMissing -url $item.Url -targetPath $item.Path
}

$shapeUrl = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
$shapeCompressed = Join-Path $checkpoints "shape_predictor_68_face_landmarks.dat.bz2"
$shapeOutput = Join-Path $checkpoints "shape_predictor_68_face_landmarks.dat"
if (-not (Test-Path $shapeOutput)) {
    Download-IfMissing -url $shapeUrl -targetPath $shapeCompressed
    Write-Info "Decompressing shape_predictor_68_face_landmarks.dat.bz2"
    $repoRoot = (Get-Location).Path
    $pythonCandidates = @(
        (Join-Path $repoRoot ".venv\Scripts\python.exe"),
        "python"
    )
    $pythonExe = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $pythonExe) {
        throw "Python executable not found for decompression. Activate the virtual environment first."
    }
    & $pythonExe (Join-Path $repoRoot "decompress.py") $shapeCompressed
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to decompress shape predictor."
    }
} else {
    Write-Info "Already present: $shapeOutput"
}

Write-Info "SadTalker setup completed."
Write-Info "Ensure the Python environment is prepared via scripts/setup_env.ps1 before running the controller."
