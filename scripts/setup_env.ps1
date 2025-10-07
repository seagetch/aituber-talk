param(
    [string]$EnvPath = ".venv",
    [switch]$Force,
    [string]$TorchChannel = "cu121",
    [switch]$CpuTorch
)

$ErrorActionPreference = "Stop"

function Write-Info($message) {
    Write-Host "[setup] $message"
}

function Install-PyTorch {
    param(
        [string]$PythonExe,
        [string]$TorchChannel,
        [switch]$CpuTorch
    )

    if ($CpuTorch) {
        Write-Info "Installing PyTorch (CPU build)"
        & $PythonExe -m pip install --upgrade torch torchvision torchaudio
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install CPU PyTorch packages"
        }
        return
    }

    if (-not $TorchChannel) {
        Write-Info "TorchChannel parameter empty; installing CPU build"
        & $PythonExe -m pip install --upgrade torch torchvision torchaudio
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install CPU PyTorch packages"
        }
        return
    }

    Write-Info "Installing PyTorch (nightly channel: $TorchChannel)"
    $nightlyIndex = "https://download.pytorch.org/whl/nightly/$TorchChannel"
    & $PythonExe -m pip install --upgrade --pre torch torchvision torchaudio --index-url $nightlyIndex --extra-index-url https://pypi.org/simple
    if ($LASTEXITCODE -ne 0) {
        Write-Info "CUDA nightly installation failed; falling back to CPU build"
        & $PythonExe -m pip install --upgrade torch torchvision torchaudio
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install PyTorch packages"
        }
    }
}

if (Test-Path $EnvPath) {
    if ($Force) {
        Write-Info "Removing existing environment at $EnvPath"
        Remove-Item -Recurse -Force $EnvPath
    } else {
        Write-Info "Environment already exists at $EnvPath; reuse it. Use -Force to recreate."
    }
}

if (-not (Test-Path $EnvPath)) {
    Write-Info "Creating virtual environment"
    python -m venv $EnvPath
}

$pythonExe = Join-Path $EnvPath "Scripts/python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Failed to locate python executable at $pythonExe"
}

Write-Info "Upgrading pip"
& $pythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip"
}

if (-not (Test-Path "requirements.txt")) {
    throw "requirements.txt not found in repo root"
}

Write-Info "Installing requirements"
& $pythonExe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install requirements"
}

Install-PyTorch -PythonExe $pythonExe -TorchChannel $TorchChannel -CpuTorch:$CpuTorch

Write-Info "Setup complete. Activate the environment with `& \"$EnvPath\Scripts\Activate.ps1\"`."
