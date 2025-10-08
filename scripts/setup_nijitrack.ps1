# This script installs the nijitrack dependency.

# --- Helper functions ---

# Function to print messages
function Info {
    param ([string]$message)
    Write-Host "[INFO] $message"
}

# --- Main script ---

# Check if nijitrack directory exists
if (Test-Path -Path "nijitrack" -PathType Container) {
    Info "nijitrack directory already exists. Skipping clone."
} else {
    Info "Cloning nijitrack repository..."
    git clone https://github.com/nijigenerate/nijitrack.git
}

# Check for virtual environment
if (-not (Test-Path -Path ".venv" -PathType Container)) {
    Write-Error "Virtual environment .venv not found. Please run setup_env.ps1 first."
    exit 1
}

# Activate virtual environment and install
Info "Activating virtual environment and installing nijitrack..."

# PowerShell execution policy might prevent script execution.
# We can try to bypass it for this process.
try {
    & .\.venv\Scripts\Activate.ps1
    Info "Installing nijitrack..."
    pip install .\nijitrack
} catch {
    Write-Error "Failed to install nijitrack. Ensure your PowerShell execution policy allows script execution (`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`) or run this script in a terminal that supports it."
    exit 1
}

Info "nijitrack installation complete."

