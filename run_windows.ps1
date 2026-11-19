<#
.SYNOPSIS
    Windows launcher for Adaptive Sweet Spot (tools/app.py).
.EXAMPLE
    .\run_windows.ps1
    .\run_windows.ps1 --camera 0 --audio-device WASAPI
    .\run_windows.ps1 --no-camera
    .\run_windows.ps1 -SetupDepth
#>
param(
    [switch]$SetupDepth,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Push-Location $Root

try {
    function Get-SystemPython {
        if (Get-Command py -ErrorAction SilentlyContinue) { return @("py", "-3") }
        if (Get-Command python -ErrorAction SilentlyContinue) { return @("python") }
        throw "Python 3.11+ not found. Install Python (or uv) and re-run."
    }

    function Invoke-Step {
        param([string]$Exe, [string[]]$StepArgs)
        & $Exe @StepArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed ($LASTEXITCODE): $Exe $($StepArgs -join ' ')"
        }
    }

    # Optional one-time setup for MoGe-2 depth (used by Setup / Room scan in the web app)
    if ($SetupDepth) {
        $MogeDir = Join-Path $Root ".venvs\moge2"
        $MogePy = Join-Path $MogeDir "Scripts\python.exe"
        if (-not (Test-Path $MogePy)) {
            Write-Host "Creating MoGe-2 environment in .venvs\moge2..." -ForegroundColor Cyan
            if (Get-Command uv -ErrorAction SilentlyContinue) {
                Invoke-Step "uv" @("venv", $MogeDir, "--python", "3.12")
            } else {
                $sysPy = Get-SystemPython
                $verArgs = if ($sysPy[0] -eq "py") { @("-3.12", "-m", "venv", $MogeDir) } else { @("-m", "venv", $MogeDir) }
                & $sysPy[0] @verArgs
                if ($LASTEXITCODE -ne 0) {
                    Invoke-Step $sysPy[0] ($sysPy[1..($sysPy.Length - 1)] + @("-m", "venv", $MogeDir))
                }
            }
        }

        $torchIndex = @()
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            Write-Host "NVIDIA GPU detected - installing CUDA PyTorch for MoGe-2..." -ForegroundColor Cyan
            $torchIndex = @("--extra-index-url", "https://download.pytorch.org/whl/cu124")
        }

        $deps = @(
            "torch>=2.4", "torchvision>=0.19", "numpy>=2", "opencv-python-headless",
            "scipy", "pillow", "trimesh", "huggingface-hub", "requests", "tqdm", "click",
            "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37",
            "pipeline @ git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4"
        )
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            Invoke-Step "uv" (@("pip", "install", "--python", $MogePy) + $torchIndex + $deps)
            Invoke-Step "uv" @("pip", "install", "--python", $MogePy, "--no-deps", "moge @ git+https://github.com/microsoft/MoGe.git")
        } else {
            Invoke-Step $MogePy (@("-m", "pip", "install") + $torchIndex + $deps)
            Invoke-Step $MogePy @("-m", "pip", "install", "--no-deps", "moge @ git+https://github.com/microsoft/MoGe.git")
        }
        Write-Host "MoGe-2 environment ready. Run .\run_windows.ps1 to start the app." -ForegroundColor Green
        exit 0
    }

    # Ensure main virtual environment exists
    $VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        Write-Host "Setting up Python environment (.venv)..." -ForegroundColor Cyan
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            Invoke-Step "uv" @("sync")
        } else {
            $sysPy = Get-SystemPython
            $venvArgs = if ($sysPy.Length -gt 1) { $sysPy[1..($sysPy.Length - 1)] + @("-m", "venv", ".venv") } else { @("-m", "venv", ".venv") }
            Invoke-Step $sysPy[0] $venvArgs
            Invoke-Step $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
            Invoke-Step $VenvPython @("-m", "pip", "install", "-e", ".[audio,vision,dev]")
        }
    }

    # Ensure BlazeFace model weights are present
    $ModelPath = Join-Path $Root "assets\models\blaze_face_full_range.tflite"
    if (-not (Test-Path $ModelPath)) {
        Write-Host "Downloading BlazeFace model weights..." -ForegroundColor Cyan
        Invoke-Step $VenvPython @("tools\fetch_model.py")
    }

    # Launch Sweet Spot Live (same app as macOS/Linux)
    & $VenvPython (Join-Path $Root "tools\app.py") @AppArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
