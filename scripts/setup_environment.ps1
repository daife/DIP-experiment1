[CmdletBinding()]
param(
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $true
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$requirementsPath = Join-Path $projectRoot "requirements.txt"
$verifyScript = Join-Path $PSScriptRoot "verify_environment.py"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/ and rerun this script."
}

Push-Location $projectRoot
try {
    Write-Host "[1/4] Ensuring Python $PythonVersion is available..."
    uv python install $PythonVersion
    if ($LASTEXITCODE -ne 0) {
        throw "uv could not install or locate Python $PythonVersion."
    }

    $createEnvironment = -not (Test-Path $venvPython)
    if (-not $createEnvironment) {
        $installedVersion = & $venvPython -c "import platform; print(platform.python_version())"
        if (-not $installedVersion.StartsWith("$PythonVersion.")) {
            Write-Host "Existing .venv uses Python $installedVersion; recreating it with Python $PythonVersion."
            $createEnvironment = $true
        }
    }

    if ($createEnvironment) {
        Write-Host "[2/4] Creating .venv..."
        $venvArguments = @("venv", ".venv", "--python", $PythonVersion)
        if (Test-Path $venvPath) {
            $venvArguments += "--clear"
        }
        uv @venvArguments
        if ($LASTEXITCODE -ne 0) {
            throw "uv failed to create .venv with Python $PythonVersion."
        }
    }
    else {
        Write-Host "[2/4] Reusing .venv (Python $installedVersion)."
    }

    Write-Host "[3/4] Installing project dependencies..."
    uv pip install --python $venvPython --requirements $requirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }

    Write-Host "[4/4] Verifying imports and basic numerical operations..."
    & $venvPython $verifyScript
    if ($LASTEXITCODE -ne 0) {
        throw "Environment verification failed."
    }

    Write-Host "Environment setup completed successfully."
    Write-Host "Activate it with: .venv\Scripts\Activate.ps1"
}
finally {
    Pop-Location
}
