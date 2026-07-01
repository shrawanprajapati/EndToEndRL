$ErrorActionPreference = "Stop"

$bundledPython = "C:\Users\adwit\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$pythonCommand = $null
$pythonArgs = @()

if (Test-Path $bundledPython) {
    $pythonCommand = $bundledPython
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCommand = (Get-Command python).Source
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCommand = (Get-Command py).Source
    $pythonArgs = @("-3")
} else {
    throw "No Python runtime found. Install Python or use the bundled Codex runtime."
}

Push-Location $PSScriptRoot
try {
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"

    Write-Host "Running feature engineering..."
    & $pythonCommand @pythonArgs data/feature_engineer.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "Splitting train/test data..."
    & $pythonCommand @pythonArgs data/split_data.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "Launching Optuna tuner..."
    & $pythonCommand @pythonArgs -X utf8 -u -m models.tune_hyperparams
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
