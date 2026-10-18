# Windows: create .venv, install deps, run the smoke test.
#   powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
  foreach ($v in "-3.13","-3.12","-3.11") { & py $v -c "pass" 2>$null; if ($LASTEXITCODE -eq 0) { $py = @("py",$v); break } }
}
if (-not $py -and (Get-Command python -ErrorAction SilentlyContinue)) { $py = @("python") }
if (-not $py) { Write-Error "Python 3.11+ not found. Install from https://www.python.org/downloads/ (tick 'Add to PATH') or: winget install Python.Python.3.12"; exit 1 }
$exe = $py[0]; $pyArgs = @($py | Select-Object -Skip 1)
& $exe @pyArgs -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"
if ($LASTEXITCODE -ne 0) { Write-Error "Need Python 3.11+"; exit 1 }
if (-not (Test-Path .venv)) { & $exe @pyArgs -m venv .venv }
$vpy = ".\.venv\Scripts\python.exe"
& $vpy -m pip install -q --upgrade pip
& $vpy -m pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed"; exit 1 }
& $vpy -m mvp.smoke
if ($LASTEXITCODE -ne 0) { Write-Error "smoke test failed"; exit 1 }
Write-Host "Setup done. Next: run.bat (demo)"
