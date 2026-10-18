@echo off
REM Windows: set up if needed, run a short demo sim (~30 s CPU), open the GIF.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1 || exit /b 1
)
.venv\Scripts\python.exe -m mvp.smoke --demo --out out_demo %* || exit /b 1
start "" "out_demo\demo.gif"
