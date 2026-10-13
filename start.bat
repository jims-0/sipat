@echo off
title Tanim Guard - Crop Disease Detection
cd /d "%~dp0"
echo.
echo ========================================
echo   TANIM GUARD
echo ========================================
echo.

rem Its own Python environment in .venv, so nothing else on this PC is touched.
if not exist ".venv\Scripts\python.exe" (
    echo First run: setting up - needs internet once, about 1-2 minutes...
    where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
    if not exist ".venv\Scripts\python.exe" goto :nopython
    ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
    ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
    if errorlevel 1 goto :fail
)

if not exist "app\ml\model\tanim_guard.tflite" (
    echo.
    echo The trained model is missing: app\ml\model\tanim_guard.tflite
    echo See TRAINING.md to train it, or copy the model folder from the project.
    goto :fail
)

echo Starting... the browser opens by itself. Close this window to stop.
set TANIM_OPEN_BROWSER=1
".venv\Scripts\python.exe" run.py
goto :end

:nopython
echo.
echo Python was not found. Install Python 3.10 or newer from python.org
echo (tick "Add python.exe to PATH"), then run start.bat again.
:fail
echo.
echo SETUP FAILED - read the message above.
:end
pause
