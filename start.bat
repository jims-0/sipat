@echo off
title Tanim Guard - Crop Disease Detection
cd /d "%~dp0"
echo.
echo ========================================
echo   TANIM GUARD - Starting Server...
echo ========================================
echo.
echo Installing dependencies (first time only)...
pip install -r requirements.txt --quiet
echo.
echo Starting Flask server...
echo Open your browser to: http://localhost:5000
echo Or access from your phone: http://YOUR_IP:5000
echo.
python run.py
pause
