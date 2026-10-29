@echo off
rem Rebuild the Sipat APK from ..\pwa (run after build_pwa.py or any pwa\ change).
setlocal
set "JAVA_HOME=%LOCALAPPDATA%\Android\jdk\jdk-21.0.13+9"
set "ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk"
set "PATH=%JAVA_HOME%\bin;%PATH%"
cd /d "%~dp0"
if not exist node_modules call npm install || exit /b 1
call npm run apk || (echo BUILD FAILED & pause & exit /b 1)
echo.
echo APK: %~dp0android\app\build\outputs\apk\debug\app-debug.apk
pause
