@echo off
rem Sipat demo: start the Android emulator phone, install the APK, open Sipat.
rem Close games / heavy apps first - the phone uses about 2 GB of RAM.
setlocal
set "ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk"
set "ADB=%ANDROID_HOME%\platform-tools\adb.exe"
set "APK=%~dp0android\app\build\outputs\apk\debug\app-debug.apk"

if not exist "%APK%" (
  echo No APK yet - build it first: build_apk.bat
  pause & exit /b 1
)

echo Starting the phone (first boot takes 1-3 minutes)...
start "" "%ANDROID_HOME%\emulator\emulator.exe" -avd Sipat_Phone -no-snapshot-save -no-audio

"%ADB%" wait-for-device
:boot
"%ADB%" shell getprop sys.boot_completed 2>nul | find "1" >nul || (timeout /t 3 /nobreak >nul & goto boot)

echo Installing Sipat...
"%ADB%" install -r "%APK%"
"%ADB%" shell monkey -p ph.edu.bulsu.sipat -c android.intent.category.LAUNCHER 1 >nul
echo.
echo Sipat is open on the phone. To give it leaf photos, drag a .jpg onto the
echo emulator window (it lands in Downloads), then tap "Pumili ng picture".
pause
