<#
    Sipat - build Sipat.exe and the zip you carry to another PC.

        powershell -ExecutionPolicy Bypass -File build_exe.ps1

    Runs the tests, builds dist\Sipat\ with PyInstaller (no TensorFlow
    inside - the app runs the .tflite with LiteRT), adds a README, checks that
    no data\ (photos, database, secret key) slipped in, and zips it to
    ..\Sipat_<date>.zip. On the other PC: unzip, double-click Sipat.exe.
#>
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Test-Path 'app\ml\model\tanim_guard.tflite')) {
    throw 'No app\ml\model\tanim_guard.tflite - train and export the model first (TRAINING.md).'
}

Write-Host '[ Tests ]' -ForegroundColor Cyan
python -m pytest -p no:cacheprovider -q tests
if ($LASTEXITCODE -ne 0) { throw 'Tests failed - nothing was built.' }

Write-Host '[ Building ]' -ForegroundColor Cyan
python -m PyInstaller tanim_guard.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }
$dist = Join-Path $PSScriptRoot 'dist\Sipat'
if (-not (Test-Path (Join-Path $dist 'Sipat.exe'))) { throw 'No Sipat.exe was produced.' }

$readme = @'
Sipat
===========

1. Double-click Sipat.exe. A black window opens (keep it open) and the
   browser opens Sipat by itself.
2. To use it on a phone: connect the phone to the same Wi-Fi and open the
   "Sa cellphone" address shown in the black window.
3. Photos and history are saved in the data\ folder next to the exe. Copy
   that folder to keep them.

Optional - the adviser ("Magtanong" box) needs Ollama on this PC:
  install from https://ollama.com, then run:  ollama pull qwen2.5:3b-instruct
Without it, the written guidance still shows; only the question box is off.

Windows may warn the first time ("Windows protected your PC"): click
More info -> Run anyway. Allow the firewall prompt for phones to connect.
'@
Set-Content -Path (Join-Path $dist 'README.txt') -Value $readme -Encoding utf8

Write-Host '[ Leak check ]' -ForegroundColor Cyan
if (Test-Path (Join-Path $dist 'data')) { throw 'dist\Sipat\data exists - remove it before zipping (real photos/database).' }

$zip = Join-Path (Split-Path -Parent $PSScriptRoot) ("Sipat_{0}.zip" -f (Get-Date -Format 'yyyyMMdd'))
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $dist -DestinationPath $zip
'{0} ({1:N0} MB)' -f $zip, ((Get-Item $zip).Length / 1MB) | Write-Host -ForegroundColor Green
