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
4. Training evidence (dataset, accuracy per epoch, confusion matrix, score
   per disease, raw files): in Sipat, About -> "Training Evidence", or as
   files in the Docs folder (start with Docs\1 - START HERE.txt).

Optional - the adviser ("Magtanong" box) needs Ollama on this PC:
  install from https://ollama.com, then run:  ollama pull qwen2.5:3b-instruct
Without it, the written guidance still shows; only the question box is off.

Windows may warn the first time ("Windows protected your PC"): click
More info -> Run anyway. Allow the firewall prompt for phones to connect.
'@
Set-Content -Path (Join-Path $dist 'README.txt') -Value $readme -Encoding utf8

Write-Host '[ Docs ]' -ForegroundColor Cyan
# Training/evaluation evidence as readable files. Same layout as the repo, so
# the paths written inside TRAINING.md and the docs still point at real files.
$docsOut = Join-Path $dist 'Docs'
$docFiles = @(
    'TRAINING.md', 'requirements-train.txt',
    'docs\EVALUATION.md', 'docs\confusion_matrix.png', 'docs\SIPAT_SYSTEM_OVERVIEW.md',
    'docs\training_evidence\README.txt', 'docs\training_evidence\training_log_20261014_010227.txt',
    'app\ml\model\training_info.json', 'app\ml\model\training_history.csv',
    'app\ml\model\evaluation_report.json', 'app\ml\model\labels.json',
    'scripts\prepare_dataset.py', 'scripts\train_model.py', 'scripts\evaluate.py',
    'scripts\export_tflite.py', 'scripts\evaluate_field.py', 'scripts\tg_data.py'
)
foreach ($f in $docFiles) {
    if (-not (Test-Path $f)) { throw "Docs: missing $f" }
    $dest = Join-Path $docsOut $f
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
    Copy-Item $f $dest
}
$docsIndex = @'
Sipat - training and evaluation documents
=========================================

Read in this order:

1. TRAINING.md                         how the model was trained, step by step
                                       (dataset, split, settings, limits)
2. docs\EVALUATION.md                  test-set results: accuracy, per-disease
   docs\confusion_matrix.png           scores, the confusion matrix
3. docs\training_evidence\             the raw console log of the training run
                                       (all 11 epochs) and what each file is
4. app\ml\model\                       the run's numbers as data files:
                                       training_info.json (settings),
                                       training_history.csv (per epoch),
                                       evaluation_report.json (test results),
                                       labels.json (the 21 classes)
5. scripts\                            the code that prepared the dataset,
   requirements-train.txt              trained, evaluated and exported the model
6. docs\SIPAT_SYSTEM_OVERVIEW.md       how the whole system fits together

The .md files open in any text editor, or nicely in VS Code / GitHub.
The dataset photos are not included: PlantVillage is public, see TRAINING.md.
'@
Set-Content -Path (Join-Path $docsOut '1 - START HERE.txt') -Value $docsIndex -Encoding utf8

Write-Host '[ Leak check ]' -ForegroundColor Cyan
if (Test-Path (Join-Path $dist 'data')) { throw 'dist\Sipat\data exists - remove it before zipping (real photos/database).' }

$zip = Join-Path (Split-Path -Parent $PSScriptRoot) ("Sipat_{0}.zip" -f (Get-Date -Format 'yyyyMMdd'))
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $dist -DestinationPath $zip
'{0} ({1:N0} MB)' -f $zip, ((Get-Item $zip).Length / 1MB) | Write-Host -ForegroundColor Green
