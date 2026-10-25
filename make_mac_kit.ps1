<#
    Sipat - the small zip you copy to a Mac (run this on Windows).

        powershell -ExecutionPolicy Bypass -File make_mac_kit.ps1

    The tanim_guard folder is ~1.7 GB because of datasets\, .venv\, build\ and
    dist\ - none of which a Mac can use. This zips only what git tracks (code,
    model, phone app), so no local photos, database or .env go with it, and
    start.command / build_mac.sh keep their executable bit. Commit first:
    uncommitted changes are NOT in the kit.

    On the Mac: unzip, double-click start.command (SETUP.md, "On a Mac").
#>
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$dirty = git status --porcelain -- .
if ($dirty) { Write-Host "Warning: uncommitted changes are not in the kit:`n$($dirty -join "`n")" -ForegroundColor Yellow }

$zip = Join-Path (Split-Path -Parent $PSScriptRoot) ("Sipat_mac_kit_{0}.zip" -f (Get-Date -Format 'yyyyMMdd'))
if (Test-Path $zip) { Remove-Item $zip -Force }
# From the repo root: run inside tanim_guard\, git archive would also cut the
# tree down to the current folder and the zip comes out empty.
git -C (git rev-parse --show-toplevel) archive --format=zip --prefix=Sipat/ -o $zip HEAD:tanim_guard
if ($LASTEXITCODE -ne 0) { throw 'git archive failed.' }
'{0} ({1:N0} MB)' -f $zip, ((Get-Item $zip).Length / 1MB) | Write-Host -ForegroundColor Green
