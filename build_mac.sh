#!/bin/bash
# Sipat - build the Mac version and the zip you carry to another Mac.
# Run it ON the Mac (PyInstaller can only build for the system it runs on):
#
#     bash build_mac.sh
#
# Runs the tests, builds dist/Sipat/ with PyInstaller from the same
# tanim_guard.spec as the Windows exe, adds a double-click launcher
# (Sipat.command) and a README, checks that no data/ (photos, database,
# secret key) slipped in, and zips it to ../Sipat_mac_<date>.zip.
#
# Why a folder + Sipat.command and not a Sipat.app: Sipat is a server with a
# console, like Sipat.exe's black window. In Terminal you see the phone
# address and close the window to stop it; an .app would run with no window
# and no way to stop it, and would save photos inside itself.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f app/ml/model/tanim_guard.tflite ]; then
    echo "No app/ml/model/tanim_guard.tflite - train and export the model first (TRAINING.md)." >&2
    exit 1
fi

# The same .venv start.command makes; created here if start.command never ran.
if [ ! -x .venv/bin/python ]; then
    for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1 \
            && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
            "$candidate" -m venv .venv
            break
        fi
    done
    [ -x .venv/bin/python ] || { echo "Python 3.10+ not found - install it from python.org." >&2; exit 1; }
fi
py=.venv/bin/python
echo "[ Packages ]"
$py -m pip install --quiet --upgrade pip
$py -m pip install --quiet -r requirements.txt pytest pyinstaller

echo "[ Tests ]"
$py -m pytest -p no:cacheprovider -q tests || { echo "Tests failed - nothing was built." >&2; exit 1; }

echo "[ Building ]"
$py -m PyInstaller tanim_guard.spec --noconfirm --clean
dist=dist/Sipat
[ -x "$dist/Sipat" ] || { echo "No dist/Sipat/Sipat was produced." >&2; exit 1; }

cat > "$dist/Sipat.command" <<'EOF'
#!/bin/bash
# Double-click to start Sipat. Close this window to stop it.
cd "$(dirname "$0")" && ./Sipat
EOF
chmod +x "$dist/Sipat.command"

cat > "$dist/README.txt" <<'EOF'
Sipat (Mac)
===========

1. Double-click Sipat.command. A Terminal window opens (keep it open) and
   the browser opens Sipat by itself. Close the window to stop Sipat.
2. To use it on a phone: connect the phone to the same Wi-Fi and open the
   "Sa cellphone" address shown in the Terminal window.
3. Photos and history are saved in the data/ folder next to Sipat.command.
   Copy that folder to keep them.
4. Training evidence (dataset, accuracy per epoch, confusion matrix, score
   per disease, raw files): in Sipat, About -> "Training Evidence".

Optional - the adviser ("Magtanong" box) needs Ollama on this Mac:
  install from https://ollama.com, then run:  ollama pull qwen2.5:3b-instruct
Without it, the written guidance still shows; only the question box is off.

If macOS says Sipat.command "cannot be opened" (it came from another Mac):
right-click it -> Open -> Open. If it still refuses, open Terminal and run
  xattr -dr com.apple.quarantine <drag this folder here>
Click Allow if macOS asks about incoming connections - phones need it.
EOF

echo "[ Leak check ]"
if [ -e "$dist/data" ]; then
    echo "dist/Sipat/data exists - remove it before zipping (real photos/database)." >&2
    exit 1
fi

zip="../Sipat_mac_$(date +%Y%m%d).zip"
rm -f "$zip"
# ditto keeps the executable bits that a plain zip can lose.
ditto -c -k --keepParent "$dist" "$zip"
echo "$zip ($(du -h "$zip" | cut -f1))"
