#!/bin/bash
# Sipat - double-click on a Mac (the Mac version of start.bat).
# If macOS says it cannot be opened: right-click it -> Open -> Open.
cd "$(dirname "$0")" || exit 1
echo
echo "========================================"
echo "  SIPAT"
echo "========================================"
echo

finish() {
    echo
    read -n 1 -s -r -p "Press any key to close this window..."
    exit "$1"
}

# A Python 3.10 or newer: the model runtime (ai-edge-litert) has no build for
# the 3.9 that comes with Apple's developer tools.
find_python() {
    for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1 \
            && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# Its own Python environment in .venv, so nothing else on this Mac is touched.
if [ ! -x ".venv/bin/python" ]; then
    echo "First run: setting up - needs internet once, about 1-2 minutes..."
    if ! python=$(find_python); then
        echo
        echo "Python 3.10 or newer was not found. Install Python 3.12 from"
        echo "https://www.python.org/downloads/macos/ then open start.command again."
        finish 1
    fi
    "$python" -m venv .venv || { echo; echo "SETUP FAILED - read the message above."; finish 1; }
    .venv/bin/python -m pip install --quiet --upgrade pip
    if ! .venv/bin/python -m pip install --quiet -r requirements.txt; then
        rm -rf .venv  # so the next double-click tries the setup again
        echo
        echo "SETUP FAILED - read the message above."
        finish 1
    fi
fi

if [ ! -f "app/ml/model/tanim_guard.tflite" ]; then
    echo
    echo "The trained model is missing: app/ml/model/tanim_guard.tflite"
    echo "See TRAINING.md to train it, or copy the model folder from the project."
    finish 1
fi

echo "Starting... the browser opens by itself. Close this window to stop."
TANIM_OPEN_BROWSER=1 .venv/bin/python run.py
finish $?
