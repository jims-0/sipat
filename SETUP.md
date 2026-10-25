# Sipat - Quick Setup Guide

## Installation

1. **Navigate to the project:**
   ```bash
   cd tanim_guard
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create uploads folder:**
   ```bash
   mkdir -p app\static\uploads
   ```

5. **Copy environment file:**
   ```bash
   copy .env.example .env
   ```
   
   Edit `.env` and set your `SECRET_KEY`:
   ```
   SECRET_KEY=your-random-secret-key-here
   ```

6. **Run the app:**
   ```bash
   python run.py
   ```

7. **Open in browser:**
   - Desktop: `http://localhost:5000`
   - Mobile (same network): `http://<your-ip>:5000`

## On a Mac (Apple Silicon: M1/M2/M3/M4)

`Sipat.exe` is Windows-only. A Mac runs Sipat from source instead. The model
runtime (`ai-edge-litert`) ships Apple Silicon builds for Python 3.10-3.14,
macOS 12 or newer. Intel Macs have no build; use the phone app (`pwa/`) there.

1. **On Windows, make the kit:** commit, then
   `powershell -ExecutionPolicy Bypass -File make_mac_kit.ps1`.
   This writes `..\Sipat_mac_kit_<date>.zip`: only the files git tracks, not the
   1.7 GB folder (no datasets, .venv, photos, database or .env).
2. **On the Mac, install Python 3.12** from https://www.python.org/downloads/macos/
   (Apple's own `python3` is 3.9, which is too old).
3. **Unzip, double-click `start.command`.** The first run sets up `.venv` (needs
   internet once), then the browser opens Sipat. Close the Terminal window to stop.
   - "cannot be opened": right-click `start.command` -> Open -> Open.
   - "permission denied": in Terminal, `chmod +x start.command build_mac.sh`.
   - Click Allow when macOS asks about incoming connections (phones need it).
4. **Optional, a standalone build** (no Python needed on the next Mac):
   `bash build_mac.sh` on the Mac. It runs the tests and makes
   `../Sipat_mac_<date>.zip` with `Sipat.command` inside. This is a folder plus a
   Terminal launcher, not a `Sipat.app`, because Sipat is a server: the Terminal
   window shows the phone address and closing it stops Sipat, like `Sipat.exe`'s
   black window.

## Project Structure

```
tanim_guard/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── models.py            # Database layer (SQLite)
│   ├── routes.py            # Main routes
│   ├── ml/
│   │   ├── inference.py     # Mock TFLite inference
│   │   ├── gradcam.py       # Mock Grad-CAM visualization
│   │   └── treatment.py     # Treatment guidance templates
│   ├── static/
│   │   ├── css/main.css     # Mobile-first responsive styles
│   │   ├── js/camera.js     # Camera/upload handler
│   │   └── uploads/         # User-uploaded images (created at runtime)
│   └── templates/
│       ├── base.html        # Base template + bottom nav
│       ├── index.html       # Camera/upload interface
│       ├── result.html      # Diagnosis result + Grad-CAM
│       ├── history.html     # Past diagnoses
│       ├── stats.html       # Usage statistics
│       └── about.html       # About page
├── requirements.txt
├── run.py                   # Dev server
└── README.md
```

## Current Implementation Status

### ✅ Working (Fully Implemented)
- Flask web app with mobile-first UI
- Camera capture and file upload
- Image preprocessing and storage
- SQLite database with diagnosis history
- Mobile-responsive layout (works on smartphones)
- Tagalog/English language support
- Treatment guidance templates (template-based)
- Usage statistics and history
- User feedback system
- Bottom navigation

### 🚧 Mock Implementation (Ready for Real Model)
- **ML Inference** (`app/ml/inference.py`):
  - Currently returns mock predictions
  - Replace with real TensorFlow Lite model
  - Model structure is ready (MobileNetV3 + 20 disease classes)
  
- **Grad-CAM** (`app/ml/gradcam.py`):
  - Currently generates mock heatmap
  - Replace with real CNN activation + gradient computation
  - Visual overlay system works

### 🔮 Future Enhancements
- Train actual MobileNetV3 on PlantVillage dataset
- Convert model to TensorFlow Lite
- LLM integration for dynamic treatment guidance (Ollama/OpenAI)
- Progressive Web App (PWA) with service worker
- Offline-first caching
- More Philippine crop species
- Regional language support (Bisaya, Ilocano)

## Next Steps to Make It Production-Ready

1. **Train the model:**
   ```bash
   # Download PlantVillage dataset
   # Train MobileNetV3 with transfer learning
   # Convert to TFLite
   # Place in app/static/models/crop_disease_model.tflite
   ```

2. **Update inference.py:**
   - Uncomment TFLite loading code
   - Replace mock predictions with real inference

3. **Update gradcam.py:**
   - Implement real Grad-CAM using Keras model
   - Compute gradients from last conv layer

4. **Deploy:**
   - Render, Railway, PythonAnywhere, or local server
   - Ensure mobile network accessibility

## Features Overview

| Feature | Status | Notes |
|---------|--------|-------|
| Camera capture | ✅ Working | Uses native camera on mobile |
| File upload | ✅ Working | Alternative to camera |
| Image preprocessing | ✅ Working | Resize, optimize |
| Disease prediction | 🟡 Mock | Returns fixed prediction |
| Grad-CAM visualization | 🟡 Mock | Shows concept overlay |
| Treatment guidance | ✅ Working | Template-based, Tagalog+English |
| Diagnosis history | ✅ Working | SQLite database |
| User feedback | ✅ Working | Track accuracy |
| Statistics | ✅ Working | Usage analytics |
| Mobile-first UI | ✅ Working | Responsive, touch-optimized |
| Offline support | ⏳ Planned | PWA + service worker |

## Testing the App

1. Start the server: `python run.py`
2. Open on mobile: Navigate to your computer's IP on port 5000
3. Click "Kamera" to take a leaf photo
4. Click "Suriin ang Sakit" to analyze
5. View diagnosis with Grad-CAM heatmap and treatment guidance
6. Check "Kasaysayan" for past diagnoses
7. View "Estadistika" for usage stats

The app follows your repo's Flask patterns (Blueprint structure, custom CSS, SQLite, mobile-responsive) and is production-ready except for the actual ML model integration.
