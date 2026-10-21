"""Copy what the phone app needs out of the Flask app into pwa/.

The mobile version (pwa/) runs the SAME .tflite model in the phone's browser,
so it works in the field with no laptop and no signal once installed. It must
not drift from the desktop app, so nothing in pwa/data or pwa/model is written
by hand - this script regenerates it from the one source of truth:

  app/ml/model/tanim_guard.tflite      -> pwa/model/sipat.tflite
  app/ml/model/cam_weights.npy         -> pwa/model/cam_weights.bin  (float32, [576, 21] row-major)
  labels, classes, threshold, treatment text, adviser keywords, test-set numbers
                                       -> pwa/data/sipat.json
  app/static/library/*.jpg             -> pwa/library/

tests/test_pwa.py fails if pwa/ is out of date, so run this after retraining
or editing treatment_data.py:   python scripts/build_pwa.py
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ml import adviser  # noqa: E402
from app.ml.classes import CLASSES, CROPS, NOT_A_LEAF, OTHER_PLANT, is_diagnosis, slug  # noqa: E402
from app.ml.treatment import STATUS_MESSAGES, get_treatment_guidance  # noqa: E402
from app.ml.treatment_data import HARVEST_RULE, LABEL_RULE, TREATMENTS  # noqa: E402

MODEL_DIR = ROOT / "app" / "ml" / "model"
LIBRARY_DIR = ROOT / "app" / "static" / "library"
PWA = ROOT / "pwa"


def model_version() -> str:
    """Short hash of the model + data - the service worker cache key, so a
    retrained model or edited guidance reaches phones instead of staying cached."""
    h = hashlib.sha256((MODEL_DIR / "tanim_guard.tflite").read_bytes())
    h.update(json.dumps(TREATMENTS, sort_keys=True).encode())
    return h.hexdigest()[:12]


def build_data() -> dict:
    labels = json.loads((MODEL_DIR / "labels.json").read_text())
    report = json.loads((MODEL_DIR / "evaluation_report.json").read_text())
    training = json.loads((MODEL_DIR / "training_info.json").read_text())
    threshold = report["recommended_threshold"]
    at_threshold = next((t for t in report.get("thresholds", []) if t.get("threshold") == threshold), None)
    return {
        "model_version": model_version(),
        "img_size": training.get("image_size", 224),
        "labels": labels,
        "threshold": threshold,
        "not_a_leaf": NOT_A_LEAF,
        "other_plant": OTHER_PLANT,
        "classes": {k: {**v, "slug": slug(k), "diagnosis": is_diagnosis(k)} for k, v in CLASSES.items()},
        "crops": CROPS,
        # Pre-rendered markdown, exactly what the desktop result page shows.
        "guidance": {k: get_treatment_guidance(k) for k in TREATMENTS},
        "status_messages": STATUS_MESSAGES,
        # Raw fields for the offline adviser (same retrieval as app/ml/adviser.py).
        "treatments": TREATMENTS,
        "label_rule": LABEL_RULE,
        "harvest_rule": HARVEST_RULE,
        "adviser": {
            "intents": adviser.INTENTS,
            "labels": adviser._LABELS,
            "dont_know": adviser.DONT_KNOW,
        },
        "evaluation": {
            "test_photos": report["test_photos"],
            "accuracy": report["accuracy"],
            "top3_accuracy": report["top3_accuracy"],
            "macro_f1": report["macro_f1"],
            "at_threshold": at_threshold,
            "per_class": report["per_class"],
            "architecture": training.get("architecture"),
            "dataset": training.get("dataset"),
        },
        "library": sorted(p.stem for p in LIBRARY_DIR.glob("*.jpg")),
    }


def build() -> None:
    for sub in ("data", "model", "library"):
        (PWA / sub).mkdir(parents=True, exist_ok=True)

    (PWA / "data" / "sipat.json").write_text(json.dumps(build_data(), ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    shutil.copyfile(MODEL_DIR / "tanim_guard.tflite", PWA / "model" / "sipat.tflite")
    weights = np.load(MODEL_DIR / "cam_weights.npy").astype("<f4")
    (PWA / "model" / "cam_weights.bin").write_bytes(np.ascontiguousarray(weights).tobytes())
    for old in (PWA / "library").glob("*.jpg"):
        old.unlink()
    for thumb in LIBRARY_DIR.glob("*.jpg"):
        shutil.copyfile(thumb, PWA / "library" / thumb.name)

    # The service worker's cache name carries the version (see sw.js).
    sw = PWA / "sw.js"
    text = sw.read_text(encoding="utf-8")
    start = text.index('const VERSION = "') + len('const VERSION = "')
    end = text.index('"', start)
    sw.write_text(text[:start] + model_version() + text[end:], encoding="utf-8")
    print(f"pwa/ rebuilt (version {model_version()}, {len(list(LIBRARY_DIR.glob('*.jpg')))} library photos)")


if __name__ == "__main__":
    build()
