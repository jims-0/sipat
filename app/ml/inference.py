"""Leaf photo -> diagnosis, using the model trained by scripts/train_model.py.

There used to be a mock here that answered "Tomato Late Blight, 92%" for every
photo. Now: the real model, its top three guesses, and four possible outcomes:

  status "ok"          a crop diagnosis the model is confident about
  status "unsure"      best guess below the confidence threshold -> retake photo
  status "not_leaf"    the model sees no leaf at all
  status "other_plant" a leaf, but of a plant Tanim Guard was not trained on

The app runs the TensorFlow Lite export (scripts/export_tflite.py) with LiteRT,
so it needs a few MB of runtime instead of TensorFlow. The threshold comes from
app/ml/model/evaluation_report.json (measured on the test set by
scripts/evaluate.py), so it is a number we can defend, not a guess.
"""

import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.ml.classes import NOT_A_LEAF, OTHER_PLANT, info

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "tanim_guard.tflite"
IMG_SIZE = 224
DEFAULT_THRESHOLD = 0.6

_lock = threading.Lock()
_state = {}


class ModelMissing(RuntimeError):
    pass


def load() -> dict:
    """Interpreter, labels, threshold and CAM weights - loaded once."""
    with _lock:
        if not _state:
            if not MODEL_PATH.exists():
                raise ModelMissing("Wala pang trained model. Patakbuhin ang scripts/train_model.py "
                                   "at scripts/export_tflite.py (tingnan ang TRAINING.md).")
            from ai_edge_litert.interpreter import Interpreter

            interp = Interpreter(model_path=str(MODEL_PATH))
            interp.allocate_tensors()
            outputs = interp.get_output_details()
            report = MODEL_DIR / "evaluation_report.json"
            _state.update(
                interp=interp,
                input=interp.get_input_details()[0]["index"],
                probs=next(d["index"] for d in outputs if len(d["shape"]) == 2),
                features=next(d["index"] for d in outputs if len(d["shape"]) == 4),
                labels=json.loads((MODEL_DIR / "labels.json").read_text()),
                cam_weights=np.load(MODEL_DIR / "cam_weights.npy"),
                threshold=(json.loads(report.read_text()).get("recommended_threshold", DEFAULT_THRESHOLD)
                           if report.exists() else DEFAULT_THRESHOLD),
            )
    return _state


def preprocess_image(image_path) -> np.ndarray:
    """Same as training: RGB, 224x224, 0-255 floats (the model rescales itself).
    exif_transpose so a phone photo taken sideways is not fed in sideways."""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32)[None, ...]


def run(image_path) -> tuple[np.ndarray, np.ndarray]:
    """(probabilities [classes], feature map [h, w, C]) for one photo."""
    s = load()
    x = preprocess_image(image_path)
    with _lock:  # one interpreter, shared - calls must not interleave
        s["interp"].set_tensor(s["input"], x)
        s["interp"].invoke()
        probs = s["interp"].get_tensor(s["probs"])[0].copy()
        features = s["interp"].get_tensor(s["features"])[0].copy()
    return probs, features


def interpret(probs, labels, threshold) -> dict:
    """Pure: probabilities -> the answer the app shows. Tested without a model."""
    order = np.argsort(probs)[::-1]
    top = [{"class": labels[i], **info(labels[i]), "confidence": round(float(probs[i]), 4)}
           for i in order[:3]]
    best = top[0]
    if best["class"] == NOT_A_LEAF and best["confidence"] >= threshold:
        status = "not_leaf"
    elif best["class"] == OTHER_PLANT and best["confidence"] >= threshold:
        status = "other_plant"
    elif best["confidence"] < threshold:
        status = "unsure"
    else:
        status = "ok"
    return {
        "status": status,
        "class_key": best["class"],
        "class_idx": int(order[0]),
        "disease_name": best["name"],
        "disease_name_tl": best["name_tl"],
        "crop_type": best["crop"],
        "healthy": bool(best.get("healthy")),
        "confidence": best["confidence"],
        "threshold": threshold,
        "top3": top,
        "error": None,
    }


def predict_disease(image_path) -> dict:
    try:
        probs, _features = run(image_path)
        s = load()
        return interpret(probs, s["labels"], s["threshold"])
    except ModelMissing as exc:
        return {"status": "error", "error": str(exc)}
    except Exception:  # unreadable image, corrupt upload
        return {"status": "error",
                "error": "Hindi mabasa ang larawan. Subukang kumuha ulit ng JPG o PNG."}
