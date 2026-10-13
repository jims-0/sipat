"""Tanim Guard: the decisions around the model, the guidance, and the screens.

The model itself is exercised at the bottom only when a trained model file
exists (scripts/train_model.py); everything else runs without TensorFlow work.
"""

import io
import json
import re

import numpy as np
import pytest

from app.ml import adviser, inference
from app.ml.classes import CLASSES, NOT_A_LEAF, OTHER_PLANT, is_diagnosis
from app.ml.inference import interpret
from app.ml.treatment import get_treatment_guidance
from app.ml.treatment_data import TREATMENTS
from tests.conftest import jpeg_bytes

LABELS = list(CLASSES)


def _probs(winner: str, p: float) -> np.ndarray:
    probs = np.full(len(LABELS), (1 - p) / (len(LABELS) - 1))
    probs[LABELS.index(winner)] = p
    return probs


# --- deciding what to tell the farmer ---------------------------------------

def test_confident_disease_is_a_diagnosis():
    out = interpret(_probs("Tomato___Late_blight", 0.93), LABELS, 0.6)
    assert out["status"] == "ok" and out["disease_name"] == "Tomato Late Blight"
    assert len(out["top3"]) == 3 and out["top3"][0]["confidence"] == pytest.approx(0.93)


def test_low_confidence_is_unsure_not_a_guess():
    out = interpret(_probs("Corn___Common_rust", 0.41), LABELS, 0.6)
    assert out["status"] == "unsure"


def test_not_a_leaf_and_other_plant_are_not_diagnoses():
    assert interpret(_probs(NOT_A_LEAF, 0.9), LABELS, 0.6)["status"] == "not_leaf"
    assert interpret(_probs(OTHER_PLANT, 0.8), LABELS, 0.6)["status"] == "other_plant"
    assert not is_diagnosis(NOT_A_LEAF) and not is_diagnosis(OTHER_PLANT)


def test_healthy_is_flagged():
    assert interpret(_probs("Potato___healthy", 0.95), LABELS, 0.6)["healthy"] is True


# --- guidance ---------------------------------------------------------------

def test_every_crop_class_has_bilingual_sourced_guidance():
    for key in LABELS:
        if not is_diagnosis(key):
            continue
        assert key in TREATMENTS, key
        t = get_treatment_guidance(key)
        assert t["en"].strip() and t["tl"].strip(), key
        assert TREATMENTS[key]["sources"], key


def test_guidance_never_invents_a_dose():
    """Doses belong to the product label. A number with a unit per volume in our
    text would be an unsourced dose - exactly what a panel would catch."""
    dose = re.compile(r"\d+(\.\d+)?\s*(g|ml|mL|tbsp|tablespoons?|kg|L)\b\s*(per|/)", re.I)
    for key in TREATMENTS:
        for lang in ("en", "tl"):
            assert not dose.search(get_treatment_guidance(key)[lang]), (key, lang)


def test_statuses_get_retake_or_unsupported_messages():
    assert "Retake" in get_treatment_guidance(None, "unsure")["en"]
    assert "single leaf" in get_treatment_guidance(None, "not_leaf")["en"]
    assert "tomato" in get_treatment_guidance(None, "other_plant")["en"].lower()


def test_markdown_filter_escapes_html():
    from app import markdown_lite
    html = str(markdown_lite("**Bold** <script>x</script>\n1. one\n2. two"))
    assert "<strong>Bold</strong>" in html and "<script>" not in html and "<ol>" in html


# --- the local adviser ------------------------------------------------------

def test_adviser_is_grounded_in_the_guidance_and_the_language():
    diag = {"class_key": "Tomato___Late_blight", "status": "ok", "confidence": 0.9}
    msgs = adviser.build_messages(diag, "Kailan ako mag-spray?", "tl")
    assert "Phytophthora infestans" in msgs[0]["content"]
    assert "Tagalog" in msgs[0]["content"] and "Never give a dose" in msgs[0]["content"]
    assert msgs[1]["content"] == "Kailan ako mag-spray?"


RUST = {"class_key": "Corn___Common_rust", "status": "ok", "confidence": 0.9}


def _model_says(text):
    calls = []

    def fake(request, timeout=None):
        calls.append(request)
        return io.BytesIO(json.dumps({"message": {"content": text}}).encode())
    fake.calls = calls
    return fake


def test_adviser_offline_never_raises():
    def down(*a, **k):
        raise OSError("connection refused")
    out = adviser.ask(RUST, "kailangan ko bang sabihan ang kapitbahay?", "tl", opener=down)
    assert out["offline"] and "Ollama" in out["answer"]


def test_a_dose_question_never_reaches_the_model():
    """Live 3B answered '500 gramo' to this. It must come from the label rule."""
    fake = _model_says("500 gramo")
    out = adviser.ask({"class_key": "Tomato___Late_blight", "status": "ok"},
                      "Ilang gramo ng mancozeb sa isang litro?", "tl", opener=fake)
    assert out["source"] == "guide" and "FPA" in out["answer"] and not fake.calls
    assert "500" not in out["answer"]


def test_common_questions_come_from_the_curated_text():
    late = {"class_key": "Tomato___Late_blight", "status": "ok"}
    assert "mancozeb" in adviser.ask(late, "Anong gamot ang puwede?", "tl", opener=_model_says("x"))["answer"]
    assert "pre-harvest" in adviser.ask(late, "Can I sell the fruit?", "en", opener=_model_says("x"))["answer"]
    virus = {"class_key": "Tomato___Tomato_mosaic_virus", "status": "ok"}
    # No spray exists for a virus: the answer falls back to what to do now.
    assert "Remove" in adviser.ask(virus, "what spray should I use", "en", opener=_model_says("x"))["answer"]


def test_kailangan_is_not_a_dose_question():
    assert adviser.intent_of("Kailangan ko bang sabihan ang kapitbahay?") is None


def test_model_answer_with_a_dose_or_a_loop_is_replaced():
    q = "kailangan ko bang sabihan ang kapitbahay?"
    out = adviser.ask(RUST, q, "tl", opener=_model_says("Maglagay ng 2 kutsara bawat galon."))
    assert out["source"] == "safe" and "MAO" in out["answer"]
    out = adviser.ask(RUST, q, "en", opener=_model_says("5.000 words\n" * 20))
    assert out["source"] == "safe"


def test_adviser_returns_a_clean_model_answer():
    out = adviser.ask(RUST, "kailangan ko bang sabihan ang kapitbahay?", "tl",
                      opener=_model_says("Oo, sabihan sila dahil tinatangay ng hangin ang spores."))
    assert out == {"answer": "Oo, sabihan sila dahil tinatangay ng hangin ang spores.",
                   "offline": False, "source": "model"}


# --- screens ----------------------------------------------------------------

@pytest.fixture
def fake_model(monkeypatch):
    def predict(path):
        return interpret(_probs("Tomato___Early_blight", 0.88), LABELS, 0.6)
    monkeypatch.setattr("app.routes.predict_disease", predict)
    monkeypatch.setattr("app.routes.generate_gradcam", lambda *a: False)


def test_upload_saves_and_shows_a_diagnosis(client, fake_model):
    resp = client.post("/api/predict", data={"image": (io.BytesIO(jpeg_bytes()), "leaf.jpg")},
                       content_type="multipart/form-data")
    data = resp.get_json()
    assert resp.status_code == 200 and data["status"] == "ok"
    assert data["image_url"].startswith("/uploads/")
    assert client.get(data["image_url"]).status_code == 200
    page = client.get(f"/diagnosis/{data['diagnosis_id']}").get_data(as_text=True)
    assert "Tomato Early Blight" in page and "Alternaria" in page and "Magtanong" in page
    assert "**" not in page  # markdown rendered, not dumped
    assert client.get("/history").status_code == 200
    assert client.get("/stats").status_code == 200


def test_upload_rejects_non_images(client, fake_model):
    resp = client.post("/api/predict", data={"image": (io.BytesIO(b"not an image"), "leaf.jpg")},
                       content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "Traceback" not in resp.get_data(as_text=True)


def test_missing_model_is_a_clear_503(client, monkeypatch):
    monkeypatch.setattr("app.routes.predict_disease",
                        lambda p: {"status": "error", "error": "Wala pang trained model."})
    resp = client.post("/api/predict", data={"image": (io.BytesIO(jpeg_bytes()), "leaf.jpg")},
                       content_type="multipart/form-data")
    assert resp.status_code == 503 and "model" in resp.get_json()["error"]


def test_ask_endpoint(client, fake_model, monkeypatch):
    monkeypatch.setattr("app.routes.ask", lambda d, q, l: {"answer": f"sagot: {q}", "offline": False})
    diag_id = client.post("/api/predict", data={"image": (io.BytesIO(jpeg_bytes()), "a.jpg")},
                          content_type="multipart/form-data").get_json()["diagnosis_id"]
    assert client.post(f"/api/ask/{diag_id}", json={"question": "ano?"}).get_json()["answer"] == "sagot: ano?"
    assert client.post(f"/api/ask/{diag_id}", json={"question": "  "}).status_code == 400
    assert client.post("/api/ask/99999", json={"question": "x"}).status_code == 404


def test_old_database_gains_the_new_columns(tmp_path):
    import sqlite3
    from app import create_app
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE diagnoses (id INTEGER PRIMARY KEY, image_path TEXT NOT NULL, disease_name TEXT NOT NULL,"
                " disease_name_tl TEXT, confidence REAL NOT NULL, crop_type TEXT NOT NULL, gradcam_path TEXT,"
                " treatment TEXT, treatment_tl TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, feedback INTEGER)")
    con.execute("INSERT INTO diagnoses (image_path, disease_name, confidence, crop_type) VALUES ('x', 'old', 0.9, 'tomato')")
    con.commit(); con.close()
    create_app({"TESTING": True, "DATABASE_URL": f"sqlite:///{db}", "UPLOAD_FOLDER": str(tmp_path / "u")})
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(diagnoses)")}
    assert {"status", "class_key", "top3_json"} <= cols


# --- the real model (only when trained) --------------------------------------

needs_model = pytest.mark.skipif(not inference.MODEL_PATH.exists(), reason="no trained model yet")


@needs_model
def test_saved_labels_match_the_class_table():
    assert json.loads((inference.MODEL_DIR / "labels.json").read_text()) == LABELS


@needs_model
def test_real_model_answers_with_a_known_status(tmp_path):
    path = tmp_path / "grey.jpg"
    path.write_bytes(jpeg_bytes(color=(128, 128, 128)))
    out = inference.predict_disease(path)
    assert out["status"] in ("ok", "unsure", "not_leaf", "other_plant")
    assert abs(sum(g["confidence"] for g in out["top3"])) <= 1.0001


@needs_model
def test_real_gradcam_writes_an_overlay(tmp_path):
    from app.ml.gradcam import generate_gradcam
    src = tmp_path / "leaf.jpg"
    src.write_bytes(jpeg_bytes(color=(40, 120, 40), size=(400, 300)))
    out = tmp_path / "cam.jpg"
    assert generate_gradcam(src, out, 0) and out.stat().st_size > 0


def test_no_leaf_or_unsupported_plant_never_goes_to_the_model():
    fake = _model_says("anything")
    for status in ("not_leaf", "other_plant"):
        out = adviser.ask({"class_key": None, "status": status}, "Anong gamot?", "tl", opener=fake)
        assert out["source"] == "guide" and "**" not in out["answer"]
    assert not fake.calls
