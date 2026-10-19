"""Sipat 2026-10-19 features: field test set, plots + follow-ups, library,
MAO printout, stats. The model is faked; these test what the app does with
its answers."""

import csv
import io
import sqlite3
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.ml.classes import CLASSES, NOT_A_LEAF, SLUGS, is_diagnosis
from app.ml.inference import interpret
from tests.conftest import jpeg_bytes

LABELS = list(CLASSES)
ROOT = Path(__file__).resolve().parent.parent


def _probs(winner, p):
    probs = np.full(len(LABELS), (1 - p) / (len(LABELS) - 1))
    probs[LABELS.index(winner)] = p
    return probs


@pytest.fixture
def model(monkeypatch):
    """The fake model answers whatever the test sets in model.answer."""
    class Fake:
        answer = ("Tomato___Early_blight", 0.88)
    fake = Fake()
    monkeypatch.setattr("app.routes.predict_disease",
                        lambda path: interpret(_probs(*fake.answer), LABELS, 0.6))
    monkeypatch.setattr("app.routes.generate_gradcam", lambda *a: False)
    return fake


def scan(client, **form):
    data = {"image": (io.BytesIO(jpeg_bytes()), "leaf.jpg"), **form}
    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return resp.get_json()["diagnosis_id"]


def row(app, diagnosis_id):
    with app.app_context():
        from app.models import get_diagnosis_by_id
        return get_diagnosis_by_id(diagnosis_id)


# --- field test set ---------------------------------------------------------

def test_thumbs_up_confirms_the_model_answer(app, client, model):
    d = scan(client)
    assert client.post(f"/api/feedback/{d}", json={"helpful": 1}).status_code == 200
    r = row(app, d)
    assert r["feedback"] == 1 and r["true_class_key"] == "Tomato___Early_blight"


def test_thumbs_down_records_the_real_disease(app, client, model):
    d = scan(client)
    resp = client.post(f"/api/feedback/{d}", json={"helpful": 0, "true_class_key": "Tomato___Late_blight"})
    assert resp.get_json()["true_class_key"] == "Tomato___Late_blight"
    assert row(app, d)["feedback"] == 0


def test_thumbs_down_without_knowing_leaves_no_label(app, client, model):
    d = scan(client)
    client.post(f"/api/feedback/{d}", json={"helpful": 0})
    assert row(app, d)["true_class_key"] is None


def test_unsure_result_can_still_be_labelled(app, client, model):
    model.answer = ("Corn___Common_rust", 0.4)
    d = scan(client)
    assert row(app, d)["status"] == "unsure"
    client.post(f"/api/feedback/{d}", json={"true_class_key": "Corn___Common_rust"})
    r = row(app, d)
    assert r["true_class_key"] == "Corn___Common_rust" and r["feedback"] == 1
    # a thumbs up on an unsure result does not invent a label
    d2 = scan(client)
    client.post(f"/api/feedback/{d2}", json={"helpful": 1})
    assert row(app, d2)["true_class_key"] is None


def test_feedback_rejects_unknown_class_and_missing_scan(client, model):
    d = scan(client)
    assert client.post(f"/api/feedback/{d}", json={"true_class_key": "Mango___rot"}).status_code == 400
    assert client.post("/api/feedback/9999", json={"helpful": 1}).status_code == 404
    assert client.post(f"/api/feedback/{d}", json={}).status_code == 400


def test_result_page_offers_the_label_picker(client, model):
    page = client.get(f"/diagnosis/{scan(client)}").get_data(as_text=True)
    assert 'id="true-class"' in page and "Late Blight sa Kamatis" in page and NOT_A_LEAF in page


def test_field_set_export_is_a_class_per_folder_zip(app, client, model):
    labelled = scan(client)
    scan(client)  # unlabelled: not exported
    client.post(f"/api/feedback/{labelled}", json={"helpful": 0, "true_class_key": "Potato___Late_blight"})
    resp = client.get("/export/field-set.zip")
    assert resp.status_code == 200 and resp.mimetype == "application/zip"
    with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
        photos = [n for n in zf.namelist() if n.endswith(".jpg")]
        table = list(csv.DictReader(io.StringIO(zf.read("labels.csv").decode())))
    assert len(photos) == 1 and photos[0].startswith("Potato___Late_blight/")
    assert table[0]["model_class"] == "Tomato___Early_blight" and table[0]["true_class"] == "Potato___Late_blight"


def test_stats_show_field_accuracy(client, model):
    right, wrong = scan(client), scan(client)
    client.post(f"/api/feedback/{right}", json={"helpful": 1})
    client.post(f"/api/feedback/{wrong}", json={"helpful": 0, "true_class_key": "Tomato___Late_blight"})
    page = client.get("/stats").get_data(as_text=True)
    assert "50.0%" in page and "Tama: 1 sa 2" in page and "field-set.zip" in page


# --- evaluate_field.py --------------------------------------------------------

sys.path.insert(0, str(ROOT / "scripts"))


def test_evaluate_field_report(tmp_path):
    import evaluate_field
    answers = {"a.jpg": ("Tomato___Late_blight", 0.9), "b.jpg": ("Tomato___Early_blight", 0.9),
               "c.jpg": ("Tomato___Late_blight", 0.3)}
    items = [(tmp_path / name, "Tomato___Late_blight") for name in answers]
    report = evaluate_field.evaluate(items, predict=lambda p: interpret(_probs(*answers[p.name]), LABELS, 0.6))
    assert report["photos"] == 3
    assert report["accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert report["answered_share"] == pytest.approx(2 / 3, abs=1e-3)
    assert report["accuracy_when_answered"] == 0.5
    late = report["per_class"][0]
    assert late["photos"] == 3 and late["unsure"] == 1 and late["enough_data"] is False
    assert report["top_mistakes"][0]["predicted"] == "Tomato___Early_blight"
    evaluate_field.write_markdown(report, tmp_path / "FIELD.md")
    assert "Tomato Late Blight" in (tmp_path / "FIELD.md").read_text(encoding="utf-8")


def test_evaluate_field_reads_the_app_zip_and_refuses_zip_slip(tmp_path):
    import evaluate_field
    good = tmp_path / "set.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("Corn___Common_rust/x.jpg", jpeg_bytes())
        zf.writestr("Not_a_class/y.jpg", jpeg_bytes())
        zf.writestr("labels.csv", "id\n")
    out = tmp_path / "out"
    out.mkdir()
    items = evaluate_field.load_items(good, str(out))
    assert [(p.name, k) for p, k in items] == [("x.jpg", "Corn___Common_rust")]

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../escape.jpg", b"x")
    with pytest.raises(SystemExit):
        evaluate_field.load_items(bad, str(out))


# --- plots, follow-ups, reminders -------------------------------------------

def test_new_plot_is_created_and_reused(app, client, model):
    a = scan(client, new_plot="Lote 2")
    b = scan(client, new_plot="lote 2")  # same plot, any case
    assert row(app, a)["plot_id"] == row(app, b)["plot_id"] is not None
    assert row(app, a)["plot_name"] == "Lote 2"
    page = client.get("/plots").get_data(as_text=True)
    assert "Lote 2" in page and "2 scan" in page


def test_unknown_plot_id_is_ignored(app, client, model):
    assert row(app, scan(client, plot_id="999"))["plot_id"] is None


def test_followup_links_to_its_first_scan_and_inherits_the_plot(app, client, model):
    first = scan(client, new_plot="Likod-bahay")
    model.answer = ("Tomato___healthy", 0.95)
    second = scan(client, parent_id=str(first))
    r = row(app, second)
    assert r["parent_id"] == first and r["plot_id"] == row(app, first)["plot_id"]
    after = client.get(f"/diagnosis/{second}").get_data(as_text=True)
    assert "Bago at Ngayon" in after and "Mukhang gumaling" in after
    before = client.get(f"/diagnosis/{first}").get_data(as_text=True)
    assert "Mga Follow-up" in before


def test_followup_that_got_worse_says_so(client, model):
    model.answer = ("Tomato___healthy", 0.95)
    first = scan(client)
    model.answer = ("Tomato___Late_blight", 0.95)
    page = client.get(f"/diagnosis/{scan(client, parent_id=str(first))}").get_data(as_text=True)
    assert "Lumala" in page


def test_index_prefills_a_followup(client, model):
    first = scan(client)
    page = client.get(f"/?followup={first}").get_data(as_text=True)
    assert f'id="parent-id" value="{first}"' in page


def test_recheck_reminder_shows_when_due_and_clears_after_rescan(app, client, model):
    d = scan(client)
    resp = client.post(f"/api/diagnosis/{d}/recheck", json={"days": 7})
    assert resp.status_code == 200 and resp.get_json()["recheck_due"]
    assert "Dapat Balikan" not in client.get("/").get_data(as_text=True)  # not due yet
    with app.app_context():
        from app.models import set_recheck
        set_recheck(d, "2026-01-01")  # overdue
    assert "Dapat Balikan" in client.get("/").get_data(as_text=True)
    scan(client, parent_id=str(d))
    assert "Dapat Balikan" not in client.get("/").get_data(as_text=True)


def test_recheck_validation_and_clearing(app, client, model):
    d = scan(client)
    assert client.post(f"/api/diagnosis/{d}/recheck", json={"days": 0}).status_code == 400
    assert client.post(f"/api/diagnosis/{d}/recheck", json={"days": "x"}).status_code == 400
    assert client.post("/api/diagnosis/999/recheck", json={"days": 3}).status_code == 404
    client.post(f"/api/diagnosis/{d}/recheck", json={"days": 3})
    client.post(f"/api/diagnosis/{d}/recheck", json={"days": None})
    assert row(app, d)["recheck_due"] is None


def test_deleting_a_plot_keeps_its_scans(app, client, model):
    d = scan(client, new_plot="Tabi ng ilog")
    plot_id = row(app, d)["plot_id"]
    client.post(f"/plots/{plot_id}/delete")
    assert row(app, d) is not None and row(app, d)["plot_id"] is None


# --- history ----------------------------------------------------------------

def test_history_filters(client, model):
    scan(client, new_plot="A")
    model.answer = ("Corn___Common_rust", 0.95)
    scan(client)
    model.answer = ("Corn___Common_rust", 0.3)
    scan(client)

    def items(qs):
        return client.get("/history" + qs).get_data(as_text=True).count('class="history-item"')
    assert items("") == 3
    assert items("?crop=corn") == 2
    assert items("?status=unsure") == 1
    assert items("?q=rust") == 2
    assert items("?q=Kamatis") == 1
    assert items("?disease=corn-common-rust") == 2
    assert items("?q=walangganito") == 0


def test_notes_are_saved_searchable_and_printed(client, model):
    d = scan(client)
    client.post(f"/api/diagnosis/{d}/notes", json={"notes": "10 halaman, nag-spray noong Lunes"})
    assert client.get("/history?q=nag-spray").get_data(as_text=True).count('class="history-item"') == 1
    assert "nag-spray noong Lunes" in client.get(f"/diagnosis/{d}/print").get_data(as_text=True)
    assert client.post("/api/diagnosis/999/notes", json={"notes": "x"}).status_code == 404


def test_delete_removes_the_scan_and_its_photo(app, client, model):
    first = scan(client)
    child = scan(client, parent_id=str(first))
    photo = Path(app.config["UPLOAD_FOLDER"]) / Path(row(app, first)["image_path"]).name
    assert photo.exists()
    resp = client.post(f"/diagnosis/{first}/delete")
    assert resp.status_code == 302
    assert row(app, first) is None and not photo.exists()
    assert row(app, child)["parent_id"] is None
    assert client.post(f"/diagnosis/{first}/delete").status_code == 404
    assert client.get(f"/diagnosis/{first}").status_code == 404


# --- MAO printout -------------------------------------------------------------

def test_print_page_both_languages(client, model):
    d = scan(client, new_plot="Lote 7")
    tl = client.get(f"/diagnosis/{d}/print").get_data(as_text=True)
    en = client.get(f"/diagnosis/{d}/print?lang=en").get_data(as_text=True)
    assert "Early Blight sa Kamatis" in tl and "Para sa MAO" in tl and "Lote 7" in tl
    assert "Tomato Early Blight" in en and "For the MAO" in en and "FPA" in en
    assert "**" not in tl and "bottom-nav" not in tl  # standalone page, markdown rendered
    assert client.get("/diagnosis/999/print").status_code == 404


# --- library ----------------------------------------------------------------

def test_library_lists_every_crop_disease_with_a_page(client):
    page = client.get("/library").get_data(as_text=True)
    for key, info in CLASSES.items():
        if is_diagnosis(key):
            assert info["name_tl"] in page, key
    for name, key in SLUGS.items():
        resp = client.get(f"/library/{name}")
        assert resp.status_code == (200 if is_diagnosis(key) else 404), name
    entry = client.get("/library/tomato-late-blight").get_data(as_text=True)
    assert "Phytophthora infestans" in entry and "FPA" in entry
    assert client.get("/library/no-such-thing").status_code == 404


def test_library_counts_your_scans(client, model):
    scan(client)
    page = client.get("/library/tomato-early-blight").get_data(as_text=True)
    assert "1 sa iyong mga scan" in page


# --- stats + misc -------------------------------------------------------------

def test_weekly_counts_cover_eight_weeks(app, client, model):
    from datetime import date
    scan(client)
    with app.app_context():
        from app.models import get_stats
        stats = get_stats()
    assert len(stats["by_week"]) == 8 and stats["by_week"][-1]["count"] == 1
    assert date.fromisoformat(stats["by_week"][-1]["week"]).weekday() == 0
    assert {s["status"] for s in stats["by_status"]} == {"ok"}


def test_empty_stats_page_renders(client):
    assert client.get("/stats").status_code == 200


def test_localtime_filter():
    from app import local_time
    assert local_time("not a date") == "not a date"
    assert local_time(None) == ""
    assert "2026" in local_time("2026-10-19 02:00:00")


def test_old_database_gains_plots_and_label_columns(tmp_path):
    from app import create_app
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE diagnoses (id INTEGER PRIMARY KEY, image_path TEXT NOT NULL, disease_name TEXT NOT NULL,"
                " disease_name_tl TEXT, confidence REAL NOT NULL, crop_type TEXT NOT NULL, gradcam_path TEXT,"
                " treatment TEXT, treatment_tl TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, feedback INTEGER)")
    con.commit()
    con.close()
    create_app({"TESTING": True, "DATABASE_URL": f"sqlite:///{db}", "UPLOAD_FOLDER": str(tmp_path / "u")})
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(diagnoses)")}
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert {"true_class_key", "plot_id", "notes", "recheck_due", "parent_id"} <= cols and "plots" in tables


def test_unknown_page_is_a_friendly_404(client):
    resp = client.get("/no-such-page")
    assert resp.status_code == 404 and "Hindi makita" in resp.get_data(as_text=True)
