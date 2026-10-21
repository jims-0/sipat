"""The phone app (pwa/) must give the same answers as the desktop app.

- pwa/data + pwa/model are exactly what scripts/build_pwa.py makes from the
  Python sources (so nobody edited a copy by hand, and a retrain was rebuilt);
- the JS ports in pwa/sipat-core.js agree with the Python originals:
  interpret() on the same probabilities, the Pillow BILINEAR resize on the
  same pixels, and the field-set zip opens in Python's zipfile.

The JS checks run under Node; they are skipped if Node is not installed.
The browser-level check (real model, real photos) is scripts/pwa_parity.py.
"""

import base64
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.ml.inference import interpret
from scripts import build_pwa

ROOT = Path(__file__).resolve().parents[1]
PWA = ROOT / "pwa"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js not installed")


def run_js(code: str, tmp_path: Path) -> str:
    script = tmp_path / "run.js"
    script.write_text(f"const C = require({json.dumps(str(PWA / 'sipat-core.js'))});\n{code}", encoding="utf-8")
    return subprocess.run([NODE, str(script)], capture_output=True, text=True, check=True).stdout


def test_pwa_data_is_built_from_current_sources():
    on_disk = json.loads((PWA / "data" / "sipat.json").read_text(encoding="utf-8"))
    fresh = json.loads(json.dumps(build_pwa.build_data(), ensure_ascii=False))
    assert on_disk == fresh, "pwa/ is stale - run: python scripts/build_pwa.py"


def test_pwa_model_and_cam_weights_match_the_app():
    assert (PWA / "model" / "sipat.tflite").read_bytes() == (build_pwa.MODEL_DIR / "tanim_guard.tflite").read_bytes()
    weights = np.load(build_pwa.MODEL_DIR / "cam_weights.npy").astype("<f4")
    assert (PWA / "model" / "cam_weights.bin").read_bytes() == weights.tobytes()


def test_service_worker_cache_carries_the_model_version():
    assert f'const VERSION = "{build_pwa.model_version()}"' in (PWA / "sw.js").read_text(encoding="utf-8")


def test_service_worker_precaches_every_file_the_page_loads():
    sw = (PWA / "sw.js").read_text(encoding="utf-8")
    html = (PWA / "index.html").read_text(encoding="utf-8")
    for ref in re.findall(r'(?:src|href)="([^"#:]+)"', html):
        assert f'"{ref}"' in sw, f"{ref} is loaded by index.html but not cached for offline use"
    for path in re.findall(r'"((?:vendor|model|data|icons)/[^"]+)"', sw):
        assert (PWA / path).is_file(), f"sw.js caches {path}, which does not exist"


@needs_node
def test_js_interpret_matches_python(tmp_path):
    data = json.loads((PWA / "data" / "sipat.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(7)
    cases = []
    for i in range(60):
        logits = rng.normal(0, 1 + i % 5, len(data["labels"]))
        if i % 6 == 0:
            logits[data["labels"].index(data["not_a_leaf"])] += 8
        if i % 6 == 1:
            logits[data["labels"].index(data["other_plant"])] += 8
        p = np.exp(logits - logits.max())
        cases.append((p / p.sum()).round(6).tolist())
    out = json.loads(run_js(
        f"const data = require({json.dumps(str(PWA / 'data' / 'sipat.json'))});\n"
        f"console.log(JSON.stringify({json.dumps(cases)}.map(p => C.interpret(p, data))));", tmp_path))
    statuses = set()
    for probs, js in zip(cases, out):
        py = interpret(np.array(probs), data["labels"], data["threshold"])
        statuses.add(py["status"])
        assert (js["status"], js["class_key"], js["confidence"]) == (py["status"], py["class_key"], py["confidence"])
        assert [g["class"] for g in js["top3"]] == [g["class"] for g in py["top3"]]
    assert statuses == {"ok", "unsure", "not_leaf", "other_plant"}


@needs_node
@pytest.mark.parametrize("size", [(256, 256), (640, 480), (300, 1000), (180, 150)])
def test_js_resize_matches_pillow_bilinear(tmp_path, size):
    w, h = size
    rng = np.random.default_rng(w * h)
    # Smooth gradients + noise: like a photo, and exercises rounding.
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.stack([(xx * 255 / w), (yy * 255 / h), ((xx + yy) % 97) * 2.6], -1)
    rgb = np.clip(rgb + rng.normal(0, 20, rgb.shape), 0, 255).astype(np.uint8)
    rgba = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], -1)
    (tmp_path / "px.bin").write_bytes(rgba.tobytes())
    expected = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)
    out = run_js(
        f"const px = new Uint8ClampedArray(require('fs').readFileSync({json.dumps(str(tmp_path / 'px.bin'))}));\n"
        f"process.stdout.write(Buffer.from(C.resizeBilinear(px, {w}, {h}, 224, 224).buffer).toString('base64'));",
        tmp_path)
    got = np.frombuffer(base64.b64decode(out), dtype=np.float32).reshape(224, 224, 3)
    diff = np.abs(got - expected)
    assert diff.max() == 0, f"max diff {diff.max()} on {(diff > 0).mean():.2%} of values"


@needs_node
def test_js_field_set_zip_opens_in_python(tmp_path):
    target = tmp_path / "field.zip"
    run_js(
        "const enc = new TextEncoder();\n"
        "const z = C.zip([{name: 'Tomato___Late_blight/sipat_phone_1.jpg', bytes: new Uint8Array([255, 216, 1, 2, 3])},\n"
        "                 {name: 'labels.csv', bytes: enc.encode(C.csvRow(['id', 'note']) + '\\n' + C.csvRow([1, 'a, \"b\"']) + '\\n')}]);\n"
        f"require('fs').writeFileSync({json.dumps(str(target))}, z);", tmp_path)
    with zipfile.ZipFile(target) as zf:
        assert zf.testzip() is None
        assert zf.read("Tomato___Late_blight/sipat_phone_1.jpg") == bytes([255, 216, 1, 2, 3])
        assert zf.read("labels.csv").decode() == 'id,note\n1,"a, ""b"""\n'


@needs_node
def test_js_unit_tests_pass():
    result = subprocess.run([NODE, "--test", str(ROOT / "tests" / "pwa_core.test.js")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_flask_serves_the_phone_app_at_m(client):
    page = client.get("/m/")
    assert page.status_code == 200 and b"sipat-core.js" in page.data
    assert client.get("/m/model/sipat.tflite").status_code == 200
    assert client.get("/m/_parity/expected.json").status_code == 404
    assert client.get("/m/../run.py").status_code == 404
