"""Does the phone app answer exactly like the desktop app? Measure it.

    python scripts/pwa_parity.py            # 5 test-split photos per class (105)
    python -m http.server 8765 --bind 127.0.0.1 -d pwa
    open http://127.0.0.1:8765/parity.html  (desktop Chrome, or a phone on the same network)

Step 1 runs the desktop pipeline (LiteRT, app/ml/inference.py) on held-out
test photos and saves its answers to pwa/_parity/expected.json together with
copies of the photos. parity.html then runs the PHONE pipeline (the same
.tflite in the browser, pwa/app.js) on the same files and reports how often
the class and status agree and how far the probabilities differ.

pwa/_parity/ is git-ignored and never cached by the service worker.
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ml import inference  # noqa: E402
from scripts.tg_data import split_files  # noqa: E402

OUT = ROOT / "pwa" / "_parity"


def main(per_class: int = 5) -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    state = inference.load()
    taken, rows = {}, []
    for path, label in split_files()["test"]:
        if taken.get(label, 0) >= per_class:
            continue
        taken[label] = taken.get(label, 0) + 1
        name = f"{len(rows):03d}{Path(path).suffix.lower()}"
        shutil.copyfile(path, OUT / name)
        probs, _ = inference.run(path)
        d = inference.interpret(probs, state["labels"], state["threshold"])
        rows.append({"file": name, "true": state["labels"][label], "class_key": d["class_key"],
                     "status": d["status"], "confidence": d["confidence"],
                     "probs": [float(p) for p in probs]})
    (OUT / "expected.json").write_text(json.dumps(rows))
    print(f"{len(rows)} photos -> {OUT.relative_to(ROOT)}. Now serve pwa/ and open /parity.html")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
