# Sipat phone app (pwa/)

The same trained model as the desktop app, running **inside the phone's browser**.
No laptop, no server and no signal needed once it is installed: the model
(`model/sipat.tflite`, 3.8 MB), the TFLite runtime (`vendor/`, WebAssembly),
the treatment guidance and the library photos are all cached on the phone.
Scan history stays on the phone (IndexedDB); no photo is uploaded anywhere.

Features: camera / gallery scan, top-3 guesses, heat map (CAM), "not sure" /
"no leaf" / "unsupported plant" handling, Tagalog + English guidance, the
retrieval adviser (curated answers only, never a dose), plots, follow-up
re-scans with before/after trend, recheck reminders, notes, disease library,
"was this right?" labels, and **field test set export** (.zip in the same
layout as the desktop export, so `scripts/evaluate_field.py` reads it).

What it does not have: the Ollama free-text adviser (no local LLM on a phone;
unrecognised questions get the safe "ask your MAO" answer), the A4 printout,
and the stats page.

## Does it answer like the desktop app? (measured)

`python scripts/pwa_parity.py`, then open `/parity.html` (2026-10-22, Chrome
on Windows, 105 held-out test photos, 5 per class):

| | |
|---|---|
| same top class as desktop | 105 / 105 |
| same status (ok / unsure / not leaf / other) | 105 / 105 |
| largest probability difference | 0.0000024 |
| correct vs true label | 103 / 105 (desktop: 103 / 105) |
| time per photo | ~40-80 ms on a laptop CPU |

This needed an exact port of Pillow's BILINEAR resize
(`sipat-core.js resizeBilinear`): with the browser's own canvas resize the
probabilities differed by up to 0.15. `tests/test_pwa.py` checks the port
pixel-for-pixel against Pillow, and that `interpret()` gives the same status,
class and confidence as the Python version.

Run the parity page on a real phone before quoting phone timings.

## Run it

| Where | How | Offline install? |
|---|---|---|
| Laptop, testing | `python -m http.server 8765 --bind 127.0.0.1 -d pwa` → http://127.0.0.1:8765 | yes (localhost counts as secure) |
| Phone on the same Wi-Fi as a PC running Sipat | open `http://<PC address>:5000/m/` (desktop.py prints it) | **no** - works while connected |
| Phone anywhere | host `pwa/` on any HTTPS static host (GitHub Pages, Netlify, Cloudflare Pages), open the link once, then "Add to Home screen" | yes |

Browsers only allow the offline cache (service worker) on HTTPS or localhost,
which is why the Wi-Fi route cannot install.

## After changing the model or the guidance

```
python scripts/build_pwa.py      # regenerates data/, model/, library/ and the cache version in sw.js
python -m pytest tests/test_pwa.py
```

Changed `app.js`, `style.css` or `index.html`? Bump `APP_REV` in `sw.js`, or
installed phones keep the old copy.

## Files

- `index.html`, `app.js`, `style.css` - the app (`style.css` starts as a copy of the desktop CSS)
- `sipat-core.js` - logic with no DOM (decision rule, resize, CAM, adviser, zip); tested under Node
- `sw.js` - offline cache; `manifest.webmanifest`, `icons/` - home-screen install
- `data/`, `model/`, `library/` - **generated** by `scripts/build_pwa.py`, do not edit
- `vendor/` - TensorFlow.js core 4.9.0, CPU backend 4.9.0 and tfjs-tflite 0.0.1-alpha.10 (Apache-2.0), copied from npm so nothing loads from a CDN
- `parity.html` - phone vs desktop check (needs `scripts/pwa_parity.py` first)
