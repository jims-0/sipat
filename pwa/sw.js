/* Offline cache. Everything the app needs - the model, the TFLite runtime, the
 * treatment text, the library photos - is stored on the phone at install, so a
 * farmer can scan in the field with no signal.
 *
 * VERSION is rewritten by scripts/build_pwa.py from the model + guidance hash:
 * a retrained model gets a new cache, and the old one is deleted on activate.
 * Bump APP_REV by hand when you change app.js / style.css / index.html.
 */
const VERSION = "931d2fddfa74";
const APP_REV = "7";
const CACHE = `sipat-${VERSION}-${APP_REV}`;

const CORE = [
  "./", "index.html", "style.css", "app.js", "sipat-core.js", "manifest.webmanifest",
  "icons/icon-192.png", "icons/icon-512.png", "icons/icon-maskable-512.png",
  "fonts/outfit-latin-wght.woff2",
  "data/sipat.json", "model/sipat.tflite", "model/cam_weights.bin",
  "vendor/tf-core.min.js", "vendor/tf-backend-cpu.min.js", "vendor/tf-tflite.min.js",
  "vendor/tflite_web_api_client.js",
  "vendor/tflite_web_api_cc.js", "vendor/tflite_web_api_cc.wasm",
  "vendor/tflite_web_api_cc_simd.js", "vendor/tflite_web_api_cc_simd.wasm",
];

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    await cache.addAll(CORE);
    const data = await (await cache.match("data/sipat.json")).json();
    await cache.addAll(data.library.map((name) => `library/${name}.jpg`));
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) {
      if (key.startsWith("sipat-") && key !== CACHE) await caches.delete(key);
    }
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  event.respondWith((async () => {
    const cached = await caches.match(event.request, { ignoreSearch: true });
    return cached || fetch(event.request);
  })());
});
