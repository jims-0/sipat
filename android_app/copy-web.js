// Copy ../pwa into www/ for the APK, without the parity test page and its photos.
const fs = require("fs");
const path = require("path");
const src = path.join(__dirname, "..", "pwa");
const dst = path.join(__dirname, "www");
const skip = new Set(["_parity", "parity.html", "README.md"]);
fs.rmSync(dst, { recursive: true, force: true });
fs.cpSync(src, dst, { recursive: true, filter: (p) => !skip.has(path.basename(p)) });
console.log("www/ <- pwa/");
