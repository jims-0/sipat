// Unit tests for pwa/sipat-core.js - run: node --test tests/pwa_core.test.js
// (tests/test_pwa.py runs this too, and checks the JS against the Python code).
const test = require("node:test");
const assert = require("node:assert");
const path = require("node:path");
const C = require(path.join(__dirname, "..", "pwa", "sipat-core.js"));
const DATA = require(path.join(__dirname, "..", "pwa", "data", "sipat.json"));

const late = { status: "ok", class_key: "Tomato___Late_blight" };

test("adviser never gives a dose - it points to the label and the MAO", () => {
  for (const q of ["ilang gramo ng mancozeb sa isang litro?", "how much fungicide do I mix", "gaano karami"]) {
    const a = C.ask(late, q, "tl", DATA);
    assert.strictEqual(a.source, "guide");
    assert.ok(a.answer.startsWith(DATA.label_rule.tl), q);
    assert.doesNotMatch(a.answer, /\d+\s*(g|ml|gramo|litro)\b/i);
  }
});

test("adviser answers recognised questions from the curated text, word for word", () => {
  const a = C.ask(late, "Paano maiwasan?", "tl", DATA);
  const first = DATA.treatments.Tomato___Late_blight.prevention.tl[0];
  assert.ok(a.answer.includes(first));
  assert.strictEqual(C.ask(late, "can I sell the fruit?", "en", DATA).answer, DATA.harvest_rule.en);
});

test("short Tagalog keywords need word boundaries ('ilan' inside 'kailangan')", () => {
  assert.notStrictEqual(C.intentOf("kailangan ko bang mag-spray?", DATA), "dose");
  assert.strictEqual(C.intentOf("ilan ang ilalagay?", DATA), "dose");
});

test("unrecognised question -> safe answer, not a guess", () => {
  const a = C.ask(late, "anong kulay ng langit", "tl", DATA);
  assert.strictEqual(a.source, "safe");
  assert.strictEqual(a.answer, DATA.adviser.dont_know.tl);
});

test("not a leaf / other plant -> the fixed guidance, whatever the question", () => {
  const a = C.ask({ status: "not_leaf", class_key: DATA.not_a_leaf }, "ano ang gamot?", "en", DATA);
  assert.ok(a.answer.startsWith("Leaf not clear."));
});

test("CAM: relu, scaled so the peak is 1, uses the predicted class column", () => {
  // 2 cells x 2 channels, 2 classes; weights[k * classes + c]
  const features = new Float32Array([1, 0, 0, 1]);
  const weights = new Float32Array([2, -1, -1, 3]);
  assert.deepStrictEqual(Array.from(C.camMap(features, weights, 0, 2, 2)), [1, 0]);
  assert.deepStrictEqual(Array.from(C.camMap(features, weights, 1, 2, 2)), [0, 1]);
  assert.deepStrictEqual(Array.from(C.camMap(new Float32Array(4), weights, 0, 2, 2)), [0, 0]);
});

test("colorize: low is blue, high is red", () => {
  assert.deepStrictEqual(C.colorize(0), [0, 0, 128]);
  assert.deepStrictEqual(C.colorize(1), [128, 0, 0]);
});

test("trend between a scan and its follow-up", () => {
  const healthy = { status: "ok", class_key: "Tomato___healthy" };
  assert.strictEqual(C.trend(late, healthy, DATA).kind, "better");
  assert.strictEqual(C.trend(healthy, late, DATA).kind, "worse");
  assert.strictEqual(C.trend(late, late, DATA).kind, "same");
  assert.strictEqual(C.trend(late, { status: "unsure", class_key: "Tomato___healthy" }, DATA).kind, "same");
});

test("markdown escapes HTML and renders the treatment subset", () => {
  const html = C.markdown("**Ano ito:** <b>x</b>\n\n1. una\n2. dalawa\n\n_Batayan: UC IPM_");
  assert.ok(html.includes("<strong>Ano ito:</strong> &lt;b&gt;x&lt;/b&gt;"));
  assert.ok(html.includes("<ol><li>una</li><li>dalawa</li></ol>"));
  assert.ok(html.includes("<em>Batayan: UC IPM</em>"));
});

test("every diagnosis class has guidance in both languages", () => {
  for (const [key, c] of Object.entries(DATA.classes)) {
    if (!c.diagnosis) continue;
    assert.ok(DATA.guidance[key].tl && DATA.guidance[key].en, key);
  }
});

test("crc32 known value", () => {
  assert.strictEqual(C.crc32(new TextEncoder().encode("123456789")), 0xcbf43926);
});

test("dashboard: counts, top diseases, weeks and plots (sick plots first)", () => {
  const ts = (d) => `${d}T04:00:00.000Z`;
  const dayOf = (iso) => iso.slice(0, 10);
  const scans = [
    { id: 1, ts: ts("2026-10-20"), status: "ok", class_key: "Tomato___Late_blight", plot: "Lote A", recheck_due: "2026-10-21" },
    { id: 2, ts: ts("2026-10-21"), status: "ok", class_key: "Tomato___Late_blight", plot: "Lote B" },
    { id: 3, ts: ts("2026-10-22"), status: "ok", class_key: "Tomato___healthy", plot: "Lote B" },
    { id: 4, ts: ts("2026-10-13"), status: "unsure", class_key: "Corn___Common_rust" },
  ];
  const d = C.dashboard(scans, DATA, "2026-10-22", dayOf);
  assert.deepStrictEqual([d.total, d.healthy, d.sick, d.unsure, d.due], [4, 1, 2, 1, 1]);
  assert.deepStrictEqual(d.top.map((t) => [t.key, t.count]), [["Tomato___Late_blight", 2]]);
  assert.strictEqual(d.weeks.length, 8);
  assert.deepStrictEqual(d.weeks.slice(-2), [{ week: "2026-10-12", count: 1 }, { week: "2026-10-19", count: 3 }]);
  assert.deepStrictEqual(d.plots.map((p) => [p.name, p.kind]), [["Lote A", "sick"], ["Lote B", "healthy"]]);
  assert.deepStrictEqual(d.recent.map((r) => [r.scan.id, r.kind]), [[4, "unsure"], [3, "healthy"], [2, "sick"]]);
  assert.deepStrictEqual(d.week, { total: 3, sick: 2, top: DATA.classes.Tomato___Late_blight.name_tl });
});
