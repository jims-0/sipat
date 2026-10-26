/* Sipat phone app - the parts with no DOM, so Node can test them (tests/pwa_core.test.js).
 *
 * Each function mirrors one in the Flask app, and says which. The data they
 * read (labels, threshold, treatment text, adviser keywords) is generated from
 * the Python sources by scripts/build_pwa.py, so the phone and the laptop give
 * the same answer for the same probabilities.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.SipatCore = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // app/ml/inference.py interpret(): probabilities -> what the app shows.
  function interpret(probs, data) {
    const order = Array.from(probs, (p, i) => i).sort((a, b) => probs[b] - probs[a]);
    const top3 = order.slice(0, 3).map((i) => {
      const key = data.labels[i];
      const c = data.classes[key] || { name: "Unknown", name_tl: "Hindi kilala", crop: "unknown" };
      return { class: key, name: c.name, name_tl: c.name_tl, crop: c.crop, healthy: !!c.healthy,
               // toFixed rounds the exact binary value, like Python's round(x, 4)
               confidence: Number(Number(probs[i]).toFixed(4)) };
    });
    const best = top3[0];
    const t = data.threshold;
    let status;
    if (best.class === data.not_a_leaf && best.confidence >= t) status = "not_leaf";
    else if (best.class === data.other_plant && best.confidence >= t) status = "other_plant";
    else if (best.confidence < t) status = "unsure";
    else status = "ok";
    return {
      status, class_key: best.class, class_idx: order[0],
      disease_name: best.name, disease_name_tl: best.name_tl, crop_type: best.crop,
      healthy: best.healthy, confidence: best.confidence, threshold: t, top3,
    };
  }

  // inference.py preprocess_image() resizes with Pillow's BILINEAR filter, which
  // (unlike a browser canvas) widens the filter when shrinking so every source
  // pixel counts. Same algorithm here, down to Pillow's 22-bit fixed-point
  // weights and its rounding (Resample.c: horizontal pass to 8 bits, then
  // vertical), so the model sees the same pixels as on the laptop.
  // rgba: Uint8ClampedArray [h * w * 4] -> Float32Array [oh * ow * 3] of 0-255.
  const PRECISION = 4194304; // 1 << 22
  const HALF = 2097152;      // 1 << 21
  const clip8 = (ss) => { const v = Math.floor(ss / PRECISION); return v < 0 ? 0 : v > 255 ? 255 : v; };

  function pilWeights(inSize, outSize) {
    const scale = inSize / outSize;
    const support = scale > 1 ? scale : 1;
    const ss = 1 / Math.max(scale, 1);
    const taps = [];
    for (let o = 0; o < outSize; o++) {
      const center = (o + 0.5) * scale;
      const lo = Math.max(Math.trunc(center - support + 0.5), 0);
      const hi = Math.min(Math.trunc(center + support + 0.5), inSize);
      const w = [];
      let sum = 0;
      for (let j = lo; j < hi; j++) {
        const x = Math.abs((j - center + 0.5) * ss);
        const v = x < 1 ? 1 - x : 0;
        w.push(v); sum += v;
      }
      // normalise, then to fixed point exactly as Pillow does (truncate after +-0.5)
      taps.push({ lo, w: w.map((v) => {
        const n = (sum ? v / sum : 0) * PRECISION;
        return Math.trunc(n < 0 ? n - 0.5 : n + 0.5);
      }) });
    }
    return taps;
  }

  function resizeBilinear(rgba, w, h, ow, oh) {
    const tx = pilWeights(w, ow), ty = pilWeights(h, oh);
    const mid = new Uint8Array(h * ow * 3);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < ow; x++) {
        const { lo, w: k } = tx[x];
        let r = HALF, g = HALF, b = HALF;
        for (let i = 0; i < k.length; i++) {
          const p = (y * w + lo + i) * 4;
          r += rgba[p] * k[i]; g += rgba[p + 1] * k[i]; b += rgba[p + 2] * k[i];
        }
        const q = (y * ow + x) * 3;
        mid[q] = clip8(r); mid[q + 1] = clip8(g); mid[q + 2] = clip8(b);
      }
    }
    const out = new Float32Array(oh * ow * 3);
    for (let y = 0; y < oh; y++) {
      const { lo, w: k } = ty[y];
      for (let x = 0; x < ow; x++) {
        let r = HALF, g = HALF, b = HALF;
        for (let i = 0; i < k.length; i++) {
          const q = ((lo + i) * ow + x) * 3;
          r += mid[q] * k[i]; g += mid[q + 1] * k[i]; b += mid[q + 2] * k[i];
        }
        const o = (y * ow + x) * 3;
        out[o] = clip8(r); out[o + 1] = clip8(g); out[o + 2] = clip8(b);
      }
    }
    return out;
  }

  // app/ml/gradcam.py heatmap(): CAM_c(x, y) = relu(sum_k w[k, c] * A_k(x, y)), scaled to 0..1.
  // features: [h * w * C] (NHWC, batch dropped), weights: [C * classes] row-major.
  function camMap(features, weights, classIdx, numClasses, channels) {
    const cells = features.length / channels;
    const cam = new Float32Array(cells);
    let peak = 0;
    for (let p = 0; p < cells; p++) {
      let s = 0;
      const base = p * channels;
      for (let k = 0; k < channels; k++) s += features[base + k] * weights[k * numClasses + classIdx];
      cam[p] = s > 0 ? s : 0;
      if (cam[p] > peak) peak = cam[p];
    }
    if (peak > 0) for (let p = 0; p < cells; p++) cam[p] /= peak;
    return cam;
  }

  // gradcam.py _colorize(): blue (low) -> yellow -> red (high).
  function colorize(v) {
    const ch = (c) => Math.round(Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - c))) * 255);
    return [ch(3), ch(2), ch(1)];
  }

  // app/ml/adviser.py retrieve(): recognised question -> curated text, word for word.
  // There is no local language model on a phone, so anything unrecognised gets
  // the safe "ask your MAO" answer instead of a model guess.
  function intentOf(question, data) {
    for (const [name, pattern] of data.adviser.intents) {
      if (new RegExp(pattern, "i").test(question)) return name;
    }
    return null;
  }

  function section(t, field, lang) {
    const v = t && t[field] && t[field][lang];
    if (v == null) return null;
    return Array.isArray(v) ? v.map((s, i) => `${i + 1}. ${s}`).join(" ") : v;
  }

  function ask(diagnosis, question, lang, data) {
    lang = lang === "en" ? "en" : "tl";
    question = (question || "").trim().slice(0, 400);
    if (!question) return { answer: "", source: "safe" };
    const dontKnow = data.adviser.dont_know[lang];
    if (diagnosis.status === "not_leaf" || diagnosis.status === "other_plant") {
      return { answer: data.status_messages[diagnosis.status][lang].replace(/\*\*/g, ""), source: "guide" };
    }
    const intent = intentOf(question, data);
    if (intent === "dose") {
      return { answer: data.label_rule[lang] + " " + dontKnow.split(". ").slice(1).join(". "), source: "guide" };
    }
    if (intent === "harvest") return { answer: data.harvest_rule[lang], source: "guide" };
    const t = data.treatments[diagnosis.class_key];
    if (!intent || !t) return { answer: dontKnow, source: "safe" };
    const text = section(t, intent, lang) || section(t, "do_now", lang);
    const label = data.adviser.labels[lang][intent];
    let answer = label ? `${label}: ${text}` : text;
    if (intent === "chemical" && t.chemical) answer += " " + data.label_rule[lang];
    return { answer, source: "guide" };
  }

  // templates/result.html trend(): what changed between a scan and its follow-up.
  function trend(before, after, data) {
    const b = data.classes[before.class_key] || {};
    const a = data.classes[after.class_key] || {};
    if (before.status !== "ok" || after.status !== "ok")
      return { kind: "same", text: 'Hindi maihambing: may "hindi sigurado" o walang dahon sa isa sa mga larawan.' };
    if (before.class_key === after.class_key)
      return a.healthy ? { kind: "better", text: "Malusog pa rin ✓" }
                       : { kind: "same", text: "Parehong sakit pa rin - ituloy ang gabay o magtanong sa MAO." };
    if (a.healthy && !b.healthy) return { kind: "better", text: "Mukhang gumaling ✓ - wala nang nakitang sakit." };
    if (b.healthy && !a.healthy) return { kind: "worse", text: "Lumala: may nakitang sakit na wala noon." };
    return { kind: "worse", text: "Ibang sakit na ang nakita - basahin ang bagong gabay." };
  }

  // Home-screen dashboard (the phone's version of models.get_dashboard).
  // scans: the stored scans; today: local "YYYY-MM-DD"; dayOf: ISO timestamp -> local "YYYY-MM-DD".
  const DAY = 864e5;
  const dayNum = (d) => Date.UTC(+d.slice(0, 4), +d.slice(5, 7) - 1, +d.slice(8, 10)) / DAY;
  const mondayOf = (d) => { const n = dayNum(d); return n - ((new Date(n * DAY).getUTCDay() + 6) % 7); };
  const isoOf = (n) => new Date(n * DAY).toISOString().slice(0, 10);

  function kindOf(scan, data) {
    if (scan.status !== "ok") return "unsure";
    return (data.classes[scan.class_key] || {}).healthy ? "healthy" : "sick";
  }

  function dashboard(scans, data, today, dayOf) {
    const counts = { healthy: 0, sick: 0, unsure: 0 };
    const diseases = {};
    for (const s of scans) {
      const k = kindOf(s, data);
      counts[k]++;
      if (k === "sick") diseases[s.class_key] = (diseases[s.class_key] || 0) + 1;
    }
    const due = scans.filter((s) => s.recheck_due && s.recheck_due <= today
      && !scans.some((f) => f.parent_id === s.id)).length;
    const top = Object.entries(diseases)
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 5)
      .map(([key, count]) => ({ key, count, name_tl: data.classes[key].name_tl, name: data.classes[key].name }));
    // Monday-start weeks, last 8, oldest first (same as models.get_stats).
    const thisMonday = mondayOf(today);
    const weeks = [];
    for (let i = 7; i >= 0; i--) weeks.push({ week: isoOf(thisMonday - 7 * i), count: 0 });
    for (const s of scans) {
      const w = weeks.find((x) => x.week === isoOf(mondayOf(dayOf(s.ts))));
      if (w) w.count++;
    }
    // Latest scan per plot; plots with a disease first.
    const latest = {};
    for (const s of scans) if (s.plot && (!latest[s.plot] || s.id > latest[s.plot].id)) latest[s.plot] = s;
    const order = { sick: 0, unsure: 1, healthy: 2 };
    const plots = Object.entries(latest).map(([name, s]) => ({ name, kind: kindOf(s, data), scan: s }))
      .sort((a, b) => order[a.kind] - order[b.kind] || a.name.localeCompare(b.name));
    // Last 3 scans, newest first.
    const recent = [...scans].sort((a, b) => b.id - a.id).slice(0, 3).map((s) => ({ scan: s, kind: kindOf(s, data) }));
    // "This week" = since Monday: how many scans, how many sick, the most common disease.
    const weekSick = {};
    let weekTotal = 0;
    for (const s of scans) {
      if (mondayOf(dayOf(s.ts)) !== thisMonday) continue;
      weekTotal++;
      if (kindOf(s, data) === "sick") weekSick[s.class_key] = (weekSick[s.class_key] || 0) + 1;
    }
    const topWeek = Object.entries(weekSick).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0];
    const week = { total: weekTotal, sick: Object.values(weekSick).reduce((a, b) => a + b, 0),
                   top: topWeek && data.classes[topWeek[0]] ? data.classes[topWeek[0]].name_tl : null };
    return { total: scans.length, ...counts, due, top, weeks, plots, recent, week };
  }

  // The small markdown subset treatment.py writes: **bold**, _italic_, "1. " lists, blank-line paragraphs.
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function inline(s) {
    return escapeHtml(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/^_(.+)_$/, "<em>$1</em>");
  }

  function markdown(md) {
    const out = [];
    let list = null;
    for (const line of String(md).split("\n")) {
      const item = line.match(/^\d+\.\s+(.*)$/);
      if (item) {
        (list = list || []).push(`<li>${inline(item[1])}</li>`);
        continue;
      }
      if (list) { out.push(`<ol>${list.join("")}</ol>`); list = null; }
      if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
    if (list) out.push(`<ol>${list.join("")}</ol>`);
    return out.join("\n");
  }

  // A store-only zip (photos are already JPEG, so deflate would save nothing).
  // Same layout as the desktop /export/field-set.zip: <true class>/<photo>.jpg + labels.csv,
  // which is what scripts/evaluate_field.py reads.
  const CRC_TABLE = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      t[n] = c >>> 0;
    }
    return t;
  })();

  function crc32(bytes) {
    let c = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }

  function zip(files) { // [{name, bytes: Uint8Array}] -> Uint8Array
    const enc = new TextEncoder();
    const parts = [], central = [];
    let offset = 0;
    for (const f of files) {
      const name = enc.encode(f.name);
      const crc = crc32(f.bytes);
      const local = new DataView(new ArrayBuffer(30));
      local.setUint32(0, 0x04034b50, true); local.setUint16(4, 20, true); local.setUint16(6, 0x0800, true);
      local.setUint32(14, crc, true); local.setUint32(18, f.bytes.length, true); local.setUint32(22, f.bytes.length, true);
      local.setUint16(26, name.length, true);
      const cd = new DataView(new ArrayBuffer(46));
      cd.setUint32(0, 0x02014b50, true); cd.setUint16(4, 20, true); cd.setUint16(6, 20, true); cd.setUint16(8, 0x0800, true);
      cd.setUint32(16, crc, true); cd.setUint32(20, f.bytes.length, true); cd.setUint32(24, f.bytes.length, true);
      cd.setUint16(28, name.length, true); cd.setUint32(42, offset, true);
      parts.push(new Uint8Array(local.buffer), name, f.bytes);
      central.push(new Uint8Array(cd.buffer), name);
      offset += 30 + name.length + f.bytes.length;
    }
    const cdSize = central.reduce((n, p) => n + p.length, 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
    end.setUint32(12, cdSize, true); end.setUint32(16, offset, true);
    const all = [...parts, ...central, new Uint8Array(end.buffer)];
    const out = new Uint8Array(all.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of all) { out.set(p, at); at += p.length; }
    return out;
  }

  function csvRow(values) {
    return values.map((v) => {
      const s = v == null ? "" : String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    }).join(",");
  }

  return { interpret, resizeBilinear, camMap, colorize, intentOf, ask, trend, dashboard, markdown, escapeHtml, crc32, zip, csvRow };
});
