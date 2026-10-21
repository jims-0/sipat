/* Sipat on a phone: the same trained model as the desktop app, run in the browser.
 *
 * Nothing is uploaded. The photo, the model and the history all stay on the
 * phone (IndexedDB). Screens are hash routes so the whole app is one cached page:
 *   #/                 scan (camera / gallery), recheck reminders, recent scans
 *   #/scan/<id>        result: diagnosis, heat map, guidance, adviser, label, notes
 *   #/history          all scans, filter by plot / status
 *   #/library[/slug]   disease library
 *   #/about            how it was tested, export field test set, install help
 */
(function () {
  "use strict";
  const C = window.SipatCore;
  const $ = (sel, el = document) => el.querySelector(sel);
  const view = $("#view");
  const esc = C.escapeHtml;

  let DATA = null;       // data/sipat.json (built by scripts/build_pwa.py)
  let CAMW = null;       // Float32Array [576 * 21]
  let modelPromise = null;
  let lang = "tl";
  try { lang = localStorage.getItem("sipat-lang") || "tl"; } catch (e) { /* private mode */ }

  // ---------- storage (IndexedDB) ----------
  let dbPromise = null;
  function db() {
    dbPromise = dbPromise || new Promise((resolve, reject) => {
      const req = indexedDB.open("sipat", 1);
      req.onupgradeneeded = () => req.result.createObjectStore("scans", { keyPath: "id", autoIncrement: true });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    return dbPromise;
  }
  async function tx(mode, fn) {
    const d = await db();
    return new Promise((resolve, reject) => {
      const t = d.transaction("scans", mode);
      const req = fn(t.objectStore("scans"));
      t.oncomplete = () => resolve(req && req.result);
      t.onerror = () => reject(t.error);
    });
  }
  const getScan = (id) => tx("readonly", (s) => s.get(Number(id)));
  const putScan = (scan) => tx("readwrite", (s) => s.put(scan));
  const deleteScan = (id) => tx("readwrite", (s) => s.delete(Number(id)));
  const allScans = async () => ((await tx("readonly", (s) => s.getAll())) || []).sort((a, b) => b.id - a.id);

  // Object URLs for stored photos; revoked when the screen changes.
  let urls = [];
  function blobUrl(blob) {
    if (!blob) return "";
    const u = URL.createObjectURL(blob);
    urls.push(u);
    return u;
  }

  // ---------- model ----------
  function loadModel() {
    modelPromise = modelPromise || (async () => {
      // Absolute: the runtime resolves a relative path against the full URL,
      // hash included, so "#/scan/3" + "vendor/" fetched the home page instead.
      tflite.setWasmPath(new URL("vendor/", location.href.split("#")[0]).href);
      const [model, w] = await Promise.all([
        tflite.loadTFLiteModel("model/sipat.tflite", { numThreads: 1 }),
        fetch("model/cam_weights.bin").then((r) => r.arrayBuffer()),
      ]);
      CAMW = new Float32Array(w);
      return model;
    })();
    modelPromise.catch(() => { modelPromise = null; }); // let a later scan retry
    return modelPromise;
  }

  function loadImage(file) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const u = URL.createObjectURL(file);
      img.onload = () => { URL.revokeObjectURL(u); resolve(img); };
      img.onerror = () => { URL.revokeObjectURL(u); reject(new Error("unreadable")); };
      img.src = u; // modern browsers apply the photo's EXIF rotation when drawing
    });
  }

  function canvasOf(img, w, h) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, w, h);
    return c;
  }

  const toBlob = (canvas) => new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.85));

  // Same preprocessing as inference.py: RGB, squashed to 224x224, 0-255 floats.
  async function analyse(file) {
    const model = await loadModel();
    const img = await loadImage(file);
    const size = DATA.img_size;
    // Full resolution, so the Pillow-style resize matches the laptop. Phones cap
    // canvas size (iOS ~16 MP), so very large photos are pre-shrunk to 4096 px first.
    const fit = Math.min(1, 4096 / Math.max(img.naturalWidth, img.naturalHeight));
    const sw = Math.round(img.naturalWidth * fit), sh = Math.round(img.naturalHeight * fit);
    const px = canvasOf(img, sw, sh).getContext("2d").getImageData(0, 0, sw, sh).data;
    const input = C.resizeBilinear(px, sw, sh, size, size);
    const x = tf.tensor4d(input, [1, size, size, 3]);
    const out = model.predict(x);
    const tensors = out instanceof tf.Tensor ? [out] : Object.values(out);
    const probsT = tensors.find((t) => t.shape.length === 2);
    const featT = tensors.find((t) => t.shape.length === 4);
    const probs = await probsT.data();
    const feats = await featT.data();
    const [, fh, fw, ch] = featT.shape;
    tf.dispose([x, ...tensors]);

    const d = C.interpret(probs, DATA);
    const scale = Math.min(1, 1024 / Math.max(img.naturalWidth, img.naturalHeight));
    const photo = canvasOf(img, Math.round(img.naturalWidth * scale), Math.round(img.naturalHeight * scale));
    const cam = C.camMap(feats, CAMW, d.class_idx, DATA.labels.length, ch);
    return { d, probs: Array.from(probs), photo: await toBlob(photo), heat: await toBlob(overlay(photo, cam, fh, fw)) };
  }

  // gradcam.py generate_gradcam(): blend weighted by the heat, so cool areas keep the photo.
  function overlay(photo, cam, fh, fw) {
    const out = document.createElement("canvas");
    out.width = photo.width; out.height = photo.height;
    const ctx = out.getContext("2d");
    ctx.drawImage(photo, 0, 0);
    const im = ctx.getImageData(0, 0, out.width, out.height);
    const p = im.data;
    for (let y = 0; y < out.height; y++) {
      const fy = Math.min(fh - 1, Math.max(0, (y + 0.5) * fh / out.height - 0.5));
      const y0 = Math.floor(fy), y1 = Math.min(fh - 1, y0 + 1), ty = fy - y0;
      for (let x = 0; x < out.width; x++) {
        const fx = Math.min(fw - 1, Math.max(0, (x + 0.5) * fw / out.width - 0.5));
        const x0 = Math.floor(fx), x1 = Math.min(fw - 1, x0 + 1), tx = fx - x0;
        const v = (cam[y0 * fw + x0] * (1 - tx) + cam[y0 * fw + x1] * tx) * (1 - ty)
                + (cam[y1 * fw + x0] * (1 - tx) + cam[y1 * fw + x1] * tx) * ty;
        const [r, g, b] = C.colorize(v);
        const a = 0.55 * v, i = (y * out.width + x) * 4;
        p[i] = p[i] * (1 - a) + r * a; p[i + 1] = p[i + 1] * (1 - a) + g * a; p[i + 2] = p[i + 2] * (1 - a) + b * a;
      }
    }
    ctx.putImageData(im, 0, 0);
    return out;
  }

  // ---------- helpers ----------
  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  const today = () => new Date().toLocaleDateString("en-CA"); // YYYY-MM-DD, local
  const when = (iso, opts) => new Date(iso).toLocaleString("en-PH", opts || { dateStyle: "medium", timeStyle: "short" });
  const STATUS_LABEL = { unsure: "Hindi sigurado", not_leaf: "Walang dahon", other_plant: "Hindi suportado" };
  const nameOf = (s) => (s.status === "ok" ? s.disease_name_tl : STATUS_LABEL[s.status] || s.status);
  function guidanceOf(scan) {
    if (DATA.status_messages[scan.status]) return DATA.status_messages[scan.status];
    return DATA.guidance[scan.class_key] || { en: "No guidance on file.", tl: "Walang gabay." };
  }
  function toast(msg) {
    const t = document.createElement("div");
    t.className = "toast"; t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2600);
  }
  function download(blob, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }
  function langToggle() {
    return `<div class="lang-toggle">
      <button class="lang-btn ${lang === "tl" ? "active" : ""}" data-lang="tl">Tagalog</button>
      <button class="lang-btn ${lang === "en" ? "active" : ""}" data-lang="en">English</button></div>`;
  }
  function bindLang(rerender) {
    view.querySelectorAll("[data-lang]").forEach((b) => b.addEventListener("click", () => {
      lang = b.dataset.lang;
      try { localStorage.setItem("sipat-lang", lang); } catch (e) { /* ignore */ }
      rerender();
    }));
  }
  function scanItem(s) {
    return `<a href="#/scan/${s.id}" class="history-item">
      <img src="${blobUrl(s.photo)}" alt="" class="history-thumb">
      <div class="history-details">
        <div class="history-title">${esc(nameOf(s))}</div>
        <div class="history-date">${when(s.ts)}${s.plot ? " • " + esc(s.plot) : ""}</div>
      </div>
      <span class="history-badge ${s.status === "ok" ? "" : "muted"}">${s.status === "ok" ? pct(s.confidence) : "?"}</span>
    </a>`;
  }

  // ---------- dashboard ----------
  // Same cards as the desktop home (templates/_dashboard.html), from the scans on this phone.
  const localDay = (iso) => new Date(iso).toLocaleDateString("en-CA");
  const KIND_LABEL = { healthy: "Malusog", sick: "May sakit", unsure: "Hindi sigurado / walang dahon" };

  function dashboardHtml(scans) {
    const d = C.dashboard(scans, DATA, today(), localDay);
    if (!d.total) {
      return `<div class="card dash"><div class="card-title">📊 Dashboard</div>
        <p class="result-meta">Dito lalabas ang buod ng mga scan mo: ilan ang malusog, ilan ang may sakit, at ano ang pinakamadalas na sakit.
        Ang mga picture ay naka-save lang sa teleponong ito; walang ipinapadala sa internet.</p></div>`;
    }
    const share = (n) => `${(100 * n / d.total).toFixed(0)}%`;
    const peakWeek = Math.max(1, ...d.weeks.map((w) => w.count));
    const peakTop = Math.max(1, ...d.top.map((t) => t.count));
    const weekLabel = (iso) => new Date(iso + "T00:00:00").toLocaleDateString("en-PH", { month: "short", day: "numeric" });
    return `<div class="card dash">
      <div class="card-title">📊 Dashboard</div>
      <div class="tiles tiles-4">
        <a class="tile" href="#/history"><div class="num">${d.total}</div><div class="lbl">Lahat ng scan</div></a>
        <div class="tile tile-good"><div class="num">${d.healthy}</div><div class="lbl">Malusog</div></div>
        <div class="tile tile-warm"><div class="num">${d.sick}</div><div class="lbl">May sakit</div></div>
        <div class="tile ${d.due ? "tile-alert" : ""}"><div class="num">${d.due}</div><div class="lbl">Balikan na</div></div>
      </div>
      <div class="mix" role="img" aria-label="Malusog ${share(d.healthy)}, may sakit ${share(d.sick)}, hindi sigurado ${share(d.unsure)}">
        ${["healthy", "sick", "unsure"].map((k) => d[k] ? `<span class="mix-${k}" style="flex:${d[k]}"></span>` : "").join("")}
      </div>
      <div class="mix-legend">
        ${["healthy", "sick", "unsure"].map((k) => `<span><i class="dot mix-${k}"></i>${KIND_LABEL[k]} ${share(d[k])}</span>`).join("")}
      </div>

      <div class="dash-title">Scan bawat linggo</div>
      <div class="weeks">${d.weeks.map((w) => `<div><b>${w.count || ""}</b>
        <span class="bar" style="height:${Math.round(100 * w.count / peakWeek)}%"></span>${weekLabel(w.week)}</div>`).join("")}</div>

      ${d.top.length ? `<div class="dash-title">Pinakamadalas na sakit</div>
        <ul class="bars">${d.top.map((t) => `<li><span>${esc(t.name_tl)}</span>
          <span class="track"><span class="bar-sick" style="width:${Math.round(100 * t.count / peakTop)}%"></span></span>
          <strong>${t.count}</strong></li>`).join("")}</ul>` : ""}

      ${d.plots.length ? `<div class="dash-title">Kalagayan ng bawat taniman</div>
        <div class="plot-list">${d.plots.map((p) => `<a href="#/history?plot=${encodeURIComponent(p.name)}" class="plot-row">
          <i class="dot mix-${p.kind}"></i><strong>${esc(p.name)}</strong>
          <span>${esc(nameOf(p.scan))} • ${when(p.scan.ts, { month: "short", day: "numeric" })}</span></a>`).join("")}</div>` : ""}
    </div>`;
  }

  // ---------- screens ----------
  async function home(params) {
    const scans = await allScans();
    const followOf = params.get("follow") ? await getScan(params.get("follow")) : null;
    const due = scans.filter((s) => s.recheck_due && s.recheck_due <= today() && !scans.some((f) => f.parent_id === s.id));
    const plots = [...new Set(scans.map((s) => s.plot).filter(Boolean))];
    view.innerHTML = `
      ${followOf ? `<div class="card follow-banner">🔁 Follow-up ng scan noong ${when(followOf.ts, { dateStyle: "medium" })}:
        <strong>${esc(nameOf(followOf))}</strong>. Kunan ulit ang parehong halaman.</div>` : ""}
      <div class="card scan-card">
        <div class="card-title">Kunan ang dahon</div>
        <p class="result-meta">Isang dahon lang, halos buong picture, sa liwanag ng araw, kita ang mga batik.</p>
        <label class="field-label" for="plot">Taniman (opsyonal)</label>
        <input id="plot" class="text-input" list="plot-list" maxlength="60" placeholder="hal. Kamatis sa likod-bahay" value="${esc(followOf ? followOf.plot || "" : "")}">
        <datalist id="plot-list">${plots.map((p) => `<option value="${esc(p)}">`).join("")}</datalist>
        <div class="scan-buttons">
          <label class="btn btn-primary btn-large">📷 Kunan ng picture
            <input type="file" accept="image/*" capture="environment" class="file-input" data-pick></label>
          <label class="btn btn-secondary">🖼️ Pumili sa gallery
            <input type="file" accept="image/*" class="file-input" data-pick></label>
        </div>
        <div id="busy" class="busy" hidden><div class="spinner"></div><span>Tinitingnan ng AI…</span></div>
        <p id="model-state" class="small-print"></p>
      </div>
      ${due.length ? `<div class="card due-card"><div class="card-title">⏰ Balikan Na</div>
        <div class="history-list">${due.map(scanItem).join("")}</div></div>` : ""}
      ${dashboardHtml(scans)}
      ${scans.length ? `<div class="card"><div class="card-title">Mga huling scan</div>
        <div class="history-list">${scans.slice(0, 3).map(scanItem).join("")}</div>
        <a href="#/history" class="small-print">Tingnan lahat →</a></div>` : ""}`;

    const state = $("#model-state");
    state.textContent = "Nilo-load ang AI…";
    loadModel().then(() => { state.textContent = "✓ Handa ang AI (gumagana kahit offline)"; })
      .catch(() => { state.textContent = "Hindi ma-load ang AI. Buksan ulit ang app habang may internet nang isang beses."; });

    view.querySelectorAll("[data-pick]").forEach((input) => input.addEventListener("change", async () => {
      const file = input.files && input.files[0];
      if (!file) return;
      $("#busy").hidden = false;
      view.querySelectorAll(".btn").forEach((b) => b.classList.add("disabled"));
      try {
        const { d, photo, heat } = await analyse(file);
        const id = await putScan({
          ts: new Date().toISOString(), photo, heat, ...d,
          plot: $("#plot").value.trim() || null, notes: "", recheck_due: null,
          parent_id: followOf ? followOf.id : null, true_class_key: null,
        });
        location.hash = `#/scan/${id}`;
      } catch (e) {
        console.error(e);
        $("#busy").hidden = true;
        view.querySelectorAll(".btn").forEach((b) => b.classList.remove("disabled"));
        toast("Hindi mabasa ang larawan. Subukang kumuha ulit.");
      }
    }));
  }

  async function result(id) {
    const s = await getScan(id);
    if (!s) return notFound();
    const parent = s.parent_id ? await getScan(s.parent_id) : null;
    const followups = (await allScans()).filter((f) => f.parent_id === s.id).sort((a, b) => a.id - b.id);
    const c = DATA.classes[s.class_key] || {};
    const g = guidanceOf(s);
    const labelled = s.true_class_key;
    view.innerHTML = `
      <div class="card result-card">
        ${s.status === "ok"
          ? `<div class="confidence-badge">${pct(s.confidence)} Kumpiyansa</div>
             <div class="disease-name">${esc(s.disease_name)}</div>
             <div class="disease-name-tl">${esc(s.disease_name_tl)}</div>
             <div class="result-meta">Pananim: <strong>${esc(DATA.crops[s.crop_type] || s.crop_type)}</strong></div>`
          : `<div class="status-banner status-${s.status}">${{
              unsure: "Hindi sigurado ang AI - kumuha ulit ng larawan",
              not_leaf: "Walang nakitang dahon sa larawan",
              other_plant: "Hindi suportadong halaman" }[s.status]}</div>`}
        <div class="result-meta">${when(s.ts)}${s.plot ? ` • Taniman: <a href="#/history?plot=${encodeURIComponent(s.plot)}">${esc(s.plot)}</a>` : ""}</div>
        <div class="card-title mt">Top 3 hula ng AI</div>
        <ul class="top3">${s.top3.map((t) => `<li>
          <span>${esc(t.name_tl)} <small>(${esc(t.name)})</small></span>
          <span class="top3-bar"><span style="width:${Math.round(t.confidence * 100)}%"></span></span>
          <strong>${pct(t.confidence)}</strong></li>`).join("")}</ul>
        <div class="card-title mt">Paliwanag ng AI (heat map)</div>
        <p class="result-meta">Ang mapula/dilaw na bahagi ang pinagbatayan ng AI. Kung nasa labas ng dahon ito, kumuha ulit ng mas malinaw na larawan.</p>
        <div class="image-comparison">
          <figure><img src="${blobUrl(s.photo)}" alt="Orihinal na dahon"><figcaption>Orihinal</figcaption></figure>
          <figure><img src="${blobUrl(s.heat)}" alt="Heat map ng AI"><figcaption>AI Heat map</figcaption></figure>
        </div>
      </div>

      ${parent ? `<div class="card"><div class="card-title">🔁 Bago at Ngayon</div>
        <div class="compare">
          <figure><a href="#/scan/${parent.id}"><img src="${blobUrl(parent.photo)}" alt="Dati"></a>
            <figcaption><strong>Dati</strong>${esc(nameOf(parent))}<br>${when(parent.ts, { dateStyle: "medium" })}</figcaption></figure>
          <figure><img src="${blobUrl(s.photo)}" alt="Ngayon">
            <figcaption><strong>Ngayon</strong>${esc(nameOf(s))}<br>${when(s.ts, { dateStyle: "medium" })}</figcaption></figure>
        </div>${trendHtml(parent, s)}</div>` : ""}

      ${followups.length ? `<div class="card"><div class="card-title">🔁 Mga Follow-up</div>
        <div class="history-list">${followups.map(scanItem).join("")}</div>${trendHtml(s, followups[followups.length - 1])}</div>` : ""}

      <div class="card">
        <div class="card-title">🌱 Ano ang Gagawin</div>
        ${langToggle()}
        <div class="treatment-content">${C.markdown(g[lang])}</div>
        ${c.diagnosis ? `<p class="small-print"><a href="#/library/${c.slug}">📖 Basahin pa tungkol dito</a></p>` : ""}
      </div>

      <div class="card">
        <div class="card-title">💬 Magtanong</div>
        <p class="result-meta">Sumasagot mula sa nakasulat na gabay lamang. Hindi ito magbibigay ng dami ng gamot: sundin ang label at magtanong sa MAO.</p>
        <div class="chips">
          ${["Ano ang gagawin ko?", "Paano maiwasan?", "May organikong paraan ba?", "Puwede pa bang ibenta ang bunga?", "Kailan hihingi ng tulong?"]
            .map((q) => `<button class="chip" data-q="${esc(q)}">${esc(q)}</button>`).join("")}
        </div>
        <form id="ask-form" class="ask-form">
          <input id="ask-q" class="text-input" maxlength="400" placeholder="Isulat ang tanong…" autocomplete="off">
          <button class="btn btn-primary" type="submit">Itanong</button>
        </form>
        <div id="answers"></div>
      </div>

      <div class="card">
        <div class="card-title">✅ Tama ba ang hula?</div>
        <p class="result-meta">Ang mga sagot dito ang bumubuo ng <em>field test set</em>: sinusukat nito kung gaano katama ang AI sa totoong bukid.</p>
        ${labelled ? `<p class="label-done">Na-save: <strong>${esc((DATA.classes[labelled] || {}).name_tl || labelled)}</strong></p>` : ""}
        <div class="feedback-row">
          ${s.status === "ok" ? `<button class="btn btn-secondary" id="fb-yes">👍 Tama</button>` : ""}
          <button class="btn btn-secondary" id="fb-no">${s.status === "ok" ? "👎 Mali" : "🏷️ Ilagay ang totoong sakit"}</button>
        </div>
        <div id="fb-pick" hidden>
          <label class="field-label" for="fb-class">Ano talaga ito? (mas mabuti kung kinumpirma ng MAO)</label>
          <select id="fb-class" class="text-input">
            ${Object.entries(DATA.classes).map(([k, v]) => `<option value="${esc(k)}" ${k === (labelled || s.class_key) ? "selected" : ""}>${esc(v.name_tl)} (${esc(v.name)})</option>`).join("")}
          </select>
          <button class="btn btn-primary" id="fb-save">I-save</button>
        </div>
      </div>

      <div class="card">
        <div class="card-title">📝 Notes at Paalala</div>
        <label class="field-label" for="notes">Tala</label>
        <textarea id="notes" class="text-input" rows="3" maxlength="1000" placeholder="hal. Nag-spray noong Lunes">${esc(s.notes || "")}</textarea>
        <label class="field-label" for="recheck">Balikan sa petsang ito</label>
        <input id="recheck" type="date" class="text-input" value="${s.recheck_due || ""}">
        <div class="chips">${[3, 7, 14].map((n) => `<button class="chip" data-days="${n}">+${n} araw</button>`).join("")}</div>
        <button class="btn btn-primary" id="save-notes">I-save ang tala</button>
      </div>

      <div class="actions">
        <a href="#/?follow=${s.id}" class="btn btn-primary">🔁 Follow-up na scan</a>
        <a href="#/" class="btn btn-secondary">📷 Bagong scan</a>
        <button class="btn btn-danger" id="delete">🗑️ I-delete</button>
      </div>`;

    bindLang(() => result(id));
    const answers = $("#answers");
    function askQ(q) {
      const a = C.ask(s, q, lang, DATA);
      if (!a.answer) return;
      answers.insertAdjacentHTML("afterbegin",
        `<div class="qa"><div class="qa-q">${esc(q)}</div><div class="qa-a">${esc(a.answer)}</div></div>`);
    }
    view.querySelectorAll("[data-q]").forEach((b) => b.addEventListener("click", () => askQ(b.dataset.q)));
    $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); askQ($("#ask-q").value); $("#ask-q").value = ""; });

    const yes = $("#fb-yes");
    if (yes) yes.addEventListener("click", async () => { s.true_class_key = s.class_key; await putScan(s); toast("Salamat! Na-save."); result(id); });
    $("#fb-no").addEventListener("click", () => { $("#fb-pick").hidden = false; });
    $("#fb-save").addEventListener("click", async () => { s.true_class_key = $("#fb-class").value; await putScan(s); toast("Na-save."); result(id); });

    view.querySelectorAll("[data-days]").forEach((b) => b.addEventListener("click", () => {
      const d = new Date(); d.setDate(d.getDate() + Number(b.dataset.days));
      $("#recheck").value = d.toLocaleDateString("en-CA");
    }));
    $("#save-notes").addEventListener("click", async () => {
      s.notes = $("#notes").value.trim(); s.recheck_due = $("#recheck").value || null;
      await putScan(s); toast("Na-save.");
    });
    $("#delete").addEventListener("click", async (e) => {
      const btn = e.currentTarget;
      if (btn.dataset.armed !== "1") { btn.dataset.armed = "1"; btn.textContent = "Pindutin ulit para i-delete"; return; }
      for (const f of followups) { f.parent_id = null; await putScan(f); }
      await deleteScan(s.id);
      toast("Na-delete.");
      location.hash = "#/history";
    });
  }

  function trendHtml(before, after) {
    const t = C.trend(before, after, DATA);
    return `<div class="trend trend-${t.kind}">${esc(t.text)}</div>`;
  }

  async function history(params) {
    const scans = await allScans();
    const plot = params.get("plot") || "";
    const status = params.get("status") || "";
    const plots = [...new Set(scans.map((s) => s.plot).filter(Boolean))];
    const shown = scans.filter((s) => (!plot || s.plot === plot) && (!status || s.status === status));
    view.innerHTML = `
      <div class="card">
        <div class="card-title">History (${shown.length})</div>
        <div class="filters">
          <select id="f-plot" class="text-input" aria-label="Taniman"><option value="">Lahat ng taniman</option>
            ${plots.map((p) => `<option ${p === plot ? "selected" : ""}>${esc(p)}</option>`).join("")}</select>
          <select id="f-status" class="text-input" aria-label="Resulta"><option value="">Lahat ng resulta</option>
            ${[["ok", "May diagnosis"], ["unsure", "Hindi sigurado"], ["not_leaf", "Walang dahon"], ["other_plant", "Hindi suportado"]]
              .map(([v, l]) => `<option value="${v}" ${v === status ? "selected" : ""}>${l}</option>`).join("")}</select>
        </div>
        <div class="history-list">${shown.map(scanItem).join("") || `<p class="result-meta">Walang tugma.</p>`}</div>
      </div>`;
    const go = () => {
      const q = new URLSearchParams();
      if ($("#f-plot").value) q.set("plot", $("#f-plot").value);
      if ($("#f-status").value) q.set("status", $("#f-status").value);
      location.hash = `#/history${q.toString() ? "?" + q : ""}`;
    };
    $("#f-plot").addEventListener("change", go);
    $("#f-status").addEventListener("change", go);
  }

  function library() {
    const groups = {};
    for (const c of Object.values(DATA.classes)) {
      if (c.diagnosis) (groups[c.crop] = groups[c.crop] || []).push(c);
    }
    view.innerHTML = `
      <div class="card"><div class="card-title">📖 Mga Sakit</div>
        <p class="result-meta">Ang mga sakit na kilala ng Sipat. Halimbawang larawan: PlantVillage (CC BY 4.0).</p></div>
      ${Object.entries(groups).map(([crop, items]) => `
        <div class="card"><div class="card-title">${esc(DATA.crops[crop] || crop)}</div>
          <div class="library-grid">${items.map((c) => `
            <a href="#/library/${c.slug}" class="library-tile">
              ${DATA.library.includes(c.slug) ? `<img src="library/${c.slug}.jpg" alt="" loading="lazy">` : `<div class="tile-blank">🌿</div>`}
              <span>${esc(c.name_tl)}</span></a>`).join("")}
          </div></div>`).join("")}`;
  }

  function libraryEntry(slug) {
    const entry = Object.entries(DATA.classes).find(([, c]) => c.slug === slug && c.diagnosis);
    if (!entry) return notFound();
    const [key, c] = entry;
    const m = DATA.evaluation.per_class.find((p) => p.class === key);
    view.innerHTML = `
      <div class="card">
        ${DATA.library.includes(slug) ? `<img class="library-hero" src="library/${slug}.jpg" alt="">` : ""}
        <div class="disease-name">${esc(c.name)}</div>
        <div class="disease-name-tl">${esc(c.name_tl)}</div>
        ${m ? `<p class="result-meta">Sa test set: nakita ng AI ang <strong>${pct(m.recall)}</strong> ng ${m.support} larawan nito;
          kapag sinabi ng AI na ito, tama ito <strong>${pct(m.precision)}</strong> ng beses (lab na larawan).</p>` : ""}
      </div>
      <div class="card">${langToggle()}<div class="treatment-content">${C.markdown(DATA.guidance[key][lang])}</div></div>
      <a href="#/library" class="btn btn-secondary">← Mga Sakit</a>`;
    bindLang(() => libraryEntry(slug));
  }

  async function about() {
    const e = DATA.evaluation;
    const scans = await allScans();
    const labelled = scans.filter((s) => s.true_class_key);
    let usage = "";
    try {
      const est = await navigator.storage.estimate();
      usage = `${(est.usage / 1048576).toFixed(1)} MB ang gamit sa telepono`;
    } catch (err) { /* not supported */ }
    view.innerHTML = `
      <div class="card">
        <div class="card-title">Tungkol sa SIPAT 🌱🔍</div>
        <p>Ang <strong>Sipat</strong> ay tumutukoy sa sakit ng kamatis, sili, patatas at mais mula sa larawan ng isang dahon.</p>
        <ul class="plain-list">
          <li><strong>Sa telepono tumatakbo ang AI.</strong> Walang larawang ipinapadala sa internet.</li>
          <li><strong>Gumagana offline</strong> pagkatapos itong mabuksan nang isang beses habang may internet.</li>
          <li><strong>"Hindi sigurado"</strong> kapag mas mababa sa ${Math.round(DATA.threshold * 100)}% ang kumpiyansa, sa halip na manghula.</li>
          <li><strong>Heat map</strong> na nagpapakita kung saan tumingin ang AI.</li>
        </ul>
      </div>
      <div class="card">
        <div class="card-title">📏 Gaano Katama ang AI</div>
        <p class="result-meta">Sinukat sa ${e.test_photos.toLocaleString()} test na larawan na hindi ginamit sa training:</p>
        <ul class="plain-list">
          <li>Accuracy: <strong>${pct(e.accuracy)}</strong></li>
          <li>Tamang sagot nasa top 3: <strong>${pct(e.top3_accuracy)}</strong></li>
          <li>Macro F1: <strong>${e.macro_f1.toFixed(3)}</strong></li>
          ${e.at_threshold ? `<li>Sa threshold ${DATA.threshold}: sumasagot sa ${pct(e.at_threshold.answered_share)} ng larawan, tama ang ${pct(e.at_threshold.accuracy_when_answered)}</li>` : ""}
        </ul>
        <p class="result-meta">Lab na larawan ito (isang dahon, payak na background). Hindi pa nasusukat ang accuracy sa larawang kuha sa bukid: iyon ang layunin ng field test set sa ibaba.</p>
        <p class="small-print">${esc(e.architecture || "")} • ${esc(e.dataset || "")}</p>
      </div>
      <div class="card">
        <div class="card-title">📦 Field test set</div>
        <p class="result-meta">${labelled.length} sa ${scans.length} scan ang may label ("Tama ba ang hula?").
          I-download at patakbuhin sa laptop: <code>python scripts/evaluate_field.py &lt;zip&gt;</code></p>
        <div class="actions">
          <button class="btn btn-primary" id="export-zip" ${labelled.length ? "" : "disabled"}>⬇️ Field set (.zip)</button>
          <button class="btn btn-secondary" id="export-csv" ${scans.length ? "" : "disabled"}>⬇️ Lahat ng scan (.csv)</button>
        </div>
        <p class="small-print">${usage}</p>
      </div>
      <div class="card">
        <div class="card-title">📲 Ilagay sa home screen</div>
        <p class="result-meta"><strong>Android (Chrome):</strong> ⋮ menu → "Add to Home screen" / "Install app".<br>
          <strong>iPhone (Safari):</strong> Share → "Add to Home Screen".</p>
        <button class="btn btn-primary" id="install" hidden>📲 I-install ang Sipat</button>
      </div>
      <div class="card centered small-print">
        <p>Sipat • Binuo para sa Magsasakang Pilipino • bersyon ${esc(DATA.model_version)}</p>
        <p>Gabay: UC IPM, Cornell, CIP, CIMMYT; kumpirmahin sa inyong MAO</p>
      </div>`;

    $("#export-zip").addEventListener("click", async () => {
      const files = [];
      const rows = [["id", "file", "true_class", "model_class", "model_status", "confidence", "timestamp"]];
      for (const s of labelled) {
        const name = `${s.true_class_key}/sipat_phone_${s.id}.jpg`;
        files.push({ name, bytes: new Uint8Array(await s.photo.arrayBuffer()) });
        rows.push([s.id, name, s.true_class_key, s.class_key, s.status, s.confidence, s.ts]);
      }
      files.push({ name: "labels.csv", bytes: new TextEncoder().encode(rows.map(C.csvRow).join("\n") + "\n") });
      download(new Blob([C.zip(files)], { type: "application/zip" }), `sipat_field_set_${today().replace(/-/g, "")}.zip`);
    });
    $("#export-csv").addEventListener("click", () => {
      const rows = [["id", "timestamp", "plot", "status", "model_class", "confidence", "true_class", "parent_id", "recheck_due", "notes"]];
      for (const s of scans.slice().reverse())
        rows.push([s.id, s.ts, s.plot, s.status, s.class_key, s.confidence, s.true_class_key, s.parent_id, s.recheck_due, s.notes]);
      download(new Blob([rows.map(C.csvRow).join("\n") + "\n"], { type: "text/csv" }), `sipat_scans_${today().replace(/-/g, "")}.csv`);
    });
    if (deferredInstall) {
      const b = $("#install");
      b.hidden = false;
      b.addEventListener("click", () => { deferredInstall.prompt(); deferredInstall = null; b.hidden = true; });
    }
  }

  function notFound() {
    view.innerHTML = `<div class="card"><div class="card-title">Wala rito</div>
      <p class="result-meta">Hindi makita ang hinahanap mo.</p><a href="#/" class="btn btn-primary">Bumalik</a></div>`;
  }

  // ---------- router ----------
  async function route() {
    urls.forEach((u) => URL.revokeObjectURL(u));
    urls = [];
    const [path, query] = (location.hash.slice(1) || "/").split("?");
    const params = new URLSearchParams(query || "");
    const parts = path.split("/").filter(Boolean);
    const nav = { history: "history", library: "library", about: "about" }[parts[0]] || (parts.length ? "" : "home");
    document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === nav));
    window.scrollTo(0, 0);
    try {
      if (!parts.length) await home(params);
      else if (parts[0] === "scan") await result(parts[1]);
      else if (parts[0] === "history") await history(params);
      else if (parts[0] === "library") parts[1] ? libraryEntry(parts[1]) : library();
      else if (parts[0] === "about") await about();
      else notFound();
    } catch (e) {
      console.error(e);
      view.innerHTML = `<div class="card"><div class="card-title">May mali</div><p class="result-meta">${esc(e.message || e)}</p></div>`;
    }
  }

  let deferredInstall = null;
  window.addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); deferredInstall = e; });

  async function start() {
    DATA = await (await fetch("data/sipat.json")).json();
    window.addEventListener("hashchange", route);
    await route();
    if ("serviceWorker" in navigator && window.isSecureContext) {
      navigator.serviceWorker.register("sw.js").catch((e) => console.warn("service worker:", e));
    }
  }

  // Used by scripts/pwa_parity.py to compare phone vs laptop answers on the same photos.
  window.SipatDebug = { analyse, loadModel, data: () => DATA };

  start().catch((e) => {
    view.innerHTML = `<div class="card"><div class="card-title">Hindi mabuksan ang Sipat</div>
      <p class="result-meta">${esc(e.message || e)}</p></div>`;
  });
})();
