// Balance Band recorder — page logic: connect (Bluetooth or simulated), guide each step, record, save, analyse.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);

  const STEP_INFO = {
    los: { title: "1 · Limits of stability",
      instr: "Feet together, arms by the side, looking ahead. First stand still. Then follow the screen: lean " +
             "as far as you can FORWARD, BACKWARD, LEFT and RIGHT — from the ankles, body straight, without " +
             "stepping or lifting heels or toes — and come back to the centre each time. Guard the person." },
    eo_firm: { title: "2 · Eyes open, firm floor",
      instr: "On the floor. Feet together, arms by the side, eyes OPEN, looking at a point on the wall. Stand as still as you can." },
    ec_firm: { title: "3 · Eyes closed, firm floor",
      instr: "On the floor. Feet together, arms by the side, eyes CLOSED. Stand as still as you can." },
    eo_foam: { title: "4 · Eyes open, on foam",
      instr: "On the FOAM. Feet together, arms by the side, eyes OPEN, looking at a point on the wall. Stand as still as you can." },
    ec_foam: { title: "5 · Eyes closed, on foam",
      instr: "On the FOAM. Feet together, arms by the side, eyes CLOSED. Stand as still as you can. Guard closely." },
  };
  const PHASE_TEXT = { centre: "STAND STILL — centre", forward: "LEAN FORWARD", backward: "LEAN BACKWARD",
                       left: "LEAN LEFT", right: "LEAN RIGHT", stand: "STAND STILL" };
  const LEAN_SUB = "as far as you can — don't step";
  const STEPS = ["los", "eo_firm", "ec_firm", "eo_foam", "ec_foam"];

  let cfg = null, session = null, sel = null, status = {};
  let parser = null, source = null, sim = null;
  let rec = null;                                   // current recording
  const live = [];                                  // last ~10 s of samples for the live trace
  let base = null, rateCount = 0;

  // ---------------------------------------------------------------- band connection
  function onBytes(bytes) {
    const samples = parser.feed(bytes);
    rateCount += samples.length;
    for (const s of samples) {
      live.push(s); if (live.length > 1000) live.shift();
      if (rec && rec.recording) addRow(s);
    }
  }

  function connected(label) {
    $("bandDot").classList.add("on"); $("bandState").textContent = label;
    $("btConnect").disabled = true; $("simConnect").disabled = true; refresh();
  }

  function disconnected(msg) {
    $("bandDot").classList.remove("on"); $("bandState").textContent = msg || "Band not connected";
    $("btConnect").disabled = false; $("simConnect").disabled = false; source = null;
    if (sim) { sim.stop(); sim = null; }
    if (rec) abort("The band disconnected during the recording — this step was NOT saved. Please redo it.");
    refresh();
  }

  async function connectBluetooth() {
    if (!navigator.bluetooth) {
      alert("This browser cannot use Bluetooth. Please use Google Chrome or Microsoft Edge (Windows, Mac or Android).");
      return;
    }
    try {
      const dev = await navigator.bluetooth.requestDevice({
        filters: [{ name: BandProtocol.DEVICE_NAME }], optionalServices: [BandProtocol.NUS_SERVICE] });
      dev.addEventListener("gattserverdisconnected", () => disconnected("Band disconnected"));
      $("bandState").textContent = "Connecting…";
      const server = await dev.gatt.connect();
      const svc = await server.getPrimaryService(BandProtocol.NUS_SERVICE);
      const tx = await svc.getCharacteristic(BandProtocol.NUS_TX);
      parser = new BandProtocol.FrameParser();
      tx.addEventListener("characteristicvaluechanged", (e) => {
        const v = e.target.value; onBytes(new Uint8Array(v.buffer, v.byteOffset, v.byteLength)); });
      await tx.startNotifications();
      source = "ble"; connected("Band connected: " + (dev.name || "BalanceBand"));
    } catch (err) {
      if (err && err.name !== "NotFoundError") alert("Could not connect: " + err.message);
      disconnected();
    }
  }

  function connectSim() {
    parser = new BandProtocol.FrameParser();
    sim = new SimBand(onBytes); sim.start(); source = "sim";
    connected("SIMULATED band (practice only)");
  }

  // ---------------------------------------------------------------- session + steps
  async function api(path, body) {
    const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" },
                                         body: JSON.stringify(body) } : undefined);
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  }

  async function newSession() {
    const person = $("person").value.trim();
    if (!person) { alert("Enter the person's ID code first."); return; }
    if (session && Object.values(status).some(Boolean) && !confirm("Start a new session? The current one stays saved.")) return;
    try {
      const j = await api("/api/session", { person, site: $("site").value });
      session = j.session; status = {}; $("report").textContent = "Record the steps, then press “Analyse session”.";
      $("sessionInfo").textContent = "Session " + session + " — saved in " + cfg.data_dir;
      select("los");
    } catch (e) { alert("Could not start the session: " + e.message); }
  }

  function renderSteps() {
    $("steps").innerHTML = "";
    for (const s of STEPS) {
      const li = document.createElement("li");
      li.className = (s === sel ? "sel " : "") + (status[s] ? "done" : "");
      li.innerHTML = `<span>${STEP_INFO[s].title}</span><span class="st">${status[s] ? "recorded ✓" : "—"}</span>`;
      li.onclick = () => { if (!rec) select(s); };
      $("steps").appendChild(li);
    }
  }

  function select(s) {
    sel = s; $("stepTitle").textContent = STEP_INFO[s].title; $("instr").textContent = STEP_INFO[s].instr;
    $("prompt").textContent = ""; $("subprompt").textContent = status[s] ? "Already recorded — Start again to redo it." : "";
    $("progress").style.width = "0"; refresh();
  }

  function refresh() {
    renderSteps();
    const ready = !!(source && session && sel && !rec);
    $("start").disabled = !ready; $("cancel").disabled = !rec;
    $("fall").disabled = !(rec && rec.recording && !rec.fallAt);
    $("newSession").disabled = !!rec;
    $("analyse").disabled = !(session && status.eo_firm && !rec);
  }

  // ---------------------------------------------------------------- recording
  function script(step) {
    return step === "los" ? cfg.los_script : [["stand", cfg.duration_s]];
  }

  function start() {
    const sc = script(sel);
    rec = { step: sel, script: sc, total: sc.reduce((a, p) => a + p[1], 0), rows: [], recording: false,
            fallAt: null, t0: null, lost0: parser.lost };
    if (sim) { sim.condition = sel === "los" ? "" : sel; sim.phase = ""; }
    refresh();
    let n = 3;
    const count = () => {
      if (!rec) return;
      if (n > 0) { $("prompt").textContent = "Get ready… " + n; $("prompt").className = ""; $("subprompt").textContent = "";
                   n--; rec.timer = setTimeout(count, 1000); return; }
      rec.recording = true; rec.startWall = performance.now(); refresh(); tickPhase();
    };
    count();
  }

  function phaseAt(elapsed) {
    let s = 0;
    for (const [name, d] of rec.script) { if (elapsed < s + d) return { name, left: s + d - elapsed }; s += d; }
    return null;
  }

  function tickPhase() {
    if (!rec || !rec.recording) return;
    const el = (performance.now() - rec.startWall) / 1000, ph = phaseAt(el);
    if (!ph || (rec.fallAt && performance.now() - rec.fallAt > 1000)) { finish(); return; }
    rec.phase = ph.name;
    if (sim) sim.phase = rec.step === "los" ? ph.name : "";
    const lean = ["forward", "backward", "left", "right"].includes(ph.name);
    $("prompt").textContent = rec.fallAt ? "LOST BALANCE" : PHASE_TEXT[ph.name];
    $("prompt").className = lean ? "lean" : "";
    $("subprompt").textContent = rec.fallAt ? "" : (lean ? LEAN_SUB : "") + `  (${Math.ceil(ph.left)} s)`;
    $("progress").style.width = Math.min(100, 100 * el / rec.total) + "%";
    $("recInfo").textContent = `${rec.rows.length} samples` + (parser.lost - rec.lost0 ? `, ${parser.lost - rec.lost0} lost` : "");
    rec.timer = setTimeout(tickPhase, 100);
  }

  function addRow(s) {
    if (rec.t0 === null) rec.t0 = s.t;
    rec.rows.push([+(s.t - rec.t0).toFixed(4), s.ax, s.ay, s.az, s.gx, s.gy, s.gz, rec.fallAt ? 1 : 0,
                   rec.step === "los" ? (rec.phase || "centre") : ""]);
  }

  function fall() {
    if (!rec || !rec.recording || rec.fallAt) return;
    rec.fallAt = performance.now(); refresh();
  }

  async function finish() {
    const r = rec; clearTimeout(r.timer); r.recording = false;
    if (sim) { sim.condition = ""; sim.phase = ""; }
    $("prompt").textContent = "Saving…"; $("prompt").className = ""; $("subprompt").textContent = "";
    try {
      const lost = parser.lost - r.lost0;
      const j = await api("/api/save", { session, step: r.step, rows: r.rows });
      status = j.status; rec = null;
      $("prompt").textContent = r.fallAt ? "Saved (lost balance marked)" : "Saved ✓";
      $("subprompt").innerHTML = `${j.saved} samples` + (lost ? ` · <span class="warn">${lost} lost in transmission</span>` : "");
      const next = STEPS.find((s) => !status[s]);
      if (next) setTimeout(() => { if (!rec) select(next); }, 1500);
    } catch (e) {
      rec = null; $("prompt").textContent = "NOT saved";
      $("subprompt").textContent = e.message; alert("Could not save this step: " + e.message);
    }
    $("progress").style.width = "100%"; refresh();
  }

  function abort(msg) {
    if (!rec) return;
    clearTimeout(rec.timer); rec = null;
    if (sim) { sim.condition = ""; sim.phase = ""; }
    $("prompt").textContent = "Cancelled — not saved"; $("prompt").className = "";
    $("subprompt").textContent = ""; $("progress").style.width = "0";
    if (msg) alert(msg);
    refresh();
  }

  async function analyse() {
    $("report").textContent = "Analysing…";
    try { const j = await api("/api/analyse", { session }); $("report").textContent = j.report + "\n\nSaved in: " + j.folder; }
    catch (e) { $("report").textContent = "Analysis failed: " + e.message; }
  }

  // ---------------------------------------------------------------- live trace (tilt of the band)
  function drawLive() {
    const c = $("live"), g = c.getContext("2d"), W = c.width, H = c.height;
    g.clearRect(0, 0, W, H);
    if (live.length > 50) {
      if (!base || (rec && rec.recording && !rec.baseSet)) {         // reference = the latest half second
        const last = live.slice(-50); base = ["ax", "ay", "az"].map((k) => last.reduce((a, s) => a + s[k], 0) / 50);
        if (rec && rec.recording) rec.baseSet = true;
      }
      const nb = Math.hypot(...base), pts = live.slice(-1000);
      const tilt = pts.map((s) => { const n = Math.hypot(s.ax, s.ay, s.az) || 1;
        const cos = (s.ax * base[0] + s.ay * base[1] + s.az * base[2]) / (n * nb);
        return Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI; });
      const top = Math.max(4, ...tilt);
      g.strokeStyle = "#1f5fbf"; g.lineWidth = 2; g.beginPath();
      tilt.forEach((v, i) => { const x = W * (i + 1000 - tilt.length) / 1000, y = H - 6 - (H - 12) * v / top;
                               i ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.stroke();
      g.fillStyle = "#5b6475"; g.font = "13px system-ui"; g.fillText(`${top.toFixed(0)}°`, 6, 16);
      $("liveInfo").textContent = `tilt now ${tilt[tilt.length - 1].toFixed(1)}° (last 10 s)`;
    }
    requestAnimationFrame(drawLive);
  }

  setInterval(() => { $("rate").textContent = source ? `${rateCount} samples/s` : "";
                      if (source && rateCount < 80) $("rate").className = "muted warn"; else $("rate").className = "muted";
                      rateCount = 0; }, 1000);

  // ---------------------------------------------------------------- wiring
  $("btConnect").onclick = connectBluetooth;
  $("simConnect").onclick = connectSim;
  $("newSession").onclick = newSession;
  $("start").onclick = start;
  $("cancel").onclick = () => abort();
  $("fall").onclick = fall;
  $("analyse").onclick = analyse;
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && rec && rec.recording) { e.preventDefault(); fall(); }
  });

  api("/api/config").then((c) => { cfg = c; refresh(); requestAnimationFrame(drawLive); })
    .catch((e) => { $("instr").textContent = "Cannot reach the recorder server: " + e.message; });
})();
