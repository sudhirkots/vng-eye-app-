// Simulated Balance Band: produces the same Bluetooth byte stream as the real firmware, so the whole app
// (decoder included) can be practised and tested without hardware. The "person" sways a little, leans when
// the screen says so, sways more with eyes closed / on foam, and wears the band at a random angle.
(function (root) {
  "use strict";
  const FS = 100, DT = 1 / FS;
  const SWAY = { "": 0.25, eo_firm: 0.25, ec_firm: 0.35, eo_foam: 0.4, ec_foam: 0.8 };   // deg, rough SD
  const LEAN = { forward: [6, 0], backward: [-4, 0], left: [0, 4.5], right: [0, -4.5] };  // deg AP, ML

  function randomRotation() {                 // rows = body F, L, U in sensor coords
    const n = () => { let v = [Math.random() - .5, Math.random() - .5, Math.random() - .5];
                      const m = Math.hypot(...v); return v.map(x => x / m); };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const F = n(); let L = cross(n(), F); const m = Math.hypot(...L); L = L.map(x => x / m);
    return [F, L, cross(F, L)];
  }

  class SimBand {
    constructor(onBytes) {
      this.onBytes = onBytes; this.mount = randomRotation();
      this.condition = ""; this.phase = "";
      this.ap = 0; this.ml = 0; this.nap = 0; this.nml = 0; this.lap = 0; this.lml = 0;
      this.seq = 0; this.tUs = Math.floor(Math.random() * 1e9); this.timer = null;
    }
    start() { this.timer = setInterval(() => this.tick(10), 100); }
    stop() { clearInterval(this.timer); this.timer = null; }
    tick(n) {
      const chunks = [];
      for (let k = 0; k < n; k++) {
        const sd = SWAY[this.condition] ?? 0.25, a = Math.exp(-2 * Math.PI * 0.5 * DT);
        this.nap = a * this.nap + Math.sqrt(1 - a * a) * gauss() * sd;     // slow random sway
        this.nml = a * this.nml + Math.sqrt(1 - a * a) * gauss() * sd * 0.7;
        const [tap, tml] = LEAN[this.phase] || [0, 0], b = Math.exp(-DT / 0.6);
        this.lap = b * this.lap + (1 - b) * tap; this.lml = b * this.lml + (1 - b) * tml;   // voluntary lean
        const ap = this.nap + this.lap, ml = this.nml + this.lml;
        const dap = (ap - this.ap) / DT, dml = (ml - this.ml) / DT; this.ap = ap; this.ml = ml;
        const r = Math.PI / 180, aF = -Math.sin(ap * r), aL = -Math.sin(ml * r);
        const body = [aF, aL, Math.sqrt(Math.max(0, 1 - aF * aF - aL * aL))];
        const gb = [-dml, dap, 0];
        const toSensor = (v) => [0, 1, 2].map(j => v[0] * this.mount[0][j] + v[1] * this.mount[1][j] + v[2] * this.mount[2][j]);
        const acc = toSensor(body).map(v => v + 0.003 * gauss());
        const gyr = toSensor(gb).map(v => v + 0.3 * gauss());
        chunks.push(BandProtocol.encodeFrame(this.seq++, this.tUs,
          { ax: acc[0], ay: acc[1], az: acc[2], gx: gyr[0], gy: gyr[1], gz: gyr[2] }));
        this.tUs = (this.tUs + 10000) >>> 0;
      }
      const out = new Uint8Array(chunks.length * BandProtocol.FRAME_LEN);
      chunks.forEach((c, i) => out.set(c, i * BandProtocol.FRAME_LEN));
      this.onBytes(out);
    }
  }
  function gauss() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random();
                     return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  root.SimBand = SimBand;
})(window);
