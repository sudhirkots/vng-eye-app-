// Balance Band stream protocol: decode (and, for the simulator and tests, encode) the 20-byte frames sent
// by the firmware. Frame, little-endian:
//   0xA5 0x5A | seq u8 | t_us u32 | ax ay az i16 (milli-g) | gx gy gz i16 (0.01 deg/s) | checksum u8
// Works in the browser (window.BandProtocol) and in node (module.exports) for the tests.
(function (root) {
  "use strict";
  const FRAME_LEN = 20, SYNC0 = 0xA5, SYNC1 = 0x5A;

  class FrameParser {
    constructor() {
      this.buf = new Uint8Array(0);
      this.lastSeq = null; this.lastT = null; this.tHigh = 0;
      this.frames = 0; this.bad = 0; this.lost = 0;
    }
    // Feed any chunk of bytes; returns decoded samples {t (s), ax, ay, az (g), gx, gy, gz (deg/s)}.
    feed(chunk) {
      const b = new Uint8Array(this.buf.length + chunk.length);
      b.set(this.buf); b.set(chunk, this.buf.length);
      const out = []; let i = 0;
      while (i + FRAME_LEN <= b.length) {
        if (b[i] !== SYNC0 || b[i + 1] !== SYNC1) { i++; continue; }        // hunt for sync
        let sum = 0;
        for (let k = 2; k < FRAME_LEN - 1; k++) sum = (sum + b[i + k]) & 0xFF;
        if (sum !== b[i + FRAME_LEN - 1]) { this.bad++; i++; continue; }  // corrupt: resync one byte on
        const dv = new DataView(b.buffer, b.byteOffset + i, FRAME_LEN);
        const seq = dv.getUint8(2), tRaw = dv.getUint32(3, true);
        if (this.lastSeq !== null) this.lost += (seq - this.lastSeq - 1 + 256) % 256;
        this.lastSeq = seq;
        if (this.lastT !== null && tRaw < this.lastT) this.tHigh += 4294967296;   // micros() wrapped
        this.lastT = tRaw;
        out.push({
          t: (this.tHigh + tRaw) / 1e6,
          ax: dv.getInt16(7, true) / 1000, ay: dv.getInt16(9, true) / 1000, az: dv.getInt16(11, true) / 1000,
          gx: dv.getInt16(13, true) / 100, gy: dv.getInt16(15, true) / 100, gz: dv.getInt16(17, true) / 100,
        });
        this.frames++; i += FRAME_LEN;
      }
      this.buf = b.slice(i);
      return out;
    }
  }

  function clip16(v) { return Math.max(-32768, Math.min(32767, Math.round(v))); }

  // Encode one sample exactly as the firmware does (used by the simulated band and the tests).
  function encodeFrame(seq, tUs, s) {
    const f = new Uint8Array(FRAME_LEN), dv = new DataView(f.buffer);
    f[0] = SYNC0; f[1] = SYNC1; dv.setUint8(2, seq & 0xFF); dv.setUint32(3, tUs >>> 0, true);
    [s.ax, s.ay, s.az].forEach((v, k) => dv.setInt16(7 + 2 * k, clip16(v * 1000), true));
    [s.gx, s.gy, s.gz].forEach((v, k) => dv.setInt16(13 + 2 * k, clip16(v * 100), true));
    let sum = 0;
    for (let k = 2; k < FRAME_LEN - 1; k++) sum = (sum + f[k]) & 0xFF;
    f[FRAME_LEN - 1] = sum;
    return f;
  }

  const api = { FrameParser, encodeFrame, FRAME_LEN,
                NUS_SERVICE: "6e400001-b5a3-f393-e0a9-e50e24dcca9e",
                NUS_TX: "6e400003-b5a3-f393-e0a9-e50e24dcca9e", DEVICE_NAME: "BalanceBand" };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.BandProtocol = api;
})(typeof window !== "undefined" ? window : this);
