// node balance_band/recorder/test_protocol.js
"use strict";
const assert = require("assert");
const P = require("./static/protocol.js");

const sample = (k) => ({ ax: 0.01 * k, ay: -0.02, az: 0.998, gx: 1.25, gy: -0.5, gz: 0.01 * k });
const frames = (n, t0 = 0, seq0 = 0) =>
  Array.from({ length: n }, (_, k) => P.encodeFrame(seq0 + k, t0 + k * 10000, sample(k)));
const concat = (arrs) => { const o = new Uint8Array(arrs.reduce((a, b) => a + b.length, 0)); let i = 0;
  for (const a of arrs) { o.set(a, i); i += a.length; } return o; };

function roundTrip() {
  const p = new P.FrameParser(), got = p.feed(concat(frames(5)));
  assert.strictEqual(got.length, 5);
  assert.ok(Math.abs(got[3].t - 0.03) < 1e-9);
  assert.ok(Math.abs(got[3].ax - 0.03) < 1e-3 && Math.abs(got[3].az - 0.998) < 1e-3);
  assert.ok(Math.abs(got[3].gx - 1.25) < 1e-2 && Math.abs(got[3].gy + 0.5) < 1e-2);
  assert.strictEqual(p.lost, 0); assert.strictEqual(p.bad, 0);
}

function arbitrarySplits() {                  // Bluetooth may cut the stream anywhere
  const all = concat(frames(50)), p = new P.FrameParser(); let got = [], i = 0;
  for (const n of [1, 7, 19, 20, 21, 3, 40, 999]) { got = got.concat(p.feed(all.slice(i, i + n))); i += n; }
  assert.strictEqual(got.length, 50);
}

function corruptionAndLoss() {
  const fs = frames(10); fs[4] = fs[4].slice(); fs[4][10] ^= 0xFF;   // corrupt frame 4
  const p = new P.FrameParser(), got = p.feed(concat([new Uint8Array([1, 2, 3]), ...fs.slice(0, 7), ...fs.slice(8)]));
  assert.strictEqual(got.length, 8);          // frame 4 rejected, frame 7 never arrived
  assert.strictEqual(p.bad >= 1, true);
  assert.strictEqual(p.lost, 2);              // one corrupt + one missing
}

function timestampWrap() {
  const p = new P.FrameParser(), got = p.feed(concat(frames(3, 4294967296 - 15000, 250)));
  assert.ok(got[2].t > got[1].t && got[1].t > got[0].t);
  assert.ok(Math.abs(got[2].t - got[0].t - 0.02) < 1e-6);
  assert.strictEqual(p.lost, 0);              // seq also wrapped 255 -> 0 cleanly
}

for (const f of [roundTrip, arbitrarySplits, corruptionAndLoss, timestampWrap]) { f(); console.log("ok ", f.name); }
console.log("4 passed");
