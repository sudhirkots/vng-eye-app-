# HANDOFF — Video Frenzel with IMU (live head position)

**Date:** 2026-09-07  **Branch:** `claude/parkinsons-band-design-te98aj` (branch name is an artefact of how the
session opened — the project is the Video Frenzel IMU)
**Status:** DESIGN ONLY — no code written, no hardware ordered, nothing built or tested.

Adds a head-position channel to the video Frenzel goggles. Complements, and does not alter, the active
clinical path in `HANDOFF_NYSTAGMUS_DETECTOR.md`.

## Goal

Mount a small sensor on the Frenzel goggles that shows, **live, the actual head angle at each step** of a
positional manoeuvre (Dix-Hallpike, supine roll / BBQ, Epley).

Clinical motivation: a large share of failed Epley manoeuvres are simply performed at the wrong head angles,
and at present nobody measures them. The clinician is working blind on the one variable that decides whether
the manoeuvre can work.

---

## 1. Decision — what to buy

| Item | Notes |
|---|---|
| **Seeed XIAO nRF52840 Sense** | ~₹2,000. **Must say "Sense"** — the plain XIAO nRF52840 has no IMU. |
| Short USB-C cable (~50 cm) | Runs the board. No battery needed. |
| *Optional:* small 4-port USB hub board (CH334 / FE1.1s) | ~25 × 20 mm, few hundred rupees. Merges 3 cables into 1. **Check camera formats first** — §4. |

The IMU chip on that board is an **LSM6DS3TR-C**: **6-axis** (3-axis accelerometer + 3-axis gyroscope).
Not bought separately — already soldered on.

**Size / weight:** board 21 × 17.8 × 3.5 mm, ~3 g *(weight is an estimate — Seeed's site was unreachable from
the session to verify)*. Goggles weigh 200–500 g, so this adds ~1%. Mechanically irrelevant.

### Why this board and not the alternatives

1. **It is a complete unit.** Sensor + processor + USB + (unused) Bluetooth on one thumbnail board. The
   BNO055 breakout considered first (`evelta.com/7semi-bno055-…`) is *only a sensor* — it needs a separate
   microcontroller, wiring and power built around it. Four things to buy and assemble instead of one.
2. **6-axis is sufficient — no magnetometer needed.** See §2. The BNO055's selling point is its built-in
   compass, which here adds nothing and would actively mislead near a metal examination couch, the goggle's
   own IR LED array, and a laptop.
3. **The BNO055 caps fused output at 100 Hz**, which rules it out if head-impulse (vHIT-style) work is ever
   wanted. Adequate for positional angles, wrong for dynamics.

---

## 2. KEY DERIVATION — why no magnetometer is required

*(The reasoning that justifies the cheaper, simpler part. Record it so it is not re-litigated.)*

The only rotation gravity **cannot** observe is yaw about the gravity vector itself. Working through the
manoeuvres:

- **Dix-Hallpike** — the initial 45° head turn *while sitting upright* is the one unobservable step.
  Everything after it is observable: once the patient is supine with the head turned, that 45° appears as a
  change in the gravity vector within the head frame.
- **Supine roll / BBQ** — fully observable. Turning the head left/right while lying down rotates the head's
  superior axis through gravity.

So the only gap is a **~10-second gyro integration** from "zeroed, sitting, facing forward" to "head turned
45°". Gyro drift over 10 s is a fraction of a degree. **A 6-axis IMU covers the whole exam.**

---

## 3. Wiring — USB, NOT Bluetooth

BLE was proposed first; **Dr. K correctly overruled it** — the goggles already carry two USB camera cables, so
a third wire costs nothing and the "avoid a dangling cable" argument was weak. USB is better on the merits:

- **Timing (the strongest reason).** The point is knowing head angle *at a given video frame*. BLE delivers on
  7.5–50 ms connection intervals with jitter and can batch packets, so a sample may land ~30 ms late by a
  varying amount. USB serial is ~1–3 ms and consistent — arrival time can be treated as sample time.
- No battery to charge or die mid-clinic.
- No Windows pairing / reconnect logic to fight in a busy clinic; it just appears as a COM port.
- Simpler firmware (USB CDC serial print vs BLE service/characteristic setup).
- Headroom for 500–1000 Hz if head-impulse work is added later; BLE gets awkward at those rates.

**Plan:** small USB hub on the headband → both cameras + IMU plug in → **one** cable to the laptop. Mount the
hub on the strap (not dangling), with strain relief where the single cable leaves it.

The IMU's own bandwidth is trivial (~20 KB/s) and is never the constraint.

---

## 4. RISK — USB camera bandwidth (pre-existing, not caused by the IMU)

Cameras reserve *isochronous* bandwidth up front; USB 2.0 provides ~280–320 Mbps of it in practice.

| Camera format | Per camera @ 640×480, 30 fps | Two cameras |
|---|---|---|
| **MJPEG** (compressed) | ~15 Mbps | OK |
| **YUY2** (uncompressed) | ~147 Mbps | **Likely fails** |

With YUY2 the second camera typically just won't open (Windows fails silently; Linux reports "failed to
allocate bandwidth").

- This may **already** apply — many laptops put all USB-A ports on one controller.
- A **USB 3.0 hub does not rescue this**: USB 2.0 devices go through the 2.0 hub inside it and still share the
  same 480 Mbps.
- **Check before buying the hub** (Windows):
  `ffmpeg -f dshow -list_options true -i video="<camera name>"`
  MJPEG listed → fine (ensure the capture code requests it). YUY2 only → drop to 320×240 or 15 fps, or keep
  the cameras on separate laptop ports and hub only the IMU.
- **Fallback:** bundle the three cables in spiral wrap / braided sleeving. Handles like one cable, three plugs,
  no shared-hub bandwidth issue.

---

## 5. Mounting — the real engineering constraint

Weight is irrelevant; **rigidity is everything.** Any flex, wobble or creep in the mount becomes fake head
movement — the sensor cannot distinguish "the head rotated 2°" from "the module rocked 2° on its tape."

- Bond to a **flat, rigid part of the goggle frame** — not the strap, not foam, not anything that flexes.
- Thin, firm double-sided tape for v1. **Not thick foam tape** — it is compliant and will wobble.
- A 3D-printed clip gripping the frame once placement is proven.
- Anchor the cable to the frame a few cm from the board, so tugs pull on the goggles rather than the mount.
- **Orientation need not be precise** — the zeroing step (§6) handles it.

---

## 6. Software design

**Zero button (required).** Patient sits upright looking straight ahead → press zero → all subsequent angles
are relative to that reference. This absorbs mounting misalignment, goggle fit and inter-patient variation in
one step, and is why the mechanical alignment can be casual. **Re-zero per patient.**

**Core maths for positional angles is trivial** — drift-free and absolute against gravity:
```python
pitch = degrees(atan2(-ax, hypot(ay, az)))
roll  = degrees(atan2(ay, az))
```
Yaw = gyro integration, reset at each zeroing.

**Read angles when static, not during transit.** While the patient is being moved, linear acceleration
contaminates the gravity estimate. Show a **"settled / moving"** indicator so the clinician knows when the
number is trustworthy — the same signal-quality discipline already used in the nystagmus detector, and the
same reason never to report a number the data cannot support.

**Expected accuracy:** pitch/roll **±1–2°, no drift**; yaw ±1–3° over a manoeuvre. Dix-Hallpike tolerance is
roughly ±10°, so this is comfortably clinical-grade.

---

## 7. Two repo facts that matter

1. **`src/core/head_pose.py` cannot work inside Frenzel goggles.** It derives yaw/pitch/roll from MediaPipe
   face landmarks (nose, chin, eye corners). Inside goggles the camera sees one eye — there is no face to
   landmark. So the IMU is not an upgrade to an existing signal, it **fills a genuine hole**. It also yields
   *absolute angle against gravity*, which face landmarks could never provide.
2. **Live camera capture does not exist yet.** Every `VideoCapture` in the repo reads from a **file**
   (`src/modules/ingestion/video_ingestion.py`, `tools/*`, etc.). Live head-angle display requires building a
   live-capture path. **This is the larger piece of work, not the sensor.**

---

## 8. OPEN QUESTION — decide before building

**Does the Frenzel currently record through its own bundled software?**

- **If yes** → the IMU must log alongside those recordings, requiring **post-hoc sync** (harder: clock
  alignment, no shared timebase).
- **If Python takes over capture directly** → sync is free: timestamp frames and IMU samples on arrival with
  `time.perf_counter()`.

This decision shapes the whole software design and should be settled first.

---

## 9. Payoff for the existing nystagmus detector

Once the IMU stream is timestamp-synced to video frames:

- **Live angle readout with per-step target zones** — "Dix-Hallpike right: turn 45°, extend 20°", numbers
  turning green when in range. This is the primary deliverable.
- **Automatic segmentation of the positional exam** by head angle (sitting → DH-right → held 58 s → sit up),
  feeding `segments.py` instead of manual marking.
- **Empirical test of head-motion invariance.** The sclera-balance gaze-zone rule is currently *asserted* to
  be head-motion invariant. A synced head-angle channel could demonstrate it rather than assume it, and flag
  epochs where head motion makes the eye signal untrustworthy.

---

## 10. Next steps

1. **Answer §8** — bundled recorder vs Python capture.
2. **Run the `ffmpeg` camera-format check** (§4) — decides hub vs cable-bundle.
3. **Order the XIAO nRF52840 Sense** — allow ~2 weeks to India.
4. **Build software against a simulated IMU stream** so it is finished and tested before hardware lands:
   live capture + head-angle calculation + zero button + settled/moving indicator + target-zone overlay.

---

## Honesty notes

- The XIAO board **weight (~3 g) is an estimate** — `wiki.seeedstudio.com` was blocked by the session's
  network proxy and could not be verified. Dimensions (21 × 17.8 × 3.5 mm) are reliable.
- The Evelta BNO055 listing was likewise unreachable, so its **price and stock were never confirmed**. The
  assessment of the BNO055 rests on its datasheet characteristics, not that listing.
- Accuracy and bandwidth figures are **design estimates, not measurements**. Nothing has been built, tested
  or ordered.
