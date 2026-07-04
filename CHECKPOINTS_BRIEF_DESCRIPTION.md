# Checkpoints — Brief Description

One section per locked checkpoint (a named git tag). Each is a **known-good state** you can return to:

```
git fetch origin --tags
git checkout checkpoint/<name>
```

Checkpoints are listed **oldest → newest**. They build on each other.

---

## checkpoint/ellseg-anchor
**Step 1 — Location of Iris by EllSeg (LOCKED).** *(alias tag: `iris-step1-ellseg-location`)*

**What was achieved:** established that the pretrained **EllSeg** net gives a reliable, continuous
*location* of the iris (its disc centroid stays on the iris front-on, at side gaze, and half-covered; it
never jumps to canthus/lid). Its raw *mask/outline* is jagged and its size/shape flicker, so those are **not**
used. Framing locked: **"EllSeg tells us WHERE the iris is; anatomy tells us WHAT it is."**

**Use from EllSeg:** approximate iris centre, radius, frame-to-frame anchor, search seed.
**Do not use:** its jagged mask, exact outline, fluctuating centre, or side-gaze dropouts.

**Key files:** `tools/sudhir_ellseg_raw_video.py` (raw disc over the whole clip), rules #17–#21 in
`docs/CLINICAL_NYSTAGMUS_DETECTOR.md`.

---

## (retired) accurate iris marking — `checkpoint/limbus-arc-rescue-gate`
**Removed 2026-07-04.** We built a fixed-radius iris circle from the sclera-facing limbus arc with a side-gaze
rescue gate (`tools/step2_fixed_base_circle.py`), but then **gave up on tracking/marking the iris accurately**
— the clinical goal only needs a rough eye-position signal (the EllSeg centroid), not a precise iris outline.
The checkpoint tag was deleted; the code remains in history. We jump straight from the EllSeg anchor to the
nystagmus-direction pipeline below.

---

## checkpoint/nystagmus-direction-primary-gaze-zones
**Clinical nystagmus-direction detector with gaze-zone classification (LOCKED — current).**

**What was achieved — the goal was reframed** from perfect iris geometry to the **clinical question:** *is
there rhythmic jerk nystagmus, and what is the fast-phase direction, in each gaze position?*

Pipeline:
- **Clean eye-position signal** — `tools/ellseg_centroid_trace.py`: the plain EllSeg disc **centroid** per eye
  (reliable and smooth), plus per-eye **sclera balance** (sclera pixels on each side of the iris) and disc
  area. This proved cleaner than any oval-fit marker.
- **Windowed Slow-Phase Asymmetry detector** — `tools/nystagmus_direction.py`: combines both eyes, keeps the
  slow-phase drift (removes only head/camera drift slower than ~3 s), and reads the fast-phase **direction**
  from the agreement of time-asymmetry + speed-asymmetry + velocity-skew. It does **not** require a crisp
  fast-phase spike (which 30 fps cannot resolve).
- **Gaze-Zone classification** — anatomical and head-motion-invariant: gaze zone (primary / left / right /
  extreme) is read from **sclera balance + canthus proximity** (where the iris sits inside the eye opening),
  **not** from raw image position. The **Sustained Gaze Zone rule** suppresses the within-beat oscillation
  over a ~2 s window and references the patient's habitual (median) position = primary, so the nystagmus's own
  slow drift no longer moves the gaze zone.

**Result (right vestibular neuritis clip, 30 fps):** **primary gaze, 21 s → LEFT-beating** (confidence 0.34) —
the clinically correct answer.

**Known limit:** 30 fps undersamples the fast phase, so we get **direction only, no beat rate**; 60/120/240 fps
is recommended for rate/velocity. Output is a **table only** (no graph).

**Also built this session (not required for the direction answer, available for later):** the Step-2c
gaze-oval experiments and the **Orbit Lock** (rigid canthus frame with absolute inter-canthus distance, nose-
bridge anchor, mark-once tracker, and the `tools/rit_orbit_lock_marker_v2.html` marker). These can feed a
head-free eye position in later if confidence needs raising.

**Key files:** `tools/ellseg_centroid_trace.py`, `tools/nystagmus_direction.py`.

---

## checkpoint/nystagmus-single-eye-calibration
**Single-eye support + one-click seeding + first multi-clip calibration (LOCKED — current).**

**What was achieved:**
- **Single-eye mode** — the detector uses whichever eye(s) are present (nystagmus is conjugate, so one *clear*
  iris is enough; both eyes are averaged when available). Lets you skip a ptotic/occluded/poorly-tracked eye.
- **One-click iris seed marker** (`tools/rit_iris_seed_marker.html`) — click the clearer iris on the first
  frame; window auto-sizes from that iris; saved to `manual_seeds.json` (1- or 2-eye entries). Seeding
  priority: manual → clinician orbit marks → automatic dark-blob finder.
- **First calibration** (`tools/nystagmus_calibrate.py`, 8 clinician-seeded clips; results in
  `docs/CALIBRATION_RESULTS.md`): **4/8 fully correct, incl. 3/3 true negatives** (normal clips correctly
  "no clear nystagmus"). **Confidence separates cleanly** — real nystagmus 0.60–0.63, normal ≤ 0.26.
- **Known remaining bugs:** vertical channel over-calls (fistula "UP-beating" false positive); gaze-evoked not
  detected (gaze zones don't split); some positives missed (head-impulse clip). Next work targets these.

**Key files:** `tools/rit_iris_seed_marker.html`, `manual_seeds.json`, `tools/nystagmus_calibrate.py`,
`docs/CALIBRATION_RESULTS.md`.

---

*All the detailed rules behind these checkpoints live in `docs/CLINICAL_NYSTAGMUS_DETECTOR.md`. The checkpoint convention
itself is in `CLAUDE.md`. Update this file whenever a new `checkpoint/<name>` is locked.*
