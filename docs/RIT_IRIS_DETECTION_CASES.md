# RIT — Iris Detection Cases (paired with Orbit Lock)

*Spec only — design document. No code/threshold/tracker changes are implied by this file. Source:
Dr. Kothari's hand sketch, 2026-06-30, of "the ways the iris might be seen." Pairs with the validated
**RIT Orbit Lock** probe (`src/core/rit_orbit_lock.py`).*

---

## Core principle

The iris is **not one fixed shape**. The constant is:

> **a DARK structure — disc, arc, or crescent — lying inside the WHITE oval of the sclera (the eye opening).**

What varies is (a) **how much of the iris circle is visible** and (b) **where it sits** in the opening.
The eyeball radius is constant, so the **iris radius is constant** — only the *visible portion* changes
as the eye rotates and the lids cover part of it.

This is exactly why iris and orbit must be solved **together**: when the iris is reduced to a sliver,
the iris alone can't say where it is, but the **sclera oval still frames the opening** and tells the
detector *where to look* and *what radius to expect*. **Orbit first, then iris inside it.**

---

## The five presentation cases (from the sketch)

| # | Sketch | Presentation | What's visible | Detector job | Difficulty |
|---|--------|--------------|----------------|--------------|------------|
| **A** | top-right | **Full / near-full dark disc** centred in sclera | Whole limbus circle, sclera all around | Fit the full limbus circle → centre | Easy (textbook) |
| **B** | top-middle | **Partial dark arc, MEDIAL** | Inner portion of the disc; lateral edge cut by gaze/lid | Fit a circle of the **known radius** to the visible **arc** | Moderate |
| **C** | bottom-right | **Partial dark arc, LATERAL** | Outer portion of the disc; medial edge cut | Same — complete the circle from the visible arc | Moderate |
| **D** | top-left & bottom-left | **Dark crescent / region at a MARGIN** (down-gaze) | Almost no circle — just darkness hugging the lower lid | Fall back to "**darkest region of ~known radius inside the oval**"; place the centre from the crescent + lid geometry | Hard |

Position within the oval is independent of shape: the dark structure may be **centred, medial,
lateral, or bottom**. The detector must not assume centred.

---

## Detection strategy (inside the locked orbit)

1. **Orbit lock first.** The moving orbit (Orbit Lock) gives the current **sclera oval** → the search
   region and the **expected iris radius** (Stage-0 limbus radius, which is constant).
2. **Find the dark iris inside the oval**, in priority of evidence:
   - **Limbus arc fit** (cases A–C): radial dark-iris→bright-sclera edges, RANSAC circle/ellipse of the
     known radius. **Accept partial arcs — never require a full circle.** (`limbus.py` already
     "completes the circle from the visible arc when a lid covers part" — this spec is the set of cases
     it must cover.)
   - **Dark-region fallback** (case D): when too little arc exists, take the **darkest blob of plausible
     radius inside the oval**, near the lower/medial margin, and place the centre by fitting the known
     radius to it. This is the crescent case where pure circle-fitting fails.
3. **Confirm it's iris, not skin/shadow:** dark inside, **scleral white outside** along whatever arc is
   visible. (The current validator's `_arc_support` already needs only `ARC_MIN_FRACTION ≈ 0.17` of the
   boundary to show the dark→bright transition — i.e. it is *already* arc-tolerant; cases A/B/C pass it,
   case D is the one that stresses it.)
4. **Measurement = the iris CENTRE**, reconstructed from the fitted known-radius circle — even when only
   an arc or crescent is visible. Report eye-local position of that centre.

---

## Rules

- **Never require a full circle.** A short arc, correctly placed, is a valid iris. Requiring a full disc
  rejects cases B–D — which are the majority during eye movement.
- **The known radius is the anchor.** The visible portion shrinks; the radius does not. Fit the constant
  radius to whatever is visible.
- **The crescent extremes (case D) are diagnostically important, not edge cases.** They are the
  nystagmus end-points (eye slammed fully to one side / down). They must be handled, not skipped.
- **Couple iris ↔ orbit.** Where the iris is, the orbit is around it, and vice versa. Use the locked
  orbit to locate/size the iris; use the found iris to confirm/refine the orbit.
- **Never invent.** If no dark structure of plausible radius is found inside the oval → mark the frame a
  **gap** (blink / occlusion / lost), do not guess a centre.

---

## Acceptance test

On the overlay, the iris centre is placed correctly across **all five presentations** — full disc,
medial arc, lateral arc, and dark crescent (down-gaze) — and the marker stays anatomically attached
through the extremes. Judged **visually** (Dr. K watches), not by confidence numbers — a high arc/fit
score is not proof of attachment (the lesson from the iris-template drift).

---

## Observed failures + hardened rules (vestibular-neuritis clip, scrubbed 2026-06-30)

Scrubbing the orbit-lock overlay: the **moving orbit (green) holds correctly on the eye**, but the
**existing iris tracker (white circle, old `iris_tracker.py`) fails** — confirming the iris must be
re-derived INSIDE the locked orbit. Concrete failures of the white iris marker:

- **f188** (gaze left): comes off the iris, cannot recover.
- **f192, f195**: re-attempts land OFF the iris (doesn't recognise the dark arc as iris).
- **f227, f228, f240, f244, f255**: marks part-iris / part-sclera — wrong size/placement on the
  partially-hidden iris.
- **f262**: spans iris + sclera + eyelid (no single dark structure there).
- **f269**: leaves the eye-opening oval entirely, onto the medial eyelid.
- **f341**: lands on peri-orbital skin / cheek, OUTSIDE the orbit (verified: green orbit on eye, white
  iris on skin below).
- **f348**: catches specks of white on the eyelid.
- **f350, f353**: outside the oval.
- **f382 → 460, 469, 475+**: eye is back, iris clearly visible, but the tracker **gives up** — no iris
  attempt at all (verified: green orbit correct, `iris:NONE`).

### Hardened rules (Dr. Kothari, from these failures)
1. **Hard containment.** The iris centre may NEVER be placed outside the (now moving) orbit oval. Search
   ONLY inside the locked orbit. Anything on sclera-edge, eyelid, cheek, nose, or skin is rejected by
   construction — the f269/f341/f350/f353 failures become impossible.
2. **Dark-only.** Within the orbit, mark ONLY the dark circular/arc region bounded by sclera. Never
   sclera, eyelid, or skin; never a part-iris/part-sclera mix (f227/f262).
3. **Imagine the circle from the visible arc.** On side gaze the iris is partly hidden — it reads as a
   **partial arc, foreshortened to an OVAL**. Fit the visible dark arc, **reconstruct the full (imagined)
   ellipse/circle**, and take **its centre** as the tracked iris centre — even though only the curve is
   visible (f188/f192/f195).
4. **Never give up — reacquire.** When the eye returns to view, re-find the iris inside the orbit.
   Sustained "no attempt" (f382→475) is a failure.

### Side-gaze foreshortening (Dr. Kothari sketch, 2026-06-30)
The circular iris is seen at an angle as the eye rotates, and **foreshortens into an OVAL**:
- **FRONT gaze** → full **circle** (fit a circle; establishes the true iris radius).
- **SIDE gaze** → iris at the margin, partly hidden; fit/imagine the full circle from the visible **arc**
  (centre-from-arc with the radius held constant — occlusion-invariant).
- **EXTREME side ("VP")** → strongly foreshortened → fit an **ELLIPSE**: **major axis ≈ the true iris
  diameter** (constant — the iris does NOT shrink), **minor axis compressed along the gaze/horizontal
  direction**, long axis ≈ vertical. **Track the CENTRE of this imagined oval.**

So: the radius (major axis) is fixed once the iris is well seen near front gaze; at side/extreme gaze
only the **centre** and the **foreshortening (minor axis)** are fit from the visible arc. Seeding from
the **orbit centroid** (not the darkest blob) is what makes the fit reliable — the dark blob lands on
lashes/corners (verified f382/f341, 2026-06-30); multi-seed (last-good → orbit-centroid → dark) cut the
gaps from ~40% to ~5%.

### Detection method — what works, what's hard (2026-06-30)
Building the iris-in-orbit detector probe (`tools/rit_iris_in_orbit_probe.py`) established:
- **Seed from the orbit centroid / last-good centre, NOT the darkest blob** → gaps fell ~40%→~5%.
- **Hold the iris radius** once a full iris is seen; reject shrunk radii → marker covers the whole dark
  iris (fixed f378 r=62→81).
- **Centre = centroid of the VISIBLE dark iris; shape = an ellipse covering it** (circle when full, oval
  when foreshortened) — matches the clinician marks. Works for full/near-front and moderate side gaze.
- **Visualisation:** hatch-shade the region the detector thinks is iris + a red centre dot, so the
  clinician can correct BOTH the region and the centre (not just a ring).

**The hard case = extreme side gaze (heavily-occluded sliver, e.g. f267/f311).** Two dead ends:
- *Largest/nearest dark blob* grabs lashes/shadow.
- *Whole-blob convexity (solidity)* rejects the iris too, because **the iris is connected to the dark
  upper-lid lashes** — the combined blob is non-convex (even a full iris at f187 was rejected).

**Corrected rule (Dr. Kothari):** the convex feature is the **LIMBUS** — the boundary where the dark
iris meets the **bright sclera** bulges convexly *away from the lid edge*. Lashes/shadow border skin,
not sclera, and have no such convex sclera-bordering arc.

**UNIFIED sclera-anchored detector — IMPLEMENTED & validated 2026-06-30** (`fit_iris` in
`tools/rit_iris_in_orbit_probe.py`). Per frame, inside the moving orbit:
1. Find the bright **sclera** (whitish, low-saturation in HSV) and the **dark** regions.
2. A **strong morphological open** (~0.22·R) detaches the dark upper-lid lashes from the iris.
3. **Select the dark component that BORDERS the sclera** (the iris — lashes/shadow border skin) nearest
   the tracked centre. This is the operational form of "a convex dark area protruding into the sclera".
4. **Centre = its centroid; shape = an ellipse covering it** (circle when full, foreshortened oval when
   a sliver), **capped at the held iris diameter**.
Validated against the clinician marks on f187 (full disc), f267 & f311 (extreme nasal slivers): centre
lands in the visible dark, ellipse covers it. The earlier single approaches each failed half the cases
(sclera-limbus-only fit broke on the full iris's one-sided arc; dark-blob-only grabbed lashes) — the
UNION (sclera to SELECT the region, dark region for the CENTRE/shape) is what handles both.
**Visualisation:** hatch-shade the selected region + red centre dot so the clinician corrects both.

---

## Hardened rules round 2 (Dr. Kothari, scrubbing the UNIFIED overlay, 2026-06-30)

Reviewing `_rit_iris_in_orbit_probe/iris_in_orbit_overlay.mp4` (frames 187–382). The unified detector
no longer gives up and stays inside the orbit, but the **iris boundary and centre are still loose**.
New rules, to be applied when the detector is revised (NOT yet implemented):

1. **Dark-brown/black ONLY — colour is the iris definition.** Most Indian irises are dark brown or
   black. **If a pixel is not dark brown / dark black, it is NOT iris.** Skin can be brown but is never
   as dark as the iris. → gate the iris region on darkness/low-value brown; **no sclera (no non-dark
   pixel) may lie inside the iris mark.** (f187 both eyes: mark spilled into sclera, esp. medially;
   f262-L: a sliver of iris left *outside* the mark — the mark must fully cover the dark and nothing but
   the dark.)
2. **The iris mark is a VERTICAL ellipse, not horizontal, not a circle.** On side/extreme gaze the iris
   foreshortens with the **long (major) axis vertical** and the minor axis compressed horizontally
   (gaze direction). Concrete corrections:
   - f187–195 LEFT: drawn as a **horizontal** oval → WRONG, must be **vertical**.
   - f227: drawn as a **circle** → WRONG, must be a **vertical ellipsoid**.
   - f269: starting point (medial) ok but the **oval axis is too horizontal — rotate ~20° clockwise**.
   So: ellipse orientation must default to ~vertical major axis and tilt only as the gaze geometry
   dictates; never default to horizontal or circular under occlusion.
3. **Centre sits INSIDE the dark, not at/over its edge.** f311 & f187-R: centre pushed too far out
   (toward/into sclera) → pull the centre **inward** into the dark region. The reconstructed-circle
   centre must still land on dark, not on the limbus/sclera.
4. **Already good (keep):** f262-R ("almost perfect"), f269 (good), f267-R (covers the dark). The fix is
   tightening boundary + orientation + centre, not redesigning the detector.

Clean per-eye crops of the original video for Dr. K to hand-mark the ground truth are in
`~/Desktop/RIT_iris_tomark/` (f227,262,267,341,348,350,353,378,382 × L/R). His returned marks define the
target the revised `fit_iris` must reproduce. f378 = "terrible", f382-L = bad, f341-R = wrong — priority cases.

---

## Eyeball-projection model + hatch rule (Dr. Kothari diagram, 2026-06-30)

Dr. K's hand diagram (FRONT / SIDE / VP=extreme) fixes the iris-shape geometry definitively:

- **The iris is a flat disc on the eyeball sphere.** Its image shape is the disc projected.
- **FRONT gaze → circle.** **SIDE → foreshortened oval.** **VP (extreme side) → thin oval at the orbit
  edge.** "OVAL when it goes to the extreme side."
- **The oval foreshortens ALONG the gaze direction and stays full diameter PERPENDICULAR to it**, so the
  long (major) axis is perpendicular to gaze and the oval **TILTS** as gaze goes oblique: look right →
  tall vertical oval; look down-and-right → oval tilts along that diagonal; look up → tilts up.
- **Use the visible dark bordered by white to complete the (dotted) imagined circle** — the "Track"
  sketches show the visible dark arc + a dotted completion of the full iris.

**Implemented in `fit_iris` (probe), 2026-06-30:** gaze vector = (iris centre − orbit centroid); major
axis = held diameter ⟂ gaze; minor = D·cos(gaze), with cos from the displacement magnitude
(sin = |g|/R_eb, R_eb ≈ 1.9·iris-radius) — NOT measured from the dark width (which the canthus shadow
inflates); tilt = ⟂ to gaze. This replaced earlier horizontal-oval / circle / vertical-only fits.

**Hatch rule (verification, rule #2):** the detector must **hatch-shade the exact region it calls iris**
on the overlay, so the *region* (not just the outline) is judged. The hatch must contain **only dark iris
bordered by white sclera** (sometimes brown lid skin); never canthus shadow, sclera, or skin. The clean,
reliable signal is **dark-surrounded-by-white** — anchor on that and complete the circle/oval from it.
Implemented: diagonal hatch over the selected dark component in `iris_in_orbit_overlay.mp4`.

**Open calibration (awaiting Dr. K review of `~/Desktop/RIT_compare`):** (a) foreshortening amount (R_eb);
(b) tilt magnitude; (c) segmentation — where the hatch grabs shadow or misses dark. The hatch separates
the two knobs: fix *segmentation* (what is iris) vs *geometry* (oval shape).

---

## Status

Spec only. Pairs with the **RIT Orbit Lock** probe (validated 2026-06-30: the moving sclera oval stays
on the eye through down/medial extremes where the static oval stranded on the brow). Not implemented;
no validator wiring; awaiting approval to proceed.
