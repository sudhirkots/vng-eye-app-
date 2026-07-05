# Clinical Nystagmus Detector — Dr. Kothari's method

*The goal is a **qualitative clinical nystagmus reader** (see CLINICAL OBJECTIVE below), not precise iris
tracking. The current pipeline: EllSeg iris centroid → windowed slow-phase asymmetry direction → sustained
sclera-balance gaze zones. Earlier sections on accurate iris marking (limbus arc, fixed circle, orbit lock)
are **retired history** — kept for reference, no longer the goal. (Renamed 2026-07-04 from RIT_IRIS_METHOD.md.)*

---

## Hard boundary (most important)
**The iris is ALWAYS inside the almond eye-opening.** Never mark the iris, and never let the completed
circle's centre, fall outside that oval. Brow / cheek / nose / lid skin outside the opening are
disqualified by definition. (This alone kills the eyelid/brow failures.)

## Step 0 — the eyeball oval (DONE)
Track the **almond eye-opening** (palpebral fissure) frame-by-frame — the RIT Orbit Lock already does
this. Everything below happens ONLY inside this oval.

## Step 1 — find the white sclera
The sclera is **one large, UNIFORMLY white patch**. It may carry a few small dark spots (vessels) but is
almost all uniform white. **Discard blotchy white specks** (reflections on brown skin are little white
dots on brown — not uniform). Within the opening, the sclera is the **only purely-white thing**.

## Step 2 — find the iris (dark) against that white
Three tests identify the iris and separate it from lid shadow / nose / brow:
1. **Convex edge abutting the white (ALWAYS true — the definitive test).** The iris's boundary with the
   sclera is a **convex arc** (part of a circle/oval) that **abuts the uniform white**. A dark lid/nose
   meets the white along a straight lid-line or a ragged edge — never a clean convex arc. This convex
   edge is the **limbus**.
2. **Darkest (generally).** The iris is generally darker than any shadow.
3. **Uniformly dark (usually).** Solid dark — may have a few internal blotches, but far more uniform than
   a dark eyelid or nose.

So: darkness + uniformity narrow it down; the **convex edge meeting the white confirms it.**

## Step 3 — complete the iris from the convex arc
The convex limbus arc is **always part of the full iris circle/oval**:
- **Complete the circle/oval from the arc** — never average the visible dark.
- **Centre is on the INSIDE of the convex curve** (toward the iris interior), even when lids hide part.
- **Shape**: a **circle** looking straight at us; a **narrower oval** the more the eye turns; **tilted**
  per gaze direction (the 15-image chart). The major axis ⟂ gaze; thinner the further the eye has gone.

## How the iris appears (all one rule: convex arc → complete the shape → centre inside)
- **A — full:** dark circular island fully surrounded by white (straight gaze). Side gaze → oval island.
- **B — one lid cuts it:** "setting sun" / semicircle (flat lid-line + convex limbus arc).
- **C — both lids cut it (extreme side):** only a **thin convex arc** shows → complete a **narrow oval**.

---

## Step 3 — locate the iris centre by VOTING (validated mechanism, 2026-07-01)
Do NOT pre-pick "which dark blob is the iris" (brow/canthus shadow fool it). Instead:
- Take **every sclera-edge point that faces dark**, and cast a vote **one known iris-radius inward** (along
  the outward-from-sclera normal, onto the dark side).
- The iris's **convex arc** makes all its votes **converge to one centre**; the **straight lid/canthus
  edges scatter**, and deep canthus shadow (no convex arc) contributes nothing. **Peak of the vote map =
  iris centre.** Constrain the centre to the dark side (never inside the sclera).
- Validated on Dr. K's sclera marks: correct on full circle (460-L), side arc + canthus shadow (382-R),
  and the brow-triangle (341-L) — the three cases every prior method failed.

**Concavity rule (confirmed):** the sclera is **concave where it faces the iris** (convex iris arc),
**straight where it faces the lid**. When the iris is far to one side, go by the **iris's convex edge
against the sclera** and complete the oval — that avoids confusion with deep canthus shadow.

**Open item — foreshortening at extreme gaze (227-R):** the iris is an OVAL, so the centre is only a
*minor*-radius from the limbus, not the full radius; casting the full radius overshoots. Must complete an
**oval** (vertical major = known radius, horizontal minor = foreshortened by gaze), not a circle.

## Step 2 (reframed) — IRIS = ALMOND − SCLERA (Dr. K, 2026-07-01)
The almond (eye opening, set by the two eyelids) contains **only sclera or iris**. So don't hunt the
iris as "dark" — **subtract**: `iris = almond − sclera`. This removes the brow/canthus/"which dark blob"
problem by construction. Practical caveat: the tracked orbit almond is **looser** than the true
lid-margin opening, so also **exclude skin colour** (orange-brown lid) → `iris = almond − sclera − skin`.
Lashes are thin → removed by a morphological open. Validated: clean for front/moderate gaze; at extreme
gaze the iris sliver is found but deep **canthus shadow** at the very corner still leaks in slightly —
**temporal continuity** (previous frame's iris side) is the clean fix in the real detector.

## Exact partition (Dr. K, 2026-07-01): almond = iris + sclera, nothing more, nothing less
The almond is tiled **completely and exactly** by iris + sclera: nothing outside it, AND no gaps inside —
every almond pixel is either iris or sclera, and together they fill it exactly. Consequences:
- There is **no "skin" inside a correct almond.** If skin appears inside, the **almond is too big** — tighten
  it. (The earlier skin-subtraction was a patch for a loose orbit, not a real step.)
- **Build the almond as sclera ∪ iris** (the union of the two things actually identified — white/pink +
  round dark), not from the loose tracked orbit. Then iris = almond − sclera and sclera = almond − iris
  hold **exactly**, with no skin term and no gaps.
- **Self-check:** if iris + sclera do not exactly fill one clean almond shape, the frame's read is wrong —
  re-examine it.

## The three-way cross-check (Dr. K, 2026-07-01) — the unifying logic
Three mutually-constraining things; use whichever is CLEAREST in a frame to lock the other two:
1. **Almond** = the eye opening, and **almond = iris + sclera** (so iris = almond − sclera; sclera =
   almond − iris).
2. **Iris** = always **ROUND** (circle front-on; foreshortened oval to the side).
3. **Sclera** = the part of the almond that is not iris.

Strategy: if the almond is clear, lock it (everything is inside it); if the sclera is clear, subtract it
for the iris; if the round iris is obvious, the rest is sclera. Each frame take the most reliable cue and
let the invariants reconstruct + check the others.

**The limbus is ALWAYS CIRCULAR** — a full circle, half circle, or arc. So the **circular** stretch of the
sclera's edge is the limbus (the iris); the **straight** stretches are lid margins. Fit/complete the circle
(foreshortened to an oval at the side) from that circular arc; the centre is inside it. This is the test
that separates iris-edge from lid-edge, and lets a partial arc reconstruct the whole iris.

**Limbus curvature (the wrong-side disambiguator):** at the limbus the **iris side is CONVEX**, the
**sclera side is CONCAVE** (the sclera cups the round iris). So the **centre of the limbus circle is on
the convex / iris side** — dark iris inside the arc, white sclera outside. This fixes which way to
complete the circle and where the centre lies, from the arc alone (no more circles bulging into the sclera).

**Specular spots on the eyelids — outside the almond, so excluded by construction.** Shiny highlights on
the lid skin lie OUTSIDE the almond; since nothing outside the almond can be sclera or iris, they can never
mislead the read (this was the old "blotchy white specks" error). Specular spots that fall ON the sclera or
iris inside the almond are absorbed by the uniform-fill / ignore-thin rules.

**Eyelashes — ignore them.** Lashes are thin dark strands projecting into the almond; they blur the limbus
and the lid margin. Strip them first (they are thin → removed by a morphological open), then read the
**smooth** eyelid margin (the proper almond) and the **smooth circular** limbus from the shapes underneath.
Complete the almond as if there were NO lashes — only an eyelid. A lash strand is never iris and never the
limbus; only the smooth round arc is.

**Hard invariants (never violated):** nothing outside the almond; the iris is round; the limbus is a
circular arc, convex toward the iris / concave toward the sclera, centre on the iris side; almond =
iris + sclera (exact partition, no gaps); eyelashes ignored.

## Build order / status
- [x] Step 0 — orbit lock (almond opening), validated.
- [x] Step 1 — sclera = bright-or-pink between the lids, filled, lid skin excluded. *(good; pink-tail /
      upper-lid edge still being tuned against Dr. K's 10 marks)*
- [x] Step 2 — iris = darkest/uniform with **convex edge abutting white** (the limbus).
- [~] Step 3 — locate centre by known-radius **voting**; works for circle + moderate arcs; **oval
      completion for extreme gaze still to add**.
- Hard boundary enforced throughout: nothing outside the almond.

---

## Sudhir's iris–sclera–almond method — clinician sketches + finalized rules (2026-07-03)

*"Sudhir's iris–sclera–almond method" = EllSeg locates → find sclera (white/pink) → iris is the uniformly-dark
region abutting the sclera along a convex arc → iris = almond − sclera → mark only the visible part.
Implemented in `tools/sudhirs_iris_sclera_almond_method.py`.*

Three hand sketches ("the ways the iris might be seen"), to be stored in `docs/iris_sketches/`
(`iris_ways_seen_1.jpg`, `_2.jpg`, `_3.jpg`). What they encode:

- **Sheet 1 & 2 — the iris can sit ANYWHERE in the almond**, and be any *portion*: a full disc high/central,
  an oval against a side, a crescent at the top or bottom margin, a **tiny nub in a corner**, or **absent
  entirely** (empty almond = iris hidden/blink). The **hatch-ticks along the lid lines are lashes — never
  iris.** The visible iris is drawn **filled solidly / uniformly** (one piece, not blotches).
- **Sheet 3 (FRONT / SIDE / VP + "Track")** — the iris is a full circle seen front-on and **foreshortens to
  an OVAL at the extreme side**. The dotted circle + "Track" is the *completion / centre* step: imagine the
  whole iris from the visible arc. **This completion is deferred** — see the current-task note below.

### Finalized iris-MARKING rules (this is the spec the marker must follow)
1. **EllSeg = approximate location only** (region + rough size); never its exact outline as the mark.
2. **Find the sclera first** — the uniform **white or pink/reddish (at least whitish/pinkish)** patch inside
   the opening.
3. **Iris = the uniformly-dark region that ABUTS the sclera** — dark against white/pink, one solid piece
   (not scattered blotches).
4. **Convex-arc discriminator** — the iris↔sclera edge is a **smooth convex arc** (limbus); a **lid/lash edge
   is straight or ragged**. Only the smooth-convex-against-sclera boundary counts as iris edge.
5. **Almond = sclera + iris** → **iris = almond − sclera**; nothing else lives inside the opening. Lashes
   (thin) are stripped first.
6. **Shape = round / arc / crescent / oval** (oval at extreme side). The iris may be any visible portion.
7. **Empty almond → mark NOTHING** (hidden/blink). Never invent an iris.

### Shape rule (Dr. K, 2026-07-03) — the iris is a CIRCLE; complete it
The iris shape is **always a circle** (an **oval** at the sides, a **sector/arc** when the lid covers part).
When ~60–70% of the circumference is tracked, **fit and COMPLETE the circle/oval** to that arc. Consequences
for the mark:
- **Nothing outside the fitted circle** — no protrusions / "horns" running along the lid or lash margin.
- **Fill the interior** — no gray gaps left inside the circle (catch-light, mid-tone iris → still iris).
- Covered iris = a clean **sector** of that circle, not a blob.
Centre estimation from the completed circle is the later step; for now the *mark* is the completed circle/oval
(clipped off the white/pink sclera), painted red.

### Priority + oval geometry (Dr. K, 2026-07-03) — SHAPE OVERRIDES DARKNESS
**Order of rules (overriding first):**
1. **Complete the circle/oval** from the clearest stretch of limbus you can make out. The smooth shape is the law.
2. **THEN** select dark *within* that shape. Selecting dark FIRST lets the iris grow toward the **medial canthus
   / lid / lashes**; completing the shape first prevents it. Dark that falls OUTSIDE the completed oval (e.g.
   canthal shadow) is **not iris — leave it out**, even though it's dark.

**Oval geometry:** the iris is a fixed-radius circle; at gaze it foreshortens to an oval. The **LONG axis =
perpendicular to the gaze direction** (keeps the full iris width); the **SHORT axis = along the gaze direction**
(compressed). Looking left → tall vertical oval; looking down-and-right → oval tilted so its long axis is
perpendicular to that diagonal. Angulation follows gaze. (Foreshorten more the further the eye has turned.)

### CONFIRMED iris rules (Dr. K, four Q&A, 2026-07-03) — the definitive spec
**What the iris IS:**
- **Shape:** a **circle** (front), an **oval** (side gaze), or **part of that shape** (arc/sector when the lid
  covers it). **Nothing outside that round shape** counts — Q4: *strictly the round shape* (drop any dark that
  isn't part of it: canthus, lash).
- **Darkness:** the iris is **dark and contains nothing LIGHT — where "light" = sclera-bright.** The **mid-tone
  rim** near the limbus is **still iris** (Q2: *include out to the limbus*). Not-all-dark-is-iris → shape decides.

**What to MARK:**
- **Extent (Q1): only what is SEEN.** Where the lid covers the iris, the mark **stops at the lid line** (an
  arc/sector). Do NOT paint under the eyelid.
- **Rim (Q2):** include the whole iris out to the limbus (the boundary with the white/pink sclera), even where
  the rim is lighter than the centre.
- **Catch-light (Q3): FILL IT IN.** The bright corneal reflection sits inside the iris shape → no hole in the
  mark. (Only sclera-bright *outside* the shape is excluded.)
- **Empty opening / no iris arc abutting sclera → mark NOTHING.**

**HOW (procedure, maps to `tools/sudhirs_iris_sclera_almond_method.py`):**
1. EllSeg → approximate location only.
2. Sclera = uniform white/pink patch. Limbus = the smooth **convex arc** where dark iris meets sclera (RANSAC on
   sclera-facing edge points; lid line & lashes are straight/ragged → excluded).
3. **Complete the circle/oval** from that limbus arc (radius = known iris radius; long axis ⟂ gaze).
4. **Clip to SEEN:** intersect the completed oval with the eye OPENING (`close(sclera∪dark)` minus sclera) →
   cuts at the lid line, keeps the full iris incl. the mid-tone rim (opening bridges it), off the sclera.
5. **Fill holes** (largest contour filled) → catch-light filled; nothing outside the oval.

---

# Sudhir_iris_method — CANONICAL rules (Dr. K, 2026-07-03)

*This is the single source of truth; it supersedes the evolving sections above. "LSIG" (as Dr. K says it) =
EllSeg, the pretrained locate-and-propose model. Implemented in `tools/sudhirs_iris_sclera_almond_method.py`.*

## Pipeline
- **Step 1 — Orbit Lock:** the clinician marks the orbit (the moving search frame).
- **Step 2 — EllSeg (LSIG) locate:** inside the orbit, EllSeg locates the eye, finds the **sclera** and the
  **convex limbus arc**, narrows the area, and proposes/fills what it thinks is the iris.
- **Step 3 — apply the rules below to mark the visible iris.**

## The rules
1. **Shape is the law.** The iris mark can be **nothing except a circle, an oval, or a part of one.**
2. **Circle → oval → sector with gaze.** Whole iris (front) = **circle**; to the side = **oval**; at the
   extreme = only a **sector**.
3. **Oval orientation.** The oval's **long (vertical) axis is PERPENDICULAR to the direction of gaze.** Looking
   right → an upright/perpendicular oval; looking to the side **and down** → the long axis **tilts outward**
   accordingly.
4. **Lid cuts make sectors.** The eyelid cuts the circle/oval at various positions: a **central cut = "setting
   sun"** (semicircle); a **cut toward the canthus = a "pizza-wedge" sector.**
5. **Mark only what is SEEN.** The iris **under the eyelids is NOT marked**; the iris **behind the canthus is
   NOT marked.**
6. **The iris margin is circular/oval.** The visible iris's boundary against the sclera (the limbus) can be
   **nothing but a circle / part of a circle / oval / part of an oval.** (The lid is where visibility ends,
   not part of the iris circle.)
7. **Dark, and darker than sclera.** The iris is **dark** — a little **less dark toward the limbus** but
   **always darker than the sclera** (which is white or pink). The iris is always a different, darker colour.
8. **Nothing white or pink inside the mark.** If in doubt, **follow the shape of the remaining clearly-dark
   part** of the iris.
9. **Dark ≠ iris.** Dark that is merely **contiguous** to the iris (canthus, under-lid shadow) is **excluded**
   if it does not fit the shape. **SHAPE OVERRIDES COLOUR.**
10. **Catch-light is filled.** The bright corneal reflection sits inside the iris shape → fill it, no hole
    (only sclera-bright *outside* the shape is excluded).
11. **Nothing there → mark NOTHING** (hidden / blink / no iris arc abutting sclera).

## Corrections from the f330–341 review (Dr. K, 2026-07-03)
12. **CONTINUITY RULE** (overrides rule 8's "nothing light"). Once a valid iris arc is found, **continue the
    smooth circle/oval THROUGH any interruption.** A bright patch that falls **INSIDE** the completed circle/
    oval is **overridden** — treated as iris (a reflection), not sclera; do not cut a notch at it. Bright is
    only sclera when it lies **OUTSIDE** the completed shape. (This is why the *top* of the iris was left out
    in almost every frame — the reflection near the limbus broke the completion.)

*Naming convention (Dr. K): every rule gets a short descriptive NAME (e.g. "Continuity Rule") that states
what it does. Named rules so far: Continuity Rule (#12), Almond-Containment Rule (#14), Canthus-Complete-Only
Rule (#13). More to be named as we lock them.*
13. **CANTHUS = complete only.** On the medial side use just **enough** of the dark to complete the circle/
    oval; dark *beyond* the completed shape (canthus shadow) is **not iris — leave it out**, even though it
    is dark. The shape bounds it. (f338 canthus side = the correct example.)
14. **STAY INSIDE THE ALMOND.** First identify the **two eyelids and the sclera between them** = the opening,
    and search **only** within that oval. **Never** mark on the eyelid, below the eyelid, or a stray dark
    spot outside the opening — there is no round dark-abutting-sclera there (fixed #2/#4/#5, which jumped to
    the lower lid).
15. **Shape must be a true oval, not a blob.** e.g. f330 should be an oval whose long axis is ~20° off
    vertical toward ~10 o'clock — not an enlarged blob from grabbing extra dark.
16. **Almond-Context Rule** (fundamental, for Steps 2–4). EllSeg locates the iris well, but sclera-vs-dark
    separation must be done over the **full eye-opening almond** — both eyelids, the visible sclera, the iris,
    and the canthus boundary — **not** an iris-sized crop. **Never compute sclera from a disc-sized region**,
    or the sclera detector is starved. (Diagnostic 2026-07-03: with a disc-sized context the sclera mask sat
    on the iris and missed the real white sclera to the side.) EllSeg = locator only; analysis region = almond.

## What raw EllSeg actually gives (whole-clip review, 2026-07-04)

Ran raw EllSeg over the entire 640-frame clip (`tools/sudhir_ellseg_raw_video.py` → `ellseg_raw.mp4`),
both eyes, **640/640 frames found**. Watching the raw disc alone, with no rules applied:

- **EllSeg reliably LOCATES the iris — always.** Front-on, side gaze, even half-covered, the disc's
  **centroid stays on the iris**. It never jumps to the canthus, the lid, or a stray dark spot. As a
  **location anchor it is trustworthy the whole clip.**
- **But its SHAPE keeps changing.** On open/forward frames the disc is a clean round patch the right
  size. As the eye goes to the side — and especially **coming BACK from the side** — the disc collapses
  to a **thin sliver / crescent**, small and irregular. It tracks only the *visible* dark iris, not the
  full circle.
- **Consequence — CENTRE JITTER.** Because the disc's shape (and area) changes frame to frame, the disc
  *centroid* wobbles even when the true iris centre moves smoothly. The disc area also **under-estimates
  the iris radius at side gaze** (a sliver has small area) — so its size cannot be trusted for the radius.
- **Almost all the sclera lies OUTSIDE the disc** (the disc is just the iris) → confirms the
  Almond-Context Rule: the sclera/almond search must be expanded *around* the anchor, not read from the disc.

17. **Shape-Stabilization Rule** (Dr. K, 2026-07-04) — *the core of the next step.* EllSeg's **location is
    good and continuous**; its **shape is not.** So **do not take EllSeg's disc shape as the iris.** Use
    EllSeg only for the **centre anchor and region**, then **STABILIZE the iris shape** — a **fixed-radius
    circle/oval** whose size does not shrink with the visible sliver — and **REFINE it by the anatomy**
    (darkness, the smooth convex limbus arc/curves, almond = iris + sclera). The steady geometric shape,
    not the flickering disc, is what fixes the centre jitter — especially when the eye is **returning from
    a side gaze**. (Founding reframe restated: optimise the **iris-centre trace**, learn **geometry/motion**,
    not the raw net's appearance.)

18. **Fixed-Radius Foreshortening Rule** (Dr. K, 2026-07-04) — *the geometry that makes #17 concrete.* The
    iris is a flat disc of **ONE fixed radius R**. Model it as a disc on a sphere seen in projection:
    - **Lock R once.** As soon as EllSeg gives a clean **frontal circle**, fix R (e.g. the median iris
      radius over the good open frames). **R never fluctuates afterward** — the *circle* size is constant
      for the whole clip; only its projection changes.
    - **Side gaze → oval, tied to R by ONE number.** At gaze angle **θ** off primary position, the iris
      projects to an ellipse: **major axis = R** (**⟂ to the gaze direction — keeps the full iris width**),
      **minor axis = R·cos θ** (**along the gaze direction — foreshortened**). The further to the side, the
      larger θ, the smaller cos θ, the flatter the oval. "The radius gets smaller as it goes to the side"
      means the **along-gaze (minor) radius shrinks; the cross-gaze (major) radius stays R.**
    - **Give R the oval, not the circle.** The mark/limbus fit uses **(major=R, minor=R·cos θ, angle⟂gaze)**,
      i.e. the **oval's radii**, never a plain circle of radius R at side gaze.
    - **The oval is mathematically related to the fixed circle** by the single foreshortening ratio
      **minor/major = cos θ** (0 = edge-on, 1 = frontal). So we estimate one scalar (θ, or the ratio) per
      frame — from the gaze/centre displacement, with EllSeg's disc eccentricity only as a hint — and the
      whole oval follows; R is not re-estimated.
    - **Division of labour:** EllSeg supplies **tracking + location (centre, θ hint)**; the **anatomy rules**
      (limbus arc, darkness, almond) **refine** the fit; **R is held fixed.** Anatomy modifies EllSeg's
      proposal — it does not let the size drift.
    - *TODO (rules to write): the exact θ→foreshortening schedule (how much it flattens per degree/pixel of
      side travel), the θ estimator (gaze from centre offset vs. disc aspect), and how far anatomy may pull
      the oval before the frame is flagged needs_rescue.*

---

# Next step: How to refine EllSeg localization using the various anatomy rules (shape, colour, eyelid, canthus, …)

**Step 1 is done and LOCKED: EllSeg localizes.** EllSeg gives a good, continuous approximation of *where*
the iris is. We accept that as the starting point. **Everything from here is refinement of that anchor by
anatomy.** The fundamental statement:

> **EllSeg tells us WHERE the iris is. Anatomy tells us WHAT the iris is.**

## EllSeg Anchor Rule (#19, Dr. K, 2026-07-04)
> EllSeg provides the **approximate iris location, centre, and size**. Its mask is only a **proposal**. The
> final iris must be **reconstructed from anatomy**: a smooth circle / oval / sector inside the eyelid-opening
> almond, bounded by sclera / canthus / lids, with reflection ignored by the arc-continuity rule.

**USE from EllSeg:** approximate iris centre · approximate iris radius/diameter · frame-to-frame location
anchor · search seed for the almond/iris analysis.

**DO NOT use directly:** EllSeg's jagged mask · its exact outline · its fluctuating centre · its accidental
inclusion of canthus/lid/shadow · its missing chunks during side gaze / reflection.

## Fixed Iris Disc / Projected Oval Rule (#20, Dr. K, 2026-07-04)
> The real iris is a **fixed-size circular disc**. Its true diameter does **not** fluctuate frame to frame.
> When the eye turns to the side, that same fixed circle is seen in projection as an **oval**. The oval's
> **long axis remains the true iris diameter**; only the **short axis foreshortens** with gaze angle.

**One sentence to record:** *The iris size is anatomically fixed; EllSeg may move the centre, but it may not
resize or distort the iris. The final mark is a fixed-size circle projected into an oval according to gaze
and clipped by eyelids / canthus.* (Do NOT accept frame-to-frame size shimmer from EllSeg.)

## The refinement algorithm (high level)
1. **EllSeg locates** — approximate iris position and size.
2. **Stabilize radius** — keep iris size largely constant across frames; the iris cannot shrink/grow rapidly.
3. **Find almond context** — around EllSeg's location, identify the eyelid-opening space: upper lid, lower
   lid, canthi, sclera, iris.
4. **Find true iris evidence** — the dark iris arc/region within the almond, especially where it touches
   true sclera.
5. **Reconstruct shape** — replace EllSeg's irregular mask with a **smooth circle / oval / sector** of the
   expected size and orientation.
6. **Clip to what is visible** — cut only where eyelid / canthus hides the iris.
7. **Stabilize centre** — use the **reconstructed smooth shape + previous frame**, NOT the raw EllSeg mask
   centroid, to avoid flutter.

## Practical size/shape procedure (the fixed-disc → oval mechanics)
1. **Estimate true iris circle size** from good frontal/open frames.
2. **Lock that size** as the anatomical iris diameter (this is R of #18).
3. **Use EllSeg for centre/location hint**, not for per-frame radius.
4. **Convert circle → oval by gaze:** front = circle; side = tall/vertical oval; extreme side = narrower oval.
5. **Long axis stays fixed** at the true iris diameter.
6. **Short axis shrinks mathematically** as the eye turns away (minor = R·cos θ, per #18).
7. **Clip the oval** by lids/canthus so only the visible part is painted.
8. **Do NOT accept frame-to-frame size shimmer** from EllSeg.

*These are the rules to fill in next — shape, colour (sclera white/pink vs. iris dark), eyelid line, canthus
boundary, arc-continuity through reflection — each becomes a named refinement rule as we lock it on the clip.*

## Two-step naming (Dr. K, 2026-07-04)
- **Step 1 — Location of Iris by EllSeg** (LOCKED): EllSeg localizes; accept centre/size/anchor.
- **Step 2 — Refinement of the Iris** (starts here): refine EllSeg's Step-1 location by anatomy. Rule #21 is
  the FIRST refinement pass — deliberately kept simpler than the full oval/anatomy model.

## Rule #21 — Refinement of the Iris: Fixed Circle / Sclera-Bounded Iris Rule (Dr. K, 2026-07-04)
> After EllSeg locates the iris, **do not trust EllSeg's changing outline or changing radius.** Use EllSeg
> only as a **centre/size hint.** Then refine by looking for a **dark circular region with a sharp limbus
> margin, bordered by relatively uniform white or pink sclera**, especially on the **medial and lateral**
> sides. **Fit and complete a circle of FIXED radius** from that sclera-facing limbus arc.

**One-line record:** *First refinement pass = fixed circle from the sclera-facing limbus. The iris is a dark
circular structure with a sharp margin against relatively white/pink sclera. EllSeg gives the seed; the final
iris circle has FIXED radius and is fitted from the limbus arc, not from EllSeg's jagged mask.*

**This first version DELIBERATELY does NOT handle** (deferred to later refinement rules):
oval foreshortening · perfect eyelid clipping · perfect canthus boundary · side-gaze oval geometry ·
torsion / texture · exact visible-sector shape. **Just lock the iris as a stable circle.**

**Procedure:**
1. Pick a **stable radius R** from good EllSeg / frontal frames.
2. Each frame: use the **EllSeg centre only as a seed**.
3. Search nearby for **dark pixels forming a curved/convex limbus edge**.
4. **Prefer edges where dark iris touches white/pink sclera** (medial & lateral limbus).
5. **Fit a circle of FIXED radius R** to that limbus evidence.
6. **Complete the circle** even if part is hidden or broken by reflection (arc-continuity, Rule #12).
7. Use the **fitted circle centre** as the refined iris centre.
8. **Do not let the circle radius change** frame by frame.

---

## STEP 2 = Refinement of the EllSeg anchor. Sub-step **2a — Fixed Base Circle From Sclera-Bounded Iris** (Dr. K, 2026-07-04)

*Step 2 as a whole = refine the locked EllSeg anchor by anatomy. This is its first part, **2a**. Implemented
FRESH in `tools/step2_fixed_base_circle.py` — it does NOT use the old iris-sclera-almond method.*

**The rule (Dr. K's words):**
> Look in and around the EllSeg-selected iris region. Find the dark circular iris region whose **left/right
> margins are bounded by relatively uniform white or pink-white sclera.** Use the **clearest, largest valid
> full-circle evidence across the clip** to estimate the base iris radius. Once this base radius is chosen,
> **LOCK it. Do not allow the iris radius to change frame by frame.**

**More precise:**
> The true iris is the **largest stable dark circle**, with a **sharp limbus margin against sclera on either
> side.** EllSeg gives the approximate location; the final **base radius** comes from the **best
> sclera-bounded circular iris evidence**, not from EllSeg's fluctuating mask.

**Future note (do NOT implement yet in 2a):** Side gaze may later make the fixed circle appear as an **oval by
foreshortening**, but the underlying **anatomical iris size stays fixed** (Rule #18). Foreshortening,
eyelid clipping, canthus exclusion, sector shape, torsion/texture are all **deferred**.

**Diagnostic output (per `tools/step2_fixed_base_circle.py`):** original frame · EllSeg disc/anchor ·
detected sclera-facing limbus points · fitted fixed-radius circle · chosen base radius R · circle-centre
trace (+ a radius-stability plot: per-frame free fit vs the locked flat R).

**PASS/FAIL question:** *Is the fitted circle stable in size and roughly centred on the true iris, without
shrinking/growing with EllSeg shimmer?*

**Result of the first 2a run (2026-07-04, clip = right vestibular neuritis, 640 frames, both eyes):**
- **Size stability — PASS.** Locked R = **L 76.5px, R 84.6px**. The free per-frame radius thrashes to ~236px
  and back (the EllSeg shimmer); the locked flat R ignores it. (`step2_radius_stability.png`.)
- **Centre trace — smooth & conjugate** (both eyes move together; clear side saccade ~f230, plateau, return).
  Fixed-R centre fit gives a clean signal despite the garbage free radius. (`step2_centre_trace.png`.)
- **Centring — good on open/moderate frames; drifts into the sclera at EXTREME side gaze** (f220, f286),
  because (i) the sclera mask still grabs bright brown skin/lid so the limbus points scatter onto brow/lid,
  and (ii) no foreshortening yet. Both are deferred to later Step-2 parts. (`step2_diag.png`.)
- **Verdict:** 2a's core claim holds — fixed R eliminates the size shimmer and yields a smooth conjugate
  centre trace. Residual side-gaze drift = the next refinement's target.

**What we LOCK here — narrowly (Dr. K, 2026-07-04):** lock **only** *"Fixed base radius works — the iris size
must NOT follow EllSeg's frame-by-frame shimmer; use a stable base R from good open/frontal frames."*
Checkpoint = **`checkpoint/fixed-base-radius`**. We do **NOT** yet lock a "sclera-bounded circle"
(`checkpoint/sclera-bounded-circle` is NOT claimed), because the diagnostic shows the **limbus points are
still badly polluted by skin/lid/brow edges** — the yellow circle is good on open frames but is pulled by
false boundary evidence at f220/f286/f335.

### Step 2b (next single rule) — True Limbus Evidence Rule (Dr. K, 2026-07-04)
> **Fit the fixed-radius circle ONLY from boundary points where dark iris touches TRUE white/pink sclera.**
> **Ignore red/brown skin, lid crease, eyebrow, lashes, and eyelid edges** even if they create strong contrast.

**Plain instruction:** keep the fixed base radius; now **clean the limbus evidence.** The red limbus points
should lie **only on the iris↔sclera margin, mainly medial/lateral**, not on skin folds or eyelid/brow edges.
**Regenerate the same diagnostic** and judge whether the red points **collapse onto the true limbus**. This
is the correction to do **before** oval foreshortening.

### Partial Arc Completion Rule (CORE rule, Dr. K, 2026-07-04)
> A full iris circle does **not** need to be visible. If even **one reliable part of the limbus arc** is
> found, the whole **fixed-radius circle can be reconstructed from it.** The most reliable arc is the
> **boundary where dark iris meets TRUE white/pink sclera.**

> **One clean limbus arc is stronger than many noisy edges.** Prefer a **short, anatomically correct
> iris-sclera arc** over a larger collection of edge points from eyelid, skin, lashes, or canthus. Use that
> clean arc to place the fixed-radius circle and complete the rest.

### Medial/Lateral Limbus Arc Rule (Dr. K, 2026-07-04)
> After EllSeg gives the iris anchor, **look FIRST on the medial and lateral sides** of the iris for a clean
> limbus arc: **dark iris on one side, white/pink sclera on the other.** If a valid arc is found on **either**
> side, use that arc to fit the fixed-radius circle and **complete the whole iris circle.**

**Exclusions:** do NOT use upper/lower eyelid edges as limbus · do NOT use lashes · do NOT use brown skin
folds · do NOT require a full circle · do NOT let many noisy edges overpower one clean sclera-facing arc.

**Practical:** search for **short curved arc segments near the left or right** side of the EllSeg iris anchor.
**Score highest** when the arc is **smooth, convex, at the expected fixed radius, and has dark iris inward /
white-pink sclera outward.** Use the best medial/lateral arc to place the fixed-radius circle.

**One sentence:** *Find one clean iris-sclera arc medially or laterally, lock onto it, and complete the
fixed-radius circle from that arc.*

**Algorithm consequence (what to change in `tools/step2_fixed_base_circle.py`):** do NOT least-squares-fit a
big cloud of edge points. Walk the iris-blob contour, keep only vertices that are **medial/lateral AND abut
TRUE sclera**, group them into **contiguous runs (arcs)**, drop short/scattered runs (skin/lash noise never
forms a long smooth arc), **score each arc** (length · sclera-contact · convexity · consistency with radius
R), and complete the fixed-R circle from the **best arc(s)** — one clean arc is enough.

### Side-Gaze Rescue Gate (safety gate, Dr. K, 2026-07-04)
> Add a conservative **Side-Gaze Rescue Gate** BEFORE drawing the fixed-radius circle. Keep the fixed base
> radius and the medial/lateral limbus arc rule **unchanged.** If the visible iris evidence is **too
> narrow/partial**, OR if the fixed circle would sit **mostly outside** the true dark iris / almond evidence,
> do **NOT** draw a confident full circle: mark the frame **`needs_rescue`** and **carry forward the last
> reliable centre/radius.** Do NOT implement oval foreshortening yet.

**This is only a safety gate** (matches the project rule: false negatives OK, false positives dangerous — a
withheld frame is safe, a confidently-wrong circle is not). **Do not change the radius, do not change the
EllSeg anchor, do not add oval projection.**

### Step 2c — Gaze-Oval Foreshortening Rule (Dr. K, 2026-07-04) — the next build (makes #18 concrete)
> When the person looks **straight ahead** the iris is a **true circle** (radius R, locked). As the eye turns,
> the **same fixed circle foreshortens into an oval**: the **LONG axis stays = 2R and is ALWAYS
> PERPENDICULAR to the direction of gaze**; the **SHORT axis foreshortens along the gaze direction**
> (short = 2R·cos θ). The **more the eye turns, the more it foreshortens** (smaller cos θ). Diagonal gaze
> **tilts** the oval: looking down-and-right, the long axis tilts to ~10–11 o'clock ↔ 4–5 o'clock (i.e. at a
> right angle to the down-right gaze). So the tracked iris **shape must change continuously with gaze.**

**How the gaze is read (EllSeg location is the reliable signal — Step 1):** gaze = the **iris centre's
displacement `g` from the straight-ahead (primary) position** (primary ≈ median iris centre over the good
`fixed_circle_ok` frames, per eye).
- **Direction** = `atan2(g_y, g_x)` → the **long-axis angle = gaze angle + 90°.**
- **Magnitude** `|g|` → how far turned → sets **cos θ**. Model the eye as a sphere: the iris centre projects
  to `|g| = D·sin θ`, so **cos θ = √(1 − (|g|/D)²)**, giving **minor = 2R·√(1 − (|g|/D)²)**, major = 2R.
- **`D` (the displacement at which the iris would be edge-on)** is calibrated from data: on the two-sided
  frames we already measure the visible medial-lateral width `w` (= the foreshortened minor at horizontal
  gaze), so `D = |g| / √(1 − (w/2R)²)`, taken as a robust median. (Physiological sanity: `D ≈ 2R`.)

**Build consequences:** replace the drawn fixed *circle* with this gaze-driven *oval* (major=2R fixed,
minor=2R·cos θ, angle ⟂ gaze). Expected effect: the **currently-rescued / too-wide side-gaze frames**
(f220, f264, f575 …) become **correctly drawn foreshortened ovals**; the Side-Gaze Rescue Gate then only has
to catch true blinks/occlusions. Keep everything in `checkpoint/limbus-arc-rescue-gate` intact; add the oval
on top. Still deferred after this: eyelid/canthus clipping, torsion, exact visible-sector shape.

**Orientation correction (Dr. K, 2026-07-04) — the first oval build failed on TILT.** The oval **angle must
come from anatomical gaze direction, NOT a noisy fitted-mask angle.** If the eye looks **down-right**, the
long axis must tilt **anti-clockwise toward ~10 o'clock** (perpendicular to the down-right gaze); a
**clockwise 1–2 o'clock tilt is anatomically wrong.** Frame corrections: **f220** = do NOT draw (keep
`needs_rescue`, the oval is mis-placed); **f235** = gaze down-right → long axis anti-clockwise, not
vertical/clockwise; **f286** = gaze down-right → long axis ~10 o'clock, not 1–2 o'clock.
- **Split the two estimates:** **EllSeg disc *aspect* (temporally smoothed) → "how flat" (cos θ = minor/major
  of the disc).** **Anatomical gaze → "which way tilted"** = direction from the eye-opening / sclera context
  to the current iris (implemented: iris centroid − sclera centroid; long axis = that angle + 90°).
- **EllSeg ellipse *angle* must NOT override anatomy** if it gives the wrong clockwise/anti-clockwise sense.
- **Short version: use EllSeg for "how flat," anatomy for "which way tilted."**
- **Confidence rescue:** if the app cannot determine the gaze *direction* confidently (oval is clearly flat
  but the anatomical gaze vector is too weak/ambiguous), mark **`needs_rescue`** rather than drawing a
  wrongly-tilted oval.

### Step 2c-i — History-Based Iris Gaze Rule (Dr. K, 2026-07-04) — build the GAZE DIRECTION first, no oval yet
*The first oval attempts failed on gaze DIRECTION (sclera-centroid gaze was noisy / sign-ambiguous). So PAUSE
the oval and build a reliable gaze-direction layer on its own, judged on its own diagnostic.*
> **Gaze direction is determined by the iris centre's MOVEMENT HISTORY from the primary/straight-ahead
> position. Do NOT use the sclera centroid to decide gaze direction. Use sclera only for limbus/boundary
> evidence.**

**Implementation (`tools/step2c_gaze_history.py`):**
1. Define the **primary (straight-ahead) iris centre** per eye from reliable open/frontal frames (frontal =
   roundest EllSeg disc, aspect ≈ 1).
2. Track the iris centre over time from the **EllSeg anchor** + the **refined fixed-radius circle centre when
   reliable**.
3. **Smooth the centre trajectory** over time so single-frame EllSeg shimmer cannot swing the gaze direction.
4. Per frame: **gaze vector = smoothed current iris centre − primary centre.**
5. If the current frame is uncertain/rescued, **infer gaze from the recent reliable trajectory**, not the
   noisy current mask.
6. If the trajectory direction itself is weak/unstable → **`ambiguous_gaze` / `needs_rescue`.**
7. **Diagnostic shows ONLY the gaze layer** (no foreshortened oval): original frame, iris-centre
   trace/history, primary centre, current smoothed centre, gaze arrow; and prints per key frame the primary,
   raw EllSeg centre, refined centre, smoothed centre, gaze angle, magnitude, confidence, status.

**PASS/FAIL:** does the gaze arrow point in the clinically correct direction through the side-gaze sequence,
especially **f220 / f235 / f286 / f335**? If yes, this gaze vector later orients + foreshortens the oval.

**Result (2026-07-04):** history-based trajectory is **smooth & confident** and the horizontal (L/R) sense
is right, BUT it was measured as iris centre in **image coordinates** vs a fixed image-space primary, so
**camera/head motion contaminated it** — max |gaze| ≈ 3.5× iris radius (physiologically impossible for eye
rotation alone), inflating the vertical swings; f286 came out down-left vs the expected down-right. → gaze
must be measured in a **head/camera-fixed eye-local frame**, not image coords. Hence Step 2c-ii below.

### Step 2c-ii — Orbit Lock reference (Dr. K, 2026-07-04) — build BEFORE gaze/foreshortening
*The apparent "head movement" is largely the **camera moving** while the head is still. Fix it with a
head/camera-fixed **Orbit Lock**: an eye-local coordinate frame. Gaze = iris movement **inside the orbit
frame**, not in image coordinates.*
> Before computing gaze/foreshortening, build an **Orbit Lock** reference for each eye from clinician-approved
> landmarks: **medial canthus, lateral canthus, upper eyelid margin, lower eyelid margin.** Track this orbit
> frame through the clip. Gaze is iris movement inside this orbit frame.

- **Horizontal anchors = medial & lateral canthus** (stable; define the eye's L↔R axis). **Vertical = upper &
  lower eyelid margins** (NOT the eyebrow — brow moves independently). 4-point lock gives a true vertical
  opening axis and lets us measure whether the iris moved up/down *inside the lids*.
- **Eye-local frame:** origin = opening centre; x-axis = medial→lateral canthus; y-axis ⟂ x; scale_x =
  canthus distance, scale_y = lid opening height. Iris-in-orbit gaze = iris centre projected onto (x,y),
  normalized by (scale_x/2, scale_y/2) → **head/camera-motion cancelled.**
- **Start from the clinician marks we already have:** the RIT Orbit Lock ground truth (`meta.json`,
  42 marked eye-opening almonds, L 23 / R 19 frames) → derive the 4 landmarks per marked frame (PCA principal
  axis = canthi; minor axis = lid mids) and **interpolate/track** them across the clip.
- **Diagnostic (`tools/orbit_lock.py`):** overlay on the video the **canthus line, upper/lower lid reference,
  eye-local axes, and iris centre**; confirm the orbit frame **follows the eye/head/camera smoothly** (canthus
  distance + orbit centre trajectories should be smooth) BEFORE using it for gaze. Once stable, gaze
  direction becomes much cleaner.

### Rigid Canthus Frame Rule (Dr. K, 2026-07-04) — the orbit lock must be RIGID, not loose
*First orbit-lock attempt was too flexible: interpolating the 4 landmarks independently let the inter-canthus
distance wander 540→676px (25%). The canthi were sliding along the lids — that is NOT a lock.*
> The **medial and lateral canthus points define the fixed eye-socket frame.** They may move together with the
> head/camera, but their **distance and relative position to each other must stay stable.** They cannot drift
> independently or slide along the eyelids.

**The app must NOT re-detect canthi loosely each frame. It must:**
1. Mark/approve medial + lateral canthus **once.**
2. Track them as a **RIGID PAIR.**
3. Preserve: **canthus-to-canthus distance, canthus-line direction, midpoint relationship, eye-local scale.**
4. Allow only **global** motion of the whole pair: translation, small rotation, and *very small* scale change
   only if the video actually zooms.
5. **Reject independent canthus jumps.** If the tracker cannot preserve the rigid pair, **carry forward the
   previous canthus frame and mark low confidence.**

**Implementation:** parameterise the pair by **midpoint + angle + inter-canthus distance** (not the two points
separately). Reconstruct canthi = midpoint ± (dist/2)·(cos θ, sin θ) so the pair is rigid BY CONSTRUCTION;
distance/angle are heavily smoothed (slow zoom OK, jitter rejected). **Diagnostic must plot inter-canthus
distance and canthus angle over time** — the eye may move *inside* the frame, but the canthus anchors must not
wander (distance ≈ flat, angle ≈ flat).

### Clinician-Marked Canthus Lock (Dr. K, 2026-07-04) — TRACK the marked texture, don't interpolate
*Treat the clinician's canthus marks as FIXED anatomical anchors, not points to rediscover freely. In the
video image the canthi may move (head/camera); in the eye-local frame they must not move relative to each
other.*
> Once the clinician marks medial + lateral canthus, those two points define a **rigid eye frame**. The
> tracker may move the **entire frame** with the face/head/camera, but it may **not change the canthus
> distance or let one canthus drift independently.**

**Enforcement (`tools/orbit_lock.py`):** (1) **template-track small patches** around each canthus (local
skin/canthus texture, optical flow) frame-to-frame; (2) **rigid-pair constraint** — solve ONE shared
transform (translation + small rotation + optional tiny scale) for both canthi, never independent motion;
(3) **distance lock** — inter-canthus distance ~constant, reject updates beyond a small tolerance; (4)
**angle smoothing** — canthus-line angle changes smoothly, never jumps; (5) **carry-forward on failure** —
if tracking confidence drops, freeze/keep the previous canthus frame rather than invent new points; (6)
**manual-correction checkpoints** — clinician can correct the canthi on a frame and restart the rigid lock
from there. The ~20 clinician marks per eye are the re-anchor points that correct accumulated drift.

### Absolute Intercanthus Distance Lock (Dr. K, 2026-07-04) — HARD constraint
> The distance between medial and lateral canthus is **fixed**. Once marked, it must remain **constant** for
> that eye throughout the clip. The tracker may **translate or rotate** the canthus pair with the head/camera,
> but it may **not stretch, shrink, or let either canthus move independently.**

**Allowed:** whole pair shifts together · whole pair rotates slightly. **NOT allowed:** distance changes ·
one canthus slides while the other stays · scaling · stretching · re-detecting a canthus independently.

**Implementation:** store `D_canthi` (median of the clinician marks) ONCE per eye. Every frame estimates only
rigid **translation** (optical flow of the midpoint) + **rotation** (smoothed mark angle) — **no scale.**
Canthi are reconstructed **CL/CR = midpoint ± (D_canthi/2)·(cos θ, sin θ)**, so the pair *cannot* stretch or
slide independently by construction; any flow that would change the distance is projected back onto the
fixed-distance pair, and implausible jumps freeze the frame. Verified: inter-canthus distance spread = **0.0%**
across all 640 frames (568px L, 575px R). This rigid canthus frame is the head-fixed reference for iris motion.

## DEFINITION OF NYSTAGMUS (operational, Dr. K, 2026-07-05) — the anchor for everything

> **Nystagmus = a JERK pattern — slow drift one way + brief faster jerk the other way — repeating AT LEAST 3
> TIMES IN SUCCESSION, IN THE SAME DIRECTION** (consistent fast-phase direction).

- **Fewer than 3 beats in succession → NOT nystagmus** (a stray drift or single flick doesn't count).
- **Inconsistent / changing direction within the run → NOT nystagmus** (irregular movement, saccadic
  intrusions, voluntary gaze shifts are excluded).
- Named by the **fast-phase** direction (left-beating = fast phase to the left). Slow phase is the opposite.
- Scope: **jerk nystagmus only.** (Pendular is out of scope.)

**How the app implements this definition:** at 30 fps the individual fast jerks are *undersampled* (they fall
between frames), so we cannot literally count the ≥3 beats — instead we detect the **equivalent sustained
signature**: a repeated, same-direction **slow-phase asymmetry** (slow drift consistently one way + brief
resets the other), measured by the windowed slow-phase asymmetry detector. That asymmetry is the proxy for
"≥3 consecutive same-direction jerks." At **higher fps (60/120+)** the jerks resolve and the app can count the
≥3 beats directly (and report beat rate). Either way, the definition above is the target.

## CLINICAL OBJECTIVE — a qualitative clinical NYSTAGMUS READER (Dr. K, 2026-07-04)

> The app is **NOT** trying to produce a precise VNG waveform. It is trying to answer the **same first-pass
> clinical questions a neurologist asks while watching the eyes.** That is a far more achievable and useful
> product.

**Core outputs (the whole product spec):**
1. **Nystagmus present or absent.**
2. **If present, direction:** left-beating · right-beating · up-beating · down-beating · mixed/oblique if needed.
3. **Gaze condition:** primary gaze · left gaze · right gaze · extreme gaze (if relevant).
4. **Gaze effect:** stronger on left gaze · stronger on right gaze · present in primary only · suppressed in one
   direction · direction-changing.
5. **Confidence:** clear · probable · unclear / insufficient data.

**Explicitly NOT in scope:** exact velocity, beat rate, degrees, a perfect trace, or any precise iris outline —
a rough eye-position signal (the EllSeg centroid) is enough. **Pipeline mapping:** present/absent + confidence
→ (1),(5); slow-phase asymmetry direction → (2); sustained sclera-balance gaze zones → (3); comparing
strength/confidence across primary vs left vs right zones → (4). The confidence bands (clear/probable/unclear)
are what the pending **multi-clip threshold calibration** will set.

## GOAL REFRAME — Clinical Nystagmus Direction Rule (Dr. K, 2026-07-04) — the new primary output
*Stop chasing a perfect VNG-grade continuous trace / perfect iris boundary. Aim for the clinically useful
answer instead.*
> The primary output is **NOT** a perfect VNG trace or a pixel-perfect iris mask. The primary output is
> **whether there is rhythmic jerk nystagmus and, if present, the fast-phase direction.**

**Why this is achievable now:** it needs only a **reasonably stable eye-position signal** (roughly attached to
the iris, smooth after filtering, not jumping to lid/skin), then a **pattern** on top: slow drift one way →
fast corrective jerk the other way → repeated rhythm → consistent direction. Crucially, **fast phases are
FAST and head/camera drift is SLOW**, so fast-jerk detection is inherently robust to the head-motion problem —
we do NOT need the perfect orbit lock for direction.

**Ignore:** isolated voluntary saccades · irregular non-rhythmic movement · blinks · low-confidence /
`needs_rescue` frames · gross head/camera motion (slow).

**Practical algorithm (`tools/nystagmus_direction.py`):**
1. Use the EllSeg / fixed-radius centre only as a **rough eye-position signal** (the existing marker's centre
   trace is enough if it's reasonably stable).
2. **Smooth** to remove shimmer; **detrend** slow head/camera drift (high-pass) so only eye movement remains.
3. **Detect fast jumps** (fast phases) in the trace by velocity peaks.
4. Check whether the jumps **repeat rhythmically** (regular inter-jerk intervals).
5. Check whether their **direction is consistent.**
6. If repeated + consistent → report **right- / left- / up- / down- / oblique-beating.**
7. If irregular → **irregular / voluntary-like / no rhythmic nystagmus detected.**

**New standard:** *"Does the trace show rhythmic fast jerks with a consistent direction?"* — NOT *"does the red
mask perfectly match the iris boundary?"* **Test:** run the marker → plot horizontal + vertical iris position
over time → overlay detected fast phases as arrows → judge: rhythmic? consistent direction? matches clinically?
irregular/voluntary rejected? (Ground truth for this clip: right vestibular neuritis → **left-beating**
horizontal nystagmus.) If yes, use this marker + build the classifier on top.

### Fast-Phase Acceptance Rule (Dr. K, 2026-07-04) — detect BEATS, not velocity spikes
*First detector over-called: it marked every sharp slope change as a jerk (both red and blue arrows) when the
clinical pattern is **slow rightward drift + fast leftward corrective jerks** (left-beating, unidirectional).
The fix: don't label every velocity spike a fast phase.*
> First establish the **dominant slow-phase direction** over a local window. Then accept only fast phases that
> are **brisk, conjugate in both eyes, and OPPOSITE to that slow phase.** Rightward candidate jerks are rejected
> unless they form their OWN repeated conjugate rhythm.

**A true fast phase must:** (1) be **much faster** than the surrounding slow drift; (2) occur in **both eyes at
nearly the same time**; (3) have the **same direction in both eyes**; (4) be **preceded/followed by slow drift
in the opposite direction**; (5) **repeat rhythmically**; (6) obey a **refractory interval** (one beat counted
once). **Reject reasons to log:** `wrong_direction`, `not_conjugate`, `too_small`, `not_rhythmic`,
`low_confidence`, `duplicate`. **Plot:** accepted beats red, rejected candidates grey. **Print the dominant
diagnosis:** left-beating / right-beating / vertical / no rhythmic nystagmus. (This clip → **left-beating**.)

### Windowed Slow-Phase Asymmetry Rule (Dr. K, 2026-07-04) — the clinician-style LOW-FPS detector
*Do NOT conclude "undetectable at 30 fps" just because strict fast-phase detection finds few crisp spikes. A
human can SEE the nystagmus in this clip → the DIRECTION is there. 30 fps limits precise fast-phase VELOCITY,
but usually not DIRECTION. The fix: behave like a clinician — use context over a window, not individual
spikes.*
> At 30 fps, do NOT detect fast phases directly. **Detect the rhythmic slow-phase drift + reset pattern.**
> Report the fast-phase direction as **OPPOSITE the dominant slow-phase drift**, with a **confidence** — do
> not require a crisp sawtooth spike in every beat.

**Method:** on the **combined (both-eyes)** eye-velocity (keep the slow-phase drift — only remove head/camera
drift slower than ~3 s), measure **motion asymmetry** over the clip and in sliding windows: (a) **time
asymmetry** — the eye spends MORE frames drifting one way (slow phase); (b) **speed asymmetry** — the opposite
direction is FASTER but briefer (fast phase); (c) **velocity skew** — the rare high-velocity tail points to the
fast phase. When these **agree**, the fast phase = that direction; **confidence** = agreement × strength ×
**cross-window consistency**. Pick the axis (H/V) with the stronger, more consistent asymmetry; if below
threshold → "no clear directional nystagmus." (Higher fps 60/120 is better for rate/velocity, but must NOT be
required for a directional answer from a visibly-positive 30 fps clip.) *Result (2026-07-04, ellseg_centroid,
30 fps): horizontal **LEFT-beating** correctly dominant over vertical (conf 0.21 vs 0.06) — direction IS
recoverable at 30 fps; confidence modest due to residual head/gaze motion in the raw centroid.*

### Gaze-Zone Nystagmus Classification Rule (Dr. K, 2026-07-04) — classify by WHERE the eye looks
*Do NOT collapse the whole clip into one label. A gaze-evoked / direction-changing diagnosis depends on gaze
position (Alexander's law).*
> Divide the video into **primary, left-gaze, and right-gaze** segments based on iris position. Run the
> windowed slow-phase asymmetry detector **separately in each segment.** Report whether nystagmus is present
> and its direction in each gaze position.

**Method:** use the **smoothed horizontal iris position** (excursion from the clip's primary/median, both eyes,
beats removed) to bin frames into **primary / left / right** gaze zones. Run the asymmetry detector in each
zone; a zone needs **enough data** (≥ ~1.5 s) or report "not enough data." **Clinical summary table:**
`gaze zone | frames | direction (L/R/U/D-beating) | H-vs-V | confidence | rhythmic/irregular | enough data`.
E.g. left-beating in primary + left gaze but absent in right gaze = Alexander's-law spontaneous nystagmus;
direction that changes with gaze = gaze-evoked.

**FIRST ATTEMPT FAILED — image-frame zoning is head-motion-contaminated (2026-07-04):** binning by absolute
image position split a *"nystagmus at rest"* clip (patient in primary throughout) into false L/R gaze zones —
the "gaze" wander was head/camera motion, not gaze. Fix = the anatomical rule below.

### Sclera-Balance / Canthus-Proximity Gaze Zone Rule (Dr. K, 2026-07-04)
> Gaze zone is classified by **where the iris sits inside the eye-opening almond**, using **visible sclera
> balance and canthus proximity** — NOT from raw image-frame motion. (This is head/camera-motion invariant by
> construction: it measures the iris *within the eye's own anatomy*.)

1. **Primary / center:** iris near the middle of the almond; sclera present on **both sides, roughly
   balanced.** Small deviations up to ~15–20° still count as primary.
2. **Looking right:** both irises shift toward the patient's right; sclera becomes more visible on the
   opposite side; iris clearly off-centre but **not at the canthus.**
3. **Looking left:** mirror of (2) toward the patient's left.
4. **Extreme right / extreme left:** iris **near or partly hidden by the canthus**, may be a narrow
   oval/sliver → **label separately** (tracking + nystagmus interpretation less reliable).

**Implementation:** for each eye estimate iris position within the almond from **relative sclera width on the
two sides of the iris** + **proximity to medial/lateral canthus**; combine both eyes → one gaze zone per
frame (primary / left / right / extreme-left / extreme-right); if eyes disagree or confidence low → **uncertain**.
Then run the asymmetry detector per zone. **Output ONLY a table:** `zone | time | nystagmus direction |
confidence | notes`.

### Sustained Gaze Zone Rule (Dr. K, 2026-07-04) — don't let the nystagmus move the gaze zone
*Frame-by-frame zoning was fooled by the nystagmus itself: in left-beating nystagmus the eye spends most of
each beat drifting rightward (slow phase), so a per-frame sclera-balance binned an at-rest clip as "right
gaze." The nystagmus was split across zones by its own drift.*
> Gaze zone is **NOT assigned frame by frame** from the instantaneous iris position. It is assigned over a
> **sustained time window.** Nystagmus oscillations within that window must **not** move the patient between
> primary/left/right zones.

**Method:** take the sclera-balance signal, **suppress within-beat oscillation** (smooth over a multi-second
window, ~2 s) to get the **sustained/held** eye position; reference it to the clip's **habitual baseline**
(median sustained position = primary); classify the **held** gaze from that (primary / left / right /
extreme). Then run the nystagmus detector inside each **sustained gaze segment.** For a spontaneous-nystagmus
at-rest clip the patient is clinically in **primary gaze throughout** (even though the iris drifts right in
the slow phase) → the whole clip = **primary gaze, left-beating.**

### Nose-Bridge Anchor + Mostly-Rigid Nasal Canthus Frame (Dr. K, 2026-07-04)
*The eye-region texture (lids/lashes/skin) is NOT rigid — it moves with blinks/lids — so the flow-tracked
midpoint drifts and the medial canthus wanders (seen at f12/f591). Anchor to the NOSE BRIDGE instead: rigid
bone/skin that does not move with gaze, lids, or blinks, and is easy to track.*
> The nasal-canthus geometry is anatomically fixed, but its 2D projection may change slightly with head yaw.
> Keep the triangle **rigid for ordinary frames**, but allow small smooth perspective changes when the whole
> head turns. **Do not allow sudden independent drift.**

- **Track the nose bridge** (stable central landmark, shared by both eyes = one rigid skull) and pin each eye's
  **medial canthus at a fixed offset from the bridge** (rotated by the small head angle). Lateral canthus =
  medial + `D_canthi` along the axis. One stable bridge point + fixed D_canthi + small angle fully determines
  the 4-point frame; nothing floats.
- **Perspective note:** under head **yaw** the 2D triangle genuinely changes (far side shorter, bridge shifts
  vs canthi, line tilts/compresses). So a pure 2D no-scale triangle is best only when yaw is small.
- **Staging (Dr. K):** (1) **NOW — strict no-scale rigid nasal-canthus frame** (best for stabilizing this
  mostly-frontal video, prevents drift); (2) later — 2D similarity/affine face transform from more landmarks
  for head pose; (3) large yaw — 3D face model (overkill now). **At this checkpoint prioritize preventing
  medial-canthus drift over modeling yaw.**

**Support must be EllSeg-disc / limbus evidence — NOT generic Otsu dark** (Dr. K, 2026-07-04). Lid shadow and
skin shadow count as "dark", so a circle drawn **below the eyelid / on skin** can score high against a dark
mask (this wrongly passed f220). Measure support against the **EllSeg iris disc** and the **true medial/lateral
sclera-facing limbus**, never against all dark pixels.

**A frame is `fixed_circle_ok` ONLY IF BOTH hold:**
1. the proposed circle **stays near the EllSeg iris anchor** (centre-to-anchor distance small) **AND overlaps
   the EllSeg iris disc sufficiently** (`support ≥ SUPPORT_MIN`); **and**
2. it has a **valid medial/lateral sclera-facing limbus arc** (a real contiguous arc, per the Medial/Lateral
   Limbus Arc Rule).

**If EITHER fails → `needs_rescue`** (do not draw a confident circle; carry forward the last reliable
centre/radius). At extreme side gaze the disc is a sliver and the rigid circle drifts off it → both the
anchor-distance and disc-overlap conditions fail → withheld. Diagnostic is regenerated with a per-frame
**`fixed_circle_ok` vs `needs_rescue`** label; the centre trace uses the carried-forward centre on rescued
frames (so no spike).
