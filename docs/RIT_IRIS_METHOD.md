# RIT — How to mark the iris (Dr. Kothari's method, clean rebuild 2026-06-30)

*Supersedes the earlier ad-hoc iris fitting. This is the method dictated step by step by Dr. Kothari.
Build it ONE step at a time; verify each step visually on the real frames before adding the next.*

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
