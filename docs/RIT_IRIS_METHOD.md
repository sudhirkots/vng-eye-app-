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
