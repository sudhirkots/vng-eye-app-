# Balance Band — case design brief (for the final design)

The starting geometry is `case.scad` (parametric, first draft, not yet rendered or printed). Parts are in
`BOM.md`. This brief says what the final design must do and what is free to change.

## What the device is
A small sensor pod worn on a belt over the **lower back (L3–L5)**. It measures how far the body sways, in
degrees, while the person stands still with feet together. Any wobble of the pod on the body shows up as
false sway, so **rigidity and a fixed position on the belt matter more than anything else.**

Used on anyone: healthy people, people who feel unsteady, and patients (older people, neuropathy,
vestibular, parkinsonism). Worn over thin clothing, in a clinic, for about 15 minutes per session.

## Frame (seen from behind the person)
- **+X** along the belt, toward the person's RIGHT (the person's LEFT is −X)
- **+Y** UP
- **+Z** away from the body; z = 0 is the back plate that touches the person

## Must have (fixed requirements)
1. **Belt tunnel.** The belt (40 mm non-stretch webbing) passes *through* the pod, not over a clip, so the
   pod can neither slide along the belt nor rotate. Tunnel height = belt width + small clearance.
2. **Board flat and rigid.** The XIAO board (21 × 17.8 mm) lies flat on the floor of its bay, parallel to
   the back plate, held by a locating frame plus foam tape. No flex, no rattle.
3. **Battery bay** for a 502535 LiPo (≈ 35 × 25 × 5.5 mm incl. protection board), padded, fully enclosed.
4. **USB-C access** for charging, through the **top** wall (+Y).
5. **Power switch** reachable from the side (−X wall).
6. **Status LED visible** through a small window in the lid.
7. **Orientation marks on the outer face:** an **"UP" arrow** (+Y) and an **"L" arrow** pointing to the
   person's left (−X), so it is always worn the same way round. The software is told the sensor axes once;
   the marks keep them true.
8. **Closed with screws** (2 × M2 self-tapping), so it opens for battery replacement but cannot open by itself.
9. **Skin-side comfort:** the back plate is smooth, with rounded edges and corners; nothing sharp.
10. **Cleanable:** smooth surfaces that can be wiped with alcohol between patients; no deep crevices.
11. **Printable in PETG** on an ordinary FDM printer.

## Free to change
- The overall shape and styling (the draft is a plain box).
- A gentle curve on the back plate to fit the lower back (but keep the board flat inside).
- Snap-fit instead of screws, *if* it stays secure and can still be opened.
- Wall thicknesses, fillets, colour, a name or logo on the lid.
- Where the switch and LED sit, if the parts still fit.

## Dimensions in the draft
| Item | Value |
|------|-------|
| Body, outer (X × Y) | ≈ 39.8 × 56.2 mm |
| Total depth incl. back plate, belt tunnel and lid | ≈ 16.1 mm |
| Belt tunnel | 40.8 mm (Y) × 4 mm (Z), open at both X ends |
| Wall / floor / lid | 2 / 1.6 / 1.6 mm |
| USB-C opening | 10 × 4.5 mm in the top wall, centred on the board |
| Switch slot | 9 × 4 mm in the left wall, level with the battery |

## Watch points for printing
- The body's floor bridges across the belt tunnel (~41 mm span). Either print with supports under that
  bridge, or make the back plate a separate part that is screwed or glued on.
- Print the lid outer face down, so the engraved arrows come out clean.

## After the first print
Wear it, lean forward and to the left, and note which sensor axes change sign. Record them here and in the
software call (`--forward`, `--left`). They then stay fixed for every unit built to this design.
