# Balance Band — parts list (one waist unit)

Prices are rough guides. "Checked" = seen on an Indian listing on 2026-09-23; confirm stock and price with the
seller before ordering.

| # | Part | Specification | Qty | Where (India) | Approx. cost |
|---|------|---------------|-----|---------------|--------------|
| 1 | Main board | **Seeed Studio XIAO nRF52840 Sense**. Built-in 6-axis IMU, Bluetooth LE, LiPo charger on board, USB-C. 21 × 17.8 mm | 1 | The Engineer Store, Amazon.in, Fab.to.Lab, Compo India, element14 India | ~₹2,300 (checked) |
| 2 | Battery | **3.7 V single-cell LiPo, 300–500 mAh, WITH protection circuit (PCM)**, e.g. size 502535 (~5 × 25 × 35 mm, ~400 mAh). Wire leads, no connector needed | 1 | Robu.in, Amazon.in, local electronics shops | a few hundred ₹ (not checked) |
| 3 | Power switch | Miniature slide switch, e.g. SS12D00 (SPDT, ~8.5 × 3.5 mm body), in the battery + lead | 1 | Robu.in, local shops | a few ₹ |
| 4 | Screws | M2 × 6 mm self-tapping (pan head), to close the lid | 2 | Local hardware / electronics shops | a few ₹ |
| 5 | Mounting | Thin double-sided foam tape (board and battery), thin silicone wire | — | Local | a few ₹ |
| 6 | Case | 3D-printed, **PETG**, from `case.scad` (after the final design) | 1 | Local 3D-printing service | ~₹200–500 (estimate) |
| 7 | Belt | **40 mm non-stretch webbing** (nylon or polypropylene) with a Velcro or buckle closure, long enough for the largest waist | 1 | Local tailors / webbing sellers; or a ready-made 40 mm belt | ~₹100–300 (estimate) |

**Per unit, roughly ₹3,000–3,500**, most of it the XIAO board.

## Needed once, not per unit
| Part | Notes |
|------|-------|
| **Foam pad** | Standard medium-density balance pad, e.g. **Airex Balance Pad** (~₹14,000 on IndiaMART, checked). Use the **same pad** for every person; foam softness changes the results. |
| **Laptop or phone** | Runs the recording app (start/stop, condition name, the examiner's "lost balance" button, CSV export). |
| **USB-C cable** | For charging. |

## Optional upgrade
| Part | When |
|------|------|
| **ICM-42688-P** IMU on a breakout board | Only if the XIAO's built-in IMU proves too noisy for the small sway angles. Chip on Robu.in; breakout boards via Desertcart India, Tindie, Elecrow. |

## Second unit (later)
A second identical unit for the **shin**, to separate ankle from hip strategy. Buy two XIAO boards at the
start so the pair are the same.
