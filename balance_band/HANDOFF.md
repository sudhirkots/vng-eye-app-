# Balance Band — handover (2026-09-23)

**Separate project from the nystagmus detector**; it only lives in this repository for now.
Branch: `claude/balance-angular-posturography-pae9k5` (not merged). Start with `README.md` in this folder.

## What it is
A motion sensor (IMU) worn on a belt over the lower back, with software that runs a five-step standing
balance test and gives a qualitative read. For anyone — healthy people, people who feel unsteady, any patient.

1. **Limits of stability:** stand still (the centre), then lean as far as possible forward, backward, left,
   right without stepping.
2. **Eyes open, firm floor** — the reference.
3. **Eyes closed, firm floor** — how much vision matters.
4. **Eyes open, on foam** — how much somatosensory input matters.
5. **Eyes closed, on foam** — balance on mainly vestibular input.

Report: the four limits (flags a reduced backward limit and left–right asymmetry); sway in each condition as
× the reference and as % of the person's own limit; a stability score /100 per condition and the three
**sensory ratios** — somatosensory, visual, vestibular, as % of eyes-open-firm — with any LOW one named under
**"Focus for rehabilitation"**; the verdict (WEAK … USE / NO WEAK SENSE / UNSTEADY EVEN WITH ALL SENSES);
falls with time and direction. Poor or missing data → INSUFFICIENT / UNCLEAR, never "normal".

## Decisions made (and why)
| Decision | Reason |
|----------|--------|
| **Waist (L3–L5), not the ankle/shin** | Close to the centre of gravity; a shin band under-reads sway once the person switches to hip strategy (foam, near the limits). Validated trunk-sway systems use the lower back. |
| **One band now; a second shin band later** | Waist alone cannot tell ankle from hip strategy. Deferred. |
| **No tremor analysis** | Not seen at the waist. Tremor will be a separate wrist-band project. |
| **Sensory ratios (SOT method) drive the verdict** | Dr K wants a % per sense to target rehabilitation. The ratios name the sense used *poorly* (the one to train); the older "× reference" wording named the sense relied on, which read as the opposite. Each sense has its own cut-off since eyes-closed foam is hardest even for healthy people. Over-reliance on vision (visual preference) is not measurable without a moving visual scene. |
| **Band can be strapped on any way round** | Orientation is found from step 1: "down" from the quiet stance, "forward" from the forward lean; the other leans are checked. |
| **Board: Seeed XIAO nRF52840 Sense** | Built-in IMU + Bluetooth + LiPo charger; sold in India (~₹2,300). Research sensors (Movesense, MbientLab, Movella DOT) are not stocked in India. |
| **Case design not critical** | Dr K: "I can just put it in a belt and strap it." The draft case files stay but are optional. |
| **Recording app = Chrome/Edge page + small local Python server** | Chrome talks Bluetooth directly (Web Bluetooth); Python saves the files and runs the analysis. |
| **Patient data outside the repo** | Recordings go to `~/BalanceBand/sessions/`; person entered as an ID code, not a name. |

## Files
| Path | What |
|------|------|
| `sway_analysis.py` | The analysis (+ demo generator: `--demo normal/weak_vestibular/weak_somatosensory/weak_visual/backward`, `--demo-tilted`) |
| `recorder/server.py`, `recorder/static/` | Recording app (page, Bluetooth decoder, simulated band) |
| `Start_Recorder.bat` | Windows launcher for the recorder |
| `firmware/balance_band_fw/balance_band_fw.ino` | Band firmware; setup steps in `firmware/README.md` |
| `hardware/` | Parts list (`BOM.md`), case brief, draft `case.scad` — optional now |
| Tests | `test_sway_analysis.py` (22), `recorder/test_server.py` (3), `recorder/test_protocol.js` (4) — all pass |

## What is verified, and what is not
- **Verified:** analysis on made-up recordings; automatic orientation (straight vs randomly tilted band gives
  the same result); server; Bluetooth frame decoding; a full five-step session end-to-end in headless Chrome
  with the **simulated** band.
- **NOT verified:** the firmware has **never been compiled or run on a board**; Web Bluetooth with a real band;
  anything on a real person. **All thresholds are PROVISIONAL placeholders** (sensory-ratio cut-offs
  80 / 70 / 50 %, 70 % of limit, backward < 50 % of forward, side < 60 %; ×2 / ×4 only without step 1).

## Next steps
1. Buy the XIAO nRF52840 Sense (two, so a future shin unit matches), a protected 3.7 V LiPo (~400 mAh), a
   slide switch, a 40 mm non-stretch belt, and the foam pad.
2. Load the firmware (Arduino IDE, "Seeed nRF52 Boards" — not mbed — + "Seeed Arduino LSM6DS3"). Expect small
   compile fixes. Check the LED goes green-blinking, then blue once connected.
3. Run `Start_Recorder.bat`, **Connect band**, confirm ~100 samples/s and no losses on the page.
4. Record a session on yourself; check the limits and the report make sense.
5. Record healthy age-matched controls → replace the provisional thresholds with real limits; add absolute
   grading.
6. Record people who feel unsteady and known groups (vestibular loss, neuropathy, cerebellar, parkinsonism ON
   and OFF, older fallers) → check each gives the expected pattern.
7. Later: second (shin) band for ankle vs hip strategy; battery level in the firmware.

## Environment notes
- Needs Python 3 + numpy for the server/analysis (works with the project's numpy<2 environment); no other
  Python packages. Node is only for `recorder/test_protocol.js`.
- Browser: Chrome or Edge (Web Bluetooth). Firefox and Safari cannot connect to the band.
