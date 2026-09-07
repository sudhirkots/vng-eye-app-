# HANDOFF — Parkinson's band (ON/OFF periods + dyskinesias)

**Date:** 2026-09-07  **Status:** PARKED — separate future project, not the Video Frenzel work.
No decisions locked, nothing ordered, no code written.

Kept only so the reasoning is not lost. The active hardware project is `HANDOFF_VIDEO_FRENZEL_IMU.md`.

## Goal

A qualitative wearable read of Parkinson's motor fluctuations: ON periods, OFF periods, and dyskinesias
across the day, plotted against dose times — to time medication, not to measure it precisely.

## Recommended approach — do NOT build hardware first

Buy a raw-data wrist logger, write the algorithm in Python, validate against a levodopa challenge. Custom
hardware later, if ever.

| Tier | Option | Comment |
|---|---|---|
| 0 | Apple Watch (Movement Disorder API / StrivePD) | Already outputs per-minute tremor + dyskinesia likelihood. Stop here if only the clinical answer is wanted. |
| **1** | **Axivity AX6** (~£170) — RECOMMENDED | Wrist, accel+gyro, 50 Hz / ±8 g, ~2 weeks battery, records to onboard flash, `.cwa` over USB. No firmware, app or BLE. Read with `openmovement-python` / `actipy`. AX3 = accel-only cheaper sibling; GENEActiv / ActiGraph GT9X equivalent. |
| 2 | ESP32-C3 + LSM6DSO + microSD + LiPo (~₹1500) | Only after Tier 1 works. You then own firmware, charging, waterproofing and regulatory questions. |

Wear on the **more-affected wrist**. One band. Do not start bilateral.

## The core algorithmic insight

ON/OFF and dyskinesia separate out of a single wrist accelerometer because they occupy different places in
the same 2-minute epoch statistics:

| State | Signature at the wrist |
|---|---|
| **OFF / bradykinesia** | Long runs of *stillness*; low amplitude and low 0.5–3 Hz power when moving |
| **Dyskinesia** | Abnormally *continuous* movement — stillness runs disappear; broadband 1–3 Hz, irregular, multi-axis |
| **Rest tremor** | Narrowband 4–6 Hz peak, high peak-to-mean spectral ratio |
| **Normal ON** | Normal alternation of stillness and purposeful movement |

Essentially the published PKG logic (Griffiths et al., *J Parkinsons Dis* 2012): bradykinesia from excess
immobility, dyskinesia from the **absence** of immobility. No ML needed for v1 — thresholds first, then a
small logistic regression once labels exist.

## Algorithm sketch (per 2-min epoch, 50 Hz)

1. High-pass 0.2 Hz (remove gravity), band-pass 0.2–15 Hz, vector magnitude.
2. Features: **immobility fraction** (% of 2 s sub-windows below threshold → bradykinesia); **run-length
   distribution** of movement vs stillness (→ dyskinesia); power in 0.5–3 Hz and 3.5–7.5 Hz plus their ratio;
   dominant frequency and peak sharpness (narrow = tremor, broad = dyskinesia); spectral entropy; jerk; SMA.
3. Output per-epoch **BKS, DKS, tremor** → smooth over 30 min → state label.
4. Plot the day with dose times marked. **That chart is the clinical deliverable** — it is what changes the
   prescription.

**Two mandatory additions or the whole thing is garbage:**
- **Non-wear detection** (flat signal + no posture change) — else a watch on the bedside table reads as
  profound OFF.
- **Sleep detection** (immobility + gravity-vector posture + time of day) — same failure mode.

## Ground truth — the part that decides whether it works

- **Best value: in-clinic levodopa challenge.** Patient arrives in practically-defined OFF (overnight
  withdrawal), band recording. Give levodopa. Video throughout, with timestamped UPDRS-III and UDysRS every
  20–30 min through ON and peak-dose dyskinesia. ~4 h per patient yields densely, expertly labelled data
  across all three states. **10–15 patients** is enough for v1.
- **Then home data:** 5–7 days with a **Hauser diary** in 30-min bins (OFF / ON without dyskinesia / ON with
  non-troublesome dyskinesia / ON with troublesome dyskinesia / asleep).
- **Metrics:** sensitivity/specificity for OFF and for dyskinesia vs diary bins; Cohen's κ; correlation of
  daily %OFF with UPDRS-IV / WOQ-9.

## Pitfalls

- **Voluntary activity vs dyskinesia is the real confound** — walking, brushing teeth and a car ride all look
  dyskinetic to a naive detector. Partial fixes: reject epochs with a clean rhythmic gait signature (~2 Hz,
  strong harmonics, high periodicity); lean on dyskinesia clustering at predictable post-dose intervals.
- **Tremor-dominant patients in OFF** move a lot at the wrist — do not let tremor power read as "not
  bradykinetic". Score the bands separately.
- **Same honesty rule as the nystagmus detector:** when wear time or signal quality is poor, output
  **"insufficient"** — never "no OFF periods detected". A false "well controlled" is the dangerous error.
- Decision-support for medication timing, **not** a diagnostic. Ethics committee approval before any patient
  wears it; CDSCO only if productised.
- Storage: 50 Hz × 3 axes × 2 bytes ≈ 26 MB/day, ~180 MB/week — needs microSD or nightly offload if building
  custom hardware.

## Fastest concrete start

Buy one AX6, wear it yourself for two days, and write the epoch feature extractor against your own data. The
pipeline then works before the first patient is recruited, and your own recording is the control distribution
to threshold against.
