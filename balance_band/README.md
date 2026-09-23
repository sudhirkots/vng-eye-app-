# Balance Band — a waist-worn IMU to evaluate anyone's balance system

*Separate project from the nystagmus detector. It only lives in this repository for now.*

**Who it is for:** anyone. A healthy person (baseline, or as a control), someone who simply feels unsteady,
or any patient with a balance complaint: vestibular, peripheral neuropathy, cerebellar, parkinsonism, the
elderly faller, and so on. It tells us **which part of the balance system the person is leaning on, and
where it fails.**

## What it does
A belt over the **lower back** (L3–L5) carries an IMU (accelerometer + gyroscope). That is close to the
body's centre of gravity, so **the belt's tilt closely tracks the centre of gravity (COG) angle**, whether
the person corrects sway at the ankles or at the hips. The band records five short steps, and the software
turns them into a bedside read.

### Why the waist, not the ankle (decided 2026-09-23)
- **It measures what we care about.** The centre of gravity sits at about the lower lumbar level. The
  lower back is also where the validated trunk-sway systems (e.g. SwayStar) put their sensor, so there is
  published work to compare with.
- **A shin band fails exactly when it matters.** The shin tracks the centre of gravity only while the body
  sways as one rigid pendulum over the ankles. On foam and near the limits of stability people switch to a
  **hip strategy**: trunk one way, legs the other. A shin band then under-reads sway, just when the person
  is least stable, and can miss a backward trunk fall.
- **Easier to wear the same way every time.** A belt goes on at the same height, over clothes, and does not
  slip like a calf strap. Consistent placement is what lets us compare a person with themselves over time.
- **What the waist alone cannot tell:** whether the person corrected at the ankles or at the hips. That
  needs a **second band on the shin** (hip angle = trunk angle − leg angle). Planned for later; the strap
  and firmware should allow a second identical unit.

`--site waist` is the default. `--site shin` still works, with a warning that it under-reads hip strategy.

| Step | File | What the person does | What it tells us |
|------|------|-----------------------|------------------|
| 1 | `los.csv` | Stand still ~3 s (**centre**), then lean as far as possible **forward, back, left, right**, returning to centre each time, without stepping | **Limits of stability**: how far the COG can go in each direction |
| 2 | `eo_firm.csv` | Feet together, eyes open, firm floor, 20 s | Reference sway (vision + somatosensory + vestibular) |
| 3 | `ec_firm.csv` | Eyes closed, firm floor | How much **vision** matters |
| 4 | `eo_foam.csv` | Eyes open, on foam | How much **somatosensory** input matters |
| 5 | `ec_foam.csv` | Eyes closed, on foam | What balance is like on mainly **vestibular** input |

## What it reports
- **Limits of stability** in degrees, forward / backward / left / right. It flags a **reduced backward
  limit** (e.g. parkinsonism, older fallers) and **left–right asymmetry**.
- For each sensory condition:
  - **sway angle** (AP and ML);
  - **× reference**: how many times the eyes-open-firm sway;
  - **% of limit**: how much of the person's *own* limit of stability the sway used. A fall counts as 100 %.
- **Pattern**: vision-dependent / somatosensory-dependent / both / fails on vestibular input alone / no
  marked dependence.
- **Falls**: when the examiner pressed the button, and **which way** they fell.
- **Honest confidence**: a missing or poor recording gives **INSUFFICIENT / UNCLEAR**, never "normal".

## The device (hardware brief)
- 6-axis IMU. A 9-axis part with on-chip fusion is fine but not needed.
- Small microcontroller with Bluetooth LE, a rechargeable battery and an elastic strap.
- Sample at **≥ 100 Hz**. The software rejects < 20 Hz and warns below 50 Hz.
- **Examiner button**: press when the person loses balance, steps, opens the eyes or grabs. This writes
  `fall = 1` from that sample on.
- One CSV per step: `t, ax, ay, az, gx, gy, gz, fall`. Units: s, g (or m/s²), deg/s.
- Wear it as a belt with the sensor centred over the lower back (L3–L5), snug, over thin clothing.
  **Do not move it between the five steps**, because the centre from step 1 is the reference for all of them.

## Run it
```
python balance_band/sway_analysis.py <session_folder> --site waist --forward +x --left +y --json result.json
python balance_band/sway_analysis.py <empty_folder> --demo backward       # synthetic example
python balance_band/test_sway_analysis.py                                  # 17 tests
```
`--forward` / `--left` say which sensor axis points to the person's front and to the person's left.
Needs only Python + numpy.

## Honest limits
- **Thresholds are PROVISIONAL**, until we record healthy controls on this band:
  - × 2 for losing one sense, × 4 for eyes closed on foam;
  - 70 % of the limit counts as near the edge;
  - backward limit < 50 % of forward counts as reduced;
  - one side < 60 % of the other counts as asymmetric.
- **Waist alone does not tell ankle from hip strategy.** It gives the centre of gravity's tilt, not how the
  person produced it. A second, shin band will add that.
- **Foam degrades, it does not abolish, foot sensation.** Eyes closed on foam leans *mainly* on vestibular
  input, not purely.
- The limits of stability are *voluntary* leans. Someone who is frightened, slow or in pain may not lean
  fully, so a small limit can mean "would not" as well as "could not".

## Next steps
1. Build the waist band, confirm its axis directions and sample rate, and record a session on myself.
2. Record healthy age-matched controls → replace the provisional thresholds with real limits.
3. Record people who feel unsteady and known patient groups (vestibular loss, neuropathy, cerebellar,
   parkinsonism ON and OFF medication, older fallers) → check that each gives the expected pattern.
4. Add the second (shin) band: synchronise the two units and report ankle vs hip strategy.
