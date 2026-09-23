# Where the videos and the marked data live

The nystagmus clips are **not in this repository** and never should be. They are
patient video, and this repo is public.

**Backed-up copy (the one that matters):**

```
C:\Users\sudhi\OneDrive\Documents\AI apps made by me\VNG-EYE app videos\
```

28 files, ~369 MB. That folder sits in OneDrive, so it is cloud-backed, and it is
deliberately **outside** any git repository — nothing here can ever pick it up.

**Working copy:** `samples/` in this folder holds the same 28 files so the tools can
run without reaching across to OneDrive. It is gitignored. Treat it as disposable —
if it is ever lost, copy it back from the OneDrive folder above.

## Hand-marked ground truth and analysis results

```
C:\Users\sudhi\OneDrive\Documents\AI apps made by me\VNG-EYE app analysis data\
```

315 files, 25 MB — every `.json`, `.csv`, `.md`, `.log` and `.html` that was under the
old `outputs/` tree, with the folder structure preserved.

This is the part of `outputs/` that is **not** regenerable. In particular
`nystagmus at rest to left in right vestibular neuritis_tracked/RIT_ground_truth/polygons.json`
holds 39 hand-marked iris polygons — re-creating it means marking frames by hand again.
Alongside it are `meta.json`, `evaluation.json`, `learned_evaluation.json`, the
`MARKING_INSTRUCTIONS.md` files, and the `orbit_lock_points.csv` probe outputs.

What was left behind was 7.9 GB of regenerable render output: mask and overlay PNGs,
review MP4s, and a 228 MB trained `model_rf.joblib` that retrains from `polygons.json`.

## Rules

- `samples/`, `*.mpg`, `*.MPG`, `*.mp4`, `*.avi` and `*.mov` are all gitignored.
  Check `.gitignore` before adding any new clip format.
- New clips go to the OneDrive folder **first**, then get copied into `samples/`.
  That way the backed-up copy is never the one that only exists on this laptop.
- Never `git add -f` anything under `samples/`.

## Copying between the two

Use robocopy, not `Copy-Item -Recurse` — this project has pnpm junctions under
`node_modules` that make Copy-Item follow the links and explode the tree.

```
robocopy "C:\Users\sudhi\OneDrive\Documents\AI apps made by me\VNG-EYE app videos" "C:\dev\vng-eye-app\samples" /E /XJ
```
