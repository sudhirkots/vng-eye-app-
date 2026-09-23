# Where the videos and the marked data live

Neither the patient video nor the hand-marked data is in this repository, and
neither ever should be — this repo is public.

Both live together in one OneDrive folder, deliberately **outside** any git
repository, so nothing here can pick them up:

```
C:\Users\sudhi\OneDrive\Documents\AI apps made by me\VNG iApp files\
├── videos\           28 files, ~369 MB
└── analysis data\    315 files, ~25 MB
```

## videos\

The 28 nystagmus clips. `samples/` in this repo holds the same 28 files so the
tools can run without reaching across to OneDrive. `samples/` is gitignored —
treat it as disposable; if it is ever lost, copy it back from `videos\`.

## analysis data\

Every `.json`, `.csv`, `.md`, `.log` and `.html` that was under the old `outputs/`
tree, with the folder structure preserved.

This is the part of `outputs/` that is **not** regenerable. In particular
`nystagmus at rest to left in right vestibular neuritis_tracked/RIT_ground_truth/polygons.json`
holds 39 hand-marked iris polygons — re-creating it means marking frames by hand
again. Alongside it are `meta.json`, `evaluation.json`, `learned_evaluation.json`,
the `MARKING_INSTRUCTIONS.md` files, and the `orbit_lock_points.csv` probe outputs.

What was left behind was 7.9 GB of regenerable render output: mask and overlay
PNGs, review MP4s, and a 228 MB trained `model_rf.joblib` that retrains from
`polygons.json`.

## Rules

- `samples/`, `*.mpg`, `*.MPG`, `*.mp4`, `*.avi` and `*.mov` are all gitignored.
  Check `.gitignore` before adding any new clip format.
- New clips go to `videos\` **first**, then get copied into `samples/`. That way
  the backed-up copy is never the one that only exists on this laptop.
- New ground-truth marking gets copied out to `analysis data\` when it is done.
- Never `git add -f` anything under `samples/`.

## Copying the videos back into this repo

Use robocopy, not `Copy-Item -Recurse` — this project has pnpm junctions under
`node_modules` that make Copy-Item follow the links and explode the tree.

```
robocopy "C:\Users\sudhi\OneDrive\Documents\AI apps made by me\VNG iApp files\videos" "C:\dev\vng-eye-app\samples" /E /XJ
```
